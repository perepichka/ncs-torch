"""Neural network model for cloth simulation (NeuralClothSim PyTorch port).

Disentangles cloth deformation into two subspaces:
  - Static  : pose-dependent displacement (from current pose alone)
  - Dynamic : momentum/inertia displacement (from change between poses)

Reference: Bertiche et al., "Neural Cloth Simulation" (NeurIPS 2022).
"""

import torch
import torch.nn as nn


def _mlp_block(in_features: int, out_features: int) -> nn.Sequential:
    """Linear -> LayerNorm -> ReLU block (used in hidden layers)."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.ReLU(inplace=True),
    )


class PoseEncoder(nn.Module):
    """Encodes a batch of joint rotation matrices to a latent vector.

    Input:  (B, J, 3, 3) rotation matrices
    Output: (B, latent_dim)

    Architecture: flatten -> Linear(J*9, 256) -> LN -> ReLU
                             -> Linear(256, 256)  -> LN -> ReLU
                             -> Linear(256, latent_dim)
    """

    def __init__(self, n_joints: int, latent_dim: int = 256) -> None:
        super().__init__()
        in_dim = n_joints * 9  # flattened rotation matrices

        self.net = nn.Sequential(
            _mlp_block(in_dim, 256),
            _mlp_block(256, 256),
            nn.Linear(256, latent_dim),  # output projection — no LN/activation
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: (B, J, 3, 3)

        Returns:
            latent: (B, latent_dim)
        """
        B = pose.shape[0]
        x = pose.reshape(B, -1)   # (B, J*9)
        return self.net(x)        # (B, latent_dim)


class StaticDecoder(nn.Module):
    """Decodes a latent pose vector to per-vertex static displacement.

    Input:  (B, latent_dim)
    Output: (B, V, 3)

    Architecture: Linear(latent_dim, 512) -> LN -> ReLU
                  -> Linear(512, 512)     -> LN -> ReLU
                  -> Linear(512, V*3)     (output layer, no LN/activation)
                  -> reshape to (B, V, 3)
    """

    def __init__(self, n_verts: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.n_verts = n_verts

        self.net = nn.Sequential(
            _mlp_block(latent_dim, 512),
            _mlp_block(512, 512),
            nn.Linear(512, n_verts * 3),  # output layer
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim)

        Returns:
            disp: (B, V, 3)
        """
        B = latent.shape[0]
        out = self.net(latent)                   # (B, V*3)
        return out.reshape(B, self.n_verts, 3)   # (B, V, 3)


class DynamicDecoder(nn.Module):
    """Decodes concatenated current and previous latents to per-vertex dynamic displacement.

    Captures momentum / inertia: the difference in motion between frames.

    Input:  latent_t (B, latent_dim), latent_prev (B, latent_dim)
    Output: (B, V, 3)

    Architecture: concat -> Linear(2*latent_dim, 512) -> LN -> ReLU
                            -> Linear(512, 512)        -> LN -> ReLU
                            -> Linear(512, V*3)        (output layer)
                            -> reshape to (B, V, 3)
    """

    def __init__(self, n_verts: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.n_verts = n_verts

        self.net = nn.Sequential(
            _mlp_block(latent_dim * 2, 512),
            _mlp_block(512, 512),
            nn.Linear(512, n_verts * 3),  # output layer
        )

    def forward(
        self, latent_t: torch.Tensor, latent_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latent_t:    (B, latent_dim)  — current-frame encoding
            latent_prev: (B, latent_dim)  — previous-frame encoding

        Returns:
            disp: (B, V, 3)
        """
        B = latent_t.shape[0]
        x = torch.cat([latent_t, latent_prev], dim=-1)  # (B, 2*latent_dim)
        out = self.net(x)                                # (B, V*3)
        return out.reshape(B, self.n_verts, 3)           # (B, V, 3)


class NCSModel(nn.Module):
    """Full NeuralClothSim model combining static + dynamic subspaces.

    Given the current and previous body pose the model predicts cloth vertex
    positions by:
      1. Encoding both poses to latent vectors.
      2. Decoding a *static* displacement from the current latent (pose shape).
      3. Decoding a *dynamic* displacement from both latents (inertia/momentum).
      4. Adding both displacements to the rest-pose template vertices.

    Args:
        n_joints   : number of skeleton joints (J)
        n_verts    : number of cloth vertices (V)
        latent_dim : dimension of the shared pose latent space (default 256)
    """

    def __init__(
        self,
        n_joints: int,
        n_verts: int,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.n_verts = n_verts
        self.latent_dim = latent_dim

        self.pose_encoder = PoseEncoder(n_joints, latent_dim)
        self.static_decoder = StaticDecoder(n_verts, latent_dim)
        self.dynamic_decoder = DynamicDecoder(n_verts, latent_dim)

    def forward(
        self,
        pose_t: torch.Tensor,
        pose_prev: torch.Tensor,
        template_verts: torch.Tensor,
    ) -> torch.Tensor:
        """Predict deformed cloth vertex positions.

        Args:
            pose_t        : (B, J, 3, 3)  current-frame rotation matrices
            pose_prev     : (B, J, 3, 3)  previous-frame rotation matrices
                            (pass zeros for the first frame of a sequence)
            template_verts: (V, 3)        rest-pose cloth vertices

        Returns:
            deformed_verts: (B, V, 3)     cloth vertices in posed space
        """
        # Encode poses to latent space
        latent_t = self.pose_encoder(pose_t)        # (B, latent_dim)
        latent_prev = self.pose_encoder(pose_prev)  # (B, latent_dim)

        # Static subspace: shape driven by current pose
        static_disp = self.static_decoder(latent_t)              # (B, V, 3)

        # Dynamic subspace: inertia driven by pose change
        dynamic_disp = self.dynamic_decoder(latent_t, latent_prev)  # (B, V, 3)

        # Combine displacements and add to rest template
        # template_verts: (V, 3) -> broadcast over batch
        deformed_verts = (
            template_verts.unsqueeze(0)  # (1, V, 3)
            + static_disp                # (B, V, 3)
            + dynamic_disp               # (B, V, 3)
        )
        return deformed_verts  # (B, V, 3)
