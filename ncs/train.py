"""Training loop for NCS-Torch.

Combines physics energy, collision penalty, and regularisation into an
unsupervised training objective:

    total_loss = w_physics * E_physics + w_collision * E_collision + w_reg * E_reg
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ncs.config import Config
from ncs.model.network import NCSModel
from ncs.mesh.mesh import compute_edges, compute_face_areas
from ncs.physics.energy import mass_spring_energy, baraff_witkin_energy, stvk_energy
from ncs.collision.collision import collision_loss


# ---------------------------------------------------------------------------
# Per-batch energy helpers
# ---------------------------------------------------------------------------

def _physics_energy(
    deformed_verts: Tensor,  # (B, V, 3)
    rest_verts: Tensor,      # (V, 3)
    faces: Tensor,           # (F, 3) long
    edges: Tensor,           # (E, 2) long
    rest_areas: Tensor,      # (F,)
    formulation: str,
    config: Config,
) -> Tensor:
    """Compute the mean physics energy over the batch."""
    energies = []
    for b in range(deformed_verts.shape[0]):
        v = deformed_verts[b]
        if formulation == "mass_spring":
            e = mass_spring_energy(v, rest_verts, edges, stiffness=config.stretch)
        elif formulation == "baraff_witkin":
            e = baraff_witkin_energy(
                v, rest_verts, faces, rest_areas,
                k_stretch=config.stretch,
                k_shear=config.shear,
            )
        elif formulation == "stvk":
            e = stvk_energy(v, rest_verts, faces, rest_areas)
        else:
            raise ValueError(f"Unknown energy formulation: {formulation!r}")
        energies.append(e)
    return torch.stack(energies).mean()


def _collision_energy(
    deformed_verts: Tensor,  # (B, V, 3)
    body_verts: Tensor,      # (B, B_V, 3)
    body_faces: Tensor,      # (B_F, 3) long
    threshold: float,
) -> Tensor:
    """Compute the mean collision loss over the batch."""
    losses = []
    for b in range(deformed_verts.shape[0]):
        loss = collision_loss(
            deformed_verts[b], body_verts[b], body_faces, threshold=threshold
        )
        losses.append(loss)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Manages unsupervised training of the NCS cloth simulation model.

    Args:
        model:         NCSModel instance.
        config:        Config dataclass with all hyperparameters.
        garment_verts: (V, 3) rest-pose cloth vertex positions.
        garment_faces: (F, 3) cloth face indices (long).
        body_faces:    (B_F, 3) body mesh face indices (long).  The body mesh
                       topology is fixed; per-frame vertex positions come from
                       the data loader.
    """

    def __init__(
        self,
        model: NCSModel,
        config: Config,
        garment_verts: Tensor,
        garment_faces: Tensor,
        body_faces: Tensor,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.model = model.to(self.device)
        self.garment_verts = garment_verts.to(self.device)
        self.garment_faces = garment_faces.to(self.device)
        self.body_faces = body_faces.to(self.device)

        # Pre-compute rest-state quantities (fixed throughout training).
        self.edges = compute_edges(self.garment_faces)
        self.rest_areas = compute_face_areas(self.garment_verts, self.garment_faces)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.step_count: int = 0
        self._loss_history: list[dict] = []

    # ------------------------------------------------------------------
    # Single gradient step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        """Run one optimisation step.

        Args:
            batch: dict with keys:
                * ``'pose'``       — (B, J, 3, 3) current-frame rotation matrices
                * ``'body_verts'`` — (B, B_V, 3)  body vertex positions

        Returns:
            Dict with scalar loss components:
            ``'physics'``, ``'collision'``, ``'reg'``, ``'total'``.
        """
        self.model.train()
        self.optimizer.zero_grad()

        pose_t = batch["pose"].to(self.device)              # (B, J, 3, 3)
        body_verts = batch["body_verts"].to(self.device)    # (B, B_V, 3)

        # Previous pose: zero for every step (frame ordering not tracked here).
        pose_prev = torch.zeros_like(pose_t)

        # ---- Forward pass ----
        deformed_verts = self.model(
            pose_t, pose_prev, self.garment_verts
        )  # (B, V, 3)

        # ---- Loss components ----
        e_physics = _physics_energy(
            deformed_verts,
            self.garment_verts,
            self.garment_faces,
            self.edges,
            self.rest_areas,
            self.config.formulation,
            self.config,
        )

        e_collision = _collision_energy(
            deformed_verts,
            body_verts,
            self.body_faces,
            threshold=self.config.collision_threshold,
        )

        # Regularisation: penalise large displacements from the rest template.
        disp = deformed_verts - self.garment_verts.unsqueeze(0)
        e_reg = (disp ** 2).mean()

        total = (
            self.config.w_physics    * e_physics
            + self.config.w_collision * e_collision
            + self.config.w_reg       * e_reg
        )

        total.backward()
        self.optimizer.step()
        self.step_count += 1

        loss_dict: dict = {
            "physics":   e_physics.item(),
            "collision": e_collision.item(),
            "reg":       e_reg.item(),
            "total":     total.item(),
        }
        self._loss_history.append(loss_dict)
        return loss_dict

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: DataLoader,
        epochs: Optional[int] = None,
    ) -> list[dict]:
        """Run training for *epochs* passes over *dataloader*.

        Args:
            dataloader: DataLoader yielding batches (see :meth:`train_step`).
            epochs:     Number of epochs.  Defaults to ``config.epochs``.

        Returns:
            List of per-step loss dicts (one entry per :meth:`train_step` call).
        """
        if epochs is None:
            epochs = self.config.epochs

        for _ in range(epochs):
            for batch in dataloader:
                self.train_step(batch)

        return self._loss_history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | os.PathLike) -> None:
        """Save model weights, optimiser state, and metadata to *path*.

        Args:
            path: Destination file path (e.g. ``checkpoints/epoch_10.pt``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step_count":           self.step_count,
                "config":               self.config,
                "model_kwargs": {
                    "n_joints":   self.model.n_joints,
                    "n_verts":    self.model.n_verts,
                    "latent_dim": self.model.latent_dim,
                },
            },
            path,
        )

    def load_checkpoint(self, path: str | os.PathLike) -> None:
        """Restore model weights and optimiser state from *path*.

        Args:
            path: Checkpoint file created by :meth:`save_checkpoint`.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step_count = ckpt.get("step_count", 0)
