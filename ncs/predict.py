"""Inference for NCS-Torch: run a trained model on a pose sequence.

Typical usage::

    predictor = Predictor.from_checkpoint("checkpoints/best.pt", garment_verts, garment_faces)
    deformed = predictor.run(pose_sequence, output_dir="out/frames/")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from ncs.config import Config
from ncs.model.network import NCSModel


# ---------------------------------------------------------------------------
# OBJ export helper
# ---------------------------------------------------------------------------

def _write_obj(path: str | os.PathLike, verts: Tensor, faces: Tensor) -> None:
    """Write a triangle mesh to a Wavefront ``.obj`` file.

    Args:
        path:  Destination path.  Parent directory is created if needed.
        verts: (V, 3) float tensor — vertex positions.
        faces: (F, 3) long tensor  — 0-indexed triangle vertex indices.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    with open(path, "w") as fh:
        for v in verts_np:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces_np:
            # .obj uses 1-based indices
            fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """Runs a trained :class:`~ncs.model.network.NCSModel` on pose sequences.

    Args:
        model:         Trained NCSModel (moved to *device* internally).
        garment_verts: (V, 3) rest-pose cloth vertex positions.
        garment_faces: (F, 3) cloth face indices (long).
        device:        Target device string or ``torch.device`` (default: CPU).
    """

    def __init__(
        self,
        model: NCSModel,
        garment_verts: Tensor,
        garment_faces: Tensor,
        device=None,
    ) -> None:
        self.device = torch.device(device or "cpu")
        self.model = model.to(self.device).eval()
        self.garment_verts = garment_verts.to(self.device)
        self.garment_faces = garment_faces.to(self.device)

    # ------------------------------------------------------------------
    # Factory: load from checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | os.PathLike,
        garment_verts: Tensor,
        garment_faces: Tensor,
        device=None,
    ) -> "Predictor":
        """Build a :class:`Predictor` from a checkpoint saved by
        :class:`~ncs.train.Trainer`.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.
            garment_verts:   (V, 3) rest-pose cloth vertices.
            garment_faces:   (F, 3) cloth face indices (long).
            device:          Target device (default: CPU).

        Returns:
            A ready-to-use :class:`Predictor`.
        """
        device = torch.device(device or "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model_kwargs: dict = ckpt.get("model_kwargs", {})
        n_joints   = model_kwargs.get("n_joints", 24)
        latent_dim = model_kwargs.get("latent_dim", 256)
        # Use the actual garment mesh to determine n_verts (authoritative).
        n_verts = garment_verts.shape[0]

        model = NCSModel(n_joints=n_joints, n_verts=n_verts, latent_dim=latent_dim)
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(model, garment_verts, garment_faces, device)

    # ------------------------------------------------------------------
    # Single-frame prediction
    # ------------------------------------------------------------------

    def predict_frame(
        self,
        pose_t: Tensor,
        pose_prev: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict cloth vertex positions for a single frame.

        Args:
            pose_t:    (J, 3, 3) or (1, J, 3, 3) — current-frame rotation matrices.
            pose_prev: (J, 3, 3) or (1, J, 3, 3) — previous-frame rotation matrices.
                       Pass ``None`` to use zeros (first frame of a sequence).

        Returns:
            (V, 3) float tensor of deformed cloth vertex positions.
        """
        if pose_t.ndim == 3:
            pose_t = pose_t.unsqueeze(0)          # (1, J, 3, 3)
        pose_t = pose_t.to(self.device)

        if pose_prev is None:
            pose_prev = torch.zeros_like(pose_t)
        elif pose_prev.ndim == 3:
            pose_prev = pose_prev.unsqueeze(0)    # (1, J, 3, 3)
        pose_prev = pose_prev.to(self.device)

        with torch.no_grad():
            deformed = self.model(
                pose_t, pose_prev, self.garment_verts
            )  # (1, V, 3)

        return deformed[0]  # (V, 3)

    # ------------------------------------------------------------------
    # Sequence prediction
    # ------------------------------------------------------------------

    def run(
        self,
        pose_sequence: Tensor,
        output_dir: str | os.PathLike,
        export_obj: bool = True,
    ) -> Tensor:
        """Run inference over a full pose sequence.

        Args:
            pose_sequence: (T, J, 3, 3) rotation matrices, one per frame.
            output_dir:    Directory for per-frame ``.obj`` exports.
            export_obj:    Write one ``frame_NNNN.obj`` file per frame when
                           ``True`` (default).

        Returns:
            (T, V, 3) tensor — cloth vertex positions for every frame.
        """
        output_dir = Path(output_dir)
        T = pose_sequence.shape[0]
        all_verts: list[Tensor] = []

        for t in range(T):
            pose_t    = pose_sequence[t]                          # (J, 3, 3)
            pose_prev = pose_sequence[t - 1] if t > 0 else None  # (J, 3, 3)

            verts = self.predict_frame(pose_t, pose_prev)         # (V, 3)
            all_verts.append(verts)

            if export_obj:
                _write_obj(output_dir / f"frame_{t:04d}.obj", verts, self.garment_faces)

        return torch.stack(all_verts, dim=0)  # (T, V, 3)
