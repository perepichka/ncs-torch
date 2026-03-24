"""
Data pipeline for NCS-Torch.

Provides:
  - rodrigues_batch: batch axis-angle -> rotation matrix conversion
  - skinning: Linear Blend Skinning (LBS)
  - PoseDataset: loads pre-computed pose sequences and body vertices from .npy files
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
import torch.utils.data


# ---------------------------------------------------------------------------
# Rodrigues formula: axis-angle -> rotation matrix
# ---------------------------------------------------------------------------

def rodrigues_batch(rvecs: Tensor) -> Tensor:
    """Convert a batch of axis-angle vectors to rotation matrices.

    Uses the Rodrigues formula:
        R = I + sin(θ) * K + (1 - cos(θ)) * K²

    where K is the skew-symmetric cross-product matrix of the unit axis
    and θ = ||rvec||.

    Args:
        rvecs: (N, 3) axis-angle vectors.  The direction encodes the rotation
               axis and the magnitude encodes the angle in radians.

    Returns:
        (N, 3, 3) rotation matrices, float32.
    """
    if rvecs.ndim != 2 or rvecs.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {tuple(rvecs.shape)}")

    rvecs = rvecs.float()
    N = rvecs.shape[0]
    device = rvecs.device

    # Rotation angle: (N,)
    theta = torch.norm(rvecs, dim=1, keepdim=True)  # (N, 1)

    # Unit axis; fall back to zero for near-zero angles to avoid division by
    # zero — when theta ≈ 0 the rotation is identity regardless of the axis.
    safe_theta = torch.where(theta < 1e-8, torch.ones_like(theta), theta)
    axis = rvecs / safe_theta  # (N, 3)

    # Skew-symmetric matrix K for each axis vector:
    #   K = [[  0, -az,  ay],
    #         [ az,   0, -ax],
    #         [-ay,  ax,   0]]
    ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]  # each (N,)
    zeros = torch.zeros(N, device=device, dtype=torch.float32)

    K = torch.stack([
        zeros, -az,   ay,
          az, zeros, -ax,
         -ay,    ax, zeros,
    ], dim=1).reshape(N, 3, 3)  # (N, 3, 3)

    # K²
    K2 = torch.bmm(K, K)  # (N, 3, 3)

    I = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)

    sin_t = torch.sin(theta).reshape(N, 1, 1)   # (N, 1, 1)
    cos_t = torch.cos(theta).reshape(N, 1, 1)   # (N, 1, 1)

    # Zero out K and K² contributions when theta is effectively zero so we
    # return exact identity matrices for the zero-vector input.
    near_zero = (theta < 1e-8).reshape(N, 1, 1)
    sin_t = torch.where(near_zero, torch.zeros_like(sin_t), sin_t)
    one_minus_cos = torch.where(near_zero, torch.zeros_like(cos_t), 1.0 - cos_t)

    R = I + sin_t * K + one_minus_cos * K2  # (N, 3, 3)
    return R


# ---------------------------------------------------------------------------
# Linear Blend Skinning (LBS)
# ---------------------------------------------------------------------------

def skinning(
    template_verts: Tensor,  # (V, 3)
    joints: Tensor,          # (J, 3) joint positions in rest pose
    weights: Tensor,         # (V, J) blend weights (should sum to 1 per vertex)
    rotations: Tensor,       # (J, 3, 3) rotation matrices (local, applied at joint)
) -> Tensor:
    """Apply Linear Blend Skinning to a template mesh.

    For each joint j the transformation is:
        T_j(v) = R_j @ (v - p_j) + p_j

    The final posed vertex is the weighted blend:
        v' = Σ_j w[v,j] * T_j(v)

    Args:
        template_verts: (V, 3) rest-pose vertex positions.
        joints:         (J, 3) joint positions in rest pose.
        weights:        (V, J) per-vertex blend weights.
        rotations:      (J, 3, 3) rotation matrices for each joint.

    Returns:
        (V, 3) posed vertex positions, float32.
    """
    template_verts = template_verts.float()
    joints = joints.float()
    weights = weights.float()
    rotations = rotations.float()

    V = template_verts.shape[0]
    J = joints.shape[0]

    if weights.shape != (V, J):
        raise ValueError(
            f"weights shape {tuple(weights.shape)} does not match (V={V}, J={J})"
        )
    if rotations.shape != (J, 3, 3):
        raise ValueError(
            f"rotations shape {tuple(rotations.shape)} does not match (J={J}, 3, 3)"
        )

    # Translate vertices relative to each joint: (V, J, 3)
    # template_verts: (V, 1, 3),  joints: (1, J, 3)
    v_centered = template_verts.unsqueeze(1) - joints.unsqueeze(0)  # (V, J, 3)

    # Rotate each vertex around each joint.
    # rotations: (J, 3, 3); v_centered: (V, J, 3)
    # Expand dims for batch matmul: (V, J, 3, 1)
    v_rot = torch.einsum("jkl,vjl->vjk", rotations, v_centered)  # (V, J, 3)

    # Translate back to world space
    v_world = v_rot + joints.unsqueeze(0)  # (V, J, 3)

    # Weighted blend: (V, J, 1) * (V, J, 3) -> sum over J -> (V, 3)
    posed = (weights.unsqueeze(-1) * v_world).sum(dim=1)  # (V, 3)
    return posed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PoseDataset(torch.utils.data.Dataset):
    """Loads pose sequence data from .npy files.

    Expected directory structure::

        data_dir/
            poses/
                seq_001.npy   # (T, J, 3, 3) rotation matrices
                seq_002.npy
            verts/
                seq_001.npy   # (T, V, 3) body vertices (pre-skinned)

    Every matching pair of files in ``poses/`` and ``verts/`` is treated as
    one sequence.  The dataset is indexed at the *frame* level: each item
    corresponds to a single frame across all sequences.

    Each item is a dict with keys:

    * ``'pose'``       — ``(J, 3, 3)`` float32 tensor, rotation matrices
    * ``'body_verts'`` — ``(V, 3)``    float32 tensor, pre-skinned body vertices
    * ``'frame_idx'``  — ``int``, global frame index within this dataset

    Args:
        data_dir: Root directory containing ``poses/`` and ``verts/``
                  sub-directories.
    """

    def __init__(self, data_dir: str | os.PathLike) -> None:
        data_dir = Path(data_dir)
        poses_dir = data_dir / "poses"
        verts_dir = data_dir / "verts"

        if not poses_dir.is_dir():
            raise FileNotFoundError(f"poses directory not found: {poses_dir}")
        if not verts_dir.is_dir():
            raise FileNotFoundError(f"verts directory not found: {verts_dir}")

        # Discover sequences that exist in both sub-directories.
        pose_files = {p.stem: p for p in sorted(poses_dir.glob("*.npy"))}
        vert_files = {p.stem: p for p in sorted(verts_dir.glob("*.npy"))}

        common_stems = sorted(set(pose_files.keys()) & set(vert_files.keys()))
        if not common_stems:
            raise RuntimeError(
                f"No matching .npy pairs found in {poses_dir} / {verts_dir}"
            )

        # Build a flat index: list of (pose_array, vert_array, local_frame_idx)
        # We keep arrays memory-mapped to avoid loading everything at once.
        self._entries: List[tuple] = []  # (poses_path, verts_path, t)
        self._sequence_info: List[Dict] = []

        for stem in common_stems:
            p_path = pose_files[stem]
            v_path = vert_files[stem]

            # Memory-map for efficient random access.
            p_arr = np.load(str(p_path), mmap_mode="r")  # (T, J, 3, 3)
            v_arr = np.load(str(v_path), mmap_mode="r")  # (T, V, 3)

            if p_arr.ndim != 4 or p_arr.shape[2:] != (3, 3):
                raise ValueError(
                    f"{p_path}: expected shape (T, J, 3, 3), got {p_arr.shape}"
                )
            if v_arr.ndim != 3 or v_arr.shape[2] != 3:
                raise ValueError(
                    f"{v_path}: expected shape (T, V, 3), got {v_arr.shape}"
                )

            T = p_arr.shape[0]
            if v_arr.shape[0] != T:
                raise ValueError(
                    f"Sequence '{stem}': poses has {T} frames but verts has "
                    f"{v_arr.shape[0]} frames"
                )

            self._sequence_info.append(
                {"stem": stem, "T": T, "J": p_arr.shape[1], "V": v_arr.shape[1]}
            )

            for t in range(T):
                self._entries.append((p_arr, v_arr, t))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        p_arr, v_arr, t = self._entries[idx]

        pose = torch.from_numpy(np.array(p_arr[t], dtype=np.float32))        # (J, 3, 3)
        body_verts = torch.from_numpy(np.array(v_arr[t], dtype=np.float32))  # (V, 3)

        return {
            "pose": pose,
            "body_verts": body_verts,
            "frame_idx": idx,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def sequence_info(self) -> List[Dict]:
        """List of per-sequence metadata dicts (stem, T, J, V)."""
        return self._sequence_info
