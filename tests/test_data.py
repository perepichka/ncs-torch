"""
Tests for ncs/data/dataset.py

All tests are self-contained — no external files are read.
Synthetic data is generated on the fly using numpy / torch primitives or
written to a temporary directory via pytest's ``tmp_path`` fixture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ncs.data.dataset import PoseDataset, rodrigues_batch, skinning


# ===========================================================================
# Helpers
# ===========================================================================

def _allclose(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> bool:
    return torch.allclose(a.float(), b.float(), atol=atol)


# ===========================================================================
# rodrigues_batch
# ===========================================================================

class TestRodriguesBatch:
    """Tests for the axis-angle -> rotation matrix conversion."""

    def test_output_shape(self):
        rvecs = torch.zeros(5, 3)
        R = rodrigues_batch(rvecs)
        assert R.shape == (5, 3, 3), f"Expected (5, 3, 3), got {R.shape}"

    def test_zero_rvec_gives_identity(self):
        """A zero rotation vector must produce the identity matrix."""
        rvecs = torch.zeros(4, 3)
        R = rodrigues_batch(rvecs)
        I = torch.eye(3).unsqueeze(0).expand(4, -1, -1)
        assert _allclose(R, I), f"Expected identity, got:\n{R}"

    def test_small_rvec_gives_identity(self):
        """Near-zero rotation vectors should also give (approximately) identity."""
        rvecs = torch.full((3, 3), 1e-9)
        R = rodrigues_batch(rvecs)
        I = torch.eye(3).unsqueeze(0).expand(3, -1, -1)
        assert _allclose(R, I, atol=1e-5)

    def test_90_degrees_around_z(self):
        """90° around Z should map x->y, y->-x, z->z."""
        angle = math.pi / 2.0
        rvec = torch.tensor([[0.0, 0.0, angle]])  # (1, 3)
        R = rodrigues_batch(rvec)[0]              # (3, 3)

        expected = torch.tensor([
            [ 0., -1.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.],
        ])
        assert _allclose(R, expected, atol=1e-6), (
            f"90° around Z failed.\nExpected:\n{expected}\nGot:\n{R}"
        )

    def test_90_degrees_around_x(self):
        """90° around X should map y->z, z->-y, x->x."""
        angle = math.pi / 2.0
        rvec = torch.tensor([[angle, 0.0, 0.0]])
        R = rodrigues_batch(rvec)[0]

        expected = torch.tensor([
            [1.,  0.,  0.],
            [0.,  0., -1.],
            [0.,  1.,  0.],
        ])
        assert _allclose(R, expected, atol=1e-6), (
            f"90° around X failed.\nExpected:\n{expected}\nGot:\n{R}"
        )

    def test_180_degrees_around_y(self):
        """180° around Y should negate x and z components."""
        angle = math.pi
        rvec = torch.tensor([[0.0, angle, 0.0]])
        R = rodrigues_batch(rvec)[0]

        expected = torch.tensor([
            [-1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0., -1.],
        ])
        assert _allclose(R, expected, atol=1e-6), (
            f"180° around Y failed.\nExpected:\n{expected}\nGot:\n{R}"
        )

    def test_output_is_rotation_matrix(self):
        """Output matrices must be orthogonal (R^T R ≈ I) with det ≈ +1."""
        torch.manual_seed(0)
        rvecs = torch.randn(16, 3)
        R = rodrigues_batch(rvecs)

        I = torch.eye(3).unsqueeze(0).expand(16, -1, -1)
        RtR = torch.bmm(R.transpose(1, 2), R)
        assert _allclose(RtR, I, atol=1e-5), "R^T R ≠ I"

        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.ones(16), atol=1e-5), "det(R) ≠ 1"

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            rodrigues_batch(torch.zeros(3))  # missing batch dim

        with pytest.raises(ValueError):
            rodrigues_batch(torch.zeros(5, 4))  # wrong last dim

    def test_batch_of_one(self):
        rvec = torch.tensor([[0.0, 0.0, 0.0]])
        R = rodrigues_batch(rvec)
        assert R.shape == (1, 3, 3)


# ===========================================================================
# skinning
# ===========================================================================

class TestSkinning:
    """Tests for Linear Blend Skinning."""

    def test_output_shape(self):
        V, J = 10, 4
        verts = torch.randn(V, 3)
        joints = torch.randn(J, 3)
        weights = torch.softmax(torch.randn(V, J), dim=1)
        rotations = torch.eye(3).unsqueeze(0).expand(J, -1, -1)
        out = skinning(verts, joints, weights, rotations)
        assert out.shape == (V, 3)

    def test_identity_rotation_preserves_vertices(self):
        """With all-identity rotations the vertices must remain unchanged."""
        V, J = 8, 3
        torch.manual_seed(1)
        verts = torch.randn(V, 3)
        joints = torch.zeros(J, 3)  # joints at origin
        weights = torch.softmax(torch.randn(V, J), dim=1)
        rotations = torch.eye(3).unsqueeze(0).expand(J, -1, -1).clone()

        out = skinning(verts, joints, weights, rotations)
        assert _allclose(out, verts, atol=1e-5), (
            "Identity rotation should leave vertices unchanged"
        )

    def test_single_joint_full_weight_rotates_correctly(self):
        """A vertex weighted 100% on joint 0 should be rotated exactly by R[0].

        Setup:
          - 1 vertex at (1, 0, 0)
          - 1 joint at origin (0, 0, 0)
          - weight = 1.0 for joint 0
          - R[0] = 90° rotation around Z  -> maps (1,0,0) to (0,1,0)
        """
        verts = torch.tensor([[1.0, 0.0, 0.0]])   # (1, 3)
        joints = torch.tensor([[0.0, 0.0, 0.0]])  # (1, 3) joint at origin
        weights = torch.tensor([[1.0]])            # (1, 1)

        angle = math.pi / 2.0
        rvec = torch.tensor([[0.0, 0.0, angle]])
        R = rodrigues_batch(rvec)                  # (1, 3, 3)

        out = skinning(verts, joints, weights, R)  # (1, 3)

        expected = torch.tensor([[0.0, 1.0, 0.0]])
        assert _allclose(out, expected, atol=1e-6), (
            f"Expected {expected}, got {out}"
        )

    def test_single_joint_offset_pivot(self):
        """Joint at (1,0,0); vertex at (2,0,0); 90° around Z at the joint.

        The vertex is 1 unit to the right of the joint.
        After rotation the offset (1,0,0) -> (0,1,0), so the final position
        should be joint + rotated_offset = (1,0,0)+(0,1,0) = (1,1,0).
        """
        verts = torch.tensor([[2.0, 0.0, 0.0]])
        joints = torch.tensor([[1.0, 0.0, 0.0]])
        weights = torch.tensor([[1.0]])

        angle = math.pi / 2.0
        rvec = torch.tensor([[0.0, 0.0, angle]])
        R = rodrigues_batch(rvec)

        out = skinning(verts, joints, weights, R)

        expected = torch.tensor([[1.0, 1.0, 0.0]])
        assert _allclose(out, expected, atol=1e-6), (
            f"Expected {expected}, got {out}"
        )

    def test_two_joints_equal_weights(self):
        """Vertex equidistant from two joints with equal weights.

        Both joints at origin; R[0]=90° around Z, R[1]=identity.
        vertex = (1, 0, 0):
          T0(v) = R0 @ v = (0, 1, 0)
          T1(v) = I  @ v = (1, 0, 0)
          blend = 0.5 * (0,1,0) + 0.5 * (1,0,0) = (0.5, 0.5, 0)
        """
        verts = torch.tensor([[1.0, 0.0, 0.0]])
        joints = torch.zeros(2, 3)
        weights = torch.tensor([[0.5, 0.5]])

        angle = math.pi / 2.0
        rvec_z90 = torch.tensor([[0.0, 0.0, angle]])
        rvec_id = torch.tensor([[0.0, 0.0, 0.0]])
        R = rodrigues_batch(torch.cat([rvec_z90, rvec_id], dim=0))  # (2,3,3)

        out = skinning(verts, joints, weights, R)

        expected = torch.tensor([[0.5, 0.5, 0.0]])
        assert _allclose(out, expected, atol=1e-6), (
            f"Expected {expected}, got {out}"
        )

    def test_wrong_weight_shape_raises(self):
        with pytest.raises(ValueError):
            skinning(
                torch.randn(5, 3),
                torch.randn(3, 3),
                torch.randn(5, 4),   # wrong J dimension
                torch.eye(3).unsqueeze(0).expand(3, -1, -1),
            )

    def test_wrong_rotation_shape_raises(self):
        with pytest.raises(ValueError):
            skinning(
                torch.randn(5, 3),
                torch.randn(3, 3),
                torch.randn(5, 3),
                torch.eye(3).unsqueeze(0).expand(4, -1, -1),  # J mismatch
            )


# ===========================================================================
# PoseDataset
# ===========================================================================

class TestPoseDataset:
    """Tests for PoseDataset using tmp_path (no real file I/O setup required)."""

    # ------------------------------------------------------------------
    # Fixtures / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_dataset_dir(
        tmp_path,
        sequences: dict[str, tuple[int, int, int]],
        # sequences: {stem: (T, J, V)}
    ) -> "Path":
        """Write synthetic .npy files and return the root data_dir."""
        data_dir = tmp_path / "data"
        poses_dir = data_dir / "poses"
        verts_dir = data_dir / "verts"
        poses_dir.mkdir(parents=True)
        verts_dir.mkdir(parents=True)

        rng = np.random.default_rng(42)
        for stem, (T, J, V) in sequences.items():
            poses = rng.standard_normal((T, J, 3, 3)).astype(np.float32)
            verts = rng.standard_normal((T, V, 3)).astype(np.float32)
            np.save(str(poses_dir / f"{stem}.npy"), poses)
            np.save(str(verts_dir / f"{stem}.npy"), verts)

        return data_dir

    # ------------------------------------------------------------------
    # __len__
    # ------------------------------------------------------------------

    def test_len_single_sequence(self, tmp_path):
        T, J, V = 20, 24, 6890
        data_dir = self._make_dataset_dir(tmp_path, {"seq_001": (T, J, V)})
        ds = PoseDataset(data_dir)
        assert len(ds) == T, f"Expected {T}, got {len(ds)}"

    def test_len_multiple_sequences(self, tmp_path):
        seqs = {"seq_001": (10, 24, 6890), "seq_002": (15, 24, 6890)}
        data_dir = self._make_dataset_dir(tmp_path, seqs)
        ds = PoseDataset(data_dir)
        assert len(ds) == 25, f"Expected 25, got {len(ds)}"

    # ------------------------------------------------------------------
    # __getitem__ shapes
    # ------------------------------------------------------------------

    def test_getitem_shapes(self, tmp_path):
        T, J, V = 8, 24, 6890
        data_dir = self._make_dataset_dir(tmp_path, {"seq_a": (T, J, V)})
        ds = PoseDataset(data_dir)

        item = ds[0]
        assert "pose" in item
        assert "body_verts" in item
        assert "frame_idx" in item

        assert item["pose"].shape == (J, 3, 3), (
            f"pose shape: expected ({J}, 3, 3), got {item['pose'].shape}"
        )
        assert item["body_verts"].shape == (V, 3), (
            f"body_verts shape: expected ({V}, 3), got {item['body_verts'].shape}"
        )

    def test_getitem_last_frame(self, tmp_path):
        T, J, V = 5, 10, 100
        data_dir = self._make_dataset_dir(tmp_path, {"seq": (T, J, V)})
        ds = PoseDataset(data_dir)

        item = ds[T - 1]
        assert item["pose"].shape == (J, 3, 3)
        assert item["body_verts"].shape == (V, 3)

    def test_getitem_dtype(self, tmp_path):
        T, J, V = 4, 6, 50
        data_dir = self._make_dataset_dir(tmp_path, {"seq": (T, J, V)})
        ds = PoseDataset(data_dir)

        item = ds[0]
        assert item["pose"].dtype == torch.float32
        assert item["body_verts"].dtype == torch.float32

    def test_frame_idx_values(self, tmp_path):
        """frame_idx should be the global dataset index (0, 1, ..., len-1)."""
        T, J, V = 6, 5, 20
        data_dir = self._make_dataset_dir(tmp_path, {"seq": (T, J, V)})
        ds = PoseDataset(data_dir)

        for i in range(len(ds)):
            assert ds[i]["frame_idx"] == i, (
                f"frame_idx mismatch at global index {i}: got {ds[i]['frame_idx']}"
            )

    # ------------------------------------------------------------------
    # Multiple sequences — data varies across frames
    # ------------------------------------------------------------------

    def test_different_frames_have_different_data(self, tmp_path):
        """Two different frames should (almost certainly) have different values."""
        T, J, V = 10, 24, 6890
        data_dir = self._make_dataset_dir(tmp_path, {"seq": (T, J, V)})
        ds = PoseDataset(data_dir)

        item0 = ds[0]
        item1 = ds[1]
        # With random data the probability of equality is astronomically small
        assert not torch.equal(item0["pose"], item1["pose"])
        assert not torch.equal(item0["body_verts"], item1["body_verts"])

    # ------------------------------------------------------------------
    # Error conditions
    # ------------------------------------------------------------------

    def test_missing_poses_dir_raises(self, tmp_path):
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        (data_dir / "verts").mkdir()
        with pytest.raises(FileNotFoundError):
            PoseDataset(data_dir)

    def test_missing_verts_dir_raises(self, tmp_path):
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        (data_dir / "poses").mkdir()
        with pytest.raises(FileNotFoundError):
            PoseDataset(data_dir)

    def test_no_matching_files_raises(self, tmp_path):
        data_dir = tmp_path / "data"
        (data_dir / "poses").mkdir(parents=True)
        (data_dir / "verts").mkdir(parents=True)
        # poses has seq_001, verts has seq_002 — no overlap
        np.save(str(data_dir / "poses" / "seq_001.npy"), np.zeros((2, 3, 3, 3), dtype=np.float32))
        np.save(str(data_dir / "verts" / "seq_002.npy"), np.zeros((2, 10, 3), dtype=np.float32))
        with pytest.raises(RuntimeError):
            PoseDataset(data_dir)

    def test_frame_count_mismatch_raises(self, tmp_path):
        data_dir = tmp_path / "data"
        (data_dir / "poses").mkdir(parents=True)
        (data_dir / "verts").mkdir(parents=True)
        # Poses has T=5, verts has T=3 — mismatch
        np.save(str(data_dir / "poses" / "seq.npy"), np.zeros((5, 3, 3, 3), dtype=np.float32))
        np.save(str(data_dir / "verts" / "seq.npy"), np.zeros((3, 10, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            PoseDataset(data_dir)

    # ------------------------------------------------------------------
    # DataLoader compatibility
    # ------------------------------------------------------------------

    def test_dataloader_batch(self, tmp_path):
        """Verify PoseDataset works with torch DataLoader (batching, collation)."""
        T, J, V = 12, 24, 100
        data_dir = self._make_dataset_dir(tmp_path, {"seq": (T, J, V)})
        ds = PoseDataset(data_dir)

        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        assert batch["pose"].shape == (4, J, 3, 3), (
            f"Expected (4, {J}, 3, 3), got {batch['pose'].shape}"
        )
        assert batch["body_verts"].shape == (4, V, 3), (
            f"Expected (4, {V}, 3), got {batch['body_verts'].shape}"
        )
        assert len(batch["frame_idx"]) == 4
