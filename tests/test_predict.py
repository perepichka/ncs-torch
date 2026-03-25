"""Tests for ncs/predict.py.

Covers:
  - predict_frame output shape (V, 3) and no NaN/Inf
  - vertex count matches garment template
  - run output shape (T, V, 3)
  - .obj export: file exists, correct vertex count, 1-indexed faces
  - from_checkpoint round-trip
"""

import pytest
import torch

from ncs.config import Config
from ncs.model.network import NCSModel
from ncs.predict import Predictor, _write_obj
from ncs.train import Trainer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

J = 6    # joints
V = 8    # cloth vertices


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def garment_mesh():
    """Tiny cloth mesh: 8 verts, 8 non-degenerate triangulated faces."""
    verts = torch.tensor([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.],
    ], dtype=torch.float32)
    faces = torch.tensor([
        [0, 1, 2], [1, 3, 2],
        [4, 5, 6], [5, 7, 6],
        [0, 1, 4], [1, 5, 4],
        [2, 3, 6], [3, 7, 6],
    ], dtype=torch.long)
    return verts, faces


@pytest.fixture()
def body_faces():
    return torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5],
        [2, 6, 3], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=torch.long)


@pytest.fixture()
def predictor(garment_mesh):
    torch.manual_seed(0)
    gv, gf = garment_mesh
    model = NCSModel(n_joints=J, n_verts=V, latent_dim=64)
    return Predictor(model, gv, gf, device="cpu")


# ---------------------------------------------------------------------------
# Tests: predict_frame
# ---------------------------------------------------------------------------

class TestPredictFrame:

    def test_output_shape(self, predictor):
        pose_t = torch.randn(J, 3, 3)
        out = predictor.predict_frame(pose_t)
        assert out.shape == (V, 3)

    def test_batched_input_accepted(self, predictor):
        """predict_frame also accepts (1, J, 3, 3) input."""
        pose_t = torch.randn(1, J, 3, 3)
        out = predictor.predict_frame(pose_t)
        assert out.shape == (V, 3)

    def test_no_nan(self, predictor):
        out = predictor.predict_frame(torch.randn(J, 3, 3))
        assert not torch.isnan(out).any()

    def test_no_inf(self, predictor):
        out = predictor.predict_frame(torch.randn(J, 3, 3))
        assert not torch.isinf(out).any()

    def test_vertex_count_matches_template(self, predictor, garment_mesh):
        gv, _ = garment_mesh
        out = predictor.predict_frame(torch.randn(J, 3, 3))
        assert out.shape[0] == gv.shape[0]

    def test_with_previous_pose(self, predictor):
        pose_t    = torch.randn(J, 3, 3)
        pose_prev = torch.randn(J, 3, 3)
        out = predictor.predict_frame(pose_t, pose_prev)
        assert out.shape == (V, 3)

    def test_none_prev_gives_zero_prev(self, predictor):
        """predict_frame(pose, None) must equal predict_frame(pose, zeros)."""
        pose_t = torch.randn(J, 3, 3)
        out_none  = predictor.predict_frame(pose_t, pose_prev=None)
        out_zeros = predictor.predict_frame(pose_t, torch.zeros(J, 3, 3))
        assert torch.allclose(out_none, out_zeros)

    def test_different_poses_give_different_outputs(self, predictor):
        torch.manual_seed(1)
        out_a = predictor.predict_frame(torch.randn(J, 3, 3))
        out_b = predictor.predict_frame(torch.randn(J, 3, 3))
        assert not torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# Tests: run (sequence prediction)
# ---------------------------------------------------------------------------

class TestRun:

    def test_output_shape(self, predictor, tmp_path):
        T = 5
        seq = torch.randn(T, J, 3, 3)
        result = predictor.run(seq, tmp_path, export_obj=False)
        assert result.shape == (T, V, 3)

    def test_no_nan_in_sequence(self, predictor, tmp_path):
        seq = torch.randn(4, J, 3, 3)
        result = predictor.run(seq, tmp_path, export_obj=False)
        assert not torch.isnan(result).any()

    def test_exports_correct_number_of_obj_files(self, predictor, tmp_path):
        T = 3
        predictor.run(torch.randn(T, J, 3, 3), tmp_path, export_obj=True)
        obj_files = sorted(tmp_path.glob("frame_*.obj"))
        assert len(obj_files) == T

    def test_obj_filenames_are_zero_padded(self, predictor, tmp_path):
        predictor.run(torch.randn(2, J, 3, 3), tmp_path, export_obj=True)
        assert (tmp_path / "frame_0000.obj").exists()
        assert (tmp_path / "frame_0001.obj").exists()

    def test_obj_vertex_count_matches_template(self, predictor, garment_mesh, tmp_path):
        gv, _ = garment_mesh
        predictor.run(torch.randn(1, J, 3, 3), tmp_path, export_obj=True)
        lines = (tmp_path / "frame_0000.obj").read_text().splitlines()
        v_count = sum(1 for l in lines if l.startswith("v "))
        assert v_count == gv.shape[0]

    def test_no_export_creates_no_files(self, predictor, tmp_path):
        predictor.run(torch.randn(2, J, 3, 3), tmp_path, export_obj=False)
        assert not any(tmp_path.iterdir())

    def test_single_frame_sequence(self, predictor, tmp_path):
        result = predictor.run(torch.randn(1, J, 3, 3), tmp_path, export_obj=False)
        assert result.shape == (1, V, 3)


# ---------------------------------------------------------------------------
# Tests: _write_obj helper
# ---------------------------------------------------------------------------

class TestWriteObj:

    def test_creates_file(self, tmp_path):
        verts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        path = tmp_path / "mesh.obj"
        _write_obj(path, verts, faces)
        assert path.exists()

    def test_vertex_count(self, tmp_path):
        verts = torch.randn(10, 3)
        faces = torch.zeros(1, 3, dtype=torch.long)
        path = tmp_path / "mesh.obj"
        _write_obj(path, verts, faces)
        lines = path.read_text().splitlines()
        assert sum(1 for l in lines if l.startswith("v ")) == 10

    def test_face_count(self, tmp_path):
        verts = torch.randn(4, 3)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
        path = tmp_path / "mesh.obj"
        _write_obj(path, verts, faces)
        lines = path.read_text().splitlines()
        assert sum(1 for l in lines if l.startswith("f ")) == 2

    def test_faces_are_1_indexed(self, tmp_path):
        verts = torch.randn(3, 3)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        path = tmp_path / "mesh.obj"
        _write_obj(path, verts, faces)
        f_lines = [l for l in path.read_text().splitlines() if l.startswith("f ")]
        assert f_lines[0] == "f 1 2 3"

    def test_creates_parent_dirs(self, tmp_path):
        verts = torch.randn(3, 3)
        faces = torch.zeros(1, 3, dtype=torch.long)
        path = tmp_path / "deep" / "nested" / "mesh.obj"
        _write_obj(path, verts, faces)
        assert path.exists()


# ---------------------------------------------------------------------------
# Tests: from_checkpoint
# ---------------------------------------------------------------------------

class TestFromCheckpoint:

    def test_round_trip_predict(self, tmp_path, garment_mesh, body_faces):
        """Train → save checkpoint → load via from_checkpoint → predict."""
        torch.manual_seed(0)
        gv, gf = garment_mesh

        model = NCSModel(n_joints=J, n_verts=V, latent_dim=64)
        cfg   = Config(formulation="mass_spring", device="cpu", lr=1e-3)
        trainer = Trainer(model, cfg, gv, gf, body_faces)
        trainer.train_step({
            "pose":       torch.randn(2, J, 3, 3),
            "body_verts": torch.randn(2, 8, 3) * 0.05 + 5.0,
        })

        ckpt_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path)

        predictor = Predictor.from_checkpoint(ckpt_path, gv, gf, device="cpu")
        out = predictor.predict_frame(torch.randn(J, 3, 3))

        assert out.shape == (V, 3)
        assert not torch.isnan(out).any()

    def test_weights_match_trainer(self, tmp_path, garment_mesh, body_faces):
        """Loaded model weights must be identical to the saved ones."""
        torch.manual_seed(0)
        gv, gf = garment_mesh

        model = NCSModel(n_joints=J, n_verts=V, latent_dim=64)
        cfg   = Config(formulation="mass_spring", device="cpu", lr=1e-3)
        trainer = Trainer(model, cfg, gv, gf, body_faces)

        ckpt_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path)

        pred = Predictor.from_checkpoint(ckpt_path, gv, gf, device="cpu")

        for (n1, p1), (n2, p2) in zip(
            trainer.model.named_parameters(),
            pred.model.named_parameters(),
        ):
            assert torch.allclose(p1.cpu(), p2.cpu()), (
                f"Weight mismatch for {n1!r}"
            )
