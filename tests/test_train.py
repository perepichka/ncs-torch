"""Tests for ncs/train.py.

Covers:
  - train_step returns expected loss keys and finite values
  - step_count increments
  - smoke test: loss decreases over 10 gradient steps on a fixed batch
  - checkpoint save / load round-trip
"""

import pytest
import torch

from ncs.config import Config
from ncs.model.network import NCSModel
from ncs.train import Trainer


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

J   = 6    # joints
V   = 8    # cloth vertices
BV  = 8    # body vertices
B   = 2    # batch size


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
    """Face topology for a tiny box body mesh."""
    return torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5],
        [2, 6, 3], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=torch.long)


@pytest.fixture()
def config():
    return Config(
        formulation="mass_spring",
        device="cpu",
        lr=1e-3,
        w_physics=1.0,
        w_collision=0.1,
        w_reg=0.01,
        collision_threshold=0.005,
    )


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return NCSModel(n_joints=J, n_verts=V, latent_dim=64)


@pytest.fixture()
def trainer(model, config, garment_mesh, body_faces):
    gv, gf = garment_mesh
    return Trainer(model, config, gv, gf, body_faces)


def _make_batch():
    """Return a fixed-seed batch with body verts far from the cloth."""
    torch.manual_seed(99)
    return {
        # Current poses
        "pose": torch.randn(B, J, 3, 3),
        # Body mesh offset far from cloth (no collision expected)
        "body_verts": torch.randn(B, BV, 3) * 0.05 + 5.0,
    }


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

class TestTrainStep:

    def test_returns_expected_keys(self, trainer):
        losses = trainer.train_step(_make_batch())
        assert set(losses.keys()) == {"physics", "collision", "reg", "total"}

    def test_all_losses_finite(self, trainer):
        losses = trainer.train_step(_make_batch())
        for key, val in losses.items():
            assert torch.isfinite(torch.tensor(val)), (
                f"loss[{key!r}] = {val} is not finite"
            )

    def test_all_losses_non_negative(self, trainer):
        losses = trainer.train_step(_make_batch())
        for key, val in losses.items():
            assert val >= 0.0, f"loss[{key!r}] = {val} is negative"

    def test_step_count_increments(self, trainer):
        assert trainer.step_count == 0
        trainer.train_step(_make_batch())
        assert trainer.step_count == 1
        trainer.train_step(_make_batch())
        assert trainer.step_count == 2

    def test_total_equals_weighted_sum(self, trainer):
        cfg = trainer.config
        losses = trainer.train_step(_make_batch())
        expected = (
            cfg.w_physics    * losses["physics"]
            + cfg.w_collision * losses["collision"]
            + cfg.w_reg       * losses["reg"]
        )
        assert abs(losses["total"] - expected) < 1e-4, (
            f"total {losses['total']:.6f} != weighted sum {expected:.6f}"
        )


# ---------------------------------------------------------------------------
# Smoke test: loss should decrease
# ---------------------------------------------------------------------------

class TestSmokeConvergence:

    def test_loss_decreases_over_10_steps(self, trainer):
        """Training on a fixed batch for 10 Adam steps must reduce the total loss."""
        torch.manual_seed(42)
        batch = _make_batch()

        losses_seq = []
        for _ in range(10):
            losses_seq.append(trainer.train_step(batch)["total"])

        assert losses_seq[-1] < losses_seq[0], (
            f"Loss did not decrease: {losses_seq[0]:.6f} → {losses_seq[-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# Formulation variants
# ---------------------------------------------------------------------------

class TestFormulations:

    @pytest.mark.parametrize("formulation", ["mass_spring", "baraff_witkin", "stvk"])
    def test_formulation_runs(self, garment_mesh, body_faces, formulation):
        torch.manual_seed(0)
        gv, gf = garment_mesh
        cfg = Config(formulation=formulation, device="cpu", lr=1e-3,
                     w_physics=1.0, w_collision=0.0, w_reg=0.01)
        m = NCSModel(n_joints=J, n_verts=V, latent_dim=64)
        t = Trainer(m, cfg, gv, gf, body_faces)
        losses = t.train_step(_make_batch())
        assert torch.isfinite(torch.tensor(losses["total"]))

    def test_unknown_formulation_raises(self, garment_mesh, body_faces):
        gv, gf = garment_mesh
        cfg = Config(formulation="unknown", device="cpu")
        m = NCSModel(n_joints=J, n_verts=V, latent_dim=64)
        t = Trainer(m, cfg, gv, gf, body_faces)
        with pytest.raises(ValueError, match="Unknown"):
            t.train_step(_make_batch())


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

class TestCheckpoint:

    def test_save_creates_file(self, trainer, tmp_path):
        trainer.train_step(_make_batch())
        path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(path)
        assert path.exists()

    def test_load_restores_step_count(self, trainer, tmp_path):
        for _ in range(3):
            trainer.train_step(_make_batch())
        path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(path)

        m2 = NCSModel(n_joints=J, n_verts=V, latent_dim=64)
        gv, gf = trainer.garment_verts.cpu(), trainer.garment_faces.cpu()
        bf = trainer.body_faces.cpu()
        t2 = Trainer(m2, trainer.config, gv, gf, bf)
        t2.load_checkpoint(path)

        assert t2.step_count == 3

    def test_load_restores_weights(self, trainer, tmp_path):
        trainer.train_step(_make_batch())
        path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(path)

        m2 = NCSModel(n_joints=J, n_verts=V, latent_dim=64)
        gv, gf = trainer.garment_verts.cpu(), trainer.garment_faces.cpu()
        t2 = Trainer(m2, trainer.config, gv, gf, trainer.body_faces.cpu())
        t2.load_checkpoint(path)

        for (n1, p1), (n2, p2) in zip(
            trainer.model.named_parameters(),
            t2.model.named_parameters(),
        ):
            assert torch.allclose(p1.cpu(), p2.cpu()), (
                f"Parameter {n1!r} mismatch after checkpoint restore"
            )

    def test_save_creates_parent_dirs(self, trainer, tmp_path):
        path = tmp_path / "deep" / "nested" / "ckpt.pt"
        trainer.save_checkpoint(path)
        assert path.exists()
