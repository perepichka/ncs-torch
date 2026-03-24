"""Tests for ncs/model/network.py.

Covers:
  - Output shape (B, V, 3)
  - No NaNs or Infs in output
  - Gradient flow through all model parameters
  - Finite outputs with all-zero pose inputs
"""

import torch
import pytest

from ncs.model.network import PoseEncoder, StaticDecoder, DynamicDecoder, NCSModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B = 4    # batch size
J = 24   # number of joints
V = 100  # number of cloth vertices
LATENT_DIM = 256


@pytest.fixture()
def model() -> NCSModel:
    """Fresh NCSModel with deterministic weights (seed fixed per test)."""
    torch.manual_seed(0)
    return NCSModel(n_joints=J, n_verts=V, latent_dim=LATENT_DIM)


@pytest.fixture()
def random_poses():
    """Return (pose_t, pose_prev) as random rotation-like (B, J, 3, 3) tensors."""
    torch.manual_seed(42)
    pose_t = torch.randn(B, J, 3, 3)
    pose_prev = torch.randn(B, J, 3, 3)
    return pose_t, pose_prev


@pytest.fixture()
def template_verts() -> torch.Tensor:
    """Rest-pose cloth vertices (V, 3)."""
    torch.manual_seed(7)
    return torch.randn(V, 3)


# ---------------------------------------------------------------------------
# Tests: NCSModel.forward
# ---------------------------------------------------------------------------

class TestNCSModelForward:

    def test_output_shape(self, model, random_poses, template_verts):
        """Forward pass must return a tensor of shape (B, V, 3)."""
        pose_t, pose_prev = random_poses
        out = model(pose_t, pose_prev, template_verts)
        assert out.shape == (B, V, 3), (
            f"Expected shape ({B}, {V}, 3), got {tuple(out.shape)}"
        )

    def test_no_nan_or_inf(self, model, random_poses, template_verts):
        """Output must contain no NaN or Inf values."""
        pose_t, pose_prev = random_poses
        out = model(pose_t, pose_prev, template_verts)
        assert not torch.isnan(out).any(), "Output contains NaN values"
        assert not torch.isinf(out).any(), "Output contains Inf values"

    def test_gradient_flow(self, model, random_poses, template_verts):
        """Gradients must flow back to every trainable parameter."""
        pose_t, pose_prev = random_poses
        out = model(pose_t, pose_prev, template_verts)
        out.sum().backward()

        # Every parameter that requires grad should have a gradient
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"No gradient for parameter '{name}'"
                )
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient for parameter '{name}'"
                )

    def test_zero_pose_finite_output(self, model, template_verts):
        """All-zero pose inputs (first frame) must still produce finite outputs."""
        pose_t = torch.zeros(B, J, 3, 3)
        pose_prev = torch.zeros(B, J, 3, 3)

        out = model(pose_t, pose_prev, template_verts)

        assert out.shape == (B, V, 3), (
            f"Expected shape ({B}, {V}, 3), got {tuple(out.shape)}"
        )
        assert not torch.isnan(out).any(), "Output contains NaN with zero pose"
        assert not torch.isinf(out).any(), "Output contains Inf with zero pose"


# ---------------------------------------------------------------------------
# Tests: sub-modules in isolation
# ---------------------------------------------------------------------------

class TestPoseEncoder:

    def test_output_shape(self):
        torch.manual_seed(0)
        enc = PoseEncoder(n_joints=J, latent_dim=LATENT_DIM)
        pose = torch.randn(B, J, 3, 3)
        latent = enc(pose)
        assert latent.shape == (B, LATENT_DIM)

    def test_no_nan(self):
        torch.manual_seed(0)
        enc = PoseEncoder(n_joints=J, latent_dim=LATENT_DIM)
        pose = torch.randn(B, J, 3, 3)
        latent = enc(pose)
        assert not torch.isnan(latent).any()


class TestStaticDecoder:

    def test_output_shape(self):
        torch.manual_seed(0)
        dec = StaticDecoder(n_verts=V, latent_dim=LATENT_DIM)
        latent = torch.randn(B, LATENT_DIM)
        disp = dec(latent)
        assert disp.shape == (B, V, 3)

    def test_no_nan(self):
        torch.manual_seed(0)
        dec = StaticDecoder(n_verts=V, latent_dim=LATENT_DIM)
        latent = torch.randn(B, LATENT_DIM)
        disp = dec(latent)
        assert not torch.isnan(disp).any()


class TestDynamicDecoder:

    def test_output_shape(self):
        torch.manual_seed(0)
        dec = DynamicDecoder(n_verts=V, latent_dim=LATENT_DIM)
        latent_t = torch.randn(B, LATENT_DIM)
        latent_prev = torch.randn(B, LATENT_DIM)
        disp = dec(latent_t, latent_prev)
        assert disp.shape == (B, V, 3)

    def test_no_nan(self):
        torch.manual_seed(0)
        dec = DynamicDecoder(n_verts=V, latent_dim=LATENT_DIM)
        latent_t = torch.randn(B, LATENT_DIM)
        latent_prev = torch.randn(B, LATENT_DIM)
        disp = dec(latent_t, latent_prev)
        assert not torch.isnan(disp).any()

    def test_different_from_static(self):
        """Dynamic decoder with different t vs prev should differ from same-input case."""
        torch.manual_seed(0)
        dec = DynamicDecoder(n_verts=V, latent_dim=LATENT_DIM)
        latent_t = torch.randn(B, LATENT_DIM)
        latent_prev = torch.randn(B, LATENT_DIM)

        disp_mixed = dec(latent_t, latent_prev)
        disp_same = dec(latent_t, latent_t)

        # With different inputs the outputs should differ (highly likely for random init)
        assert not torch.allclose(disp_mixed, disp_same), (
            "Dynamic decoder output identical for different vs same latent inputs"
        )
