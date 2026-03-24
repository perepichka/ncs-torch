"""
Tests for physics energy formulations.
Uses a minimal 2-triangle mesh (flat square split on diagonal).
"""
import pytest
import torch
from torch import Tensor

from ncs.physics.energy import (
    mass_spring_energy,
    baraff_witkin_energy,
    stvk_energy,
    bending_energy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_square():
    """
    Two-triangle flat square in XY plane.
    Vertices: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
    Faces: [0,1,2], [0,2,3]
    """
    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return verts, faces


@pytest.fixture
def edges():
    return torch.tensor([[0, 1], [1, 2], [2, 0], [2, 3], [3, 0]], dtype=torch.long)


@pytest.fixture
def rest_areas():
    # Each triangle is a right isoceles triangle with legs=1, area=0.5
    return torch.tensor([0.5, 0.5], dtype=torch.float64)


@pytest.fixture
def hinge_edges():
    # Shared edge 0-2, opposite verts 1 and 3
    return torch.tensor([[0, 2, 1, 3]], dtype=torch.long)


# ---------------------------------------------------------------------------
# Mass-Spring
# ---------------------------------------------------------------------------

class TestMassSpring:
    def test_zero_at_rest(self, flat_square, edges):
        verts, _ = flat_square
        e = mass_spring_energy(verts, verts, edges)
        assert e.item() == pytest.approx(0.0, abs=1e-10)

    def test_positive_under_stretch(self, flat_square, edges):
        rest, _ = flat_square
        deformed = rest.clone()
        deformed[1, 0] *= 2.0  # stretch vertex 1 in X
        e = mass_spring_energy(deformed, rest, edges)
        assert e.item() > 0.0

    def test_gradient_finite(self, flat_square, edges):
        verts, _ = flat_square
        v = verts.clone().requires_grad_(True)
        e = mass_spring_energy(v, verts, edges)
        e.backward()
        assert torch.isfinite(v.grad).all()


# ---------------------------------------------------------------------------
# Baraff-Witkin
# ---------------------------------------------------------------------------

class TestBaraffWitkin:
    def test_zero_at_rest(self, flat_square, rest_areas):
        verts, faces = flat_square
        e = baraff_witkin_energy(verts, verts, faces, rest_areas)
        assert e.item() == pytest.approx(0.0, abs=1e-8)

    def test_positive_under_stretch(self, flat_square, rest_areas):
        rest, faces = flat_square
        deformed = rest.clone()
        deformed[:, 0] *= 1.5  # uniform stretch in X
        e = baraff_witkin_energy(deformed, rest, faces, rest_areas)
        assert e.item() > 0.0

    def test_gradient_finite(self, flat_square, rest_areas):
        verts, faces = flat_square
        v = verts.clone().requires_grad_(True)
        e = baraff_witkin_energy(v, verts, faces, rest_areas)
        e.backward()
        assert torch.isfinite(v.grad).all()


# ---------------------------------------------------------------------------
# StVK
# ---------------------------------------------------------------------------

class TestStVK:
    def test_zero_at_rest(self, flat_square, rest_areas):
        verts, faces = flat_square
        e = stvk_energy(verts, verts, faces, rest_areas)
        assert e.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_under_stretch(self, flat_square, rest_areas):
        rest, faces = flat_square
        deformed = rest.clone()
        deformed[:, 0] *= 1.3
        e = stvk_energy(deformed, rest, faces, rest_areas)
        assert e.item() > 0.0

    def test_gradient_finite(self, flat_square, rest_areas):
        verts, faces = flat_square
        v = verts.clone().requires_grad_(True)
        e = stvk_energy(v, verts, faces, rest_areas)
        e.backward()
        assert torch.isfinite(v.grad).all()

    def test_gradcheck(self, flat_square, rest_areas):
        verts, faces = flat_square
        v = verts.clone().requires_grad_(True)
        ra = rest_areas.clone()

        def fn(v_):
            return stvk_energy(v_, verts.detach(), faces, ra)

        assert torch.autograd.gradcheck(fn, (v,), eps=1e-4, atol=1e-3)


# ---------------------------------------------------------------------------
# Bending
# ---------------------------------------------------------------------------

class TestBending:
    def test_zero_at_rest(self, flat_square, hinge_edges):
        verts, _ = flat_square
        e = bending_energy(verts, verts, hinge_edges)
        assert e.item() == pytest.approx(0.0, abs=1e-10)

    def test_positive_when_bent(self, flat_square, hinge_edges):
        rest, _ = flat_square
        deformed = rest.clone()
        # Lift vertex 1 out of plane
        deformed[1, 2] = 0.5
        e = bending_energy(deformed, rest, hinge_edges)
        assert e.item() > 0.0

    def test_gradient_finite(self, flat_square, hinge_edges):
        verts, _ = flat_square
        deformed = verts.clone()
        deformed[1, 2] = 0.3
        v = deformed.requires_grad_(True)
        e = bending_energy(v, verts, hinge_edges)
        e.backward()
        assert torch.isfinite(v.grad).all()
