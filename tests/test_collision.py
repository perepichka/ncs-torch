"""Tests for ncs.collision.collision.

Body mesh: axis-aligned unit cube centred at the origin, with vertices at
(±0.5, ±0.5, ±0.5).  The 12 triangles use consistent outward-facing winding
(CCW when viewed from outside) so that the sign estimation in
``_approximate_sdf`` is correct.

Abbreviation key used in vertex indices below
----------------------------------------------
    lbn = left-bottom-near  (-0.5, -0.5, -0.5)  index 0
    rbn = right-bottom-near (+0.5, -0.5, -0.5)  index 1
    rtn = right-top-near    (+0.5, +0.5, -0.5)  index 2
    ltn = left-top-near     (-0.5, +0.5, -0.5)  index 3
    lbf = left-bottom-far   (-0.5, -0.5, +0.5)  index 4
    rbf = right-bottom-far  (+0.5, -0.5, +0.5)  index 5
    rtf = right-top-far     (+0.5, +0.5, +0.5)  index 6
    ltf = left-top-far      (-0.5, +0.5, +0.5)  index 7
"""

from __future__ import annotations

import torch
import pytest
from torch import Tensor

from ncs.collision.collision import collision_loss, point_to_triangle_distance


# ---------------------------------------------------------------------------
# Shared cube geometry
# ---------------------------------------------------------------------------

def _make_unit_cube() -> tuple[Tensor, Tensor]:
    """Return (vertices, faces) for an axis-aligned unit cube at the origin.

    The 8 vertices are placed at (±0.5, ±0.5, ±0.5).  Each of the 6 faces is
    split into 2 triangles (12 total) with consistent outward-facing winding.
    """
    verts = torch.tensor(
        [
            [-0.5, -0.5, -0.5],  # 0  lbn
            [ 0.5, -0.5, -0.5],  # 1  rbn
            [ 0.5,  0.5, -0.5],  # 2  rtn
            [-0.5,  0.5, -0.5],  # 3  ltn
            [-0.5, -0.5,  0.5],  # 4  lbf
            [ 0.5, -0.5,  0.5],  # 5  rbf
            [ 0.5,  0.5,  0.5],  # 6  rtf
            [-0.5,  0.5,  0.5],  # 7  ltf
        ],
        dtype=torch.float32,
    )

    faces = torch.tensor(
        [
            # -Z face (near), normal points in -Z direction
            [0, 2, 1],
            [0, 3, 2],
            # +Z face (far), normal points in +Z direction
            [4, 5, 6],
            [4, 6, 7],
            # -X face (left), normal points in -X direction
            [0, 4, 7],
            [0, 7, 3],
            # +X face (right), normal points in +X direction
            [1, 2, 6],
            [1, 6, 5],
            # -Y face (bottom), normal points in -Y direction
            [0, 1, 5],
            [0, 5, 4],
            # +Y face (top), normal points in +Y direction
            [3, 7, 6],
            [3, 6, 2],
        ],
        dtype=torch.long,
    )

    return verts, faces


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cube():
    return _make_unit_cube()


# ---------------------------------------------------------------------------
# Tests: point_to_triangle_distance
# ---------------------------------------------------------------------------

class TestPointToTriangleDistance:
    def test_output_shape(self) -> None:
        """Return shape must be (N, M)."""
        N, M = 5, 7
        points = torch.randn(N, 3)
        v0 = torch.randn(M, 3)
        v1 = torch.randn(M, 3)
        v2 = torch.randn(M, 3)
        dist = point_to_triangle_distance(points, v0, v1, v2)
        assert dist.shape == (N, M), f"Expected ({N}, {M}), got {dist.shape}"

    def test_distances_non_negative(self) -> None:
        """All unsigned distances must be >= 0."""
        points = torch.randn(8, 3)
        v0 = torch.randn(4, 3)
        v1 = torch.randn(4, 3)
        v2 = torch.randn(4, 3)
        dist = point_to_triangle_distance(points, v0, v1, v2)
        assert torch.all(dist >= 0.0), "Distances must be non-negative."

    def test_point_on_vertex_gives_zero(self) -> None:
        """A point coinciding with a triangle vertex should have distance 0."""
        v0 = torch.tensor([[0.0, 0.0, 0.0]])
        v1 = torch.tensor([[1.0, 0.0, 0.0]])
        v2 = torch.tensor([[0.0, 1.0, 0.0]])
        point = v0.clone()  # exactly on v0
        dist = point_to_triangle_distance(point, v0, v1, v2)
        assert dist.shape == (1, 1)
        assert dist.item() < 1e-5, f"Expected ~0, got {dist.item()}"

    def test_point_on_face_interior_gives_zero(self) -> None:
        """A point in the interior of the triangle plane should have distance 0."""
        v0 = torch.tensor([[0.0, 0.0, 0.0]])
        v1 = torch.tensor([[1.0, 0.0, 0.0]])
        v2 = torch.tensor([[0.0, 1.0, 0.0]])
        # Centroid is inside the triangle
        centroid = torch.tensor([[(0.0 + 1.0 + 0.0) / 3, (0.0 + 0.0 + 1.0) / 3, 0.0]])
        dist = point_to_triangle_distance(centroid, v0, v1, v2)
        assert dist.item() < 1e-5, f"Expected ~0, got {dist.item()}"

    def test_point_above_face_gives_height(self) -> None:
        """A point directly above the centroid should have distance == height."""
        v0 = torch.tensor([[0.0, 0.0, 0.0]])
        v1 = torch.tensor([[1.0, 0.0, 0.0]])
        v2 = torch.tensor([[0.0, 1.0, 0.0]])
        height = 0.3
        centroid_xy = torch.tensor([[(0.0 + 1.0 + 0.0) / 3, (0.0 + 0.0 + 1.0) / 3, height]])
        dist = point_to_triangle_distance(centroid_xy, v0, v1, v2)
        assert abs(dist.item() - height) < 1e-5, (
            f"Expected height {height}, got {dist.item()}"
        )

    def test_differentiable_wrt_points(self) -> None:
        """Distances must have finite gradients with respect to input points."""
        points = torch.randn(3, 3, requires_grad=True, dtype=torch.float32)
        v0 = torch.randn(2, 3)
        v1 = torch.randn(2, 3)
        v2 = torch.randn(2, 3)
        dist = point_to_triangle_distance(points, v0, v1, v2)
        dist.sum().backward()
        assert points.grad is not None
        assert torch.all(torch.isfinite(points.grad)), (
            "Gradients contain non-finite values."
        )


# ---------------------------------------------------------------------------
# Tests: collision_loss — outside body
# ---------------------------------------------------------------------------

class TestCollisionLossOutside:
    def test_vertex_far_outside_zero_loss(self, cube) -> None:
        """A cloth vertex far outside the cube should produce zero loss."""
        body_verts, body_faces = cube
        # Place cloth vertex 2 units away from the cube surface
        cloth_verts = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        loss = collision_loss(cloth_verts, body_verts, body_faces, threshold=0.005)
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            f"Expected zero loss for vertex outside cube, got {loss.item()}"
        )

    def test_multiple_vertices_outside_zero_loss(self, cube) -> None:
        """Multiple cloth vertices all outside the cube → zero total loss."""
        body_verts, body_faces = cube
        cloth_verts = torch.tensor(
            [
                [ 2.0,  0.0,  0.0],
                [-2.0,  0.0,  0.0],
                [ 0.0,  2.0,  0.0],
                [ 0.0, -2.0,  0.0],
                [ 0.0,  0.0,  2.0],
                [ 0.0,  0.0, -2.0],
            ],
            dtype=torch.float32,
        )
        loss = collision_loss(cloth_verts, body_verts, body_faces, threshold=0.005)
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            f"Expected zero loss for vertices outside cube, got {loss.item()}"
        )

    def test_vertex_just_outside_threshold_zero_loss(self, cube) -> None:
        """A vertex just beyond the threshold distance should have zero loss."""
        body_verts, body_faces = cube
        # Surface is at x=0.5; place cloth vertex at x = 0.5 + 2*threshold
        threshold = 0.01
        cloth_verts = torch.tensor([[0.5 + 2 * threshold, 0.0, 0.0]], dtype=torch.float32)
        loss = collision_loss(cloth_verts, body_verts, body_faces, threshold=threshold)
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            f"Expected zero loss for vertex beyond threshold, got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Tests: collision_loss — inside body
# ---------------------------------------------------------------------------

class TestCollisionLossInside:
    def test_vertex_at_origin_inside_cube(self, cube) -> None:
        """The origin is the centre of the cube → loss must be > 0."""
        body_verts, body_faces = cube
        cloth_verts = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        loss = collision_loss(cloth_verts, body_verts, body_faces, threshold=0.005)
        assert loss.item() > 0.0, (
            f"Expected positive loss for vertex at cube centre, got {loss.item()}"
        )

    def test_deeper_penetration_larger_loss(self, cube) -> None:
        """A vertex deeper inside the body should produce a larger penalty."""
        body_verts, body_faces = cube
        # Vertex at centre is deeper inside than vertex near a face
        cloth_at_centre = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        cloth_near_face = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float32)
        loss_centre = collision_loss(
            cloth_at_centre, body_verts, body_faces, threshold=0.005
        )
        loss_near = collision_loss(
            cloth_near_face, body_verts, body_faces, threshold=0.005
        )
        assert loss_centre.item() > loss_near.item(), (
            "Centre vertex should have larger loss than near-surface vertex."
        )

    def test_multiple_inside_vertices_additive(self, cube) -> None:
        """Two inside vertices should produce more loss than one alone."""
        body_verts, body_faces = cube
        single = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        double = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32)
        loss_single = collision_loss(single, body_verts, body_faces, threshold=0.005)
        loss_double = collision_loss(double, body_verts, body_faces, threshold=0.005)
        assert loss_double.item() > loss_single.item(), (
            "Two inside vertices should produce larger total loss."
        )

    def test_loss_is_scalar(self, cube) -> None:
        """collision_loss must return a zero-dimensional (scalar) tensor."""
        body_verts, body_faces = cube
        cloth_verts = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        loss = collision_loss(cloth_verts, body_verts, body_faces)
        assert loss.ndim == 0, f"Expected scalar, got ndim={loss.ndim}"


# ---------------------------------------------------------------------------
# Tests: gradients
# ---------------------------------------------------------------------------

class TestCollisionGradients:
    def test_grad_flows_through_cloth_verts_inside(self, cube) -> None:
        """Gradient of loss w.r.t. cloth_verts must be finite when inside."""
        body_verts, body_faces = cube
        cloth_verts = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True
        )
        loss = collision_loss(cloth_verts, body_verts, body_faces, threshold=0.005)
        loss.backward()
        assert cloth_verts.grad is not None, "No gradient computed for cloth_verts."
        assert torch.all(torch.isfinite(cloth_verts.grad)), (
            f"Non-finite gradient: {cloth_verts.grad}"
        )

    def test_grad_flows_through_cloth_verts_outside(self, cube) -> None:
        """Gradient of loss w.r.t. cloth_verts must be finite when outside."""
        body_verts, body_faces = cube
        cloth_verts = torch.tensor(
            [[2.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True
        )
        loss = collision_loss(cloth_verts, body_verts, body_faces, threshold=0.005)
        loss.backward()
        assert cloth_verts.grad is not None
        assert torch.all(torch.isfinite(cloth_verts.grad)), (
            f"Non-finite gradient: {cloth_verts.grad}"
        )

    def test_grad_flows_through_body_verts(self, cube) -> None:
        """Gradient must also flow through body_verts when it requires grad."""
        _, body_faces = cube
        body_verts = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [ 0.5, -0.5, -0.5],
                [ 0.5,  0.5, -0.5],
                [-0.5,  0.5, -0.5],
                [-0.5, -0.5,  0.5],
                [ 0.5, -0.5,  0.5],
                [ 0.5,  0.5,  0.5],
                [-0.5,  0.5,  0.5],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        cloth_verts = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        loss = collision_loss(cloth_verts, body_verts, body_faces, threshold=0.005)
        loss.backward()
        assert body_verts.grad is not None
        assert torch.all(torch.isfinite(body_verts.grad)), (
            f"Non-finite gradient in body_verts: {body_verts.grad}"
        )

    def test_gradcheck_point_to_triangle_distance(self) -> None:
        """gradcheck on point_to_triangle_distance with a minimal 1-point, 1-triangle case."""
        # Use float64 for gradcheck (it requires higher precision)
        torch.manual_seed(0)
        # Place the point above the interior of the triangle so the closest
        # point is in the face interior — gradient is well-defined there.
        points = torch.tensor(
            [[0.25, 0.25, 0.5]], dtype=torch.float64, requires_grad=True
        )
        v0 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        v1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        v2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)

        passed = torch.autograd.gradcheck(
            point_to_triangle_distance,
            inputs=(points, v0, v1, v2),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-3,
            raise_on_error=False,
        )
        assert passed, "gradcheck failed for point_to_triangle_distance."

    def test_gradcheck_collision_loss_tiny(self) -> None:
        """gradcheck on collision_loss with one cloth vertex and a minimal body.

        The body is a single triangle (degenerate mesh, sufficient for checking
        differentiability of the distance computation path).  We use a point
        clearly above the triangle so the forward pass is in a smooth region.
        """
        torch.manual_seed(42)

        # Single equilateral-ish triangle in the XY plane
        body_verts = torch.tensor(
            [
                [0.0,  0.0, 0.0],
                [1.0,  0.0, 0.0],
                [0.5,  1.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        body_faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

        # Cloth vertex directly above the triangle — deep enough that
        # relu(threshold - sdf) > 0 so the loss is non-trivially differentiable.
        cloth_verts = torch.tensor(
            [[0.45, 0.4, -0.3]],   # below the XY plane → inside the half-space
            dtype=torch.float64,
            requires_grad=True,
        )

        def _loss(cv, bv):
            return collision_loss(cv, bv, body_faces, threshold=1.0)

        passed = torch.autograd.gradcheck(
            _loss,
            inputs=(cloth_verts, body_verts),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-3,
            raise_on_error=False,
        )
        assert passed, "gradcheck failed for collision_loss."
