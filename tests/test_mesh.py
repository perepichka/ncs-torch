"""Tests for ncs.mesh.mesh using a minimal tetrahedron mesh."""

import pytest
import torch

from ncs.mesh.mesh import (
    Mesh,
    compute_adjacency,
    compute_edge_lengths,
    compute_edges,
    compute_face_areas,
    compute_face_normals,
    compute_vertex_normals,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tetrahedron() -> Mesh:
    """Return a regular tetrahedron with 4 vertices and 4 triangular faces.

    Vertices are placed at positions that form a regular tetrahedron:
        v0 = ( 1,  1,  1)
        v1 = ( 1, -1, -1)
        v2 = (-1,  1, -1)
        v3 = (-1, -1,  1)

    The four faces use consistent outward-pointing winding.
    """
    vertices = torch.tensor(
        [
            [ 1.0,  1.0,  1.0],  # v0
            [ 1.0, -1.0, -1.0],  # v1
            [-1.0,  1.0, -1.0],  # v2
            [-1.0, -1.0,  1.0],  # v3
        ],
        dtype=torch.float32,
    )

    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
        ],
        dtype=torch.long,
    )

    return Mesh(vertices=vertices, faces=faces)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeEdges:
    def test_edge_count(self, tetrahedron: Mesh) -> None:
        """A tetrahedron has exactly 6 unique undirected edges."""
        edges = compute_edges(tetrahedron.faces)
        assert edges.shape == (6, 2), (
            f"Expected 6 edges, got {edges.shape[0]}"
        )

    def test_edges_directed_small_first(self, tetrahedron: Mesh) -> None:
        """Each edge row should satisfy edge[0] < edge[1]."""
        edges = compute_edges(tetrahedron.faces)
        assert torch.all(edges[:, 0] < edges[:, 1])

    def test_edges_unique(self, tetrahedron: Mesh) -> None:
        """No duplicate edges should appear."""
        edges = compute_edges(tetrahedron.faces)
        unique = torch.unique(edges, dim=0)
        assert unique.shape[0] == edges.shape[0]


class TestComputeEdgeLengths:
    def test_lengths_positive(self, tetrahedron: Mesh) -> None:
        edges = compute_edges(tetrahedron.faces)
        lengths = compute_edge_lengths(tetrahedron.vertices, edges)
        assert torch.all(lengths > 0.0), "All edge lengths must be positive."

    def test_lengths_shape(self, tetrahedron: Mesh) -> None:
        edges = compute_edges(tetrahedron.faces)
        lengths = compute_edge_lengths(tetrahedron.vertices, edges)
        assert lengths.shape == (edges.shape[0],)

    def test_lengths_regular_tetrahedron(self, tetrahedron: Mesh) -> None:
        """All edges of a regular tetrahedron have the same length (2√2)."""
        edges = compute_edges(tetrahedron.faces)
        lengths = compute_edge_lengths(tetrahedron.vertices, edges)
        expected = 2.0 * (2.0 ** 0.5)
        assert torch.allclose(lengths, torch.full_like(lengths, expected), atol=1e-5)


class TestComputeFaceNormals:
    def test_unit_normals(self, tetrahedron: Mesh) -> None:
        """Face normals must be unit vectors."""
        normals = compute_face_normals(tetrahedron.vertices, tetrahedron.faces)
        norms = torch.norm(normals, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
            f"Face normal norms: {norms}"
        )

    def test_shape(self, tetrahedron: Mesh) -> None:
        normals = compute_face_normals(tetrahedron.vertices, tetrahedron.faces)
        F = tetrahedron.faces.shape[0]
        assert normals.shape == (F, 3)


class TestComputeFaceAreas:
    def test_areas_positive(self, tetrahedron: Mesh) -> None:
        areas = compute_face_areas(tetrahedron.vertices, tetrahedron.faces)
        assert torch.all(areas > 0.0), "All face areas must be positive."

    def test_shape(self, tetrahedron: Mesh) -> None:
        areas = compute_face_areas(tetrahedron.vertices, tetrahedron.faces)
        F = tetrahedron.faces.shape[0]
        assert areas.shape == (F,)

    def test_equal_areas_regular_tetrahedron(self, tetrahedron: Mesh) -> None:
        """All faces of a regular tetrahedron have the same area (2√3)."""
        areas = compute_face_areas(tetrahedron.vertices, tetrahedron.faces)
        expected = 2.0 * (3.0 ** 0.5)
        assert torch.allclose(areas, torch.full_like(areas, expected), atol=1e-5)


class TestComputeVertexNormals:
    def test_unit_normals(self, tetrahedron: Mesh) -> None:
        """Vertex normals must be unit vectors after normalisation."""
        normals = compute_vertex_normals(tetrahedron.vertices, tetrahedron.faces)
        norms = torch.norm(normals, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
            f"Vertex normal norms: {norms}"
        )

    def test_shape(self, tetrahedron: Mesh) -> None:
        normals = compute_vertex_normals(tetrahedron.vertices, tetrahedron.faces)
        V = tetrahedron.vertices.shape[0]
        assert normals.shape == (V, 3)


class TestComputeAdjacency:
    def test_shape(self, tetrahedron: Mesh) -> None:
        """Adjacency table must have V rows."""
        V = tetrahedron.vertices.shape[0]
        adj = compute_adjacency(tetrahedron.faces, V)
        assert adj.shape[0] == V

    def test_regular_tetrahedron_valence(self, tetrahedron: Mesh) -> None:
        """Every vertex of a regular tetrahedron is connected to 3 others."""
        V = tetrahedron.vertices.shape[0]
        adj = compute_adjacency(tetrahedron.faces, V)
        for v in range(V):
            row = adj[v]
            n_neighbors = int((row >= 0).sum())
            assert n_neighbors == 3, (
                f"Vertex {v} has {n_neighbors} neighbours, expected 3."
            )

    def test_padding_value(self, tetrahedron: Mesh) -> None:
        """Unused adjacency slots must be -1."""
        V = tetrahedron.vertices.shape[0]
        adj = compute_adjacency(tetrahedron.faces, V)
        # For a regular tetrahedron max_neighbors == 3, so no padding here,
        # but this is still a valid structural check.
        assert torch.all(adj >= -1)

    def test_neighbors_are_valid_indices(self, tetrahedron: Mesh) -> None:
        """All non-padding neighbour indices must be in [0, V)."""
        V = tetrahedron.vertices.shape[0]
        adj = compute_adjacency(tetrahedron.faces, V)
        valid = adj[adj >= 0]
        assert torch.all(valid < V)
