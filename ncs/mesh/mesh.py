"""Mesh utilities for cloth simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Mesh:
    """Simple mesh container.

    Attributes:
        vertices: Float tensor of shape (V, 3).
        faces:    Long tensor of shape (F, 3).
    """

    vertices: Tensor  # (V, 3)
    faces: Tensor     # (F, 3) dtype=long


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_obj(path: str, device=None) -> Mesh:
    """Load a Wavefront .obj file and return a :class:`Mesh`.

    Uses *trimesh* for loading.  Quad faces are triangulated automatically.

    Args:
        path:   Path to the .obj file.
        device: Target torch device (default: CPU).

    Returns:
        A :class:`Mesh` with tensors on *device*.
    """
    import trimesh  # optional dependency – imported lazily

    scene_or_mesh = trimesh.load(path, force="mesh", process=False)

    # trimesh may return a Scene when multiple objects are present.
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = list(scene_or_mesh.geometry.values())
        if len(meshes) == 0:
            raise ValueError(f"No geometry found in {path!r}")
        # Concatenate all sub-meshes into one.
        mesh_obj = trimesh.util.concatenate(meshes)
    else:
        mesh_obj = scene_or_mesh

    # trimesh triangulates quads during load when force="mesh".
    # Make sure faces are triangles (F, 3).
    faces_np = mesh_obj.faces          # already triangulated
    verts_np = mesh_obj.vertices       # (V, 3) float64

    vertices = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces    = torch.tensor(faces_np, dtype=torch.long,    device=device)

    return Mesh(vertices=vertices, faces=faces)


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def compute_edges(faces: Tensor) -> Tensor:
    """Return unique undirected edges of a triangle mesh.

    Args:
        faces: Long tensor of shape (F, 3).

    Returns:
        Long tensor of shape (E, 2) where each row ``[i, j]`` satisfies
        ``i < j``.  Edges are unique and sorted.
    """
    # Build the three directed half-edges for each face.
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]

    all_edges = torch.cat([e01, e12, e20], dim=0)  # (3F, 2)

    # Normalise direction so that the smaller index comes first.
    all_edges, _ = torch.sort(all_edges, dim=1)

    # Remove duplicate edges.
    unique_edges = torch.unique(all_edges, dim=0)   # (E, 2)

    return unique_edges


# ---------------------------------------------------------------------------
# Geometric quantities
# ---------------------------------------------------------------------------

def compute_edge_lengths(vertices: Tensor, edges: Tensor) -> Tensor:
    """Compute the Euclidean length of each edge.

    Args:
        vertices: Float tensor of shape (V, 3).
        edges:    Long tensor of shape (E, 2).

    Returns:
        Float tensor of shape (E,).
    """
    v0 = vertices[edges[:, 0]]  # (E, 3)
    v1 = vertices[edges[:, 1]]  # (E, 3)
    return torch.norm(v1 - v0, dim=1)  # (E,)


def compute_face_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """Compute unit face normals via the cross product of two edge vectors.

    Args:
        vertices: Float tensor of shape (V, 3).
        faces:    Long tensor of shape (F, 3).

    Returns:
        Float tensor of shape (F, 3) – unit normals.
    """
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    normals = torch.linalg.cross(e1, e2)           # (F, 3)
    norms   = torch.norm(normals, dim=1, keepdim=True).clamp(min=1e-12)
    return normals / norms


def compute_face_areas(vertices: Tensor, faces: Tensor) -> Tensor:
    """Compute the area of each triangular face.

    Args:
        vertices: Float tensor of shape (V, 3).
        faces:    Long tensor of shape (F, 3).

    Returns:
        Float tensor of shape (F,) – triangle areas.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    cross = torch.linalg.cross(e1, e2)  # (F, 3)
    return 0.5 * torch.norm(cross, dim=1)  # (F,)


def compute_vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """Compute area-weighted vertex normals, then normalise.

    Each vertex normal is the weighted sum of the normals of the faces it
    belongs to, weighted by the face area.

    Args:
        vertices: Float tensor of shape (V, 3).
        faces:    Long tensor of shape (F, 3).

    Returns:
        Float tensor of shape (V, 3) – unit vertex normals.
    """
    n_verts = vertices.shape[0]

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    # Cross product magnitude is 2 * area; use directly as area weight.
    weighted_normals = torch.linalg.cross(e1, e2)  # (F, 3)

    vertex_normals = torch.zeros_like(vertices)  # (V, 3)

    # Accumulate weighted normals for each corner of every face.
    for corner in range(3):
        idx = faces[:, corner]  # (F,)
        vertex_normals.index_add_(0, idx, weighted_normals)

    norms = torch.norm(vertex_normals, dim=1, keepdim=True).clamp(min=1e-12)
    return vertex_normals / norms


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def compute_adjacency(faces: Tensor, n_verts: int) -> Tensor:
    """Build a vertex adjacency table padded with ``-1``.

    Args:
        faces:   Long tensor of shape (F, 3).
        n_verts: Total number of vertices V.

    Returns:
        Long tensor of shape (V, max_neighbors) where ``max_neighbors`` is
        the maximum vertex valence in the mesh.  Unused slots are filled
        with ``-1``.
    """
    # Collect neighbours for each vertex using the undirected edges.
    edges = compute_edges(faces)  # (E, 2)

    # Build an adjacency list as a plain Python list of sets for flexibility.
    adj: list[set[int]] = [set() for _ in range(n_verts)]
    edges_list = edges.tolist()
    for i, j in edges_list:
        adj[i].add(j)
        adj[j].add(i)

    max_neighbors = max((len(s) for s in adj), default=0)

    table = torch.full(
        (n_verts, max_neighbors),
        fill_value=-1,
        dtype=torch.long,
        device=faces.device,
    )

    for v, neighbors in enumerate(adj):
        nb_list = sorted(neighbors)
        table[v, : len(nb_list)] = torch.tensor(nb_list, dtype=torch.long)

    return table
