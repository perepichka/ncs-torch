"""
Differentiable body-cloth collision penalty.

The body mesh is represented as a triangle soup (vertices + faces). For each
cloth vertex we approximate the signed distance to the body surface and
accumulate a soft quadratic penalty for any penetration.

Signed distance approximation
------------------------------
True mesh SDF requires expensive spatial data structures. We use a tractable
two-step proxy that remains fully differentiable:

1. **Distance** – compute the unsigned point-to-triangle distance from every
   cloth vertex to every body triangle; take the per-vertex minimum.
2. **Sign** – for each cloth vertex, form the vector from the nearest body
   triangle centroid to the cloth vertex and dot it with that triangle's
   outward normal. A negative dot product means the vertex is "behind" the
   face, i.e. inside the body. When multiple triangles contribute equally we
   use a soft minimum (log-sum-exp) weighted sign.

Penalty
-------
    loss = sum_v relu(threshold - sdf_v)^2

Points outside the body (sdf > threshold) contribute zero; points inside
accumulate a quadratic cost proportional to the penetration depth.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Point-to-triangle distance
# ---------------------------------------------------------------------------

def point_to_triangle_distance(
    points: Tensor,   # (N, 3)
    tri_v0: Tensor,   # (M, 3)
    tri_v1: Tensor,   # (M, 3)
    tri_v2: Tensor,   # (M, 3)
) -> Tensor:          # (N, M)
    """Compute the unsigned squared distance from each point to each triangle.

    The implementation follows the parametric closest-point projection of
    Eberly (2001): we express a point on the triangle as
        P(s,t) = v0 + s*(v1-v0) + t*(v2-v0),  s,t >= 0, s+t <= 1
    and solve the constrained minimisation analytically, clamping to the
    correct Voronoi region.  All operations are batched over both N points
    and M triangles simultaneously and are fully differentiable.

    Returns
    -------
    Tensor of shape (N, M) containing the *unsigned* Euclidean distances
    (not squared distances).
    """
    # Expand for broadcasting: points -> (N, 1, 3), triangles -> (1, M, 3)
    p  = points.unsqueeze(1)          # (N, 1, 3)
    v0 = tri_v0.unsqueeze(0)          # (1, M, 3)
    v1 = tri_v1.unsqueeze(0)
    v2 = tri_v2.unsqueeze(0)

    # Triangle edge vectors
    e0 = v1 - v0   # (1, M, 3)
    e1 = v2 - v0   # (1, M, 3)

    # Vector from v0 to point
    w = p - v0     # (N, M, 3)

    # Gramian entries (dot products of edge vectors)
    a = (e0 * e0).sum(-1)  # (1, M)
    b = (e0 * e1).sum(-1)
    c = (e1 * e1).sum(-1)

    # Projection of w onto edge vectors
    d = (w * e0).sum(-1)   # (N, M)
    e = (w * e1).sum(-1)

    det = a * c - b * b    # (1, M)

    # Unconstrained barycentric coords
    s = b * e - c * d      # (N, M)
    t = b * d - a * e

    # We now clamp (s, t) to the triangle's parameter domain.
    # There are 7 Voronoi regions; we handle each with a mask.
    # To keep the computation differentiable we compute all branches
    # and blend via masks (no in-place ops on leaf tensors).

    # Region 0: interior (s>=0, t>=0, s+t<=det)
    # s_bar, t_bar are the interior values (un-normalised)
    s_bar = s / det.clamp(min=1e-10)
    t_bar = t / det.clamp(min=1e-10)

    # ---- Edge/vertex clamps (scalar formula per region) ----
    # Each helper returns (s_clamped, t_clamped) for its region.

    def _clamp_edge01(s_, t_):
        """Region where t < 0: project onto edge v0-v1 (t=0)."""
        s_c = (d / a.clamp(min=1e-10)).clamp(0.0, 1.0)
        t_c = torch.zeros_like(s_c)
        return s_c, t_c

    def _clamp_edge02(s_, t_):
        """Region where s < 0: project onto edge v0-v2 (s=0)."""
        t_c = (e / c.clamp(min=1e-10)).clamp(0.0, 1.0)
        s_c = torch.zeros_like(t_c)
        return s_c, t_c

    def _clamp_edge12(s_, t_):
        """Region where s+t > det: project onto edge v1-v2 (s+t=1)."""
        # Along edge v1->v2: s+t=1, parameterised by s in [0,1].
        numer = c + e - b - d   # (N, M)
        denom = (a - 2.0 * b + c).clamp(min=1e-10)
        s_c = (numer / denom).clamp(0.0, 1.0)
        t_c = 1.0 - s_c
        return s_c, t_c

    # Compute all candidate (s, t) pairs
    s0, t0 = s_bar, t_bar                    # interior
    s1, t1 = _clamp_edge01(s, t)             # t < 0
    s2, t2 = _clamp_edge02(s, t)             # s < 0
    s3, t3 = _clamp_edge12(s, t)             # s+t > det

    # Determine regions via boolean masks (float for smooth blending)
    inside   = ((s >= 0) & (t >= 0) & (s + t <= det)).float()   # (N, M)
    neg_t    = (t < 0).float()
    neg_s    = ((s >= 0) & (t < 0)).float()   # will be overridden below
    # Standard region classification (Eberly):
    #   R0: s>=0, t>=0, s+t<=det        → interior projection
    #   R1: s>=0, t< 0                  → project to edge v0-v1
    #   R2: s< 0, t>=0                  → project to edge v0-v2
    #   R3: s>=0, t>=0, s+t> det       → project to edge v1-v2
    #   Corners at v0, v1, v2 are handled by the edge clamps automatically.

    r0 = ((s >= 0) & (t >= 0) & ((s + t) <= det)).float()
    r1 = (t < 0).float() * (1.0 - r0)
    r2 = (s < 0).float() * (1.0 - r0)
    r3 = ((s + t) > det).float() * (1.0 - r0)

    # If multiple non-interior regions overlap (e.g. s<0 and t<0), the
    # nearest vertex is v0. Both r1 and r2 will activate; both clamps return
    # (0,0) in that case so the result is correct.

    s_final = r0 * s0 + r1 * s1 + r2 * s2 + r3 * s3
    t_final = r0 * t0 + r1 * t1 + r2 * t2 + r3 * t3

    # Closest point on triangle
    closest = v0 + s_final.unsqueeze(-1) * e0 + t_final.unsqueeze(-1) * e1  # (N, M, 3)

    # Euclidean distance
    diff = p - closest   # (N, M, 3)
    dist = (diff * diff).sum(-1).clamp(min=0.0).sqrt()   # (N, M)
    return dist


# ---------------------------------------------------------------------------
# Signed distance approximation
# ---------------------------------------------------------------------------

def _face_normals(v0: Tensor, v1: Tensor, v2: Tensor) -> Tensor:
    """Outward face normals (unit length) for a triangle soup.

    Parameters
    ----------
    v0, v1, v2 : (M, 3) – triangle vertices in consistent winding order.

    Returns
    -------
    (M, 3) unit normals.
    """
    n = torch.cross(v1 - v0, v2 - v0, dim=-1)           # (M, 3)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    return n


def _approximate_sdf(
    cloth_verts: Tensor,   # (V, 3)
    body_verts: Tensor,    # (B_V, 3)
    body_faces: Tensor,    # (B_F, 3) long
) -> Tensor:               # (V,)
    """Per-cloth-vertex signed distance approximation.

    Algorithm
    ---------
    1. Gather triangle vertex positions from body mesh.
    2. Compute unsigned point-to-triangle distances: shape (V, B_F).
    3. For sign, use a soft-minimum weighted vote over face normals:
       - Compute soft-minimum weights via softmin over distances.
       - For each cloth vertex, compute dot(centroid->vertex, face_normal)
         weighted by softmin weights.
       - If weighted sum < 0, vertex is inside (negative SDF).
    4. sdf = sign * min_distance.

    Using softmin rather than a hard argmin keeps the sign estimation
    smooth and avoids discontinuities that would break gradients.

    Note: gradients flow only through the *distance* component; the sign
    is treated as a detached indicator so that the penalty gradient always
    points toward the body surface.
    """
    v0 = body_verts[body_faces[:, 0]]   # (B_F, 3)
    v1 = body_verts[body_faces[:, 1]]
    v2 = body_verts[body_faces[:, 2]]

    # (V, B_F) unsigned distances
    dists = point_to_triangle_distance(cloth_verts, v0, v1, v2)

    # Minimum unsigned distance per cloth vertex
    min_dist, _ = dists.min(dim=-1)   # (V,)

    # ---- Sign estimation via softmin-weighted normal vote ----
    normals = _face_normals(v0, v1, v2)                          # (B_F, 3)
    centroids = (v0 + v1 + v2) / 3.0                            # (B_F, 3)

    # Vector from each face centroid to each cloth vertex: (V, B_F, 3)
    # cloth_verts: (V, 3) -> (V, 1, 3); centroids: (1, B_F, 3)
    to_vert = cloth_verts.unsqueeze(1) - centroids.unsqueeze(0)  # (V, B_F, 3)

    # Dot with face normal: positive = outside face, negative = inside
    dot = (to_vert * normals.unsqueeze(0)).sum(-1)               # (V, B_F)

    # Softmin weights: faces closest to each vertex get highest weight.
    # Temperature controls sharpness; small positive keeps it stable.
    temperature = 0.01
    weights = F.softmax(-dists / temperature, dim=-1)            # (V, B_F)

    # Weighted vote: positive → outside, negative → inside
    signed_vote = (weights * dot).sum(-1)                        # (V,)

    # Sign: +1 outside, -1 inside (detached so gradient flows through dist)
    sign = torch.sign(signed_vote).detach()
    # Treat exactly zero as outside (conservative — no penalty for boundary)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    sdf = sign * min_dist   # (V,)
    return sdf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collision_loss(
    cloth_verts: Tensor,   # (V, 3)
    body_verts: Tensor,    # (B_V, 3)
    body_faces: Tensor,    # (B_F, 3) long
    threshold: float = 0.005,
) -> Tensor:               # scalar
    """Differentiable body–cloth collision penalty.

    Computes an approximate signed distance for each cloth vertex relative to
    the body mesh and penalises penetration with a soft quadratic loss:

        loss = sum_v relu(threshold - sdf_v)^2

    A cloth vertex is considered in collision when its signed distance to the
    body surface falls below *threshold* (i.e. it is either inside the body or
    within *threshold* metres of the surface).

    Parameters
    ----------
    cloth_verts : (V, 3) float tensor — current cloth vertex positions.
    body_verts  : (B_V, 3) float tensor — body mesh vertex positions.
    body_faces  : (B_F, 3) long tensor — triangle face indices into body_verts.
    threshold   : Safety margin in metres (default 0.005 m = 5 mm).

    Returns
    -------
    Scalar loss tensor.  Gradient flows through *cloth_verts* (and optionally
    *body_verts* if it requires grad).
    """
    sdf = _approximate_sdf(cloth_verts, body_verts, body_faces)   # (V,)

    # Penetration: sdf < threshold  →  penalty = relu(threshold - sdf)^2
    penetration = F.relu(threshold - sdf)   # (V,)
    loss = (penetration ** 2).sum()
    return loss
