"""
Differentiable cloth energy formulations.
All functions are pure (no nn.Module) to support torch.autograd.gradcheck.
"""
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triangle_bases(verts: Tensor, faces: Tensor):
    """Return edge vectors u=(v1-v0), v=(v2-v0) for each face. (F,3),(F,3)"""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return v1 - v0, v2 - v0


def _deformation_gradient(verts: Tensor, rest_verts: Tensor, faces: Tensor):
    """
    Compute per-face 3x2 deformation gradient F mapping rest→deformed.
    Returns F: (F, 3, 2)
    """
    u_def, v_def = _triangle_bases(verts, faces)       # (F, 3)
    u_rest, v_rest = _triangle_bases(rest_verts, faces)

    # Dm: rest-space 2x2 matrix [u_rest_2d | v_rest_2d]
    # We use 3D coordinates directly and compute F = [u_def|v_def] @ Dm^{-1}
    # Dm in 2D: project rest edges onto local frame
    # Build Ds (deformed) and Dm (rest) as (F,3,2)
    Ds = torch.stack([u_def, v_def], dim=2)   # (F, 3, 2)
    Dm = torch.stack([u_rest, v_rest], dim=2)  # (F, 3, 2)

    # Dm_inv: (F, 2, 2) pseudo-inverse via normal equations
    DmT = Dm.transpose(1, 2)           # (F, 2, 3)
    DmTDm = torch.bmm(DmT, Dm)        # (F, 2, 2)
    DmTDm_inv = torch.linalg.inv(DmTDm)  # (F, 2, 2)
    Dm_pinv = torch.bmm(DmTDm_inv, DmT)  # (F, 2, 3)

    F = torch.bmm(Ds, Dm_pinv.transpose(1, 2))  # (F, 3, 2)
    return F


# ---------------------------------------------------------------------------
# Mass-Spring Energy
# ---------------------------------------------------------------------------

def mass_spring_energy(
    verts: Tensor,       # (V, 3)
    rest_verts: Tensor,  # (V, 3)
    edges: Tensor,       # (E, 2) long
    stiffness: float = 1.0,
) -> Tensor:
    """Elastic energy: sum_e 0.5 * k * (|e| - |e_rest|)^2"""
    v0 = verts[edges[:, 0]]
    v1 = verts[edges[:, 1]]
    lengths = torch.norm(v1 - v0, dim=1)

    r0 = rest_verts[edges[:, 0]]
    r1 = rest_verts[edges[:, 1]]
    rest_lengths = torch.norm(r1 - r0, dim=1)

    strain = lengths - rest_lengths
    return 0.5 * stiffness * (strain ** 2).sum()


# ---------------------------------------------------------------------------
# Baraff-Witkin 1998 Energy
# ---------------------------------------------------------------------------

def baraff_witkin_energy(
    verts: Tensor,        # (V, 3)
    rest_verts: Tensor,   # (V, 3)
    faces: Tensor,        # (F, 3) long
    rest_areas: Tensor,   # (F,)
    k_stretch: float = 1.0,
    k_shear: float = 1.0,
) -> Tensor:
    """
    Baraff & Witkin 1998 stretch + shear energy on triangles.

    Stretch condition: C_u = a*(|w_u| - 1), C_v = a*(|w_v| - 1)
    Shear condition:   C_sh = a*(w_u . w_v)
    where w_u, w_v are deformed edge vectors in rest UV space.
    """
    u_rest, v_rest = _triangle_bases(rest_verts, faces)  # (F, 3)
    u_def, v_def = _triangle_bases(verts, faces)

    # Rest-space local frame (orthonormal u direction)
    rest_u_len = torch.norm(u_rest, dim=1, keepdim=True).clamp(min=1e-8)
    e1 = u_rest / rest_u_len  # (F, 3)

    rest_v_proj = v_rest - (v_rest * e1).sum(dim=1, keepdim=True) * e1
    rest_v_len = torch.norm(rest_v_proj, dim=1, keepdim=True).clamp(min=1e-8)
    e2 = rest_v_proj / rest_v_len

    # Deformed coordinates in rest frame
    wu = u_def / rest_u_len  # (F, 3)
    wv_raw = v_def - (v_def * e1).sum(dim=1, keepdim=True) * e1
    wv = wv_raw / rest_v_len  # (F, 3)

    wu_norm = torch.norm(wu, dim=1)  # (F,)
    wv_norm = torch.norm(wv, dim=1)

    a = rest_areas  # (F,)

    # Stretch energy
    E_stretch = 0.5 * k_stretch * (
        a * (wu_norm - 1.0) ** 2 + a * (wv_norm - 1.0) ** 2
    ).sum()

    # Shear energy
    wu_dot_wv = (wu * wv).sum(dim=1)
    E_shear = 0.5 * k_shear * (a * wu_dot_wv ** 2).sum()

    return E_stretch + E_shear


# ---------------------------------------------------------------------------
# Saint Venant-Kirchhoff (StVK) Energy
# ---------------------------------------------------------------------------

def stvk_energy(
    verts: Tensor,        # (V, 3)
    rest_verts: Tensor,   # (V, 3)
    faces: Tensor,        # (F, 3) long
    rest_areas: Tensor,   # (F,)
    young: float = 1e5,
    poisson: float = 0.3,
) -> Tensor:
    """
    Saint Venant-Kirchhoff energy via Green-Lagrange strain tensor.
    E = 0.5 * lam * tr(S)^2 + mu * tr(S^T S)
    where S = 0.5*(F^T F - I), F is the 2D deformation gradient.
    """
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))

    F = _deformation_gradient(verts, rest_verts, faces)  # (F, 3, 2)

    # Right Cauchy-Green: C = F^T F, shape (F, 2, 2)
    C = torch.bmm(F.transpose(1, 2), F)

    # Green-Lagrange strain: S = 0.5*(C - I)
    I2 = torch.eye(2, dtype=verts.dtype, device=verts.device).unsqueeze(0)
    S = 0.5 * (C - I2)  # (F, 2, 2)

    trS = S[:, 0, 0] + S[:, 1, 1]                 # (F,)
    trSTS = (S * S).sum(dim=(1, 2))                # (F,)

    energy_density = 0.5 * lam * trS ** 2 + mu * trSTS
    return (rest_areas * energy_density).sum()


# ---------------------------------------------------------------------------
# Bending Energy (discrete shells)
# ---------------------------------------------------------------------------

def bending_energy(
    verts: Tensor,          # (V, 3)
    rest_verts: Tensor,     # (V, 3)
    hinge_edges: Tensor,    # (H, 4) long: [v0, v1, v2, v3] - shared edge v0v1, opposite v2,v3
    k_bend: float = 1.0,
) -> Tensor:
    """
    Discrete bending energy: penalises change in dihedral angle.
    E = k * (cos θ - cos θ_rest)^2 per hinge
    """
    def dihedral_cos(v, e):
        """cos of dihedral angle for hinge edges e=(H,4) from verts v."""
        a = v[e[:, 0]]
        b = v[e[:, 1]]
        c = v[e[:, 2]]
        d = v[e[:, 3]]
        ab = b - a
        n1 = torch.cross(ab, c - a, dim=1)
        n2 = torch.cross(ab, d - a, dim=1)
        n1n = torch.norm(n1, dim=1, keepdim=True).clamp(min=1e-8)
        n2n = torch.norm(n2, dim=1, keepdim=True).clamp(min=1e-8)
        return (n1 / n1n * n2 / n2n).sum(dim=1)

    cos_def = dihedral_cos(verts, hinge_edges)
    cos_rest = dihedral_cos(rest_verts, hinge_edges)
    return 0.5 * k_bend * ((cos_def - cos_rest) ** 2).sum()
