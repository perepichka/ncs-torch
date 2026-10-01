"""Prototype: auto skinning weights for a garment on Geno, headless Blender (bpy).

Methods:
  nearest : closest-point barycentric transfer only (baseline)
  rswt    : Robust Skin Weights Transfer (Abdrashitov et al. 2023): closest-point transfer where
            distance/normal tests pass, biharmonic inpainting elsewhere, then limit-4 + normalize.
Usage (venv with the `bpy` wheel + scipy, Python 3.11):  python skin_garment_prototype.py <Geno.fbx> <out_dir>
Prototype validated with bpy 5.0.1 on Geno + a procedural skirt; see docs/playground/SPEC.md section 9.
Production version (P4) takes an arbitrary garment OBJ instead of the built-in skirt.
"""
import sys, os, math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import bpy, bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

geno_fbx, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=geno_fbx)
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
body = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
bpy.context.view_layer.update()

# ---- body data in world space (rest/bind pose)
dg = bpy.context.evaluated_depsgraph_get()
bm = bmesh.new(); bm.from_mesh(body.data); bmesh.ops.triangulate(bm, faces=bm.faces[:])
bm.transform(body.matrix_world); bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
BV = np.array([v.co[:] for v in bm.verts]); BF = np.array([[v.index for v in f.verts] for f in bm.faces])
groups = [g.name for g in body.vertex_groups]
BW = np.zeros((len(body.data.vertices), len(groups)))
for v in body.data.vertices:
    for g in v.groups: BW[v.index, g.group] = g.weight
BW /= np.maximum(BW.sum(1, keepdims=True), 1e-8)
tree = BVHTree.FromBMesh(bm)
lo, hi = BV.min(0), BV.max(0)
print("Geno: %d verts, %d tris, %d groups, bbox %s..%s (m), armature '%s' %d bones" % (
    len(BV), len(BF), len(groups), np.round(lo, 3), np.round(hi, 3), arm.name, len(arm.data.bones)))
up = int(np.argmax(hi - lo)); print("up axis index:", up)

# ---- procedural garment: flared skirt from waist to knees (hardest case: legs split under it)
def bone_head(n): return np.array((arm.matrix_world @ arm.data.bones[n].head_local)[:])
hips, lknee, rknee = bone_head('Hips'), bone_head('LeftLeg'), bone_head('RightLeg')
spine = bone_head('Spine')
waist_h, hem_h = spine[up], 0.5 * (lknee[up] + rknee[up]) + 0.05
ring, rows = 64, 24
def body_radius(h):  # max horizontal extent of body verts in a slab around height h (torso/legs)
    m = np.abs(BV[:, up] - h) < 0.02
    hor = [a for a in range(3) if a != up]
    c = np.array([hips[hor[0]], hips[hor[1]]])
    d = np.linalg.norm(BV[m][:, hor] - c, axis=1)
    return d[d < 0.25].max()   # torso/legs only: A-pose hands hang at hip height
verts, faces = [], []
hor = [a for a in range(3) if a != up]
for r in range(rows):
    t = r / (rows - 1); h = waist_h + (hem_h - waist_h) * t
    rad = max(body_radius(waist_h), body_radius(h)) + 0.015 + t * 0.06
    for k in range(ring):
        a = 2 * math.pi * k / ring
        p = np.zeros(3); p[up] = h; p[hor[0]] = hips[hor[0]] + rad * math.cos(a); p[hor[1]] = hips[hor[1]] + rad * math.sin(a)
        verts.append(p)
for r in range(rows - 1):
    for k in range(ring):
        a, b = r * ring + k, r * ring + (k + 1) % ring
        faces += [(a, b, b + ring), (a, b + ring, a + ring)]
GV, GF = np.array(verts), np.array(faces)
print("skirt: %d verts, %d tris, waist %.2f m -> hem %.2f m" % (len(GV), len(GF), waist_h, hem_h))

def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for i in range(3): np.add.at(n, F[:, i], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
GN = vertex_normals(GV, GF)

# ---- closest point transfer
def closest_transfer(V, N, dist_thr, ang_thr_deg):
    W = np.zeros((len(V), BW.shape[1])); ok = np.zeros(len(V), bool)
    cos_thr = math.cos(math.radians(ang_thr_deg))
    for i, (p, n) in enumerate(zip(V, N)):
        loc, nrm, fi, d = tree.find_nearest(Vector(p))
        tri = BF[fi]; a, b, c = BV[tri]
        # barycentric of loc in triangle
        v0, v1, v2 = b - a, c - a, np.array(loc[:]) - a
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01; bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den
        bary = np.clip(np.array([1 - bv - bw, bv, bw]), 0, 1); bary /= bary.sum()
        W[i] = bary @ BW[tri]
        ok[i] = d < dist_thr and abs(np.dot(n, np.array(nrm[:]))) > cos_thr   # RSWT allows flipped normals
    return W, ok

def cotan_laplacian(V, F):
    n = len(V); I, J, X = [], [], []
    for k in range(3):
        i, j, o = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        e1, e2 = V[i] - V[o], V[j] - V[o]
        cot = np.einsum('ij,ij->i', e1, e2) / np.maximum(np.linalg.norm(np.cross(e1, e2), axis=1), 1e-12)
        I += [i, j]; J += [j, i]; X += [0.5 * cot, 0.5 * cot]
    I, J, X = np.concatenate(I), np.concatenate(J), np.concatenate(X)
    L = sp.coo_matrix((X, (I, J)), shape=(n, n)).tocsr()
    L = L - sp.diags(np.asarray(L.sum(1)).ravel())
    # lumped (barycentric) mass
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    m = np.zeros(n)
    for k in range(3): np.add.at(m, F[:, k], area / 3)
    return L, sp.diags(m)

def inpaint(V, F, W, ok):
    L, M = cotan_laplacian(V, F)
    Q = -L + L @ sp.diags(1.0 / M.diagonal()) @ L
    u, k = np.where(~ok)[0], np.where(ok)[0]
    if len(u) == 0: return W
    Quu, Quk = Q[u][:, u].tocsc(), Q[u][:, k]
    Wn = W.copy()
    Wn[u] = spla.spsolve(Quu, -(Quk @ W[k]))
    return Wn

def finalize(W, k=4):
    W = np.clip(W, 0, None)
    idx = np.argsort(-W, axis=1)[:, k:]
    np.put_along_axis(W, idx, 0, axis=1)
    return W / np.maximum(W.sum(1, keepdims=True), 1e-8)

diag = np.linalg.norm(hi - lo)
W0, ok = closest_transfer(GV, GN, dist_thr=0.05 * diag, ang_thr_deg=30)
W_near = finalize(closest_transfer(GV, GN, 1e9, 90)[0])
W_rswt = finalize(inpaint(GV, GF, W0, ok))
print("rswt: %d/%d verts matched by closest point, %d inpainted" % (ok.sum(), len(ok), (~ok).sum()))
for name, W in (("nearest", W_near), ("rswt", W_rswt)):
    print("  %-8s weights sum [%.4f, %.4f], max influences %d, finite %s" % (
        name, W.sum(1).min(), W.sum(1).max(), (W > 0).sum(1).max(), np.isfinite(W).all()))

# ---- evaluate: pose legs apart (stride) with LBS and measure skirt edge stretch
def bone_mats(rot):  # rot: dict bone -> (axis, deg) applied in pose mode; returns per-group 4x4 skinning matrices
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    for pb in arm.pose.bones: pb.rotation_mode = 'XYZ'; pb.rotation_euler = (0, 0, 0)
    for n, (ax, deg) in rot.items():
        e = [0, 0, 0]; e[ax] = math.radians(deg); arm.pose.bones[n].rotation_euler = e
    bpy.context.view_layer.update()
    mats = []
    for g in groups:
        pb = arm.pose.bones.get(g)
        mats.append(np.array(arm.matrix_world @ pb.matrix @ pb.bone.matrix_local.inverted() @ arm.matrix_world.inverted()) if pb else np.eye(4))
    bpy.ops.object.mode_set(mode='OBJECT')
    return np.array(mats)

def lbs(V, W, mats):
    Vh = np.c_[V, np.ones(len(V))]
    return np.einsum('vj,jab,vb->va', W, mats, Vh)[:, :3]

E = np.unique(np.sort(np.r_[GF[:, [0, 1]], GF[:, [1, 2]], GF[:, [2, 0]]], axis=1), axis=0)
rest_len = np.linalg.norm(GV[E[:, 0]] - GV[E[:, 1]], axis=1)
results = {}
for label, rot in (("stride", {'LeftUpLeg': (0, -45), 'RightUpLeg': (0, 35)}),
                   ("crouch", {'LeftUpLeg': (0, -80), 'RightUpLeg': (0, -80), 'LeftLeg': (0, 100), 'RightLeg': (0, 100)})):
    mats = bone_mats(rot)
    for name, W in (("nearest", W_near), ("rswt", W_rswt)):
        P = lbs(GV, W, mats)
        ratio = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1) / rest_len
        results[(label, name)] = ratio
        print("  pose %-6s %-8s edge ratio max %.2f | p99 %.2f | %%>1.2: %.1f%% | %%<0.8: %.1f%%" % (
            label, name, ratio.max(), np.percentile(ratio, 99), 100 * (ratio > 1.2).mean(), 100 * (ratio < 0.8).mean()))
bone_mats({})

# ---- write a skinned garment into the scene and export FBX (what the engine would load via ufbx)
me = bpy.data.meshes.new("skirt"); me.from_pydata(GV.tolist(), [], GF.tolist()); me.update()
ob = bpy.data.objects.new("skirt", me); bpy.context.scene.collection.objects.link(ob)
for j, g in enumerate(groups):
    vg = ob.vertex_groups.new(name=g)
    nz = np.where(W_rswt[:, j] > 0)[0]
    for i in nz: vg.add([int(i)], float(W_rswt[i, j]), 'REPLACE')
ob.parent = arm; mod = ob.modifiers.new("Armature", 'ARMATURE'); mod.object = arm
bpy.ops.object.select_all(action='DESELECT'); ob.select_set(True); arm.select_set(True)
out = os.path.join(out_dir, "skirt_skinned.fbx")
bpy.ops.export_scene.fbx(filepath=out, use_selection=True, add_leaf_bones=False)
print("exported", out, os.path.getsize(out), "bytes")
