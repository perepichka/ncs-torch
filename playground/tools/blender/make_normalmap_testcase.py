"""Normal-map validation test case: high-res garment -> simplified mesh -> baked tangent-space normal map.

Builds a high-res flared tube (skirt-like) with displaced folds, two vertical seams and a hem stitch,
decimates a copy (Blender Decimate/Collapse, UVs preserved), bakes a MikkTSpace tangent-space normal map
from high -> low with Cycles (off-the-shelf "selected to active" bake), and exports both meshes in a
tiny binary format read by playground/tools/normalmap_validate.

Usage (venv with the `bpy` wheel, Python 3.11):
    python make_normalmap_testcase.py <out_dir> [--ring 1024] [--rows 512] [--target-tris 4000] [--tex 2048]

Outputs in <out_dir>: hi.pgm, lo.pgm, normal.png, testcase.json
Coordinates are converted to the engine convention (Y-up, meters); UV v is flipped (v' = 1 - v) so that
row 0 of normal.png is sampled at v' = 0, matching how raylib uploads images.
"""
import argparse, json, math, os, struct, sys, time
import numpy as np
import bpy, addon_utils

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("--ring", type=int, default=1024)
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--target-tris", type=int, default=4000)
ap.add_argument("--tex", type=int, default=2048)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
os.makedirs(args.out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("cycles", default_set=True)

# ---------------------------------------------------------------- high-res garment (Blender Z-up)
H, R0, FLARE = 0.45, 0.17, 0.05           # height, waist radius, extra hem radius (m)
NU, NV = args.ring, args.rows

def displacement(th, t):
    """Radial displacement (m) at angle th, normalized height t (0 = waist, 1 = hem)."""
    folds = 0.006 * math.sin(14 * th + 1.5 * math.sin(3 * th + 2 * t)) * (0.3 + 0.7 * t)
    folds += 0.0025 * math.sin(31 * th - 4 * t) * t
    r = R0 + FLARE * t
    seam = 0.0
    for s in (0.25 * math.pi, 1.25 * math.pi):          # two vertical seams: 2 mm-wide grooves
        d = math.atan2(math.sin(th - s), math.cos(th - s)) * r
        seam -= 0.0015 * math.exp(-(d / 0.002) ** 2)
    hem = -0.0012 * math.exp(-((t - 0.92) * H / 0.0025) ** 2)   # hem stitch line
    waistband = 0.002 * (1.0 / (1.0 + math.exp((t - 0.08) / 0.01)))  # raised waistband
    return folds + seam + hem + waistband

verts = []
for j in range(NV + 1):
    t = j / NV
    z = H * (1 - t)
    for i in range(NU):
        th = 2 * math.pi * i / NU
        r = R0 + FLARE * t + displacement(th, t)
        verts.append((r * math.cos(th), r * math.sin(th), z))
faces, uvs = [], []
for j in range(NV):
    for i in range(NU):
        i1 = (i + 1) % NU
        a, b, c, d = j * NU + i, j * NU + i1, (j + 1) * NU + i1, (j + 1) * NU + i
        faces.append((a, b, c, d))
        u0, u1 = i / NU, (i + 1) / NU                  # wrap: last column uses u = 1.0 (UV seam, welded geometry)
        v0, v1 = 1 - j / NV, 1 - (j + 1) / NV
        uvs += [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
me = bpy.data.meshes.new("hi")
me.from_pydata(verts, [], faces)
uvl = me.uv_layers.new(name="UVMap")
uvl.data.foreach_set("uv", np.array(uvs, dtype=np.float32).ravel())
me.update()
hi = bpy.data.objects.new("hi", me); bpy.context.scene.collection.objects.link(hi)
hi.data.shade_smooth()

# ---------------------------------------------------------------- simplify (Decimate / Collapse, UV-preserving)
lo = hi.copy(); lo.data = hi.data.copy(); lo.name = "lo"; bpy.context.scene.collection.objects.link(lo)
n_tris_hi = 2 * len(faces)
mod = lo.modifiers.new("decimate", "DECIMATE")
mod.decimate_type = "COLLAPSE"; mod.ratio = min(1.0, args.target_tris / n_tris_hi)
mod.use_collapse_triangulate = True; mod.delimit = {"UV", "SEAM"}
bpy.context.view_layer.objects.active = lo
bpy.ops.object.modifier_apply(modifier=mod.name)
lo.data.shade_smooth()

# ---------------------------------------------------------------- bake tangent-space normal map (Cycles)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 1
img = bpy.data.images.new("normal", args.tex, args.tex, alpha=False, float_buffer=False)
img.colorspace_settings.name = "Non-Color"
mat = bpy.data.materials.new("bake"); mat.use_nodes = True
node = mat.node_tree.nodes.new("ShaderNodeTexImage"); node.image = img
mat.node_tree.nodes.active = node
lo.data.materials.append(mat)
bpy.ops.object.select_all(action="DESELECT"); hi.select_set(True); lo.select_set(True)
bpy.context.view_layer.objects.active = lo
t0 = time.time()
bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", normal_r="POS_X", normal_g="POS_Y", normal_b="POS_Z",
                    use_selected_to_active=True, cage_extrusion=0.012, max_ray_distance=0.03, margin=16)
bake_s = time.time() - t0
img.filepath_raw = os.path.join(args.out_dir, "normal.png"); img.file_format = "PNG"; img.save()

# ---------------------------------------------------------------- export (engine: Y-up; v' = 1 - v)
def to_engine(v):  # Blender (x, y, z) Z-up -> engine (x, z, -y) Y-up; proper rotation, bitangent sign unchanged
    return (v[0], v[2], -v[1])

def write_pgm(path, P, N, UV, T, weld, idx):
    """PGM1: u32 magic, u32 nverts, u32 nindices; per vertex 12 f32 (pos3 nrm3 uv2 tan4) + u32 weld id; u32 indices."""
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"PGM1", len(P), len(idx)))
        rec = np.zeros((len(P), 13), dtype=np.float32)
        rec[:, 0:3], rec[:, 3:6], rec[:, 6:8], rec[:, 8:12] = P, N, UV, T
        rec_u = rec.view(np.uint32); rec_u[:, 12] = weld
        f.write(rec.tobytes()); f.write(np.asarray(idx, dtype=np.uint32).tobytes())

# high: indexed per vertex (tangents unused)
mh = hi.data
P = np.array([to_engine(v.co) for v in mh.vertices], dtype=np.float32)
N = np.array([to_engine(v.normal) for v in mh.vertices], dtype=np.float32)
mh.calc_loop_triangles()
idx = np.array([t.vertices[:] for t in mh.loop_triangles], dtype=np.uint32).ravel()
write_pgm(os.path.join(args.out_dir, "hi.pgm"), P, N, np.zeros((len(P), 2)), np.zeros((len(P), 4)), np.arange(len(P)), idx)

# low: one vertex per triangle corner, with MikkTSpace tangents from Blender (same space the bake used)
ml = lo.data
ml.calc_tangents(uvmap="UVMap"); ml.calc_loop_triangles()
uvd = ml.uv_layers["UVMap"].data
P, N, UV, T, weld = [], [], [], [], []
for tri in ml.loop_triangles:
    for li in tri.loops:
        loop = ml.loops[li]
        P.append(to_engine(ml.vertices[loop.vertex_index].co))
        N.append(to_engine(ml.corner_normals[li].vector))
        u, v = uvd[li].uv
        UV.append((u, 1.0 - v))
        T.append(to_engine(loop.tangent) + (loop.bitangent_sign,))
        weld.append(loop.vertex_index)
write_pgm(os.path.join(args.out_dir, "lo.pgm"), np.array(P), np.array(N), np.array(UV), np.array(T), np.array(weld), np.arange(len(P)))

info = {"hi_verts": len(mh.vertices), "hi_tris": len(mh.loop_triangles), "lo_verts": len(ml.vertices),
        "lo_tris": len(ml.loop_triangles), "tex": args.tex, "bake_seconds": round(bake_s, 2),
        "blender": bpy.app.version_string, "tangent_space": "MikkTSpace (Blender)", "normal_map": "OpenGL (+Y green)"}
json.dump(info, open(os.path.join(args.out_dir, "testcase.json"), "w"), indent=2)
print(json.dumps(info))
