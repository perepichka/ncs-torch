# NCS Playground: Engine Spec

A small real-time engine for **evaluating neural cloth and body models** on a
character driven by **motion matching** (LAFAN1). It is a playground built for
fast iteration. It is not a game engine.

> **Status:** draft spec. Decisions marked **[D#]** are open and tracked in
> [`DECISIONS.md`](DECISIONS.md). The defaults here are *recommendations*, not
> commitments. Before you implement a phase, resolve its decisions with the user.
> **Ask. Don't assume.**

---

## 1. Goals and non-goals

### Goals
1. Drive a skinned character interactively (gamepad or keyboard) with motion
   matching built on LAFAN1.
2. Plug in neural deformers (garments, body correctives) through **ONNX
   Runtime**, **PyTorch**, or **custom CUDA**, and swap them at runtime.
3. Render with modern shading that looks good enough to judge cloth quality:
   PBR, IBL, shadows, a sheen cloth BRDF and HDR tonemapping.
4. Measure the models live (penetration, stretch, jitter, inference time) and
   record reproducible runs.
5. Keep iteration fast: hot reload shaders, configs, models and custom Python
   deformers. No bake step except the motion-matching database and the IBL
   cache.

### Non-goals (v1)
- General scene graph, editor, prefab system or asset streaming
- Rigid-body physics, environment collision or terrain adaptation (flat ground only)
- Animation state machines or blend trees (motion matching replaces them)
- Learned Motion Matching (possible later as a deformer-style plug-in)
- Deferred or clustered lighting, ray tracing, TAA, hair, audio, networking
- macOS, web or mobile targets **[D2]**

---

## 2. Stack (recommended) **[D1]**

| Concern | Choice | Why |
|---|---|---|
| Host language | **Python 3.10+** | Same language as `ncs/`, so trained models load directly and iteration is fast |
| GPU API | **OpenGL 4.5 core via `moderngl`** | Simple; has compute shaders and SSBOs; CUDA–GL interop exists; headless rendering via EGL |
| Window / input | `glfw` | Gamepad mapping (SDL DB), keyboard and mouse |
| UI | `imgui-bundle` | Dear ImGui + ImPlot (live metric plots) + ImGuizmo |
| Tensors / CUDA | `torch` (CUDA if available, CPU fallback) | Already a dependency; custom kernels via Triton or `cpp_extension` |
| NN inference | `onnxruntime-gpu` (CUDA EP, CPU EP fallback) | IOBinding on torch CUDA pointers means no host copies |
| Assets | glTF 2.0 (`.glb`), OBJ, BVH, SMPL `.npz` | glTF for props and materials, OBJ for garments (matches `ncs`) |
| Config | YAML → dataclasses | Mirrors `ncs/config.py` |
| Capture | `imageio` + ffmpeg pipe | PNG screenshots, MP4 recordings |

Hot reload uses file-mtime polling once per second, so no file-watcher
dependency is needed.

**One rule keeps this simple:** *animation math runs on the CPU in numpy
(about 22 joints). Geometry runs in torch on `sim.device`.* The renderer only
draws vertex buffers. It never skins.

---

## 3. Conventions

- World is **Y-up, right-handed, meters**. Character forward is **+Z** (glTF convention).
- LAFAN1 is in cm. Convert to m on load. SMPL is already Y-up and in meters.
- Quaternions are `(w, x, y, z)` float32, and the database stores rotations as unit quaternions.
- Joint rotations go to models as `(J, 3, 3)` rotation matrices by default
  (that is what `NCSModel` takes). The manifest can ask for quaternions or axis-angle.
- Simulation uses a **fixed tick** of `sim.tick_hz` (default 30 Hz to match LAFAN1 **[D7]**).
  Rendering runs at vsync and **interpolates** between the last two sim states:
  slerp for joints, lerp for vertices. That adds one tick of latency, which is acceptable.
- Everything is `float32`. Batch dim is 1 at runtime. Shapes follow `ncs`: `(V, 3)` verts, `(F, 3)` faces.

---

## 4. Architecture

```
playground/
├── app.py              # entry point, main loop, CLI (--scene, --headless, --replay)
├── core/               # clock (fixed tick), input (gamepad/kbm), config, hot-reload, logging
├── anim/               # quat math, Skeleton, BVH loader, FK, retarget, foot IK (later)
├── mm/                 # database build, features, search, inertialization, controller
├── body/               # BodyModel: blendshapes + LBS (torch), SMPL loader, glTF skin loader
├── deform/             # Deformer protocol, manifest, backends: lbs, torch, onnx, custom
├── render/             # renderer, materials, IBL, shadows, post, debug draw
│   └── shaders/        # GLSL (hot-reloadable), shared includes
├── eval/               # metrics, recorder (inputs/poses/meshes/video), replay
├── ui/                 # ImGui panels
├── scenes/             # scene YAMLs (default.yaml, turntable.yaml, eval_track.yaml)
└── assets/             # small redistributable assets (procedural garments, materials)
tools/
├── fetch_assets.py     # LAFAN1 + CC0 HDRIs/textures -> data/ (git-ignored)
├── build_mm_db.py      # BVH -> data/mm/<name>.npz
└── export_onnx.py      # ncs checkpoint -> .onnx + manifest
```

Location is `playground/` in this repo by default **[D3]**. It imports `ncs` only for
metrics (`ncs.physics`, `ncs.collision`, `ncs.mesh`) and the torch deformer
adapter. `ncs` never imports `playground`.

### Main loop

```python
while app.running:
    input.poll()
    acc += clock.frame_dt()
    while acc >= tick_dt:                       # fixed-step simulation
        controller.update(input, camera, tick_dt)   # desired velocity/facing, springs
        anim_pose = anim_source.step(tick_dt)       # MotionMatching | ClipPlayer | Replay
        body_pose = retarget(anim_pose)             # LAFAN skeleton -> body skeleton
        body.update(body_pose, betas)               # blendshapes + LBS  (torch)
        for g in garments: g.verts = g.deformer.step(ctx)
        metrics.update(ctx); recorder.update(ctx)
        acc -= tick_dt
    renderer.draw(scene, alpha=acc / tick_dt)   # interpolated
    ui.draw()
```

`--headless` runs the same loop. It either skips `renderer.draw` (metrics only)
or renders offscreen through EGL (captures). Use it for batch evaluation and CI.

---

## 5. Animation core (`anim/`)

- **Skeleton:** `names`, `parents (J,)`, `rest_offsets (J,3)`, `rest_rots (J,4)`.
- **Pose:** local `rots (J,4)`, `root_pos (3,)`, plus derived globals from FK.
- **BVH loader:** reads any channel order, converts to quaternions and meters,
  and returns `(Skeleton, Clip)`. Tested with round-trip and FK checks against
  known LAFAN1 frames.
- **Quat library** (vectorized numpy): mul, inv, slerp, from/to matrix,
  axis-angle, `log`/`exp`, and the helpers Holden uses (`quat_abs`,
  `quat_to_scaled_angle_axis`).
- **Retarget** (`anim/retarget.py`, runtime, per tick) **[D5]**:
  1. Map joints by name using a table in YAML (see below). Unmapped target joints stay at rest.
  2. For each mapped joint, transfer the *global* rotation delta from the rest
     pose, then convert back to local on the target.
  3. Scale root translation by `target_hip_height / source_hip_height`.

  Default map, LAFAN1 (22 joints) → SMPL (24 joints). *Check the names against
  the BVH header.*

  | LAFAN1 | SMPL | LAFAN1 | SMPL |
  |---|---|---|---|
  | Hips | pelvis | Spine / Spine1 / Spine2 | spine1 / spine2 / spine3 |
  | LeftUpLeg / RightUpLeg | L_hip / R_hip | Neck / Head | neck / head |
  | LeftLeg / RightLeg | L_knee / R_knee | LeftShoulder / RightShoulder | L_collar / R_collar |
  | LeftFoot / RightFoot | L_ankle / R_ankle | LeftArm / RightArm | L_shoulder / R_shoulder |
  | LeftToe / RightToe | L_foot / R_foot | LeftForeArm / RightForeArm | L_elbow / R_elbow |
  | — | L_hand / R_hand (rest) | LeftHand / RightHand | L_wrist / R_wrist |

- **Foot locking + two-bone IK:** optional, off by default, later phase. Uses the contact labels in the database.

---

## 6. Motion matching (`mm/`)

The design follows Clavet (GDC 2016) and Holden's open-source reference
(*Code vs Data Driven Displacement*, orangeduck/Motion-Matching). Keep it to
**brute-force search plus inertialization**. Add acceleration only if profiling
shows a need.

### 6.1 Data: LAFAN1
- Ubisoft La Forge, BVH, 30 fps, 22 joints, 5 subjects, 15 categories.
  **License: CC BY-NC-ND 4.0.** `tools/fetch_assets.py` downloads it to
  `data/lafan1/`. Do **not** commit raw or derived data **[D12]**.
- Default database subset: locomotion (`walk*`, `run*`, `sprint*`) plus
  **mirrored copies** (L/R swap + X reflect) **[D6]**. Tag each clip range with
  its category, so the UI can ban ranges (for example `pushAndStumble`).

### 6.2 Offline build (`tools/build_mm_db.py`)
1. Load BVHs, convert to meters, optionally resample to `sim.tick_hz` (slerp).
2. **Simulation bone** (root): position = `Spine2` projected to the ground;
   facing = hip forward direction projected to XZ, Gaussian-smoothed.
   Re-express `Hips` relative to it.
3. Compute local/global positions, rotations, velocities and angular velocities.
4. Label foot contacts with a velocity + height threshold on `LeftToe` and `RightToe`.
5. Compute features per frame, all expressed in the sim-bone frame:

   | Group | Dims | Default weight |
   |---|---|---|
   | Left/right foot position | 6 | 0.75 |
   | Left/right foot velocity | 6 | 1.0 |
   | Hip velocity | 3 | 1.0 |
   | Future trajectory position (XZ) at +0.33 s, +0.67 s, +1.0 s | 6 | 1.0 |
   | Future trajectory direction (XZ) at the same offsets | 6 | 1.5 |

6. Normalize each group (mean, plus a group std pooled across dims), and mark
   the frames whose future trajectory window crosses a clip end as invalid.
7. Save `data/mm/<name>.npz` with: skeleton, `pos/rot/vel/angvel (N,J,·)`,
   `contacts (N,2)`, `features (N,27)`, `mean/std`, `range_starts/stops`, `tags`, `valid (N,)`.

Weights apply at query time. Changing a weight in the UI rescales the cached
weighted feature matrix in under 50 ms and needs no rebuild.

### 6.3 Controller (`mm/controller.py`)
- Desired velocity comes from the left stick (or WASD) in **camera space**.
  Desired facing follows the move direction, or the camera when **strafe** is held.
- Gait speeds (m/s, tunable): walk 1.75 fwd / 1.5 side / 1.25 back; run 4.0 / 3.0 / 2.5.
- The simulation object uses critically damped springs for velocity and
  rotation (halflife ≈ 0.27 s). Prediction comes from the closed-form spring
  solution at the trajectory offsets.

### 6.4 Runtime search and transitions
- **Query** = the current pose's features with the trajectory groups replaced by the controller's prediction.
- **Search** every `search_interval` (0.1 s), on a large input change, or at the
  end of a range. Brute-force weighted L2 over the valid, unbanned frames
  (torch on `sim.device`).
- **Switch** if `best_cost < current_cost - eps` and the best frame is not
  within ±`k` frames of the current frame.
- **Inertialization:** keep a per-joint position/rotation offset, decayed by a
  critically damped spring (halflife 0.1 s). This avoids crossfades and keeps
  the cost to one pose per tick.
- **Root sync:** the character root integrates the animation root velocity,
  then gets pulled toward the simulation object (halflife plus a clamp of
  0.5 m / 90°). The animation stays natural and the controller stays responsive.

### 6.5 Alternative animation sources (same `AnimSource` interface)
- `ClipPlayer`: plays a BVH or `.npz` pose sequence directly. Use it for
  deterministic evaluation on fixed motions.
- `Replay`: feeds a recorded input `.jsonl` into the motion-matching controller.
  The result is deterministic given the same database and tick rate.

### 6.6 Debug view
Overlay the desired and matched trajectories, the skeleton and the contacts.
A panel shows the current clip, frame and tag, a per-group cost breakdown, the
search count per second and transition markers on a timeline.

---

## 7. Body model (`body/`)

`BodyModel` holds torch tensors on `sim.device`:

| Field | Shape | Notes |
|---|---|---|
| `v_template` | (V,3) | rest vertices |
| `faces` | (F,3) | |
| `parents` | (J,) | |
| `J_regressor` or `J_rest` | (J,V) / (J,3) | regressor if shape-dependent |
| `weights` | (V,J) | dense is fine at SMPL scale; top-4 sparse optional |
| `shapedirs` | (V,3,S) | optional, betas |
| `posedirs` | (P,V·3) | optional, pose correctives |
| `morph_targets` | dict name → (V,3) | generic blendshapes (glTF morphs, expressions) |

Forward: `v = v_template + shapedirs·β + Σ w_i·morph_i` → `J = regress(v)` →
`v += posedirs·(R − I)` → **LBS**. Reuse and extend `ncs.data.dataset.skinning`.
The forward pass also returns posed joints and **per-vertex normals**
(scatter-add, reusing `ncs.mesh.compute_vertex_normals`).

Loaders:
- **SMPL** `.npz`/`.pkl`. Default body: SMPL neutral, supplied by the user under
  `assets/external/smpl/` because its license forbids redistribution **[D4]**.
- **glTF skin + morph targets**, for any rigged character.
- **Fallback mannequin**, generated from the skeleton (capsules, rigid
  weights), so the playground runs with no licensed assets.

UI: beta sliders, morph weights and pose-correctives on/off. Changing betas
re-binds the garments (§8.2).

---

## 8. Neural deformers (`deform/`)

### 8.1 Interface
```python
class Deformer(Protocol):
    def bind(self, target: MeshAsset, body: BodyModel, cfg: dict) -> None: ...
    def reset(self) -> None: ...                  # clear temporal state (teleport, clip restart)
    def step(self, ctx: FrameContext) -> DeformOut: ...

@dataclass
class FrameContext:            # all torch, on sim.device
    dt: float; tick: int
    joint_rotmats: Tensor      # (J,3,3) current, body skeleton order
    joint_rotmats_prev: Tensor # (J,3,3)
    root_xform: Tensor         # (4,4)
    root_lin_vel: Tensor; root_ang_vel: Tensor   # (3,), (3,)
    body_verts: Tensor; body_normals: Tensor     # (Vb,3)
    joints: Tensor             # (J,3) posed
    betas: Tensor              # (S,)

@dataclass
class DeformOut:
    verts: Tensor                         # (V,3) world space
    scalars: dict[str, Tensor] = {}       # optional per-vertex channels for heatmaps
```

### 8.2 Built-in deformers
- `static`: the rest mesh. A sanity check.
- `lbs`: the garment skinned with the body's weights (nearest body vertex at
  bind, plus optional shape-displacement transfer). **This is the baseline
  every neural model is compared against.**
- `torch`: loads an `nn.Module` or checkpoint, for example `NCSModel`, through a small adapter.
- `onnx`: ONNX Runtime session described by a manifest (§8.3).
- `custom`: a Python class given as `module.path:ClassName` that implements
  `Deformer`. It can use torch ops, Triton or `torch.utils.cpp_extension` CUDA
  kernels **[D10]**. The engine reloads it when the file changes.

### 8.3 Model manifest (`*.model.yaml`)
The manifest maps model tensors to engine semantics, so a new model needs no
engine code:

```yaml
name: ncs_skirt_v1
backend: onnx                    # onnx | torch | custom
path: models/ncs_skirt_v1.onnx
target: assets/garments/skirt.obj
skeleton: smpl24                 # joint order the model expects
rate_hz: 30                      # engine warns if != sim.tick_hz
providers: [CUDAExecutionProvider, CPUExecutionProvider]
inputs:
  pose:      { semantic: joint_rotmats, frame: current }   # (1,J,3,3)
  pose_prev: { semantic: joint_rotmats, frame: previous }
  template:  { semantic: target_rest_verts }               # (V,3)
outputs:
  verts:     { semantic: verts, space: unposed }   # unposed -> engine applies garment LBS
state: {}                         # recurrent tensors carried across ticks: {in_name: out_name}
```

- Input semantics: `joint_rotmats | joint_quats | joint_axis_angle | root_lin_vel | root_ang_vel | body_verts | body_normals | betas | target_rest_verts | dt | state:<name>`.
- Output spaces: `world | unposed | offset_unposed | offset_world` **[D9]**.
- At load, check names, shapes and dtypes against the ONNX graph and fail
  loudly. Bind inputs and outputs through `IOBinding` on torch CUDA
  `data_ptr()`s, so the data never leaves the GPU.
- `tools/export_onnx.py` exports `NCSModel` (`torch.onnx.export`, opset ≥ 17),
  writes the manifest, and checks parity against torch (`atol=1e-4`).

### 8.4 Render mesh vs sim mesh
UV and normal seams make the render mesh bigger than the sim mesh. Store a
`render_to_sim (Vr,)` index and gather it each tick. Recompute normals, and
tangents (per-face UV derivatives + Gram-Schmidt), in torch after deformation.

---

## 9. Rendering (`render/`)

**Forward renderer**, one sun light plus IBL. Passes:

1. **Shadow:** sun, an orthographic frustum fitted to the character's bounds
   (plus the ground near it), 2048², PCF 5×5, slope-scaled bias.
2. **Opaque forward:** to RGBA16F with **4× MSAA**. Garments render
   **two-sided** (normal flipped on back faces).
3. **Sky:** HDRI background, with optional blur and exposure.
4. **Resolve + post:** exposure, tonemap (**AgX**, with ACES as an option), sRGB output.
5. **Overlay** (LDR, no depth test, optional): debug lines, skeleton,
   trajectories, gizmos, ImGui.

**IBL:** at load, turn an equirect HDR into a cubemap, then a diffuse SH9 /
irradiance map, a GGX prefiltered specular map and a BRDF LUT. Cache the
results in `data/cache/ibl/`.

**Shading models** (`shaders/materials/`):

| Model | Use | Notes |
|---|---|---|
| `standard` | props, glTF | metal/rough, normal, AO, emissive |
| `cloth` | garments | Charlie sheen (Estevez-Kulla) + wrap diffuse, sheen color/roughness, two-sided |
| `skin` | body | standard + wrap diffuse SSS approximation |
| `unlit` | debug, ground grid lines | |
| `heatmap` | evaluation | maps any `DeformOut.scalars[name]` with turbo/viridis, range in UI |
| `custom` | user | see below |

**Custom shaders:** a material YAML points to a GLSL file that implements
`Surface evaluate(SurfaceInputs s)`. The shared include provides lighting, IBL
and shadows. Uniforms declared in the YAML show up as UI widgets. Optional
user post-process passes take `(hdr_color, depth, normals)` as input. Shaders
hot-reload. A compile error keeps the last good program and shows the log in
the UI.

**Debug view modes:** lit, albedo, normals, wireframe overlay, heatmap, and
body/garment visibility toggles.

GPU geometry upload uses torch CUDA → GL buffer through CUDA–GL interop
(`cuda-python`, `cudaGraphicsGLRegisterBuffer`). The fallback is a host copy,
which is fine below about 50k vertices. Interop comes in a later phase.

---

## 10. Scene, assets and controls

### 10.1 Scene file (`scenes/default.yaml`)
```yaml
sim:       { tick_hz: 30, device: cuda }
scene:
  hdri: data/assets/hdri/<cc0_studio>.hdr
  ground: { size: 50, material: materials/checker_1m.yaml }
  props:  [ { mesh: assets/props/box_1m.glb, xform: [2,0,3] } ]
camera:   { mode: follow, distance: 3.0, height: 1.4 }
character:
  body: { type: smpl, path: assets/external/smpl/SMPL_NEUTRAL.npz, betas: [0,0,0,0,0,0,0,0,0,0] }
  animation: { source: motion_matching, database: data/mm/lafan_locomotion.npz }
  garments:
    - { mesh: assets/garments/skirt.obj, material: materials/cotton.yaml, deformer: lbs }
```

### 10.2 Sample assets **[D8, D11]**

| Asset | Source | License | In repo? |
|---|---|---|---|
| LAFAN1 BVH | Ubisoft La Forge | CC BY-NC-ND 4.0 | No, `fetch_assets.py` |
| SMPL neutral body | MPI-IS | SMPL license (registration) | No, user-supplied |
| Fallback mannequin | procedural | ours | Generated |
| Garments: tube skirt, cape (pinned at shoulders) | procedural, fit to SMPL rest | ours | Yes (small OBJ) |
| Real garments (e.g. T-shirt, dress) | **TBD** | TBD | **[D8]** |
| HDRIs (studio + outdoor), 2k | Poly Haven | CC0 | No, fetched |
| PBR textures: ground checker, cotton, denim | ambientCG / Poly Haven | CC0 | No, fetched (checker generated) |
| Props: boxes 0.5/1/2 m, ramp, 1 m reference pole | procedural | ours | Generated |

### 10.3 Test scenes
- `default.yaml`: the motion-matching character in the open scene.
- `turntable.yaml`: idle pose, auto-orbiting camera, for shading and static drape checks.
- `eval_track.yaml`: a scripted input (straight walk → sharp 180° → sprint →
  stop → strafe circle) run through `Replay`. It is the **standard repeatable
  benchmark**.

### 10.4 Controls

| Action | Gamepad | Keyboard / mouse |
|---|---|---|
| Move | Left stick | WASD |
| Camera orbit / zoom | Right stick / — | RMB drag / wheel |
| Run (default) / walk | — / LT | — / Ctrl |
| Strafe (lock facing) | LB | Alt |
| Pause / step one tick | Start / D-pad → | Space / `.` |
| Reset character | Back | R |
| Camera: follow / free-fly / presets (front, side, back) | Y cycles | F / 1–3 |
| Cycle deformer on selected garment | RB | Tab |
| Reload shaders, models, scene | — | F5 |
| Screenshot / toggle recording | — | F12 / F9 |
| Toggle UI | — | F1 |

Pause and single-step are first-class because they are the main tools for inspecting artifacts.

---

## 11. Evaluation (`eval/`)

**Live metrics** for each garment, every tick, toggled in the UI, plotted with
ImPlot and written to `runs/<timestamp>/metrics.csv`:

| Metric | Definition |
|---|---|
| Penetration % / depth | garment verts inside the body, using `ncs.collision` |
| Stretch | edge length ratio vs rest: mean, max, % > 1.1 |
| Physics energy | `ncs.physics` energy for the configured formulation (costly, off by default) |
| Jitter | mean ‖vertex acceleration‖ (second difference over ticks) |
| Inference ms | CUDA-event timing per deformer; frame ms total |

**Recording** (`runs/<timestamp>/`): `input.jsonl` (replayable),
`poses.npz` (body-skeleton rotmats + root, in the same format as the `ncs` data
pipeline so it can **feed training**), optional per-frame garment verts `.npz`
/ `.obj`, and `video.mp4`.

**A/B:** each garment can hold several deformers. Tab cycles them. A
*split* mode renders a second character instance offset by 1 m with deformer
B, both driven by the same pose.

**Headless:** for example,
`python -m playground --scene scenes/eval_track.yaml --headless --frames 900 --deformer models/x.model.yaml`
writes metrics with no window.

---

## 12. Performance budget (60 fps, 1080p, RTX 3060-class) **[D2]**

| Stage | Budget |
|---|---|
| Motion-matching search (amortized) + pose | ≤ 1 ms |
| Retarget + body blendshapes + LBS | ≤ 1 ms |
| Deformers | measured, shown in HUD (target ≤ 4 ms each) |
| Render | ≤ 6 ms |
| UI + metrics | ≤ 2 ms |

It must also run on CPU only (torch CPU + ORT CPU EP) at a reduced frame rate.

---

## 13. Testing

Use `pytest` in the same style as `tests/`, under `tests/playground/`. No GPU
or window is required except where a test is marked `@pytest.mark.gpu`.

| Module | Tests |
|---|---|
| anim | quat identities; BVH parse → FK matches reference joint positions; retarget of the rest pose gives the target rest pose |
| mm | feature dims = 27; searching with a DB frame's own features returns that frame; inertialization offset → 0 and no discontinuity at the switch; spring converges; mirroring is an involution |
| body | zero betas, zero morphs and rest pose → `v_template`; LBS with identity rotations is a no-op; normals are unit |
| deform | manifest validation errors; ONNX vs torch parity (`atol 1e-4`); `lbs` deformer matches body skinning on body verts |
| render | headless EGL render of the default scene: no GL errors, no NaN, non-empty image; shader compile of every material |
| eval | penetration = 0 for a garment inflated off the body; replay determinism (two runs give identical poses) |

---

## 14. Phases

Each phase ends with something you can run. **Start each phase by confirming
its open decisions with the user.**

| # | Phase | Deliverable | Gate decisions |
|---|---|---|---|
| P0 | Shell | window, fixed-tick loop, cameras, ground grid, ImGui, config + hot reload, headless flag | D1, D2, D3 |
| P1 | Animation core | quat lib, BVH loader, FK, skeleton debug draw, `ClipPlayer` on LAFAN1 | D7, D12 |
| P2 | Rendering | glTF loading, PBR + IBL + shadows + tonemap, materials incl. `cloth`, test scenes | D11, D14 |
| P3 | Body | `BodyModel` (SMPL + mannequin), blendshapes, LBS, retarget, betas UI | D4, D5 |
| P4 | Motion matching | DB build tool, features, controller, search, inertialization, debug panel, input record/replay | D6, D15 |
| P5 | Deformers + eval | protocol, `lbs`/`torch`/`onnx` backends, manifest, ONNX export, metrics HUD, recording, A/B, sample garments | D8, D9, D13 |
| P6 | Iteration extras | custom shaders + post, `custom` deformers + CUDA, CUDA–GL interop, foot IK, video capture | D10 |
