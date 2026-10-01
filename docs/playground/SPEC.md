# NCS Playground: Engine Spec

A small real-time **C++ / Windows** engine for **evaluating neural cloth and
body models** on the **Geno** character, driven by **motion matching** on
orangeduck's retargeted datasets (LAFAN1-resolved + 100STYLE-retarget). It is a
playground built for fast iteration. It is not a game engine.

> **Status:** draft spec, revision 2. Decisions marked **[D#]** are tracked in
> [`DECISIONS.md`](DECISIONS.md). Defaults here are recommendations until the
> user confirms them. Before implementing a phase, resolve its open decisions
> with the user. **Ask. Don't assume.**
>
> **No trained models ship with this project.** Every neural component is
> **placeholder code** (§8) that runs end-to-end with identity/LBS output until
> a real model file is dropped in.

---

## 1. Goals and non-goals

### Goals
1. Drive Geno interactively (gamepad or keyboard) with **motion matching**:
   idle, walk, run, **sprint**, **crouch** (idle/walk/run), and strafe.
2. **LBS + blendshapes** on the CPU, so it is simple and testable. Upload the
   result to the GPU each tick.
3. Render with modern shading that looks good enough to judge cloth: deferred
   PBR, IBL, shadows, SSAO, a sheen cloth BRDF, HDR + AgX tonemap, and a
   GenoView-style "artifact grid" view.
4. Plug-in deformers through **ONNX Runtime**, **custom GLSL compute** or
   **custom CUDA**. All of them are placeholders at first.
5. Start from **free garments**, fit them to Geno, and use **LBS as the
   baseline** that every model is compared against.
6. Live metrics, deterministic record/replay, A/B comparison and hot reload of
   shaders, configs and models.

### Non-goals (v1)
- Editor, scene graph, prefabs, asset streaming, scripting language
- Physics engine, environment collision, terrain or stairs (flat ground only)
- Animation state machines or blend trees (motion matching plus tags replaces them)
- Learned Motion Matching, jumps and vaults (later, see **[D15]**)
- Producing or shipping any trained model. Model export lives in the Python `ncs/` side, if anywhere.
- Linux, macOS, consoles, web. **Windows only.**

---

## 2. Stack **[D1, D18]**

| Concern | Choice | Why |
|---|---|---|
| Language / toolchain | **C++20, MSVC 2022, CMake ≥ 3.25, vcpkg manifest** | Standard on Windows |
| Window, GL, input, gamepad | **raylib 5.x** (OpenGL 4.3 backend) | Holden's MIT Motion-Matching demo and MIT GenoView (deferred + shadows + SSAO) are both raylib, so their code ports almost verbatim. GL 4.3 has compute shaders and SSBOs. Gamepads come through GLFW/XInput. |
| UI | Dear ImGui + **rlImGui** + **ImPlot** | Panels and live metric plots |
| Character / mesh import | **ufbx** (single-file, MIT) for `Geno.fbx`; tinyobjloader for garment OBJ | ufbx reads the skin, bind pose and blendshapes directly, with no Maya step |
| Images / HDR | stb_image, stb_image_write | |
| Config | **TOML** via toml++ (header-only) | Readable, hot-reloadable |
| Nearest-neighbor | nanoflann | Garment binding and penetration metric |
| NN inference | **ONNX Runtime** (official GPU package: CUDA EP, CPU EP fallback) | **[D10]** |
| CUDA (optional) | CUDA Toolkit 12.x, CMake option `PG_WITH_CUDA` | Custom deformer kernels, GL interop |
| Tests | doctest + CTest | |

**One rule keeps this simple:** *the CPU owns the canonical state.* Skeleton,
skinning, blendshapes and garment vertices live in CPU arrays and upload to GL
buffers once per tick. Body + garments total around 30k vertices, which is under
1 MB per tick. GPU deformers (ONNX CUDA EP, GLSL compute, CUDA) are
accelerators that write back into those buffers. Zero-copy CUDA–GL interop is
a later optimization (P6).

---

## 3. Data, character and conventions

### 3.1 Geno (body model)
Geno comes from the orangeduck dataset repos (`Geno.fbx`, identical in
`lafan1-resolved`, `zeroeggs-retarget` and `100style-retarget`). The README says
it is *"free for non-commercial research use"*.

| Property | Value (measured from the repo files) |
|---|---|
| Mesh | 10,329 verts after UV/normal split, 18,660 tris, one material, has UVs |
| Skin | 75 bones, ≤ 4 influences per vertex |
| Skeleton | `Hips → Spine → Spine1 → Spine2 → Spine3 → Neck → Neck1 → Head`, full fingers (`*Hand{Thumb,Index,Middle,Ring,Pinky}{1-4}`), `*Shoulder/Arm/ForeArm/Hand`, `*UpLeg/Leg/Foot/ToeBase`, plus `*End` leaf joints |
| Bind pose | `Geno_bind.bvh`: **A-pose** (arms about 45° down) |
| Stance pose | `Geno_stance.bvh`: **T-pose** |
| Size | about 1.66 m tall, hips at 0.855 m |
| Blendshapes | **none** (see §7) |

The same skeleton is used by every dataset below, so **no retargeting is
needed**.

### 3.2 Motion data **[D6]**

| Dataset | Content used | fps | License |
|---|---|---|---|
| **lafan1-resolved** (orangeduck) | `walk*`, `run*`, `sprint*`, `ground*` (crouch/crawl), with crawl ranges excluded | 60 | LAFAN1 terms: CC BY-NC-ND 4.0, non-commercial |
| **100style-retarget** (orangeduck) | styles **`Neutral`** and **`Crouched`**, clip types `FW BW SW FR BR SR ID TR1` (forward/back/side × walk/run, idle, transitions) | 60 | CC BY 4.0 |
| zeroeggs-retarget | not used in v1 (speech gestures); possible idle variety later | 60 | ZeroEGGS terms |

100STYLE `Crouched` gives crouch walk, run, sidestep, backwards and idle, which
LAFAN1 lacks. LAFAN1 `ground*` adds crouch variety and possibly stand↔crouch
transitions **[D16]**.

Downloads are BVH zips from `theorangeduck.com/media/uploads/Geno/<dataset>/bvh.zip`.
`tools/fetch_assets.ps1` downloads them into the git-ignored `data/`. Never
commit raw or derived motion data. *(This sandbox got HTTP 403 from that host.
On a normal Windows machine it should work. If it doesn't, the script prints
manual-download instructions.)*

### 3.3 Conventions
- World is **Y-up, right-handed, meters**. Geno faces **+Z**, with its left at +X. BVH is in cm, so convert on load.
- Quaternions are stored `(w,x,y,z)`. The math lib is Holden's `vec.h`/`quat.h`/`spring.h` style (MIT).
- **Fixed 60 Hz simulation tick**, matching the data **[D7]**. Rendering runs
  at vsync. Optional render interpolation is off by default.
- Joint order is the BVH order (75 joints). Models can take a subset (§8.3).

---

## 4. Architecture

```
playground/
├── CMakeLists.txt  vcpkg.json  CMakePresets.json
├── src/
│   ├── app/        main.cpp, App (main loop, CLI), Clock, Config (toml++), HotReload
│   ├── math/       vec, quat, mat, spring (Holden-style), transforms
│   ├── anim/       Skeleton, Pose, BVH loader, FK, mirror, ClipPlayer, FootIK
│   ├── mm/         Database, Tags, Features, Search (AABB), Controller, Inertializer, Recorder/Replay
│   ├── body/       BodyModel (ufbx loader), Morphs, LBS (CPU), Normals
│   ├── garment/    Garment asset, procedural garments, Fitter, SkinBinding
│   ├── deform/     IDeformer, registry, Lbs/Static, Placeholder, Onnx, GlslCompute, Cuda (.cu)
│   ├── render/     Renderer (deferred), GBuffer, Shadow, SSAO, IBL, Post, Materials, DebugDraw
│   ├── eval/       Metrics, CSV/NPY writers, Capture
│   └── ui/         ImGui panels
├── shaders/        GLSL: gbuffer, lighting, ssao, shadow, post, materials/, user/
├── scenes/         default.toml, turntable.toml, eval_track.toml, clip_browser.toml
├── assets/         small redistributable assets (procedural garment params, materials)
├── models/         *.model.toml manifests ONLY (placeholder.model.toml); no weights committed
├── tests/          doctest unit tests
└── tools/          fetch_assets.ps1
```

Location is `playground/` in this repo by default **[D3]**. The C++ engine has
no build or runtime dependency on the Python `ncs/` package. They exchange
files: recorded poses and meshes go out as `.npy`, models come in as `.onnx`.

### Main loop
```cpp
while (!WindowShouldClose()) {
    input.Poll();
    acc += clock.FrameDt() * timeScale;
    while (acc >= kTickDt) {                             // fixed 60 Hz
        controller.Update(input, camera, kTickDt);       // desired vel/facing/stance, springs
        animSource->Step(kTickDt, pose);                 // MotionMatching | ClipPlayer | Replay
        footIk.Apply(pose);                              // optional
        body.Update(pose, morphWeights);                 // morphs + LBS + normals (CPU)
        for (auto& g : garments) g.deformer->Step(ctx, g.out);
        metrics.Update(ctx); recorder.Update(ctx);
        acc -= kTickDt;
    }
    renderer.Upload(body, garments);
    renderer.Draw(scene, camera);
    ui.Draw();
}
```
`--headless` runs the same loop with a hidden window and no presentation (metrics and capture only).

---

## 5. Animation core (`anim/`)

- **BVH loader** handles any channel order (Geno BVHs use 6 channels on every
  joint, `Zrotation Yrotation Xrotation`), converts cm to m and outputs local
  `rot (J,4)` and `pos (J,3)` per frame.
- **FK and mirroring:** mirroring swaps the `Left*` and `Right*` names and
  reflects X, following the `animation_mirror` method from Holden's
  `generate_database.py`.
- **ClipPlayer:** play, scrub and loop any BVH. Its *clip browser* scene doubles
  as a GenoView-equivalent viewer and as the tagging UI (§6.2).
- **Foot IK:** a port of Holden's contact locking + two-bone IK
  (`ik_foot_height 0.02`, `ik_toe_length 0.15`, `ik_unlock_radius 0.2`,
  `ik_blending_halflife 0.1`). Off by default.

---

## 6. Motion matching (`mm/`)

Start by porting the **MIT reference `orangeduck/Motion-Matching`**
(`controller.cpp`, `database.h`, `character.h`, `spring.h`) to the Geno
skeleton. Then add what it lacks: **tags** (stand/crouch), **sprint** and
**crouch**.

### 6.1 Database build (`playground.exe --build-db scenes/mm_db.toml`, C++)
1. Load the configured clip list, with frame ranges, from both datasets.
2. Optionally **mirror** every clip, which doubles the data.
3. **Simulation bone:** position = `Spine2` projected to the ground and
   Savitzky-Golay smoothed. Facing = `Hips` forward on XZ, smoothed. This is
   Holden's method. Check `Spine2` vs `Spine3` on Geno.
4. Compute local/global positions and rotations, velocities and angular velocities.
5. Label **contacts** on `LeftToeBase` and `RightToeBase` with height + velocity thresholds.
6. Compute **features** (all in the sim-bone frame, 27 dims, Holden's set):

   | Group | Dims | Default weight |
   |---|---|---|
   | Foot positions (L/R) | 6 | 0.75 |
   | Foot velocities (L/R) | 6 | 1.0 |
   | Hip velocity | 3 | 1.0 |
   | Trajectory positions (XZ) at +20/+40/+60 ticks | 6 | 1.0 |
   | Trajectory directions (XZ) at +20/+40/+60 ticks | 6 | 1.5 |

7. Normalize per group, build the **AABB acceleration structure** (Holden's
   small/large bounding boxes) per tag, and mark frames whose trajectory window
   crosses a range end as invalid.
8. Write `data/mm/db.bin` (bones, contacts, ranges, tags) and `data/mm/features.bin`.
   Print **speed statistics per tag** (p50/p95 forward, side and back) so the
   controller speeds can be tuned against the real data.

### 6.2 Tags (`data/mm/tags.toml`)
- Tags per frame range: `stand`, `crouch`, `transition`, `exclude` (crawl, falls, bad frames).
- The build tool **suggests** tags automatically: crouch when hip height is
  below a threshold (default 0.65 m, tunable), and `exclude` when the head is
  below the hips. A person confirms the suggestions in the clip browser
  (timeline with colored tag ranges, set or clear on a selection, save).
- 100STYLE clips get their tags from the style name (`Neutral` → stand, `Crouched` → crouch).

### 6.3 Controller and controls
Desired velocity comes from the stick or WASD in **camera space**. Desired
facing follows the move direction, or the camera while strafing. Springs and
synchronization follow Holden: velocity/rotation halflife 0.27 s, adjustment
pos/rot halflife 0.1/0.2 s, clamping 0.15 m.

| Gait | Fwd / side / back (m/s), initial values to retune from DB stats |
|---|---|
| Walk | 1.75 / 1.5 / 1.25 |
| Run (default) | 4.0 / 3.0 / 2.5 |
| **Sprint** | 6.5 / run side / run back (forward cone only) |
| **Crouch walk** | 1.0 / 0.8 / 0.7 |
| **Crouch run** | 2.5 / 2.0 / 1.5 |

Stick magnitude blends between walk and run. A gait change uses
`gait_change_halflife` 0.1 s.

**Stance:** pressing crouch toggles `desiredStance`. The search is restricted
to frames tagged with that stance (plus `transition`). A stance change
**forces an immediate search** and uses a longer inertialization halflife
(0.2 s, tunable) **[D16]**.

| Action | Gamepad (XInput) | Keyboard / mouse |
|---|---|---|
| Move | Left stick (magnitude → walk…run) | WASD (run) |
| Walk | — (small deflection) | hold Alt |
| **Sprint** | hold RT or click LS | hold Shift |
| **Crouch** (toggle) | B | C |
| Strafe (face camera) | hold LT | hold Ctrl |
| Camera orbit / zoom | Right stick / LB+RB | RMB drag / wheel |
| Camera mode: follow → free → presets (front/side/back) | Y | F / 1–3 |
| Pause / step one tick / time scale | Start / D-pad → / D-pad ↑↓ | P / `.` / `[` `]` |
| Reset character | Back | R |
| Cycle deformer (selected garment) | D-pad ← | Tab |
| Reload shaders, configs, models | — | F5 |
| Screenshot / record toggle | — | F12 / F9 |
| Toggle UI | — | F1 |
| *(Later)* Jump | A | Space **[D15]** |

### 6.4 Runtime
- **Query** = current features, with the trajectory replaced by the controller's spring prediction.
- **Search** every `search_time` (0.1 s), on a large input change (velocity
  or rotation change thresholds), on a stance change, or at the end of a range.
  The search covers only valid frames carrying the current stance tag.
- **Switch** when the best cost beats the current cost and the best frame is
  not close to the current frame.
- **Inertialization:** offsets decayed with a critically damped spring
  (halflife 0.1 s; stance change 0.2 s).
- **Budget:** at most 1 ms per search with AABB culling. Expected size is
  about 0.4M frames including mirroring.

### 6.5 Animation sources
All three implement the same `IAnimSource`:
- `MotionMatching`
- `ClipPlayer`: a fixed BVH, for deterministic evaluation.
- `Replay`: a recorded input stream fed into the motion-matching controller.
  Deterministic given the same DB, tick and config.

### 6.6 Debug view
Overlay the desired and matched trajectories, the sim bone, contacts and foot
IK targets. A panel shows the current clip, frame and tag, a per-group cost
breakdown, live feature-weight sliders (they rebuild the weighted features in
under 100 ms, no DB rebuild), searches per second and a transition timeline.

---

## 7. Body: LBS + blendshapes (`body/`)

`BodyModel` stores, on the CPU: `restVerts (V,3)`, `restNormals`, `uvs`,
`indices (uint32)`, `boneIdx (V,4) u8`, `boneW (V,4) f32`, `bindInv (J, 4x4)`,
`parents`, and `morphs: name → sparse {vertIdx, delta}`.

Per tick:
1. `v = rest + Σ wᵢ·morphᵢ` (sparse adds)
2. LBS with `bindInv`
3. Recompute normals (area-weighted) and tangents

At Geno's size this is well under 1 ms single-threaded. Add an OpenMP or
`std::execution::par` loop only if profiling demands it.

**Blendshapes:** the loader reads FBX blend channels generically (ufbx). Geno
has **none**. To exercise the path, the engine creates **procedural test
morphs** at load (`inflate` along normals, `belly`, `chest` by
radial falloff around joints), and the **placeholder corrective deformer**
(§8) can output morph weights or per-vertex offsets for the body **[D19]**.
The UI has morph-weight sliders and a "show bind pose" toggle.

---

## 8. Deformers: neural embedding, placeholders only (`deform/`)

### 8.1 Interface
```cpp
struct FrameContext {
    float dt; int64_t tick; Stance stance;
    std::span<const quat> localRot, prevLocalRot;    // (J) Geno order
    std::span<const vec3> globalPos;  std::span<const quat> globalRot;
    vec3 rootVel, rootAngVel;
    std::span<const vec3> bodyVerts, bodyNormals;    // posed body, world space
};

struct DeformOut {
    std::vector<vec3> verts;                                  // world space, sim-mesh order
    std::unordered_map<std::string, std::vector<float>> scalars; // per-vertex, for heatmaps
};

class IDeformer {
public:
    virtual ~IDeformer() = default;
    virtual const char* Name() const = 0;
    virtual bool Bind(const GarmentAsset& g, const BodyModel& body, const toml::table& cfg) = 0;
    virtual void Reset() = 0;                                 // clear temporal state
    virtual void Step(const FrameContext& ctx, DeformOut& out) = 0;
};
```

### 8.2 Built-in deformers

| Deformer | What it does | Status |
|---|---|---|
| `static` | rest mesh | real |
| `lbs` | garment skinned with weights transferred from Geno (§9.3) | real, **the baseline** |
| `placeholder` | runs the full model path (gather inputs → "infer" → apply output) but inference returns **zero offsets**, so the result equals `lbs`. The HUD shows `PLACEHOLDER`. | **placeholder** |
| `onnx` | ONNX Runtime session from a manifest. If the `model` path is empty or missing it falls back to `placeholder` with a warning. | code real, **no model shipped** |
| `glsl_compute` | runs `shaders/user/deform_template.comp` over SSBOs (identity kernel) | **template** |
| `cuda` | runs `src/deform/cuda/deform_template.cu` (identity kernel), built only with `PG_WITH_CUDA` | **template** |
| `body_corrective` | the same pattern applied to the body: outputs morph weights or offsets (zeros) | **placeholder** |

Placeholder sketch (the actual stub to write in P5):
```cpp
class PlaceholderDeformer final : public IDeformer {
    LbsDeformer lbs_;            // baseline path
    std::vector<vec3> offsets_;  // what a model would predict (unposed space)
public:
    const char* Name() const override { return "placeholder"; }
    bool Bind(const GarmentAsset& g, const BodyModel& b, const toml::table& c) override {
        offsets_.assign(g.simVerts.size(), vec3{0, 0, 0});
        return lbs_.Bind(g, b, c);
    }
    void Reset() override { std::fill(offsets_.begin(), offsets_.end(), vec3{0, 0, 0}); }
    void Step(const FrameContext& ctx, DeformOut& out) override {
        // TODO(model): gather inputs from ctx per manifest, run inference, write offsets_
        lbs_.StepWithUnposedOffsets(ctx, offsets_, out);   // offsets are all zero → equals LBS
    }
};
```

### 8.3 Model manifest (`models/*.model.toml`)
The manifest maps model tensors to engine semantics, so a new model needs no
engine code. `models/placeholder.model.toml` ships with an empty `model` path:

```toml
name    = "placeholder_tshirt"
backend = "onnx"                      # onnx | glsl_compute | cuda
model   = ""                          # empty → placeholder behaviour
target  = "garments/tshirt"           # garment id
rate_hz = 60                          # engine resamples if different
joints  = "body22"                    # preset (no fingers/ends) or explicit list
providers = ["CUDA", "CPU"]

[inputs.pose]      semantic = "joint_rotmats"     frame = "current"   # (1,J,3,3)
[inputs.pose_prev] semantic = "joint_rotmats"     frame = "previous"
[inputs.template]  semantic = "target_rest_verts"                     # (V,3)
[outputs.verts]    semantic = "verts"             space = "unposed"   # engine applies garment LBS
```

- Input semantics: `joint_rotmats | joint_quats | joint_6d | root_vel | root_ang_vel | body_verts | body_normals | target_rest_verts | dt | stance | state:<name>`.
- Output spaces: `world | unposed | offset_unposed | offset_world`. Outputs may also include `scalars:<name>` **[D9]**.
- At load, validate names, shapes and dtypes against the session and fail
  loudly in the UI without crashing. Recurrent `state:*` tensors carry over
  between ticks and are cleared by `Reset()`.

---

## 9. Garments (`garment/`)

### 9.1 Sources (free) **[D8]**

| Source | What | License | Fits Geno? |
|---|---|---|---|
| **Procedural (ours)** | tube skirt, circle skirt, cape pinned at the shoulders, generated from Geno landmarks at load | ours | yes, by construction |
| **GarmentCode** (`maria-korosteleva/GarmentCode`) | parametric sewing patterns: T-shirt, shirt, hoodie, pants, skirts, dresses, with a built-in drape simulator (NVIDIA Warp) | **MIT** | **yes**: generate made-to-measure from Geno measurements and drape on Geno's bind-pose mesh |
| **NeuralClothSim** samples (`hbertiche/NeuralClothSim/body_models/`) | `tshirt.obj`, `pants.obj` (+`pants_pin.npy`) on SMPL; `tshirt.obj` on a Mixamo mannequin | non-commercial research | needs refit (§9.2) |

Recommended start: procedural skirt and cape (available right away), plus a
GarmentCode T-shirt and pants made for Geno (a one-off offline export to OBJ
with UVs). The NCS samples come second, through the fitter.

### 9.2 Fitter (`playground.exe --fit-garment <cfg>` with a UI)
1. Coarse align: an interactive similarity transform with an ImGui gizmo, plus
   per-axis scale. The result is saved in the garment TOML.
2. **Push-out:** move any vertex inside the body outward along the
   closest-surface normal, plus an offset (default 3 mm).
3. **Relax:** a few iterations of Laplacian smoothing with edge-length preservation, keeping pinned vertices fixed.
4. Save `assets/garments/<id>/garment.obj` and `garment.toml` (transform, pins, material, sim/render maps).

### 9.3 Skin binding (for `lbs` and `unposed` outputs)
Copy the weights of the nearest Geno vertex, then run N smoothing iterations
(default 10; skirts 50–100, as in NCS's `blend_weights_smoothing_iterations`).
Results are cached next to the garment.

### 9.4 Sim mesh vs render mesh
Seams make the render mesh larger. Store a `renderToSim` index map, gather it
each tick, and recompute normals and tangents after deformation.

---

## 10. Rendering (`render/`)

Start from GenoView's deferred renderer (MIT) and upgrade it:

1. **Shadow:** sun, an orthographic frustum fitted to the character + garment
   bounds, 2048², PCF.
2. **GBuffer:** albedo, normal (octahedral), roughness / metallic / sheen /
   **shading-model id**, depth. Garments are **two-sided**: flip the normal on
   back faces and optionally tint the back side.
3. **SSAO + blur:** GenoView's.
4. **Lighting:** GGX + **IBL** (HDRI → cubemap, irradiance SH9, GGX prefilter,
   BRDF LUT, cached in `data/cache/ibl/`). Shading models: `standard`, `cloth`
   (Charlie sheen + wrap diffuse), `skin` (wrap diffuse), `unlit`, `heatmap`.
5. **Sky:** the HDRI background.
6. **Post:** exposure → **AgX** tonemap (ACES optional) → FXAA (GenoView's).
7. **Overlay:** debug lines, trajectories, skeleton, gizmos, ImGui.

**Artifact view:** GenoView's procedural grid can be applied to the ground,
the body and the garments. It makes sliding, stretching and penetration easy
to see.

**Custom shaders:** a material TOML references a GLSL function
`Surface evaluate(SurfaceIn)` that is injected into the GBuffer pass. Its
uniforms appear in the UI automatically. Users can also add post passes in
`shaders/user/post_*.fs`. Everything hot-reloads, and a compile error keeps the
last good program and shows the log.

**Debug modes:** lit, albedo, normals, AO, wireframe overlay, heatmap of any
`DeformOut.scalars` channel (turbo/viridis), and body/garment visibility.

---

## 11. Scenes and assets

```toml
# scenes/default.toml
[sim]        tick_hz = 60
[scene]      hdri = "data/assets/hdri/studio.hdr"   ground_size = 50.0   artifact_grid = true
[camera]     mode = "follow"  distance = 3.5  height = 1.3
[character]  fbx = "data/geno/Geno.fbx"  anim = "motion_matching"  db = "data/mm"
[[garments]] id = "skirt_procedural"  material = "materials/cotton.toml"  deformer = "lbs"
[[garments]] id = "tshirt_garmentcode" material = "materials/jersey.toml" deformer = "models/placeholder.model.toml"
```

Scenes:
- `default.toml`: the motion-matching playground
- `turntable.toml`: bind or idle pose with an auto-orbiting camera
- `clip_browser.toml`: GenoView-style viewer and tagging tool
- `eval_track.toml`: a scripted input (walk → 180° turn → sprint → stop →
  crouch walk → stand → strafe circle) run through `Replay`. It is the
  **standard repeatable benchmark**.

| Asset | Source | License | In git? |
|---|---|---|---|
| Geno (`Geno.fbx`, bind/stance BVH) | orangeduck/lafan1-resolved | non-commercial research | no, fetched |
| LAFAN1-resolved BVH | theorangeduck.com | CC BY-NC-ND 4.0 | no, fetched |
| 100STYLE-retarget BVH (Neutral, Crouched) | theorangeduck.com | CC BY 4.0 | no, fetched |
| Garments | §9.1 | §9.1 | procedural params only; generated OBJs git-ignored **[D11]** |
| HDRIs, fabric/ground textures | Poly Haven / ambientCG | CC0 | no, fetched |
| Props (boxes, 1 m pole, ramp) | procedural | ours | generated |
| Neural models | — | — | **none, placeholders only** |

---

## 12. Evaluation (`eval/`)

**Live metrics** for each garment, every tick, plotted with ImPlot and written
to `runs/<timestamp>/metrics.csv`:

| Metric | Definition |
|---|---|
| Penetration % / mean depth | garment verts behind the nearest body vertex's normal (NCS-style) |
| Stretch | edge length / rest: mean, max, % > 1.1 |
| Jitter | mean ‖second difference‖ of vertex positions |
| Deformer ms | CPU timer plus GPU timestamp queries |
| Frame ms | per stage: MM / body / deform / render |

**Recording** (`runs/<timestamp>/`):
- `input.bin` (replayable)
- `poses.npy` (T, J, 4) with `root.npy`, the format the Python `ncs` side reads **[D20]**
- optional `garment_<id>.npy` (T, V, 3)
- `poses.bvh`
- PNG frames or MP4 (via an ffmpeg pipe if `ffmpeg.exe` is on PATH)

**A/B:** each garment can hold several deformers, and Tab cycles them. *Split*
mode draws a second Geno 1 m to the side with deformer B, driven by the same
pose.

**Headless:** `playground.exe --scene scenes/eval_track.toml --headless --ticks 3600 --deformer <manifest>`.

---

## 13. Performance budget (60 fps @ 1080p, RTX 3060-class) **[D2]**

| Stage | Budget |
|---|---|
| Motion matching (amortized search + pose + IK) | ≤ 1 ms |
| Morphs + LBS + normals (body) | ≤ 1 ms |
| Garment deformers | measured; target ≤ 4 ms each |
| Render | ≤ 6 ms |
| UI + metrics | ≤ 2 ms |

---

## 14. Testing (doctest + CTest)

| Area | Tests |
|---|---|
| math | quat identities, slerp endpoints, spring convergence |
| anim | BVH parse of `Geno_bind.bvh`: 75 joints, FK hips height ≈ 0.855 m; mirror twice = identity |
| mm | 27 features; querying with a DB frame's own features returns that frame; AABB search = brute-force result; tag filter never returns a wrong-stance frame; inertialization offset → 0 with no pop at the switch; replay determinism |
| body | zero morphs + bind pose → rest verts; identity LBS is a no-op; normals are unit length |
| garment | push-out leaves 0 penetrating verts on the bind pose; skin weights sum to 1 |
| deform | `placeholder` == `lbs` bit-for-bit; manifest validation errors; ONNX path with a **test-only tiny identity graph generated in the test** (never shipped) |
| render | a hidden-window frame renders with no GL errors; every material shader compiles |

---

## 15. Phases

Each phase ends with something you can run. **Start each phase by asking the
user about its open decisions.**

| # | Phase | Deliverable | Gate decisions |
|---|---|---|---|
| P0 | Shell | CMake/vcpkg build, raylib window, ImGui, fixed tick, cameras, ground grid, TOML config + hot reload | D1, D3, D18 |
| P1 | Geno + clips | ufbx Geno load, CPU LBS, procedural morphs, BVH loader, FK, mirror, clip browser (GenoView parity) | D7, D19 |
| P2 | Rendering | deferred PBR + IBL + shadows + SSAO + AgX + FXAA, materials incl. cloth, artifact grid, custom shader hook | D11, D14 |
| P3 | Motion matching | tags + clip-browser tagging, DB build, AABB search, controller (walk/run/sprint/crouch/strafe), inertialization, sync/adjust/clamp, foot IK, debug panel, record/replay | D6, D15, D16 |
| P4 | Garments | procedural skirt/cape, fitter, GarmentCode + NCS imports, skin binding, `lbs` baseline | D8 |
| P5 | Deformers + eval | `IDeformer`, placeholder/onnx/body_corrective stubs, manifest, metrics HUD, recording, A/B, eval track, headless | D9, D10, D13, D20 |
| P6 | GPU extras | `glsl_compute` + `cuda` templates, CUDA–GL interop, video capture | D10 |

---

## 16. References
- orangeduck/Motion-Matching (MIT): `controller.cpp`, `database.h`, `spring.h`, `resources/generate_database.py`
- orangeduck/GenoView (MIT): deferred renderer, SSAO, shadows, grid shader, Geno binary loader
- orangeduck/lafan1-resolved, 100style-retarget, zeroeggs-retarget: Geno character + data
- Clavet, *Motion Matching and The Road to Next-Gen Animation*, GDC 2016
- Holden et al., *Learned Motion Matching*, SIGGRAPH 2020; Holden, *Code vs Data Driven Displacement*
- Mason et al., *Real-Time Style Modelling of Human Locomotion…* (100STYLE), 2022
- Harvey et al., *Robust Motion In-Betweening* (LAFAN1), SIGGRAPH 2020
- Korosteleva et al., *GarmentCode* (2023) / *GarmentCodeData* (2024)
- Bertiche et al., *Neural Cloth Simulation*, SIGGRAPH Asia 2022
