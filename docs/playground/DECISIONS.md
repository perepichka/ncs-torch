# NCS Playground: Decisions

Open questions for [`SPEC.md`](SPEC.md). Each one has a **recommended
default**, but **nothing is decided until the user confirms it.** When a phase
needs an open decision, ask the user. Don't silently pick the default. When an
answer comes in, move the item to *Resolved* along with the answer and the date.

## Open

| ID | Question | Recommended default | Alternatives | Blocks |
|---|---|---|---|---|
| D6 | Final motion DB clip list? | lafan1-resolved walk/run/sprint + scan-selected crouch/transition/jump ranges + 100STYLE `Neutral`/`Crouched`, all mirrored; `ground*` excluded (mostly crawl) | Fewer clips for faster iteration; more 100STYLE styles | P3 |
| D9 | Model output convention: does `ncs` predict unposed verts (engine applies LBS) or world space? | `unposed` + garment LBS, as in the original NCS | `world`, offsets | P5 |
| D10 | ORT execution providers? | CUDA EP only on the frame path (zero-copy via CUDA–GL interop); CPU EP only in unit tests | Add TensorRT EP. DirectML is not an option with GL: it would need D3D12 interop. | P5 |
| D11 | Asset handling? | `tools/fetch_assets.ps1` into git-ignored `data/`; generated garments not committed | Commit small CC0 assets; Git LFS | P2 |
| D13 | Which metrics matter most? | Penetration, stretch, jitter, timings, split by motion state | Port `ncs.physics` energies to C++; ground-truth sim comparison | P5 |
| D14 | Is the rendering bar enough (deferred PBR, IBL, shadows, SSAO, sheen cloth, AgX, FXAA)? | Yes for v1 | TAA, skin SSS, contact shadows | P2 |
| D15b | Jump scope? | Standing + running jumps on flat ground; no jump from crouch unless data exists; no vaults | Crouch-jump; double-jump/parkour (out of scope) | P3 |
| D18 | Is C++20 / MSVC 2022 / CMake + vcpkg OK? Which CUDA toolkit version? | Yes; CUDA 12.x behind `PG_WITH_CUDA` | Visual Studio solution; Conan | P0 |
| D19 | Geno has no blendshapes. Are procedural test morphs + placeholder correctives enough? | Yes | Sculpt real morphs; a second character with blendshapes | P1 |
| D20 | Recording format for the Python `ncs` side? | `.npy` quaternions + root trajectory, plus BVH | `.npz` with rotation matrices | P5 |
| D23 | Blender runtime for tools: the `bpy` pip wheel in a Python 3.11 venv (what the prototype used), or a Blender app install with scipy pip-installed into its Python? Which Blender version? | `bpy` wheel venv (tested with 5.0.1) | Blender 4.2 LTS app + `--python` | P4 |
| D28 | Which kinds of normal-map detail will you evaluate? | Per-garment baked maps first (validated pipeline), then a tiling fabric detail map | Dynamic wrinkle maps (weights or a texture written by a deformer) | P2/P4 |
| D29 | Should neural deformers output normals or tangent frames alongside positions? Positions-only costs ~1° mean normal accuracy under the normal-map test (0.67° → 1.44°). | Optional `normals`/`frames` outputs in the manifest; positions-only stays supported | Positions only; recompute normals on a denser sim mesh | P5 |
| D24 | Which implicit neural models will you test first? This shapes the `implicit_mlp` manifest. | Neural body SDF (collision proxy + metric) and a per-vertex deformation field | NeRF-like/appearance; hash-grid encodings (CUDA path) | P6 |

## Resolved

| ID | Decision | Date |
|---|---|---|
| D1a | Engine language is **C++**, not Python. | 2026-10-01 |
| D1 | **raylib**, provided modern shader support is validated. raylib 6.0 GL 4.3 passed T0–T8, T10 and T11 **API-only** (sandbox software rasterizer). **Hardware validation of T0–T11 on the Windows/NVIDIA target is the P0 gate.** The API choice is reopened as D26. | 2026-10-01 |
| D25 | **No software rendering, proper GPU rendering like real engines.** Hardware GPU only. The GPU owns all geometry: morphs, skinning, deformers, normals and metrics are compute passes. The CPU runs animation only. Readback is asynchronous and only for metrics/recording. CPU LBS exists only as a test oracle. | 2026-10-01 |
| D26 | Graphics API: **stay on raylib (OpenGL) for now.** Revisit D3D12/Vulkan only if needed. | 2026-10-01 |
| D27 | **Normal mapping is supported and validated:** high-res → Decimate → Cycles MikkTSpace bake → engine render compared per pixel against the high-res mesh, at rest and under deformation (`tools/normalmap_validate`, N1–N5 pass API-only; hardware rerun required in P2). | 2026-10-01 |
| D2 | **Windows only.** | 2026-10-01 |
| D3 | **Same repo** (`playground/`). | 2026-10-01 |
| D4 | Body model: **Geno** from orangeduck's retargeted datasets. | 2026-10-01 |
| D5 | Retargeting: **not needed** (shared Geno skeleton). | 2026-10-01 |
| D7 | **60 Hz** sim tick. | 2026-10-01 |
| D8 | Initial garments: **procedural skirt + cape, then GarmentCode T-shirt + pants**, with auto-generated skinning weights (Blender). | 2026-10-01 |
| D12 | **Research project**; non-commercial dataset and asset terms are acceptable. | 2026-10-01 |
| D15 | **Jumping is supported** in v1. | 2026-10-01 |
| D16 | Stand↔crouch: **use transitions found in the data** (scan: 112 candidates); **fall back to inertialization blending** (0.2 s). | 2026-10-01 |
| D17 | Motion matching based on LAFAN1, using **lafan1-resolved**, with controls for movement, crouch, sprint and jump. | 2026-10-01 |
| D21 | **No models are produced or shipped.** Neural components are placeholder code only. | 2026-10-01 |
| D22 | Garments come from **free online sources** (GarmentCode MIT, NeuralClothSim samples) plus procedural ones. | 2026-10-01 |
