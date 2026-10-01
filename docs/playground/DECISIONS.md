# NCS Playground: Decisions

Open questions for [`SPEC.md`](SPEC.md). Each one has a **recommended
default**, but **nothing is decided until the user confirms it.** When a phase
needs an open decision, ask the user. Don't silently pick the default. When an
answer comes in, move the item to *Resolved* along with the answer and the date.

## Open

| ID | Question | Recommended default | Alternatives | Blocks |
|---|---|---|---|---|
| D1 | Which graphics/window framework in C++? | **raylib 5.x (GL 4.3)** + rlImGui, reusing MIT code from Holden's Motion-Matching and GenoView | bgfx; sokol; a custom D3D11/D3D12 or Vulkan layer (more control, much more code) | P0 |
| D3 | Where does the engine live? | `playground/` in this repo, sharing data with `ncs/` through files only | Separate repo | P0 |
| D6 | Motion DB contents? | lafan1-resolved `walk*/run*/sprint*/ground*` (crawl excluded) + 100STYLE `Neutral` and `Crouched`, all mirrored | Fewer clips for faster iteration; more 100STYLE styles | P3 |
| D7 | Sim tick rate? At what fps are the target models trained? | 60 Hz, matching the data | 30 Hz (models run at a lower `rate_hz` and the engine resamples) | P1 |
| D8 | Which garments come first? | Procedural skirt + cape, then a GarmentCode T-shirt + pants made for Geno | NCS sample T-shirt/pants refit to Geno; others you have | P4 |
| D9 | Model output convention: does `ncs` predict unposed verts (engine applies LBS) or world space? | `unposed` + garment LBS, as in the original NCS | `world`, offsets | P5 |
| D10 | GPU deformer paths and ORT execution providers? | ORT CUDA EP + CPU EP; both GLSL-compute and CUDA templates | DirectML or TensorRT EP; CUDA only; GLSL only | P5/P6 |
| D11 | Asset handling? | `tools/fetch_assets.ps1` into git-ignored `data/`; generated garment OBJs not committed | Commit small CC0 assets; Git LFS | P2 |
| D13 | Which metrics matter most? Do you have any already? | Penetration, stretch, jitter, timings | Physics energy in C++ (port of `ncs.physics`); comparison against a ground-truth sim | P5 |
| D14 | Is the rendering bar enough (deferred PBR, IBL, shadows, SSAO, sheen cloth, AgX, FXAA)? | Yes for v1 | TAA, skin SSS, contact shadows, hair | P2 |
| D15 | Is the controls mapping OK (§6.3)? Do you want jump/vault in v1? | Mapping as specced; no jump in v1 | Jump using lafan1-resolved `jumps*` (needs an action-tag system) | P3 |
| D16 | Stand↔crouch transitions: is inertialization alone enough, or should transition ranges be hand-tagged from LAFAN1 `ground*`? | Tag any natural transitions found; otherwise inertialization with a 0.2 s halflife | Author or record dedicated transition clips | P3 |
| D18 | Build details: is C++20 / MSVC 2022 / CMake + vcpkg OK? Which CUDA toolkit version? | Yes; CUDA 12.x optional behind `PG_WITH_CUDA` | Visual Studio solution; Conan | P0 |
| D19 | Geno has no blendshapes. Are procedural test morphs + placeholder correctives enough for now? | Yes | Sculpt real morphs for Geno; a second character that has blendshapes | P1 |
| D20 | Recording format for feeding the Python `ncs` side? | `.npy` quaternions + root trajectory, plus BVH | `.npz` with rotation matrices; a custom format | P5 |

## Resolved

| ID | Decision | Date |
|---|---|---|
| D1a | Engine language is **C++**, not Python. | 2026-10-01 |
| D2 | **Windows only.** | 2026-10-01 |
| D4 | Body model: **Geno** from orangeduck's retargeted datasets (lafan1-resolved, zeroeggs-retarget, 100style-retarget). | 2026-10-01 |
| D5 | Retargeting: **not needed.** All datasets already share Geno's skeleton. | 2026-10-01 |
| D12 | Usage scope: **research project.** Non-commercial dataset and asset terms are acceptable. | 2026-10-01 |
| D17 | Motion matching based on LAFAN1, using **lafan1-resolved** (60 fps, Geno skeleton), with basic controls: movement, crouching, sprinting. | 2026-10-01 |
| D21 | **No models are produced or shipped.** Neural components are placeholder code only. | 2026-10-01 |
| D22 | Garments: start from **free online sources** (GarmentCode MIT, NeuralClothSim samples) plus procedural ones. | 2026-10-01 |
