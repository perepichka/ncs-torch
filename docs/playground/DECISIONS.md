# NCS Playground: Decisions

This is the list of open questions for [`SPEC.md`](SPEC.md). Each entry gives a
**recommended default**. Even so, **nothing here is decided until the user
confirms it.** When a phase needs a decision that is still open, ask the user.
Do not pick the default silently. When the user answers, move the item to
*Resolved* along with the answer and the date.

## Open

| ID | Question | Recommended default | Alternatives | Blocks |
|---|---|---|---|---|
| D1 | Host language and graphics stack? | Python + `moderngl` (GL 4.5) + `glfw` + `imgui-bundle` | Python + `wgpu-py` (no CUDA interop); C++ + sokol/bgfx/Vulkan (faster, much slower to iterate) | P0 |
| D2 | Target OS, GPU and performance floor? | Linux + Windows, NVIDIA CUDA 12, CPU fallback; 60 fps on an RTX 3060-class GPU | Add macOS (no CUDA, GL 4.1, so no compute) | P0 |
| D3 | Where does the code live? | `playground/` in this repo, importing `ncs` | Separate repo with `ncs` as a dependency | P0 |
| D4 | Which body model? Do you already have the license and files? | SMPL neutral (user-supplied), plus a procedural mannequin fallback | SMPL-X (hands + face expressions), SMPL+H, a rigged glTF (Mixamo/MPFB) | P3 |
| D5 | How do LAFAN1 motions get onto the body? | Runtime per-tick retarget LAFAN1 → body skeleton | Offline retarget of the whole DB to SMPL (simpler runtime, fixed proportions) | P3 |
| D6 | Which LAFAN1 subset goes into the motion-matching DB? | `walk*`, `run*`, `sprint*` + mirroring | All clips with tags/bans; add `jumps`/`obstacles` for extra actions | P4 |
| D7 | Sim tick rate? At what fps were the target models trained? | 30 Hz (LAFAN1 native), render interpolated | 60 Hz (resample DB; models must match) | P1 |
| D8 | Which garments ship as samples? Do NCS garment OBJs or checkpoints exist to start from? | Procedural skirt + cape now; real garments when you provide them | CLOTH3D or other dataset garments (check licenses) | P5 |
| D9 | What output space do `ncs` models use? `NCSModel` returns `template + disp` and `predict.py` applies no LBS. Is that unposed (engine skins it) or world space? | `unposed` + garment LBS, as in the original NCS | `world` | P5 |
| D10 | Custom GPU deformation path? | CUDA via torch (Triton / `cpp_extension`) inside `custom` deformers | GLSL compute deformers; also support the ORT TensorRT EP in v1? | P6 |
| D11 | Asset hosting? | CC0 HDRIs and textures fetched by script into git-ignored `data/` | Commit small 1k versions; Git LFS | P2 |
| D12 | Usage scope: is this strictly research / non-commercial? (LAFAN1 is CC BY-NC-ND; SMPL is non-commercial) | Research only; never commit raw or derived LAFAN1/SMPL data | Commercial use needs different mocap and body assets | P1 |
| D13 | Which evaluation metrics matter most? Any you use already? | Penetration, stretch, jitter, inference ms; energy optional | Add others: e.g. ground-truth comparison against a physics sim | P5 |
| D14 | Is the rendering bar enough (forward + IBL + shadow + MSAA + sheen cloth)? | Yes for v1 | Add SSAO/GTAO, skin SSS, TAA, contact shadows | P2 |
| D15 | Input devices? | Xbox-layout gamepad (via SDL mapping) + keyboard/mouse | Specific controllers; jump/action buttons (needs more LAFAN1 categories) | P4 |

## Resolved

_None yet._
