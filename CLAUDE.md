# NCS-Torch: Neural Cloth Simulation (PyTorch Port)

Porting [NeuralClothSim](https://github.com/hbertiche/NeuralClothSim) to modern PyTorch.
Original paper: *Neural Cloth Simulation* (Bertiche et al.)

## Project Goal

Reproduce the unsupervised deep-learning cloth simulation pipeline in clean PyTorch, with testable modules at each stage.

---

## Architecture Overview

```
ncs/
├── mesh/           # Mesh utilities (vertices, faces, edges, adjacency)
├── physics/        # Energy formulations (mass-spring, BW98, StVK)
├── collision/      # Collision detection & response
├── model/          # Neural network (cloth deformation predictor)
├── data/           # Dataset loading (pose sequences, body models)
├── train.py        # Training loop
├── predict.py      # Inference
└── config.py       # Hyperparameters & fabric properties
```

---

## Implementation Phases

Each phase is independently testable before moving to the next.

### Phase 1 — Mesh Utilities (`ncs/mesh/`)
**Goal:** Load and query garment meshes.

- Load `.obj` files → vertices, faces tensors
- Compute edges, edge lengths (rest state)
- Compute face normals, face areas
- Compute vertex adjacency / one-ring neighborhoods
- **Test:** assert edge lengths > 0, normals are unit vectors, areas > 0

### Phase 2 — Physics: Energy Formulations (`ncs/physics/`)
**Goal:** Differentiable cloth energy functions.

Implement three formulations (select via config):
1. **Mass-Spring** — spring energy on edges
2. **Baraff-Witkin 1998** — stretch + shear constraints on triangles
3. **Saint Venant-Kirchhoff (StVK)** — Green strain tensor on triangles

Each returns a scalar energy given deformed vertices + rest mesh.
- **Test:** energy = 0 at rest pose; energy > 0 under stretch; gradients finite

### Phase 3 — Collision (`ncs/collision/`)
**Goal:** Differentiable body–cloth collision penalty.

- Represent body as signed distance field (SDF) or triangle soup
- Compute penetration depth per cloth vertex
- Return differentiable collision loss
- **Test:** no-collision case → loss = 0; penetrating vertex → loss > 0

### Phase 4 — Data Pipeline (`ncs/data/`)
**Goal:** Load pose sequences and body model outputs.

- Parse SMPL/custom body model poses (rotation matrices or axis-angle)
- Pose body mesh given skeleton parameters
- Return batches of (posed body vertices, garment rest mesh)
- **Test:** posed body vertices change with different poses; shapes correct

### Phase 5 — Neural Network (`ncs/model/`)
**Goal:** Network that predicts cloth vertex offsets.

- Input: body pose features (joint angles or skinned vertices)
- Output: per-vertex displacement from a template (rest cloth)
- Architecture: MLP or graph network over cloth vertices
- Static/dynamic subspace disentanglement (as per paper)
- **Test:** forward pass produces output shape `(B, V, 3)`; no NaNs

### Phase 6 — Training Loop (`ncs/train.py`)
**Goal:** Unsupervised training combining physics loss + collision loss.

```
total_loss = w_physics * E_physics + w_collision * E_collision + w_reg * E_reg
```

- Gradient flows through network → deformed verts → energy
- Logging (loss curves per component)
- Checkpoint saving/loading
- **Test:** loss decreases over 10 steps on a single batch (smoke test)

### Phase 7 — Inference (`ncs/predict.py`)
**Goal:** Run trained model on a pose sequence.

- Load checkpoint
- Loop over frames, output deformed cloth meshes
- Export `.obj` per frame (or batched)
- **Test:** output mesh vertex count matches garment template

---

## Config Schema (`ncs/config.py`)

```python
@dataclass
class Config:
    # Fabric
    formulation: str   # "mass_spring" | "baraff_witkin" | "stvk"
    density: float     # kg/m²
    thickness: float   # m
    stretch: float
    shear: float
    bending: float

    # Collision
    collision_threshold: float
    collision_weight: float

    # Training
    lr: float
    batch_size: int
    epochs: int
    device: str        # "cuda" | "cpu"
```

---

## Testing Strategy

Each module has a `tests/test_<module>.py` with:
- Unit tests (pure math, no I/O)
- Smoke tests (tiny mesh, 2–4 vertices/faces)
- Gradient checks (`torch.autograd.gradcheck`) for physics energies

Run tests:
```bash
pytest tests/ -v
```

---

## Dependencies

```
torch >= 2.0
numpy
trimesh          # mesh I/O
pytest
```

---

## Development Notes

- All tensors: `float32`, device-aware
- Meshes stored as `(V, 3)` vertex tensors + `(F, 3)` face index tensors
- Batched ops over poses: leading batch dim `B`
- No TensorFlow/Keras — pure PyTorch throughout
- Keep physics functions pure (no `nn.Module`) for easy `gradcheck`

---

## Playground Engine (`playground/`, spec only so far)

Real-time **C++ / Windows-only** playground (raylib 6.0, GL 4.3) for evaluating neural cloth/body
models: the Geno character with motion matching on orangeduck's lafan1-resolved + 100style-retarget
data (walk/run/sprint/crouch/jump, 60 Hz), **GPU-resident** compute skinning + blendshapes (hardware
GPU only, no software rendering; CPU runs animation only), deferred PBR rendering, and
ONNX Runtime / GLSL-compute / implicit-MLP / CUDA deformers. Garments are auto-skinned offline in
Blender (Robust Skin Weights Transfer). **No trained models are shipped. Neural parts are placeholder code.**
The engine shares only files (`.npy`, `.onnx`) with the Python `ncs/` package.

- Spec: `docs/playground/SPEC.md`
- Open decisions: `docs/playground/DECISIONS.md`
- Existing tools: `playground/tools/gpu_validate` (raylib/GL/CUDA checks), `playground/tools/normalmap_validate`
  (normal-map bake/render validation), `playground/tools/blender` (skinning prototype, normal-map test case),
  `playground/tools/analysis` (LAFAN1 transition/jump scan)

**Rule: ask questions instead of making assumptions.** Before you implement a playground phase,
check its gate decisions in `DECISIONS.md`. If any are still open, ask the user. Don't silently
pick the recommended default. Record each answer under *Resolved* with the date. Keep the design
simple: it is a fast-iteration playground, not a game engine.
