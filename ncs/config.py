from dataclasses import dataclass, field


@dataclass
class Config:
    # Fabric
    formulation: str = "baraff_witkin"  # "mass_spring" | "baraff_witkin" | "stvk"
    density: float = 0.2       # kg/m²
    thickness: float = 0.001   # m
    stretch: float = 100.0
    shear: float = 10.0
    bending: float = 1.0

    # Gravity
    gravity: float = 9.81

    # Collision
    collision_threshold: float = 0.005  # m
    collision_weight: float = 1000.0

    # Training
    lr: float = 1e-4
    batch_size: int = 8
    epochs: int = 100
    device: str = "cuda"

    # Loss weights
    w_physics: float = 1.0
    w_collision: float = 1.0
    w_reg: float = 0.01

    # Paths
    garment_path: str = "body_models/garment.obj"
    body_model_path: str = "body_models/body.obj"
    data_dir: str = "data/"
    checkpoint_dir: str = "checkpoints/"
