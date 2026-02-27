"""
Pipeline-wide configuration and hyper-parameters.

All magic numbers live here so sweeps and experiments are trivial.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # backend/
PIPELINE_DIR = ROOT_DIR / "ml_pipeline"
DATA_DIR = PIPELINE_DIR / "datasets"                       # raw data root
CHECKPOINT_DIR = PIPELINE_DIR / "checkpoints"
LOG_DIR = PIPELINE_DIR / "logs"
EXPORT_DIR = ROOT_DIR / "exports"

for d in (DATA_DIR, CHECKPOINT_DIR, LOG_DIR, EXPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Room taxonomy — canonical label set
# ---------------------------------------------------------------------------
ROOM_TYPES: List[str] = [
    "living_room",
    "master_bedroom",
    "bedroom",
    "kitchen",
    "bathroom",
    "toilet",
    "dining",
    "study",
    "pooja",
    "store",
    "utility",
    "balcony",
    "staircase",
    "hallway",
    "parking",
    "porch",
    "garden",
    "wall",
    "door",
    "window",
    "exterior",
]

ROOM_TO_IDX = {r: i for i, r in enumerate(ROOM_TYPES)}
NUM_ROOM_CLASSES = len(ROOM_TYPES)

# Label mapping from CubiCasa5K / RPLAN raw annotation words
LABEL_ALIASES = {
    # CubiCasa5K words
    "living room": "living_room",
    "livingroom": "living_room",
    "sitting": "living_room",
    "drawing": "living_room",
    "drawing room": "living_room",
    "master bedroom": "master_bedroom",
    "bed room": "bedroom",
    "bed": "bedroom",
    "bedroom1": "bedroom",
    "bedroom2": "bedroom",
    "bedroom3": "bedroom",
    "room": "bedroom",
    "bath": "bathroom",
    "washroom": "bathroom",
    "wash": "bathroom",
    "wc": "toilet",
    "lavatory": "toilet",
    "cook": "kitchen",
    "pantry": "kitchen",
    "kitchenette": "kitchen",
    "dining room": "dining",
    "dinning": "dining",
    "study room": "study",
    "office": "study",
    "work": "study",
    "prayer": "pooja",
    "puja": "pooja",
    "worship": "pooja",
    "storage": "store",
    "closet": "store",
    "wardrobe": "store",
    "laundry": "utility",
    "utility room": "utility",
    "passage": "hallway",
    "corridor": "hallway",
    "lobby": "hallway",
    "entrance": "hallway",
    "foyer": "hallway",
    "verandah": "balcony",
    "terrace": "balcony",
    "deck": "balcony",
    "stairs": "staircase",
    "stair": "staircase",
    "car": "parking",
    "garage": "parking",
    "outdoor": "garden",
    "yard": "garden",
    "ext": "exterior",
    "outside": "exterior",
    "background": "exterior",
}


# ---------------------------------------------------------------------------
# Hyper-parameters dataclass
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """All training hyper-parameters in one place."""

    # --- Dataset ---
    dataset: str = "cubicasa5k"            # cubicasa5k | rplan | combined
    img_size: int = 256                    # rasterised mask resolution
    max_rooms: int = 16                    # max rooms per sample
    max_boundary_pts: int = 32             # polygon vertex budget
    normalize_scale: float = 1.0           # scale plots to this bounding dim

    # --- Condition vector ---
    cond_dim: int = 64                     # condition embedding dimension
    num_bedrooms_max: int = 8
    num_bathrooms_max: int = 6
    num_kitchens_max: int = 3
    budget_levels: int = 3                 # low / mid / high
    kitchen_types: int = 3                 # open / closed / semi
    parking_types: int = 3                 # none / covered / basement

    # --- Plot encoder ---
    plot_encoder: str = "polygon"          # polygon | grid
    polygon_feat_dim: int = 128            # polygon encoder output dim
    grid_channels: int = 1                 # occupancy-grid input channels

    # --- Model architecture ---
    model_type: str = "cgan"               # cgan | diffusion
    latent_dim: int = 128                  # noise z dimension
    gen_base_ch: int = 64                  # generator base channels
    disc_base_ch: int = 64                 # discriminator base channels
    num_res_blocks: int = 6                # residual blocks in bottleneck

    # Diffusion-specific
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # --- Graph (adjacency) head ---
    graph_hidden: int = 128
    graph_layers: int = 3

    # --- Training ---
    epochs: int = 200
    batch_size: int = 16
    lr_g: float = 2e-4                     # generator learning rate
    lr_d: float = 2e-4                     # discriminator learning rate
    beta1: float = 0.5                     # Adam β₁
    beta2: float = 0.999                   # Adam β₂
    d_steps: int = 2                       # disc steps per gen step
    use_amp: bool = True                   # mixed precision training
    grad_clip: float = 1.0                 # gradient clipping norm
    ema_decay: float = 0.999               # EMA of generator weights

    # --- Loss weights ---
    lambda_adv: float = 1.0                # adversarial loss
    lambda_recon: float = 10.0             # mask reconstruction (L1)
    lambda_adj: float = 5.0                # adjacency correctness
    lambda_boundary: float = 5.0           # boundary fitting
    lambda_diversity: float = 1.0          # diversity regulariser
    lambda_structural: float = 2.0         # structural feasibility
    lambda_gp: float = 10.0               # gradient penalty (WGAN-GP)

    # --- Rule-based constraints ---
    min_room_area_sqft: float = 25.0       # absolute minimum room area
    min_passage_width_ft: float = 3.0
    wall_external_ft: float = 0.75
    wall_internal_ft: float = 0.375
    column_grid_ft: float = 10.0           # structural column spacing

    # --- Evaluation ---
    eval_every: int = 5                    # evaluate every N epochs
    save_every: int = 10                   # checkpoint every N epochs
    num_eval_samples: int = 64             # batches for FID / metrics

    # --- Device ---
    device: str = "cuda"                   # cuda | cpu

    # --- Vastu (Indian) ---
    apply_vastu: bool = True
    vastu_rules_file: str = str(ROOT_DIR / "region_rules.json")

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
