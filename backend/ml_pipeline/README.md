# ML Floor Plan Training Pipeline

A conditional generative adversarial network (cGAN) that learns to produce dynamic CAD floor plans from irregular plot boundaries and structured user constraints.

## Architecture Overview

```
                                    ┌──────────────────┐
                                    │   Latent z~N(0,1) │
                                    └────────┬─────────┘
┌────────────────┐  ┌─────────────┐          │
│  Plot Polygon  │──┤ PolygonEnc  ├──┐       │
│  (N×2 coords)  │  │ Transformer │  │       │
└────────────────┘  └─────────────┘  ▼       ▼
                                  ┌──────────────┐
┌────────────────┐  ┌─────────────┐│  Generator   │  ┌───────────┐
│  Conditions    │──┤ ConditionEnc├┤│  5-stage     │──┤ Mask Head │──▶ (B, C, 256, 256)
│  (34-d vector) │  │ MLP         ││  up-conv +   │  │ 1×1 Conv  │   room class logits
└────────────────┘  └─────────────┘│  CondResBlk  │  ├───────────┤
                                   │              │──┤ Adj Head  │──▶ (B, C, C)
┌────────────────┐  ┌─────────────┐│              │  │ MLP       │   adjacency matrix
│  Occupancy Grid│──┤ OccGridEnc  ├┘              │  └───────────┘
│  (1×256×256)   │  │ 5-layer CNN │               │
└────────────────┘  └─────────────┘               │
                                                  │
                                   ┌──────────────┤
                              ┌────┤Discriminator  │
                              │    │PatchGAN+SN    │
                              │    └───────────────┘
                              ▼
                    WGAN-GP + Hinge Loss
```

## Directory Structure

```
ml_pipeline/
├── __init__.py
├── config.py              # PipelineConfig, room types, label aliases
├── data/
│   ├── __init__.py
│   ├── preprocessing.py   # FloorPlanSample, normalize, masks, adjacency
│   ├── cubicasa.py        # CubiCasa5K dataset loader (SVG parsing)
│   ├── rplan.py           # RPLAN dataset loader (index-map PNGs)
│   └── combined.py        # CombinedDataset + DataLoader factory
├── models/
│   ├── __init__.py
│   ├── encoders.py        # PolygonEncoder, ConditionEncoder, OccGridEncoder
│   ├── generator.py       # FloorPlanGenerator (encoder-decoder, CondResBlocks)
│   ├── discriminator.py   # PatchGAN (SpectralNorm, WGAN-GP)
│   └── constraints.py     # 7 differentiable architectural constraint losses
├── training/
│   ├── __init__.py
│   ├── losses.py          # FloorPlanLoss (7 weighted terms)
│   └── trainer.py         # FloorPlanTrainer (EMA, AMP, cosine LR, checkpointing)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py         # IoU, adjacency acc, diversity, violations, etc.
└── export/
    ├── __init__.py
    └── mask_to_cad.py     # MaskToCAD: predicted mask → DXF drawing
```

## Room Taxonomy (21 Classes)

| Idx | Room Type        | Indian Name     |
|-----|------------------|-----------------|
| 0   | living_room      | Drawing Room    |
| 1   | master_bedroom   | Master Bed Room |
| 2   | bedroom          | Bed Room        |
| 3   | kitchen          | Kitchen         |
| 4   | bathroom         | Bath            |
| 5   | toilet           | Toilet          |
| 6   | dining           | Dining Area     |
| 7   | study            | Study           |
| 8   | pooja            | Puja Room       |
| 9   | store            | Store Room      |
| 10  | utility          | Wash Area       |
| 11  | balcony          | Balcony         |
| 12  | staircase        | Staircase       |
| 13  | hallway          | Passage         |
| 14  | parking          | Parking         |
| 15  | porch            | Porch           |
| 16  | garden           | Garden          |
| 17  | wall             | —               |
| 18  | door             | —               |
| 19  | window           | —               |
| 20  | exterior         | —               |

## Datasets

### CubiCasa5K
- ~5,000 annotated floor plans (SVG + rooms.png)
- Download: [zenodo.org/record/2613548](https://zenodo.org/record/2613548)
- Place in: `data/cubicasa5k/`
- Structure: `cubicasa5k/{category}/{id}/model.svg` + `rooms.png`

### RPLAN
- ~80,000 floor plans as index-map PNGs
- Download: [rplan dataset](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/)
- Place in: `data/rplan/`
- Structure: `rplan/*.png` (pixel value = room class index)

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt

# Additional ML dependencies
pip install torch torchvision scipy scikit-learn Pillow lxml
```

### 2. Download Datasets

Place datasets under `backend/data/`:

```
backend/data/
├── cubicasa5k/
│   ├── colorful/
│   ├── high_quality/
│   └── high_quality_architectural/
└── rplan/
    ├── 0.png
    ├── 1.png
    └── ...
```

### 3. Train

```bash
cd backend

# Basic training
python train.py --data-dir ./data --epochs 200 --batch-size 16

# With GPU + mixed precision
python train.py --data-dir ./data --epochs 300 --batch-size 32 --amp

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_100.pt

# Custom learning rates
python train.py --lr-g 1e-4 --lr-d 2e-4 --d-steps 3
```

### 4. Evaluate

```bash
python train.py --eval-only --checkpoint checkpoints/best_model.pt --data-dir ./data
```

### 5. Run Inference via API

Start the backend server and use the ML endpoint:

```bash
# Start server
python main.py

# Generate floor plan
curl -X POST http://localhost:8000/api/ml/generate \
  -H "Content-Type: application/json" \
  -d '{
    "plot_width": 30,
    "plot_length": 40,
    "bedrooms": 3,
    "bathrooms": 2,
    "kitchen_type": "open",
    "parking": true,
    "budget": "medium",
    "num_variants": 3
  }'

# Check status
curl http://localhost:8000/api/ml/status
```

## Condition Vector (34 dimensions)

| Dims   | Feature           | Encoding           |
|--------|-------------------|--------------------|
| 0–5    | Bedrooms          | One-hot (1–6)      |
| 6–9    | Bathrooms         | One-hot (1–4)      |
| 10–12  | Kitchens          | One-hot (1–3)      |
| 13–16  | Entry side        | One-hot (S/N/E/W)  |
| 17–20  | Budget level      | One-hot (4 levels)  |
| 21–22  | Kitchen type      | One-hot (open/closed) |
| 23     | Has parking       | Binary             |
| 24–25  | North direction   | sin/cos encoding   |
| 26–27  | Total area        | log-normalised (2d) |
| 28     | Aspect ratio      | Continuous         |
| 29–33  | Reserved          | Zero padding       |

## Loss Functions

| Loss Term        | Weight (λ) | Description                              |
|------------------|------------|------------------------------------------|
| Adversarial      | 1.0        | WGAN-GP hinge loss                       |
| Reconstruction   | 10.0       | Cross-entropy on room masks              |
| Adjacency        | 5.0        | BCE on adjacency matrix                  |
| Boundary         | 2.0        | Rooms exceeding plot boundary            |
| Diversity        | 1.0        | Pairwise distance between variants       |
| Feature Matching | 2.0        | L1 on discriminator features             |
| Structural       | 3.0        | 7 architectural constraint penalties     |

### Architectural Constraints
1. **Boundary fitting** — rooms must stay within plot polygon
2. **Minimum area** — each room meets min. size requirements
3. **Ventilation** — bedrooms must touch exterior wall
4. **Plumbing alignment** — wet rooms (kitchen, bath) should be adjacent
5. **Column grid** — walls align to 3 ft structural grid
6. **Zoning** — public rooms (front), private rooms (rear)
7. **Circulation** — hallway provides connectivity

## Evaluation Metrics

| Metric               | Target    | Description                                 |
|----------------------|-----------|---------------------------------------------|
| Mean IoU             | > 0.65    | Pixel-level overlap with ground truth        |
| Adjacency Accuracy   | > 0.80    | Correct room-to-room adjacency predictions  |
| Diversity Score      | > 0.15    | Pairwise L1 distance across samples         |
| Violation Rate       | < 0.05    | Fraction of rooms violating constraints     |
| Room Count Accuracy  | > 0.85    | Correct number of rooms generated           |
| Area KL Divergence   | < 0.30    | Distribution match for room areas           |

## Training Tips

1. **Start with RPLAN** — larger dataset, simpler format, faster to iterate
2. **Pre-train discriminator** — run 5 D-steps per G-step initially
3. **Monitor diversity** — if all outputs look identical, increase `lambda_diversity`
4. **Watch boundary violations** — increase `lambda_structural` if rooms overflow
5. **Use EMA** — the exponential moving average generator produces smoother results
6. **Checkpoint frequently** — GANs can collapse; save every 10 epochs

## Hyperparameter Recommendations

| Plot Type       | Batch Size | λ_boundary | λ_structural | Epochs |
|----------------|------------|------------|--------------|--------|
| Rectangular    | 32         | 2.0        | 3.0          | 150    |
| Irregular      | 16         | 5.0        | 5.0          | 250    |
| Mixed          | 16         | 3.0        | 4.0          | 200    |

## API Endpoints

### `POST /api/ml/generate`
Generate floor plan(s) using the trained model. Falls back to heuristic engine if no checkpoint is found.

### `GET /api/ml/status`
Check if the ML model is loaded and available.

### `POST /api/ml/batch`
Generate 3+ variants for comparison (convenience alias).

## Integration with Existing System

The ML pipeline integrates seamlessly with the existing heuristic engine:

- **No model trained yet?** → API falls back to the 6-strategy heuristic engine
- **Model available?** → Uses learned generator for more diverse, realistic layouts
- **Same DXF export** → Both engines produce industry-standard DXF files
- **Same frontend** → React UI works with both engines transparently
