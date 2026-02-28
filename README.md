# NakshaNirman — AI-Powered Floor Plan Generator

An end-to-end platform that generates architecturally-compliant 2D floor plans and interactive 3D models from simple user inputs. Built with **FastAPI**, **React**, and **Three.js**.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Three.js](https://img.shields.io/badge/Three.js-3D-black?logo=three.js)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

| Feature | Description |
|---------|-------------|
| **PerfectCAD Engine** | Deterministic zone-based strip-packing layout engine with strict 3-band architectural compliance mode |
| **GNN Engine** | Graph Neural Network-driven room placement with adjacency-aware scoring |
| **AI Design Chat** | Natural-language floor plan generation via Groq/OpenAI LLM integration |
| **ML Pipeline** | Trainable generator + discriminator for learning layouts from CubiCasa / R-Plan datasets |
| **DXF Export** | Industry-standard AutoCAD DXF file export with walls, doors, windows, and dimension annotations |
| **3D Visualization** | Real-time interactive 3D walkthrough using React Three Fiber |
| **Boundary Upload** | Upload DXF site boundaries or draw custom plot shapes |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Nginx (port 80)                   │
├──────────────────────┬──────────────────────────────┤
│   Frontend (React)   │      Backend (FastAPI)        │
│   Vite · port 5173   │      Uvicorn · port 8000     │
│                      │                               │
│  • Landing Page      │  API Routes:                  │
│  • Workspace         │  /api/perfect/design          │
│  • 3D Viewer         │  /api/gnn/design              │
│  • Chat Interface    │  /api/ai-design/generate      │
│  • Export Panel      │  /api/ml/generate              │
│                      │  /api/engine/generate          │
│                      │  /api/boundary/upload          │
│                      │  /api/floorplan                │
│                      │  /api/model3d                  │
│                      │  /api/project                  │
│                      │  /api/requirements             │
│                      │  /api/health                   │
├──────────────────────┴──────────────────────────────┤
│           SQLite (aiosqlite) / PostgreSQL             │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
- **FastAPI** — async REST API with Pydantic validation
- **SQLAlchemy** + **aiosqlite** — async ORM with SQLite (dev) / PostgreSQL (prod)
- **ezdxf** — DXF file generation and parsing
- **Shapely** / **NetworkX** — computational geometry and graph algorithms
- **trimesh** / **manifold3d** — 3D mesh generation
- **Groq** / **OpenAI** — LLM-powered design chat
- **NumPy** / **SciPy** / **OpenCV** — numerical computation and image processing

### Frontend
- **React 18** with Vite
- **React Three Fiber** + **Drei** — declarative 3D scenes
- **Three.js** — WebGL rendering
- **Axios** — HTTP client
- **React Router** — client-side routing

### Infrastructure
- **Docker Compose** — multi-service orchestration (backend, frontend, Nginx, PostgreSQL, MinIO)
- **Nginx** — reverse proxy

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/kunj290506/Real-Time_Fraud_Detection_Microservice.git
cd Real-Time_Fraud_Detection_Microservice

# Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173` and the backend API on `http://localhost:8000`.

### Docker

```bash
docker-compose up --build
```

Access the app at `http://localhost` (Nginx proxy).

---

## Project Structure

```
├── docker-compose.yml          # Multi-service orchestration
├── nginx.conf                  # Reverse proxy config
├── backend/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Environment & app settings
│   ├── database.py             # Async SQLAlchemy setup
│   ├── models.py               # ORM models
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── routes/
│   │   ├── perfect_design.py   # PerfectCAD engine endpoint
│   │   ├── gnn_design.py       # GNN engine endpoint
│   │   ├── ai_design.py        # AI chat design endpoint
│   │   ├── ml_design.py        # ML pipeline endpoint
│   │   ├── engine.py           # Pro layout engine endpoint
│   │   ├── boundary.py         # Boundary upload/parse
│   │   ├── floorplan.py        # Floor plan CRUD
│   │   ├── model3d.py          # 3D model generation
│   │   ├── project.py          # Project management
│   │   ├── requirements.py     # Room requirements
│   │   └── chat.py             # WebSocket chat
│   ├── services/
│   │   ├── perfect_layout.py   # PerfectCAD engine (strict 3-band mode)
│   │   ├── gnn_engine.py       # GNN-based layout generator
│   │   ├── pro_layout_engine.py# Professional layout engine
│   │   ├── arch_engine.py      # Architectural rule engine
│   │   ├── cad_export.py       # DXF file generation
│   │   ├── model3d.py          # 3D mesh builder
│   │   ├── boundary.py         # Boundary parsing service
│   │   ├── layout_constants.py # Room dimension/area constants
│   │   └── layout_engine/      # Modular layout components
│   ├── ml_pipeline/            # Trainable ML layout generator
│   │   ├── models/             # Generator, discriminator, encoders
│   │   ├── training/           # Training loop & losses
│   │   ├── data/               # Dataset loaders (CubiCasa, R-Plan)
│   │   └── evaluation/         # Layout quality metrics
│   ├── samples/                # Sample DXF boundary files
│   └── exports/                # Generated DXF output files
└── frontend/
    ├── src/
    │   ├── App.jsx             # Root component with routing
    │   ├── pages/
    │   │   ├── LandingPage.jsx # Home / feature showcase
    │   │   └── Workspace.jsx   # Main design workspace
    │   └── components/
    │       ├── PlanPreview.jsx  # 2D floor plan SVG renderer
    │       ├── Viewer3D.jsx    # Three.js 3D walkthrough
    │       ├── FormInterface.jsx       # Design input form
    │       ├── RequirementsForm.jsx     # Room requirements editor
    │       ├── ExportPanel.jsx         # DXF/image export controls
    │       ├── ChatInterface.jsx       # LLM design chat
    │       ├── AIDesignChat.jsx        # AI-powered design assistant
    │       └── BoundaryPreview.jsx     # Boundary shape viewer
    └── public/
```

---

## Layout Engines

### PerfectCAD (Strict 3-Band Mode)

Deterministic, constraint-driven layout engine for architecturally-compliant floor plans.

**Hard constraints (1200 sqft reference):**

| Band | Zone | Rooms | Area % |
|------|------|-------|--------|
| Band 1 (Top) | Public | Drawing Room, Kitchen, Dining Area | 40% |
| Band 2 (Middle) | Service | Wash Area, Passage | ~13% |
| Band 3 (Bottom) | Private | Master Bedroom, Bedroom 1, Attached Bath (inside MBR) | ~47% |

- Exactly **8 rooms** — no utility rooms, no duplicates, no filler blocks
- Attached Bathroom carved **inside** Master Bedroom (top-right corner)
- All area percentages within specified ranges (e.g., Drawing Room 15–20%, MBR 18–22%)
- Band heights derived from target area ratios for mathematical compliance

### GNN Engine

Uses graph neural network principles with adjacency matrices to optimize room placement. Scores candidates on area utilization, proportion, adjacency satisfaction, and natural lighting.

### Pro Layout Engine

Zone-based proportional tiling with front/corridor/back structure. Supports 1–5 BHK configurations with automatic room sizing.

---

## API Reference

### Design Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/perfect/design` | Generate layout via PerfectCAD engine |
| POST | `/api/gnn/design` | Generate layout via GNN engine |
| POST | `/api/ai-design/generate` | AI chat-based design generation |
| POST | `/api/ml/generate` | ML pipeline layout generation |
| POST | `/api/engine/generate` | Pro layout engine |

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/boundary/upload` | Upload DXF boundary file |
| GET | `/api/health` | Health check |
| POST | `/api/project` | Create project |
| GET | `/api/floorplan/{id}` | Get floor plan by ID |

### Example Request

```bash
curl -X POST http://localhost:8000/api/perfect/design \
  -H "Content-Type: application/json" \
  -d '{
    "plot_width": 40,
    "plot_length": 30,
    "bedrooms": 2,
    "bathrooms": 1,
    "floors": 1,
    "extras": [],
    "strict_mode": true
  }'
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite+aiosqlite:///./floorplan.db` |
| `GROQ_API_KEY` | Groq API key for LLM chat | — |
| `SECRET_KEY` | App secret key | `change-me-in-production` |
| `MINIO_ENDPOINT` | MinIO object storage endpoint | `localhost:9000` |
| `CORS_ORIGINS` | Allowed CORS origins | `["*"]` |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m "Add my feature"`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.
