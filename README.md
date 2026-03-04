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
| **BSP Layout Engine** | Production layout engine using Binary Space Partitioning with zone-based band allocation, privacy gradient, and architectural compliance |
| **Engine Registry** | Unified dispatch layer supporting BSP (default), Grid, and GNN engines with automatic fallback |
| **3-Panel Workspace** | Professional CAD-style UI: left controls, center interactive canvas, right property panel |
| **Interactive CAD Canvas** | Zoom/pan/grid overlay, room selection, drag-to-move, dimension lines, north arrow |
| **AI Design Chat** | Natural-language floor plan generation via Groq LLM integration |
| **DXF Export** | Industry-standard AutoCAD DXF file export with walls, doors, windows, and dimensions |
| **3D Visualization** | Real-time interactive 3D walkthrough using React Three Fiber |
| **Boundary Upload** | Upload DXF site boundaries with automatic setback and buildable area computation |
| **Centralized State** | React useReducer store for layout, canvas, and UI state — no prop drilling |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Nginx (port 80)                       │
├────────────────────────┬─────────────────────────────────┤
│   Frontend (React 18)  │       Backend (FastAPI)          │
│   Vite · port 5173     │       Uvicorn · port 8000       │
│                        │                                  │
│  3-Panel Workspace:    │  Primary API:                    │
│  ┌──────┬───────┬────┐ │  POST /api/architect/design      │
│  │ Chat │ CAD   │Prop│ │  POST /api/architect/redesign     │
│  │ Form │Canvas │ery │ │  WS   /api/architect/ws           │
│  │      │       │    │ │                                  │
│  └──────┴───────┴────┘ │  Engine Registry:                │
│                        │  ├─ BSP  (arch_engine.py)        │
│  State Store:          │  ├─ Grid (layout_engine/)        │
│  layoutStore.jsx       │  └─ GNN  (gnn_engine.py)        │
│                        │                                  │
│  API Service:          │  Support APIs:                   │
│  api.js → fetch()      │  /api/boundary/* · /api/project  │
│                        │  /api/requirements · /api/health │
│                        │  /api/model3d · /api/floorplan   │
├────────────────────────┴─────────────────────────────────┤
│             SQLite (aiosqlite) / PostgreSQL                │
└──────────────────────────────────────────────────────────┘
```
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
- **Docker Compose** — multi-service orchestration (backend, frontend, Nginx)
- **Nginx** — reverse proxy

---

## Quick Start

### Prerequisites
- **Python 3.11+** — [python.org](https://python.org)
- **Node.js 20+** — [nodejs.org](https://nodejs.org)
- **Git**

### One-Command Start (Windows)

```powershell
# Clone and start
git clone <repo-url>
cd CAD

# Copy and edit your API keys
copy .env.example .env
notepad .env

# Start everything
.\start.bat
```

Or with PowerShell:
```powershell
.\start.ps1
```

This automatically:
1. Creates a Python virtual environment
2. Installs all backend dependencies
3. Installs all frontend dependencies
4. Starts backend (FastAPI) on **http://localhost:8000**
5. Starts frontend (Vite) on **http://localhost:5173**
6. Opens the browser

### Manual Start

```bash
# Terminal 1 — Backend
cd backend
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
python main.py

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

### Docker (Production)

```bash
docker-compose up --build
```

Access the app at `http://localhost` (Nginx reverse proxy).

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
│   │   ├── architect.py        # Primary design endpoint (uses engine_registry)
│   │   ├── boundary.py         # Boundary upload/parse/setback
│   │   ├── project.py          # Project management
│   │   ├── requirements.py     # Room requirements
│   │   ├── floorplan.py        # Floor plan CRUD
│   │   ├── model3d.py          # 3D model generation
│   │   ├── chat.py             # WebSocket chat
│   │   ├── engine.py           # Pro layout engine endpoint
│   │   ├── gnn_design.py       # GNN engine endpoint
│   │   ├── perfect_design.py   # PerfectCAD engine endpoint
│   │   ├── ai_design.py        # AI chat design endpoint
│   │   └── ml_design.py        # ML pipeline endpoint
│   ├── services/
│   │   ├── engine_registry.py  # Unified engine dispatch (BSP/Grid/GNN)
│   │   ├── arch_engine.py      # BSP production layout engine (~2500 lines)
│   │   ├── gnn_engine.py       # GNN-based layout generator
│   │   ├── perfect_layout.py   # PerfectCAD engine (3-band mode)
│   │   ├── pro_layout_engine.py# Zone-based tiling engine
│   │   ├── multi_factor_engine.py # Multi-factor reasoning engine
│   │   ├── cad_export.py       # DXF file generation
│   │   ├── model3d.py          # 3D mesh builder
│   │   ├── boundary.py         # Boundary parsing service
│   │   ├── layout_constants.py # Room dimension/area constants
│   │   └── layout_engine/      # Modular grid/treemap layout components
│   ├── ml_pipeline/            # Trainable ML layout generator
│   ├── samples/                # Sample DXF boundary files
│   └── exports/                # Generated DXF output files
└── frontend/
    ├── src/
    │   ├── App.jsx             # Root with LayoutProvider + routing
    │   ├── store/
    │   │   └── layoutStore.jsx # Centralized state (useReducer)
    │   ├── services/
    │   │   └── api.js          # Backend API service layer
    │   ├── pages/
    │   │   ├── LandingPage.jsx # Home / feature showcase
    │   │   └── WorkspaceNew.jsx# 3-panel design workspace
    │   └── components/
    │       ├── canvas/
    │       │   └── CadCanvas.jsx       # Interactive zoom/pan/drag canvas
    │       ├── toolbar/
    │       │   └── CanvasToolbar.jsx    # Grid/snap/zoom controls
    │       ├── panels/
    │       │   └── PropertyPanel.jsx    # Room properties + room list
    │       ├── PlanPreview.jsx         # 2D floor plan SVG renderer
    │       ├── Viewer3D.jsx           # Three.js 3D walkthrough
    │       ├── FormInterface.jsx      # Design input form
    │       ├── AIDesignChat.jsx       # AI-powered design assistant
    │       ├── ExportPanel.jsx        # DXF/image export controls
    │       └── BoundaryPreview.jsx    # Boundary shape viewer
    └── public/
```

---

## Layout Engines

All engines are accessed through the **Engine Registry** (`services/engine_registry.py`), which provides a unified interface with automatic fallback.

### BSP Engine (Default — Production)

The primary layout engine using Binary Space Partitioning with architectural reasoning:

- **10-Step Pipeline**: Plot analysis → Zoning → Room list → Adjacency graph → Routing → Placement → Doors/windows → Dimension chains → Validation → Summary
- **3-Zone System**: Public (front band), Service (middle), Private (rear band)
- **Adjacency Compliance**: Mandatory + preferred adjacency graphs with scoring
- **Position Data**: Every room includes `position`, `polygon`, `centroid` for interactive editing
- **Supports**: 1–5 BHK configurations, custom plot dimensions, boundary polygons

### Grid Engine

Zone-based proportional tiling using the `layout_engine/` module. Uses treemap subdivision for area allocation.

### GNN Engine

Graph Neural Network-inspired placement with adjacency-aware scoring. Generates candidates and scores them on area utilization, room proportions, and natural lighting.

---

## API Reference

### Primary Endpoints (Frontend uses these)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/architect/design` | Generate floor plan via engine registry |
| POST | `/api/architect/redesign` | New layout with same requirements |
| WS | `/api/architect/ws` | Interactive chat + design pipeline |

### Support Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/project` | Create project |
| POST | `/api/requirements` | Store room requirements |
| POST | `/api/upload-boundary` | Upload DXF boundary file |
| GET | `/api/extract-boundary/{id}` | Extract boundary polygon |
| POST | `/api/buildable-footprint/{id}` | Compute buildable area with setbacks |
| POST | `/api/generate-3d/{id}` | Generate 3D model |

### Alternative Engine Endpoints (Testing/Direct Access)

| Method | Endpoint | Engine |
|--------|----------|--------|
| POST | `/api/engine/design` | Pro Layout (Grid) |
| POST | `/api/gnn/design` | GNN |
| POST | `/api/perfect/design` | PerfectCAD |
| POST | `/api/ml/generate` | ML Pipeline |

### Example Request

```bash
curl -X POST http://localhost:8000/api/architect/design \
  -H "Content-Type: application/json" \
  -d '{
    "total_area": 1200,
    "bedrooms": 2,
    "bathrooms": 2,
    "floors": 1,
    "extras": []
  }'
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite+aiosqlite:///./floorplan.db` |
| `GROQ_API_KEY` | Groq API key for LLM chat | — |
| `SECRET_KEY` | App secret key | `change-me-in-production` |
| `GROK_API_KEY` | xAI Grok API key (primary AI) | — |
| `GROK_MODEL` | Grok model name | `grok-3-mini` |
| `CORS_ORIGINS` | Allowed CORS origins | `localhost:5173,localhost:3000` |

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
