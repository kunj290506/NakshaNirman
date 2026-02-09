# AutoArchitect AI

Update placeholder for 2026-02-10
# AutoArchitect AI

Transform user-uploaded plot boundaries (image/DXF) into complete home designs (DXF files) and cinematic 3D animations with drone-style fly-throughs.

## Project Structure

```
autoarchitect-ai/
├── frontend/          # React.js + TypeScript Application
├── backend/           # Python FastAPI Server
├── blender/           # Blender Automation Scripts
├── shared/            # Shared Assets & Templates
└── docker-compose.yml # Container Orchestration
```

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- Blender 3.6+
- Docker & Docker Compose

### Development Setup

1. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

2. **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

3. **Docker Setup (Production)**
```bash
docker-compose up -d
```

## Features

- **Boundary Processing**: Upload images or DXF files to extract plot boundaries
- **AI Design Generation**: Intelligent floor plan generation using LLMs
- **CAD Export**: Professional DXF files compatible with AutoCAD
- **3D Visualization**: Interactive Three.js models
- **Cinematic Animation**: Drone-style fly-through videos

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React.js, TypeScript, Three.js, Material-UI |
| Backend | FastAPI, OpenCV, ezdxf, Celery |
| AI | Llama 3.1, Code Llama, Mistral |
| 3D/Render | Blender Python API |
| Database | PostgreSQL, Redis |

## Documentation

- [Technical Specification](./AutoArchitect_AI_Technical_Specification.md)
- [API Documentation](http://localhost:8000/docs) (when running)

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License - See LICENSE file for details

---

*Built with <3 by AutoArchitect AI Team*
