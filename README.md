# NakshaNirman

NakshaNirman is an AI-assisted residential floor plan generator for Indian-style house planning.
It includes a FastAPI backend, a React frontend, and DXF export support.

## Current Project Status

The current codebase is focused on:

1. LLM-first plan generation with deterministic BSP fallback.
2. Detailed architect reasoning output for each generated plan.
3. Reasoning parity benchmarking against ChatGPT, Gemini, Opus, and DeepSeek style profiles.
4. Full-auto element placement without manual per-room adjustment requirement.

## Runtime Architecture

- Frontend: Vite React app on http://localhost:5173
- Backend: FastAPI app on http://localhost:8010
- API docs: http://localhost:8010/api/docs
- Frontend proxy: /api requests are proxied to backend port 8010

## Repository Layout

- docker-compose.yml
- nginx.conf
- start.bat
- start.ps1
- backend/
  - main.py
  - config.py
  - models.py
  - layout_engine.py
  - llm.py
  - plan_validator.py
  - prompt_builder.py
  - dxf_export.py
  - requirements.txt
  - reasoning_accuracy_benchmark.py
  - exports/
- frontend/
  - package.json
  - vite.config.js
  - src/
    - App.jsx
    - main.jsx
    - components/
    - pages/
    - services/
    - store/

## Key Backend APIs

### Health
- GET /api/health

### Auth
- POST /api/auth/signup
- POST /api/auth/login
- POST /api/auth/save-and-login

### Planning
- POST /api/architect/reason
  - Returns structured pre-plot architect reasoning only.
- POST /api/generate
  - Generates full plan output with architect_reasoning and optional dxf_url.
- POST /api/validate
  - Validates a plan response payload.

### Exports
- GET /api/download/{filename}

## Planning and Reasoning Engine

The backend planning flow is:

1. Build pre-plot architect reasoning (local plus optional LLM advisory).
2. Try OpenRouter planning path.
3. Try Claude path if configured.
4. Fall back to deterministic BSP path for guaranteed completion.
5. Attach detailed reasoning package to response.

Reasoning package includes:

- requirement_coverage
- assumptions
- counterfactual_options
- element_reasoning with checks and metrics
- diagnostics and quality_scores
- frontier_comparison profile scores

## Reasoning Accuracy Benchmark

A reusable benchmark script is included:

- backend/reasoning_accuracy_benchmark.py

Run benchmark (deterministic emergency mode):

```powershell
cd backend
.venv\Scripts\python.exe reasoning_accuracy_benchmark.py --json
```

Run benchmark (primary full pipeline):

```powershell
cd backend
.venv\Scripts\python.exe reasoning_accuracy_benchmark.py --mode primary --json
```

Output includes:

- overall_accuracy
- chatgpt_accuracy
- gemini_accuracy
- opus_accuracy
- deepseek_accuracy
- weak_cases and failed_cases

## Local Setup

### Option 1: One-command startup (recommended)

```powershell
.\start.ps1
```

or

```cmd
start.bat
```

This prepares dependencies and starts:

- Backend on 8010
- Frontend on 5173

### Option 2: Manual startup

Backend:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

## Environment Variables

Primary environment file:

- .env (root)
- backend/.env is copied from root by startup scripts

Common variables:

- OPENROUTER_API_KEY
- OPENROUTER_API_KEY_SECONDARY
- OPENROUTER_MODEL
- OPENROUTER_PLAN_MODEL
- OPENROUTER_BASE_URL
- ANTHROPIC_API_KEY
- CLAUDE_MODEL
- FORCE_LOCAL_PLANNER
- FAST_FALLBACK_MODE
- ARCHITECT_REASONING_ENABLED
- PUBLIC_LLM_FALLBACK_ENABLED
- PUBLIC_LLM_FALLBACK_URL
- PUBLIC_LLM_FALLBACK_MODEL
- RATE_LIMIT_GENERATE_PER_MINUTE
- RATE_LIMIT_WINDOW_SECONDS
- RATE_LIMIT_BYPASS_LOCAL
- MONGO_URI
- MONGO_DB_NAME
- MONGO_USERS_COLLECTION

## Docker

To run with docker compose:

```bash
docker-compose up --build
```

## Notes

- Generated DXF files are written to backend/exports.
- If external LLM providers are unavailable, backend returns deterministic fallback instead of hanging.
- Reasoning output is designed to be machine-readable and benchmarkable.
