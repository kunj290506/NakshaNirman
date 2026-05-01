# Core Project Structure

This directory contains the entire application, neatly separated into Backend and Frontend environments. By understanding this structure, any developer or team member can immediately grasp the logical placement of files and the overall boundaries of the codebase.

## Directory Tree

```
CAD-Floor-Plan-Generator/
|
|-- backend/               # Main python backend (FastAPI / Machine Learning / LLMs)
|   |-- app/               # Application package
|   |   |-- main.py        # Core API Routes (Start Here)
|   |   |-- core/          # Environment and shared settings
|   |   |   |-- config.py  # Environment Variables and Core App Settings
|   |   |-- services/      # External LLM and prompt services
|   |   |   |-- llm.py     # Local LLM interfacing
|   |   |   |-- prompt_builder.py  # Dynamic prompt construction
|   |   |-- engines/       # Deterministic geometry and quality logic
|   |   |   |-- layout_engine.py   # Geometry and coordinate math
|   |   |   |-- quality_engine.py  # Plan quality scoring
|   |   |-- validators/    # Plan validation helpers
|   |       |-- plan_validator.py  # Geometry rules and repairs
|   |-- requirements.txt   # Exact Python Packages needed
|   |-- scripts/           # Offline utilities and training scripts
|   |   |-- finetune.py    # ML fine-tuning operations
|   |-- tests/             # Backend test suite (future)
|
|-- frontend/              # Main modern Web App (React / Vite)
|   |-- src/               # The Source Code for the UI
|   |   |-- components/    # Reusable complex UI
|   |   |   |-- canvas/    # Sub-components specific to rendering interactive 2D/3D plans
|   |   |   |-- panels/    # High level sidebars for Properties and Actions
|   |   |   |-- toolbar/   # Main action items during interactions
|   |   |-- pages/         # Full views and URLs like the Main Dashboard and Landing screen
|   |   |-- services/      # Abstractions for fetching API data (api.js, auth.js)
|   |   |-- store/         # Global Layout data storage and state management
|   |   |-- styles/        # Global CSS classes where pure UI styling lives
|   |-- package.json       # Exact Web Packages needed
|   |-- vite.config.js     # Build and Dev configurations for the frontend
|
|-- docs/                  # Documentation for quick architecture overviews
|   |-- ARCHITECTURE.md    # What the parts do and the Data Flow
|   |-- FOLDER_STRUCTURE.md# You are reading this file
|
|-- .github/
|   |-- workflows/
|       |-- ci.yml         # CI/CD instructions for automated tests and builds
|
|-- start.ps1              # Unified Boot script for local developers
```

## Why it is Organized This Way

1. **Extensive Modularity:** The code is not packed into a single giant script. Every core domain of the platform lives in its own properly labelled file. For example, layout_engine.py handles the physical math, while llm.py handles just the AI interaction. This makes it easy to find and fix bugs.
2. **Clear Boundaries:** The Frontend has its own node environment, and the Backend has its own python virtual environment. Neither pollute the other, minimizing dependency conflicts.
3. **Automated Testing Setup:** With the CI/CD pipeline integrated via GitHub Actions, there is a guarantee that both environments compile flawlessly across anyone's machine without the need for manual testing of each push.