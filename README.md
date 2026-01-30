# NakshaNirman - CAD Floor Plan Generator

A VS Code-like web application for designing construction-ready residential floor plans using an intelligent AI chat agent.

![Floor Plan Generator](https://img.shields.io/badge/CAD-Floor%20Plan%20Generator-blue)

## Features

- 🤖 **AI Chat Agent** - Natural language interface like ChatGPT
- 🏠 **Smart Room Placement** - Bin-packing algorithm for optimal layouts
- 📐 **Architectural Constraints** - 230mm walls, 900mm corridors, proper aspect ratios
- 📥 **DXF Export** - AutoCAD-compatible floor plans
- 🎨 **Light Theme UI** - Professional, modern design

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open http://localhost:5173 in your browser.

## Usage

Simply chat with the agent:

- *"I need a 100 sqm house"*
- *"2 bedrooms of 12 sqm, 1 kitchen, 1 bathroom"*
- *"living room 20 sqm, master bedroom 16 sqm"*

The agent will parse your requirements, show a summary, and generate the floor plan.

## Tech Stack

- **Frontend**: Vanilla JavaScript + Vite
- **Styling**: Custom CSS with design tokens
- **CAD**: dxf-writer for AutoCAD-compatible exports
- **Architecture**: Multi-agent system (Requirement, Planning, Geometry, CAD agents)

## License

MIT
