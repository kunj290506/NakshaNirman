# NakshaNirman - CAD Floor Plan Generator

A VS Code-like web application for designing construction-ready residential floor plans using an intelligent AI chat agent.

## Features

- **AI Chat Agent** - Natural language interface for describing requirements
- **Smart Room Placement** - Bin-packing algorithm for optimal layouts
- **Architectural Constraints** - 230mm walls, 900mm corridors, proper aspect ratios
- **DXF Export** - AutoCAD-compatible floor plans
- **Light Theme UI** - Professional, modern design

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open http://localhost:5173 in your browser.

## Usage

Simply chat with the agent naturally:

- "I want a 100 sqm house"
- "2 bedrooms of 12 sqm, 1 kitchen, 1 bathroom"
- "living room 20 sqm, master bedroom 16 sqm"
- "help" - for guidance

The agent understands greetings, questions, and modifications too.

## Tech Stack

- **Frontend**: Vanilla JavaScript + Vite
- **Styling**: Custom CSS with design tokens
- **CAD**: dxf-writer for AutoCAD-compatible exports
- **Architecture**: Multi-agent system (Requirement, Planning, Geometry, CAD agents)

## License

MIT
