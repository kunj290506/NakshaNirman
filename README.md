# NakshaNirman - Professional AutoCAD-Level CAD Floor Plan Generator

An AI-powered architectural CAD design agent that generates highly detailed, construction-ready residential house floor plans similar in quality to professional AutoCAD drawings.

## Master Features

### Core Mission
Generate or modify **professional AutoCAD-level** house maps including:
- Clearly defined walls, room boundaries, and dimensions
- Furniture layout with realistic placement
- Toilets, kitchens, staircases, shops (if specified)
- Door and window placements
- Circulation paths and privacy zoning
- Full dimension annotations
- DXF export compatible with AutoCAD

### Design Standards (Indian Residential)
- **Metric Units**: All dimensions in millimeters (mm)
- **Wall Thickness**: 230mm (9") external, 115mm (4.5") internal partitions
- **Corridor Width**: Minimum 900mm
- **Door Width**: Minimum 800mm
- **Plot Shapes**: Supports rectangular, L-shaped, T-shaped, and irregular plots
- **Privacy Hierarchy**: PUBLIC → SEMI-PRIVATE → PRIVATE zones

## Interface

**Split Screen Application:**
- **Right Side**: Intelligent chat + form-based requirements input
- **Left Side**: Real-time visual preview of floor plan (DXF rendering)

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open http://localhost:5173 in your browser.

## Usage Examples

Chat naturally with the agent:

### Basic Requests
```
"I want a 100 sqm house with 2 bedrooms and 1 bathroom"
"Create 3BHK on 30x40 feet plot"
"Add a shop on the ground floor with street access"
"Place kitchen on north side with L-shaped platform"
```

### Advanced Requests
```
"Design for irregular L-shaped plot, 1200 sqft total area"
"Ensure master bedroom has attached bathroom"
"Add pooja room adjacent to living room"
"Include staircase for G+1 duplex structure"
```

### DXF Upload (Planned)
Upload existing DXF file → Agent analyzes plot boundary → Generates design within constraints

## Architecture

### Multi-Agent System

1. **Requirement Agent** ([requirementAgent.js](src/agents/requirementAgent.js))
   - Parses natural language input
   - Validates and structures requirements
   - Supports DXF upload analysis
   - Normalizes to internal JSON format

2. **Planning Agent** ([planningAgent.js](src/agents/planningAgent.js))
   - Performs feasibility analysis
   - Applies privacy hierarchy (public/semi-private/private)
   - Validates adjacency rules (kitchen→dining, bedroom→bathroom)
   - Ensures circulation path optimization

3. **Geometry Agent** ([geometryAgent.js](src/agents/geometryAgent.js))
   - Generates room layouts using BSP (Binary Space Partitioning)
   - Places doors and windows intelligently
   - Integrates furniture placement engine
   - Validates no overlaps or boundary violations

4. **CAD Agent** ([cadAgent.js](src/agents/cadAgent.js))
   - Generates layered DXF files
   - Separate layers for walls, doors, windows, furniture, sanitary, text, dimensions
   - AutoCAD-compatible output
   - Comprehensive dimension annotations

### AI Service ([aiService.js](src/services/aiService.js))
- Uses **Google Gemini 1.5 Flash** for natural language understanding
- Implements full master prompt for professional-grade reasoning
- Provides architectural thought process transparency
- Suggests single_floor vs duplex structures

## Key Utilities

### Constants ([constants.js](src/utils/constants.js))
Comprehensive Indian residential standards:
- **Room Types**: 13 types including shop, staircase, pooja room
- **Furniture Dimensions**: Sofa, beds (double/queen/king), dining tables, wardrobes, kitchen fixtures
- **Sanitary Fixtures**: Indian WC, Western WC, wash basin, shower, bathtub
- **Staircase Specs**: Tread depth (250mm), riser height (175mm)

### Furniture Placement ([furniturePlacement.js](src/utils/furniturePlacement.js))
Automatic realistic furniture layout:
- **Living Room**: Sofa, TV unit, coffee table
- **Bedrooms**: Bed (size based on room), wardroobe, side tables
- **Kitchen**: L-shaped or linear platform, sink, stove, refrigerator
- **Bathrooms**: WC, wash basin, shower area
- **Dining**: Table (4/6 seater) with chairs

### Plot Geometry ([plotGeometry.js](src/utils/plotGeometry.js))
Irregular plot handling:
- Detects rectangular, L-shaped, T-shaped, irregular boundaries
- Polygon area calculation (shoelace formula)
- Point-in-polygon validation (ray casting)
- Bounding box calculations

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Vanilla JavaScript + Vite |
| UI | Custom CSS with design tokens |
| CAD Export | dxf-writer (AutoCAD compatible) |
| AI | Google Gemini 1.5 Flash |
| ML Backend | Python Flask (optional, for ML predictions) |
| Architecture | Multi-agent system pattern |

## Room Types Supported

### Public Zone
- Living Room
- Dining Room
- Shop (with street access)
- Parking
- Entrance

### Semi-Private Zone
- Kitchen (L-shaped or linear layouts)
- Study
- Pooja Room
- Utility
- Balcony

### Private Zone
- Master Bedroom (with attached bathroom option)
- Bedroom(s)
- Bathroom(s)
- Storage

### Circulation
- Staircase (for duplex/G+1)
- Corridors (auto-calculated)

## Dimension Annotations

DXF output includes:
- Overall plot dimensions (width × height)
- Individual room dimensions
- Door widths
- Wall thicknesses
- Extension lines and leader arrows

## Privacy Hierarchy

Automatic zoning based on Indian residential design:

```
PUBLIC ZONE (Front)
├── Living Room
├── Dining Room
└── Shop (if specified)

SEMI-PRIVATE ZONE (Middle)
├── Kitchen
├── Study
└── Pooja Room

PRIVATE ZONE (Back)
├── Master Bedroom
├── Bedrooms
└── Bathrooms
```

## DXF Export Layers

AutoCAD-compatible layered output:

| Layer | Color | Content |
|-------|-------|---------|
| WALLS | White | External & internal walls (230mm/115mm) |
| DOORS | Cyan | Door symbols with swing arcs |
| WINDOWS | Blue | Window symbols with panes |
| FURNITURE | Green | Sofas, beds, tables, wardrobes |
| SANITARY | Magenta | WC, wash basin, shower, bathtub |
| TEXT | Yellow | Room labels and area annotations |
| DIMENSIONS | Red | Dimension lines and measurements |
| ANNOTATIONS | Cyan | Additional notes and callouts |

## Configuration

### API Key Setup
1. Click "Settings" in the app
2. Enter Google Gemini API key
3. API key stored in browser localStorage

### Customization
Edit [constants.js](src/utils/constants.js) to adjust:
- Default room sizes
- Wall thicknesses
- Furniture dimensions
- Color schemes

## Validation Rules

The system ensures:
- No room overlaps
- All rooms fit within plot boundary
- Minimum room dimensions respected
- Aspect ratios maintained (max 1:2)
- Minimum corridor widths (900mm)
- Privacy hierarchy maintained
- Adjacency requirements met
- Circulation space adequate (>10%)

## Constraints & Feasibility

If constraints conflict:
- Agent pauses design changes
- Requests clarification from user
- Never produces incorrect plans
- Provides detailed feasibility reports

Example infeasible scenario:
> "Total room area (85 m²) exceeds usable area (70 m²) after accounting for walls"

Agent response:
> "The requested 4BHK program cannot fit in a single floor. Recommend G+1 duplex structure or reduce room count."

## Design Philosophy

**Construction-Ready Quality**
- Every output matches professional CAD drawings
- Complete information for implementation
- No guessing or placeholder data
- Clear communication of limitations

**Indian Context**
- Vastu considerations (optional)
- Pooja room placement
- Indian + Western toilet options
- Kitchen layouts (L-shaped common)
- Shop integration for commercial-residential

**User Experience**
- Natural language input
- Real-time preview
- Transparent AI reasoning
- Clear error messages
- Iterative refinement support

## License

MIT

## Acknowledgments

Built with professional architectural standards and AI-powered intelligence for the next generation of residential design automation.
