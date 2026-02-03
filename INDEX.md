# Documentation Index

Welcome to **NakshaNirman v2.0.0** - Professional AutoCAD-Level CAD Floor Plan Generator

---

## Documentation Files

### Getting Started

**[README.md](README.md)** - Start Here!
- Overview of features and capabilities
- Quick start guide (install & run)
- Usage examples (chat interface)
- Tech stack and architecture
- Room types and design standards
- DXF layer specifications
- Validation rules and constraints

---

### User Guides

**[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - User Manual
- How to use enhanced features
- Room types by privacy zone
- Design standards reference table
- DXF export layers guide
- Furniture placement examples
- Example conversations
- Customization instructions
- Troubleshooting tips
- Best practices

---

### Technical Documentation

**[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Developer Reference
- Complete implementation summary
- File-by-file code changes
- Master prompt alignment checklist
- Architecture diagrams
- API reference for utilities
- Design principles applied
- Performance metrics
- Testing guidelines

---

### Version History

**[CHANGELOG.md](CHANGELOG.md)** - Release Notes
- Version 2.0.0 release notes
- New features breakdown
- Enhanced features list
- Bug fixes
- Migration guide from v1.0.0
- Future roadmap

**[RELEASE_NOTES_V2.md](RELEASE_NOTES_V2.md)** - Visual Summary
- ASCII art presentation
- Architecture visualization
- Feature matrices
- Quick start commands
- Usage examples
- Metrics and statistics

---

## File Structure

```
d:\projects\CAD\
├── Documentation
│   ├── README.md              ← Start here
│   ├── QUICK_REFERENCE.md     ← User guide
│   ├── IMPLEMENTATION.md      ← Technical details
│   ├── CHANGELOG.md           ← Version history
│   ├── RELEASE_NOTES_V2.md    ← Visual summary
│   └── INDEX.md               ← This file
│
├── Configuration
│   ├── package.json           ← Dependencies & scripts
│   ├── vite.config.js         ← Build configuration
│   └── index.html             ← Entry point
│
├── src/
│   ├── main.js                ← Application entry
│   ├── state.js               ← Global state management
│   │
│   ├── agents/
│   │   ├── requirementAgent.js   ← Parse user input
│   │   ├── planningAgent.js      ← Feasibility & allocation
│   │   ├── geometryAgent.js      ← Room placement & layout
│   │   └── cadAgent.js           ← DXF generation
│   │
│   ├── components/
│   │   ├── canvas.js          ← Visual preview
│   │   ├── chatInterface.js   ← Chat UI
│   │   ├── requirementForm.js ← Form input
│   │   ├── splitPane.js       ← Layout manager
│   │   └── statusPanel.js     ← Status display
│   │
│   ├── utils/
│   │   ├── constants.js          ← Design standards, furniture, fixtures
│   │   ├── geometry.js           ← Geometric calculations
│   │   ├── layoutPatterns.js     ← Layout algorithms
│   │   ├── furniturePlacement.js ← NEW: Auto furniture layout
│   │   └── plotGeometry.js       ← NEW: Irregular plot support
│   │
│   ├── services/
│   │   └── aiService.js       ← Gemini AI integration
│   │
│   ├── 📊 data/
│   │   └── floorPlanTemplates.js ← Reference templates
│   │
│   └── 🎨 styles/
│       ├── main.css           ← Global styles
│       └── components.css     ← Component styles
│
├── 📁 dataset/
│   └── architectural_data.json   ← ML training data
│
├── 📁 ml_engine/
│   ├── server.py              ← Flask ML server (optional)
│   ├── train.py               ← ML model training
│   └── generate_architectural_data.py
│
└── 📁 models/
    └── (ML model files)
```

---

## Quick Navigation

### For End Users:
1. **New to NakshaNirman?** → [README.md](README.md)
2. **How do I use it?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Usage Examples
3. **What room types are available?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Room Types
4. **How do I customize?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Customization
5. **Something's not working** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Troubleshooting

### For Developers:
1. **What changed in v2.0?** → [CHANGELOG.md](CHANGELOG.md)
2. **How is furniture placement implemented?** → [IMPLEMENTATION.md](IMPLEMENTATION.md) § Furniture Placement Engine
3. **How does irregular plot support work?** → [IMPLEMENTATION.md](IMPLEMENTATION.md) § Plot Geometry Utilities
4. **What are the DXF layers?** → [IMPLEMENTATION.md](IMPLEMENTATION.md) § CAD Agent Enhanced
5. **How to extend the system?** → [IMPLEMENTATION.md](IMPLEMENTATION.md) § Design Principles

### For Architects:
1. **What design standards are used?** → [README.md](README.md) § Design Standards
2. **How is privacy hierarchy enforced?** → [README.md](README.md) § Privacy Hierarchy
3. **What furniture is auto-placed?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Furniture Placement Examples
4. **Can I modify room dimensions?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Customization
5. **How accurate are the dimensions?** → [IMPLEMENTATION.md](IMPLEMENTATION.md) § Dimension Annotation System

---

## Search by Topic

### Features

| Topic | File | Section |
|-------|------|---------|
| Furniture Placement | [IMPLEMENTATION.md](IMPLEMENTATION.md) | § 3. Furniture Placement Engine |
| Irregular Plots | [IMPLEMENTATION.md](IMPLEMENTATION.md) | § 4. Plot Geometry Utilities |
| DXF Layers | [README.md](README.md) | § DXF Export Layers |
| Privacy Hierarchy | [README.md](README.md) | § Privacy Hierarchy |
| Room Types | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | § Room Types Available |
| Sanitary Fixtures | [IMPLEMENTATION.md](IMPLEMENTATION.md) | § 3. Sanitary Fixtures |
| Dimensions | [IMPLEMENTATION.md](IMPLEMENTATION.md) | § 6. Dimension Annotations |
| Windows | [IMPLEMENTATION.md](IMPLEMENTATION.md) | § 7. Window Generation |

### Technical

| Topic | File | Section |
|-------|------|---------|
| Architecture | [README.md](README.md) | § Architecture |
| AI Service | [IMPLEMENTATION.md](IMPLEMENTATION.md) | § 2. AI Service Master Prompt |
| Constants | [IMPLEMENTATION.md](IMPLEMENTATION.md) | § 1. Constants & Standards |
| Validation | [README.md](README.md) | § Validation Rules |
| Algorithms | [IMPLEMENTATION.md](IMPLEMENTATION.md) | § Design Principles |

### Usage

| Topic | File | Section |
|-------|------|---------|
| Quick Start | [README.md](README.md) | § Quick Start |
| Examples | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | § Example Conversations |
| Customization | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | § Customization |
| Troubleshooting | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | § Troubleshooting |
| Best Practices | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | § Best Practices |

---

## Design Standards Reference

**Quick Lookup:**

| Parameter | Value | File |
|-----------|-------|------|
| External Wall | 230mm | [constants.js](src/utils/constants.js) |
| Internal Wall | 115mm | [constants.js](src/utils/constants.js) |
| Corridor Width | 900mm min | [constants.js](src/utils/constants.js) |
| Door Width | 800mm min | [constants.js](src/utils/constants.js) |
| Window Height | 1200mm | [constants.js](src/utils/constants.js) |
| Window Sill | 750mm | [constants.js](src/utils/constants.js) |
| Stair Tread | 250mm | [constants.js](src/utils/constants.js) |
| Stair Riser | 175mm | [constants.js](src/utils/constants.js) |

---

## Code Examples

### 1. Generate Furniture for a Room
```javascript
import { generateFurnitureLayout } from './utils/furniturePlacement.js';

const room = { type: 'living_room', x: 0, y: 0, width: 5000, height: 4000 };
const furniture = generateFurnitureLayout(room);
// Returns: [sofa, coffeeTable, tvUnit]
```

**Reference:** [furniturePlacement.js](src/utils/furniturePlacement.js)

### 2. Validate Irregular Plot
```javascript
import { validateBoundary, calculatePolygonArea } from './utils/plotGeometry.js';

const boundary = [[0,0], [10000,0], [10000,8000], [5000,8000], [5000,12000], [0,12000]];
const validation = validateBoundary(boundary);
const area = calculatePolygonArea(boundary); // in mm²
```

**Reference:** [plotGeometry.js](src/utils/plotGeometry.js)

### 3. Generate DXF with Layers
```javascript
import { generateDXF } from './agents/cadAgent.js';

const geometryData = {
  boundary: { width: 10000, height: 12000 },
  rooms: [...],
  furniture: [...]
};

const dxfContent = generateDXF(geometryData);
// Returns: DXF string with 9 layers
```

**Reference:** [cadAgent.js](src/agents/cadAgent.js)

---

## Learning Path

### Beginner (End User)
1. Read [README.md](README.md) - Overview
2. Follow [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Basic Usage
3. Try example conversations
4. Download and review DXF in AutoCAD

### Intermediate (Power User)
1. Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Advanced Features
2. Experiment with irregular plots
3. Customize room sizes in [constants.js](src/utils/constants.js)
4. Review [CHANGELOG.md](CHANGELOG.md) for feature details

### Advanced (Developer)
1. Read [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical Details
2. Study [furniturePlacement.js](src/utils/furniturePlacement.js)
3. Study [plotGeometry.js](src/utils/plotGeometry.js)
4. Modify agents ([planningAgent.js](src/agents/planningAgent.js), [geometryAgent.js](src/agents/geometryAgent.js))
5. Extend furniture library or add new room types

---

## Contributing

To contribute to NakshaNirman:

1. **Understand the architecture** → [README.md](README.md) § Architecture
2. **Review implementation details** → [IMPLEMENTATION.md](IMPLEMENTATION.md)
3. **Follow design principles** → [IMPLEMENTATION.md](IMPLEMENTATION.md) § Design Principles
4. **Test with example cases** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Examples
5. **Document your changes** → Update relevant docs

---

## Support

| Question Type | Resource |
|--------------|----------|
| "How do I...?" | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| "What changed?" | [CHANGELOG.md](CHANGELOG.md) |
| "Why doesn't...?" | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Troubleshooting |
| "How does it work?" | [IMPLEMENTATION.md](IMPLEMENTATION.md) |
| "Can I customize...?" | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Customization |

---

## 📊 Statistics

**Documentation Coverage:**
- Total Documentation: 5 files, 1,500+ lines
- Code Coverage: All agents and utilities documented
- Example Count: 15+ usage examples
- Reference Tables: 10+ quick lookup tables

**Version 2.0.0:**
- Features: 25+ new capabilities
- Room Types: 13
- Furniture Items: 15
- Sanitary Fixtures: 6
- DXF Layers: 9
- Design Standards: Indian Residential

---

## Next Steps

**For New Users:**
1. Install: `npm install`
2. Run: `npm run dev`
3. Read: [README.md](README.md)
4. Try: Examples from [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**For Developers:**
1. Review: [IMPLEMENTATION.md](IMPLEMENTATION.md)
2. Explore: [furniturePlacement.js](src/utils/furniturePlacement.js) & [plotGeometry.js](src/utils/plotGeometry.js)
3. Extend: Add custom furniture or room types
4. Test: Generate DXF and verify in AutoCAD

---

**Last Updated:** February 3, 2026  
**Version:** 2.0.0  
**Status:** Production Ready

---

*NakshaNirman - Professional AutoCAD-Level CAD Floor Plan Generator*  
*Built with AI-powered intelligence for the next generation of residential design automation*
