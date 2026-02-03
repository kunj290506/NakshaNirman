# Quick Reference Guide - Master Prompt Features

## Master Prompt Capabilities

The CAD application now generates **professional AutoCAD-level house maps** with complete construction-ready details.

---

## How to Use Enhanced Features

### 1. **Basic House Design**
```
User: "I want a 1200 sqft house with 3 bedrooms, 2 bathrooms, kitchen, and living room"

AI Response:
- Analyzes feasibility (1200 sqft = ~111 sqm)
- Suggests room layout with privacy zoning
- Generates furniture for all rooms
- Adds sanitary fixtures to bathrooms
- Creates dimensioned DXF with 9 layers
```

### 2. **Irregular Plot**
```
User: "I have an L-shaped plot, 40 feet on one side, 30 feet on the other"

AI Response:
- Detects L-shaped geometry
- Calculates usable area
- Places rooms within boundary constraints
- Validates all rooms fit in irregular shape
```

### 3. **Shop Integration**
```
User: "Add a shop on the ground floor with street access"

AI Response:
- Places shop in PUBLIC zone (front)
- Marks requiresStreetAccess: true
- Ensures shop has window facing street
- Separates shop from private residential areas
```

### 4. **Privacy-Based Layout**
```
User: "Design with bedrooms away from the entrance"

AI Response:
- Applies privacy hierarchy:
  - PUBLIC zone (front): Living, dining, shop
  - SEMI-PRIVATE zone (middle): Kitchen, study, pooja
  - PRIVATE zone (back): Bedrooms, bathrooms
- Ensures clear zoning separation
```

### 5. **Kitchen Customization**
```
User: "I want an L-shaped kitchen on the north side"

AI Response:
- Places kitchen in north quadrant
- Generates L-shaped platform layout
- Adds sink, stove, refrigerator
- Ensures adjacency to dining room
```

### 6. **Duplex/G+1 Structure**
```
User: "Can this fit in single floor or need duplex?"

AI Response:
- Calculates density index
- Compares program vs available area
- Suggests "duplex" if tight
- Provides staircase dimensions (900mm width, 250mm tread)
```

---

## Room Types Available

### Public Zone
- **living_room**: Sofa, TV unit, coffee table
- **dining_room**: Dining table (4 or 6 seater), chairs
- **shop**: Counter, display area, street access
- **parking**: Car space (14 sqm min)

### Semi-Private Zone
- **kitchen**: L-shaped or linear platform, sink, stove, fridge
- **study**: Study table, bookshelf
- **pooja_room**: Compact prayer space (4 sqm min)
- **utility**: Washing, storage
- **balcony**: Outdoor space

### Private Zone
- **master_bedroom**: King/Queen bed, wardrobes, side tables
- **bedroom**: Double bed, wardrobe, side table
- **bathroom**: WC (Indian/Western), wash basin, shower
- **storage**: Storage cabinets

### Circulation
- **staircase**: Standard flight (900mm width, 250mm tread, 175mm riser)

---

## Design Standards (Auto-Applied)

| Parameter | Value | Notes |
|-----------|-------|-------|
| External Wall | 230mm | 9 inches (brick wall) |
| Internal Wall | 115mm | 4.5 inches (partition) |
| Corridor Width | 900mm min | Standard circulation |
| Door Width | 800mm min | Standard door |
| Window Height | 1200mm | Standard window |
| Window Sill | 750mm | From floor level |
| Units | Millimeters (mm) | Consistent throughout |

---

## DXF Export Layers

When you download the DXF file, open it in AutoCAD and you'll see **9 separate layers**:

| Layer | Color | Contains |
|-------|-------|----------|
| **WALLS** | White | External (230mm) & internal (115mm) walls |
| **DOORS** | Cyan | Door openings with swing arcs |
| **WINDOWS** | Blue | Windows with panes |
| **FURNITURE** | Green | Sofas, beds, tables, wardrobes |
| **SANITARY** | Magenta | WC, wash basin, shower, bathtub |
| **ROOMS** | Green | Room outlines |
| **TEXT** | Yellow | Room labels and area annotations |
| **DIMENSIONS** | Red | Dimension lines and measurements |
| **ANNOTATIONS** | Cyan | Additional notes |

### How to Use in AutoCAD:
1. Open DXF file in AutoCAD
2. Type `LAYER` or click Layers panel
3. Toggle visibility of layers
4. Freeze FURNITURE layer to see only walls
5. Freeze DIMENSIONS layer for cleaner view
6. Print with specific layers for different purposes

---

## Furniture Placement Examples

### Living Room (18 sqm)
```
- 3-seater sofa (2100×900mm) on longest wall
- Coffee table (1200×600mm) in front of sofa
- TV unit (1800×450mm) on opposite wall
- Clearance: 600mm from walls
```

### Master Bedroom (16 sqm)
```
- King bed (2100×2100mm) centered on back wall
- Side tables (450×450mm) on both sides
- Wardrobe (1800×600mm) on side wall
- Clearance: 600mm circulation space
```

### Kitchen (10 sqm)
```
- L-shaped platform (600mm depth)
- Sink (900×450mm) in corner
- Stove (600×600mm) on main platform
- Refrigerator (600×600mm) opposite corner
- Clearance: 100mm from walls
```

### Bathroom (5 sqm)
```
- Western WC (500×700mm) in back corner
- Wash basin (500×400mm) near entrance
- Shower area (900×900mm) on right side
- Clearance: 150mm from walls
```

---

## Validation Checks (Automatic)

The system automatically validates:

- **No Overlaps**: Rooms don't overlap with each other
- **Boundary Fit**: All rooms fit within plot boundary
- **Minimum Sizes**: Rooms meet minimum area requirements
- **Aspect Ratio**: Rooms maintain 1:2 max ratio
- **Corridor Width**: Minimum 900mm for circulation
- **Furniture Fit**: All furniture fits within room boundaries
- **Adjacency**: Kitchen near dining, bathrooms near bedrooms
- **Privacy Zoning**: Public, semi-private, private separation

If any validation fails, the AI will:
1. Report the specific constraint violation
2. Suggest alternative configuration
3. Ask for clarification or adjustment

---

## Example Conversations

### Complete House Design
```
User: "Design a 3BHK house on 30×40 feet plot"

AI: 
PHASE 1: SITE ANALYSIS
- Plot: 30×40 ft = 9.14×12.19m = 1200 sqft rectangular
- Boundary constraints: Regular rectangle, no irregularities

PHASE 2: FEASIBILITY
- 3BHK program: 3 bed + living + dining + kitchen + 2 bath = ~900 sqft
- Feasible in single floor with 25% circulation

PHASE 3: ZONING
- Front: Living room (18 sqm) with sofa, TV unit
- Middle: Dining (12 sqm) adjacent to kitchen (10 sqm)
- Back: 3 bedrooms (12 sqm each) with 2 bathrooms (5 sqm each)
- Privacy: Public → Semi-Private → Private progression

PHASE 4: SPECIFICATIONS
- External walls: 230mm
- Internal partitions: 115mm
- Doors: 800mm, Windows: 900mm on external walls
- Furniture: Full layout with collision-free placement

Ready to generate? (yes/no)
```

### Modification Request
```
User: "Move the kitchen to the north side"

AI:
- Repositioning kitchen to north quadrant
- Maintaining adjacency with dining room
- Adjusting circulation paths
- Regenerating furniture layout

Updated plan shows kitchen on north side with L-shaped platform, 
maintaining all other room positions and privacy zoning.
```

### Adding Special Room
```
User: "Add a pooja room near the living room"

AI:
- Adding pooja_room (4 sqm minimum)
- Placing in SEMI-PRIVATE zone
- Ensuring adjacency to living_room as requested
- No furniture generated (as per tradition, minimal furnishing)

Pooja room added adjacent to living room entrance.
```

---

## Customization

### Modify Default Room Sizes
Edit [constants.js](src/utils/constants.js):
```javascript
export const ROOM_TYPES = {
    bedroom: {
        label: 'Bedroom',
        defaultAreaSqm: 12,  // Change this
        minAreaSqm: 9,       // Change this
        // ...
    }
};
```

### Add New Furniture
Edit [furniturePlacement.js](src/utils/furniturePlacement.js):
```javascript
export const FURNITURE = {
    MY_CUSTOM_ITEM: { 
        width: 1000, 
        depth: 500, 
        label: 'Custom Item' 
    }
};
```

### Modify Wall Thickness
Edit [constants.js](src/utils/constants.js):
```javascript
export const WALL_THICKNESS = 230;  // Change to 300 for thicker walls
```

---

## Troubleshooting

### "Total room area exceeds usable area"
**Solution**: Reduce room sizes or reduce number of rooms or increase plot size.

### "Room overlaps detected"
**Solution**: AI will automatically retry placement. If persistent, reduce room count.

### "Kitchen should be adjacent to dining room"
**Solution**: This is a warning, not error. Design still generates but may not be optimal.

### "Plot shape not recognized"
**Solution**: For irregular plots, provide explicit boundary coordinates or use DXF upload.

---

## Performance Tips

1. **Start Simple**: Begin with basic requirements, then refine
2. **Use Natural Language**: "I want 3 bedrooms" works better than "bedroom qty=3"
3. **Iterate**: Generate, review, adjust, regenerate
4. **Check Dimensions**: Download DXF and verify in AutoCAD
5. **Review Furniture**: Toggle FURNITURE layer to see placement

---

## Best Practices

### For Best Results:
- Provide total area or plot dimensions
- Specify number of bedrooms explicitly
- Mention special requirements (shop, staircase) upfront
- Review adjacency warnings and adjust if needed
- Download DXF and verify before construction

### Avoid:
✗ Overly tight constraints (rooms won't fit)
✗ Conflicting requirements (small plot, many rooms)
✗ Vague descriptions ("make it nice")
✗ Missing critical info (no plot size, no room count)

---

## Getting Help

1. Type **"help"** in chat for basic guidance
2. Review [README.md](README.md) for full documentation
3. Check [IMPLEMENTATION.md](IMPLEMENTATION.md) for technical details
4. Examine AI thought process in chat for reasoning

---

**Happy Designing!**
