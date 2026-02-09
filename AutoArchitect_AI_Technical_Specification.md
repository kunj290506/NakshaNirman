# AutoArchitect AI
## Technical Specification Document v1.0

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Frontend Specifications](#3-frontend-specifications)
4. [Backend Specifications](#4-backend-specifications)
5. [API Reference](#5-api-reference)
6. [Animation Pipeline](#6-animation-pipeline)
7. [Performance Requirements](#7-performance-requirements)
8. [Development Roadmap](#8-development-roadmap)
9. [Quality Standards](#9-quality-standards)
10. [Deliverables](#10-deliverables)

---

# 1. Executive Summary

**Project Name:** AutoArchitect AI

**Mission Statement:** Transform user-uploaded plot boundaries into complete home designs with professional 3D animations and drone-style fly-throughs.

**Core Value Proposition:**
- AI-driven architectural planning
- Automated CAD file generation
- Cinematic 3D visualization pipeline
- Real-time interactive design modifications

---

# 2. System Architecture

## 2.1 High-Level Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   USER UPLOAD   │───▶│  AI PROCESSING  │───▶│  OUTPUT FILES   │
│  (Image/DXF)    │    │    PIPELINE     │    │  (DXF/3D/MP4)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                       │
        ▼                      ▼                       ▼
   File Upload          Design Generation       Results Display
   Validation           CAD Creation            3D Viewer
   Processing           3D Rendering            Animation Player
```

## 2.2 Technology Stack

### Frontend Technologies
| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | React.js + TypeScript | UI Development |
| State | Redux Toolkit / Zustand | State Management |
| UI Library | Material-UI / Ant Design | Components |
| 3D Engine | Three.js | WebGL Rendering |
| Canvas | Fabric.js / Konva.js | Plot Editor |
| Forms | React Hook Form | Validation |

### Backend Technologies
| Component | Technology | Purpose |
|-----------|------------|---------|
| API | Python FastAPI | Async Server |
| Server | Uvicorn + Gunicorn | Production Server |
| Image | OpenCV | Boundary Detection |
| CAD | ezdxf | DXF Processing |
| 3D | Blender Python API | Model Generation |
| AI | Llama 3.1 / Code Llama | Design Generation |
| Queue | Celery + Redis | Job Processing |
| Database | PostgreSQL | Data Storage |

---

# 3. Frontend Specifications

## 3.1 Landing Page

**Components:**
- Hero section with demo video (autoplay, muted)
- Feature showcase grid (Modern, Traditional, Minimalist)
- User testimonial carousel (auto-rotate 5s)
- Primary CTA button → Upload Dashboard

**Design Requirements:**
- Full-width hero (100vh)
- Smooth scroll animations
- Mobile-responsive breakpoints

## 3.2 Upload & Configuration Dashboard

### File Upload Module
- Drag-and-drop zone (minimum 100x100px)
- Supported formats: JPG, PNG, DXF, DWG
- Real-time file validation with error messages
- Preview thumbnail (auto-generated)
- Progress indicator for large files

### Design Requirements Form

| Field | Type | Options |
|-------|------|---------|
| Bedrooms | Slider | 1-6 |
| Bathrooms | Slider | 1-4 |
| Style | Dropdown | Modern, Traditional, Contemporary, Minimalist, Mediterranean |
| Features | Checkbox | Home Office, Garage, Garden, Pool, Solar Panels |
| Budget | Radio | Economy, Standard, Premium |
| Notes | Textarea | Free text (500 char max) |

### Interactive Plot Editor
- Canvas-based boundary editing
- Vertex manipulation (drag to move)
- Add/remove boundary points
- Rotation handle for orientation
- Real-time area calculation (sqm/sqft)
- North direction indicator

## 3.3 Processing Monitor

**5-Stage Progress Display:**
1. 📁 File Processing (0-20%)
2. 🔍 Boundary Analysis (20-40%)
3. 🧠 AI Design Generation (40-60%)
4. 📐 CAD Generation (60-80%)
5. 🎬 3D/Animation Rendering (80-100%)

**Features:**
- Real-time status messages
- Estimated time remaining
- Cancel/restart functionality
- Background processing notification

## 3.4 Results Interface

### 2D Plan Viewer
- Canvas-based DXF rendering
- Layer toggles: Walls, Doors, Windows, Dimensions
- Click-to-measure distance tool
- Export: Print, PDF, PNG, DXF

### 3D Interactive Model
- Three.js WebGL engine
- Controls: Orbit, Pan, Zoom
- Material switcher (brick, wood, glass)
- Day/Night lighting toggle
- Furniture visibility toggle
- VR-ready mode

### Animation Player
- Standard video controls
- Playback speed (0.5x - 2x)
- Quality: 480p, 720p, 1080p
- Download: MP4, GIF, Share link

---

# 4. Backend Specifications

## 4.1 File Processing Engine

### Image Processing Pipeline
```python
def process_image(file):
    # 1. Load image with OpenCV
    image = cv2.imread(file)
    
    # 2. Edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # 3. Contour finding
    contours = cv2.findContours(edges)
    
    # 4. Polygon simplification (Douglas-Peucker)
    boundary = cv2.approxPolyDP(contour, epsilon)
    
    # 5. Scale estimation
    scale = estimate_scale_from_reference(image)
    
    # 6. North direction detection
    north = detect_north_from_shadows(image)
    
    return BoundaryData(boundary, scale, north)
```

### DXF Processing Pipeline
```python
def process_dxf(file):
    # 1. Parse with ezdxf
    doc = ezdxf.readfile(file)
    
    # 2. Extract boundary layer
    boundary_layer = doc.layers.get('BOUNDARY')
    
    # 3. Get closed polylines
    polylines = msp.query('LWPOLYLINE')
    
    # 4. Unit conversion (mm → meters)
    boundary = convert_units(polylines)
    
    # 5. Validate closed polygon
    validate_closed(boundary)
    
    return BoundaryData(boundary)
```

## 4.2 AI Design Generation System

### Multi-Stage LLM Pipeline

**Stage 1: Layout Generator (Llama 3.1 70B)**
- Input: Boundary polygon + user requirements
- Output: Room adjacency graph with sizes
- Validation: Building code compliance

**Stage 2: Detail Refiner (Code Llama 34B)**
- Input: Adjacency graph
- Output: Precise dimensions, door/window placements
- Validation: Structural feasibility

**Stage 3: Style Applicator (Fine-tuned Mistral)**
- Input: Detailed layout + style preference
- Output: Material selections, aesthetic features
- Validation: Style consistency

### Design Validation Rules
- Minimum room sizes (building code)
- Circulation path requirements
- Window-to-wall ratio (15-20%)
- Door clearances (min 900mm)
- Staircase dimensions
- Emergency egress paths

## 4.3 CAD Generation Pipeline

### Layer Structure
| Layer Name | Color | Purpose |
|------------|-------|---------|
| A-WALL | White | Walls |
| A-WALL-FIRE | Red | Fire walls |
| A-DOOR | Cyan | Doors |
| A-GLAZ | Blue | Windows |
| A-DIM | Green | Dimensions |
| A-ANNO | Yellow | Annotations |

### Geometry Specifications
- Wall thickness: 200mm (default)
- Door widths: 900mm (standard), 1200mm (double)
- Window sill height: 900mm
- Floor-to-ceiling: 2700mm

## 4.4 3D & Animation Engine

### Blender Automation
```python
# Wall extrusion
def extrude_walls(boundary, height=2.7):
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, height)}
    )

# Camera path (Bezier curve)
def create_camera_path():
    curve = bpy.data.curves.new('CameraPath', 'CURVE')
    curve.dimensions = '3D'
    spline = curve.splines.new('BEZIER')
    # Add keyframes for drone-style flight
```

### Render Settings
- Resolution: 1920x1080 (1080p)
- Frame rate: 60 FPS
- Samples: 128 (production)
- Format: MP4 (H.264)

---

# 5. API Reference

## Endpoints

### POST /api/upload
Upload boundary file (image or DXF)

**Request:**
```json
{
  "file": "binary",
  "metadata": {
    "name": "string",
    "type": "image/png | application/dxf"
  }
}
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "estimated_time": 300
}
```

### POST /api/design/generate
Trigger AI design generation

**Request:**
```json
{
  "job_id": "uuid",
  "requirements": {
    "bedrooms": 3,
    "bathrooms": 2,
    "style": "modern",
    "features": ["garage", "garden"],
    "budget": "standard"
  }
}
```

### GET /api/job/{id}/status
Check processing status

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "stage": "ai_design",
  "progress": 45,
  "message": "Generating floor plan layout"
}
```

### GET /api/results/{id}
Retrieve all generated files

**Response:**
```json
{
  "dxf_url": "https://...",
  "pdf_url": "https://...",
  "gltf_url": "https://...",
  "video_url": "https://...",
  "specifications": {...}
}
```

---

# 6. Animation Pipeline

## Camera Sequence (35 seconds total)

| Time | Shot | Camera Movement |
|------|------|-----------------|
| 0-5s | Establishing | High aerial approach, slow zoom |
| 5-10s | Exterior Front | Low fly-by along façade |
| 10-15s | Exterior Corner | Corner sweep with motion blur |
| 15-18s | Entry | Push through front door |
| 18-22s | Living Room | 360° panoramic sweep |
| 22-25s | Kitchen | Tracking shot to dining |
| 25-28s | Bedrooms | Quick fly-through |
| 28-32s | Exit | Pull back through window |
| 32-35s | Finale | Ascending aerial retreat |

## Visual Effects

- **Depth of Field**: f/2.8 simulation, focus transitions
- **Motion Blur**: 180° shutter angle
- **Lens Flares**: Anamorphic style through windows
- **Volumetric Lighting**: God rays, interior light beams
- **Color Grading**: Cinematic LUT application

## Interactive Controls
- Playback speed: 0.5x, 1x, 1.5x, 2x
- Camera angle presets
- Click-to-focus points
- Real-time material swap
- Lighting adjustment
- Audio toggle (ambient/music)

---

# 7. Performance Requirements

## Frontend Targets

| Metric | Target | Maximum |
|--------|--------|---------|
| First Contentful Paint | 1.5s | 2.0s |
| Time to Interactive | 3.0s | 4.0s |
| 3D Model Load | 5.0s | 8.0s |
| Animation Start | 2.0s | 3.0s |
| Input Response | 100ms | 150ms |

## Backend Targets

| Process | Target | Maximum |
|---------|--------|---------|
| Boundary Processing | 10s | 15s |
| AI Design Generation | 60s | 90s |
| CAD File Creation | 20s | 30s |
| 3D Model Generation | 90s | 120s |
| Animation Render | 300s | 450s |

## Optimization Strategies

**Frontend:**
- Lazy loading for all components
- Web Workers for heavy processing
- Progressive image loading
- Service workers for caching
- GLTF compression for 3D
- WebP for images
- Code splitting

**Backend:**
- Job queue (Celery + Redis)
- Result caching
- Connection pooling
- Parallel rendering (GPU farm)
- Incremental DXF generation

---

# 8. Development Roadmap

## Phase Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| 0 | 2-3 days | Environment setup |
| 1 | 1-2 weeks | Core infrastructure |
| 2 | 1 week | Boundary processing |
| 3 | 2-3 weeks | AI design generation |
| 4 | 1 week | CAD generation |
| 5 | 2-3 weeks | 3D & animation |
| 6 | 1-2 weeks | Frontend integration |
| 7 | 1 week | Performance optimization |
| 8 | 1-2 weeks | Testing & refinement |
| 9 | Ongoing | Advanced features |

**Total Estimated: 10-14 weeks**

---

# 9. Quality Standards

## Design Quality Checklist
- [ ] All rooms meet minimum size requirements
- [ ] Circulation paths are logical
- [ ] Window-to-wall ratio optimized (15-20%)
- [ ] Architectural style consistency
- [ ] Building code compliance verified

## Technical Quality Checklist
- [ ] DXF files open in AutoCAD/LibreCAD
- [ ] 3D models are watertight
- [ ] Animation smooth (60fps)
- [ ] File sizes optimized
- [ ] All interactive elements respond < 100ms

## UX Quality Checklist
- [ ] Upload process intuitive
- [ ] Progress feedback clear
- [ ] Results presentation professional
- [ ] Mobile responsive
- [ ] WCAG 2.1 compliant

---

# 10. Deliverables

## User Deliverables

**Technical Drawing Package:**
- DXF file (AutoCAD compatible)
- PDF version (print-ready)
- PNG render (web sharing)
- Layer separation guide

**3D Visualization Package:**
- Cinematic MP4 video (1080p)
- Interactive GLTF model
- VR/AR compatible version
- 360° panorama images

**Design Documentation:**
- Room specifications table
- Material schedule
- Area calculations
- Construction timeline estimate

**Sharing Tools:**
- Unique project URL
- Website embed code
- Social media preview images
- Downloadable presentation

---

# Appendix A: Data Models

```python
class Project:
    id: UUID
    boundary_data: GeoJSON
    user_requirements: Dict
    generated_design: Dict
    cad_files: List[URL]
    animation: URL
    status: Literal["processing", "completed", "failed"]
    created_at: datetime
    processing_time: int  # seconds

class Room:
    name: str
    area: float  # sqm
    dimensions: Tuple[float, float]
    doors: List[Door]
    windows: List[Window]
    
class Door:
    width: float
    position: Point
    type: Literal["single", "double", "sliding"]
    
class Window:
    width: float
    height: float
    position: Point
    sill_height: float
```

---

*End of Document*

**Version:** 1.0  
**Date:** February 9, 2026  
**Project:** AutoArchitect AI
