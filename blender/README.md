# AutoArchitect AI - Blender Scripts

This directory contains Python scripts for Blender automation.

## Scripts

### generate_3d.py
Main script for 3D model generation and animation rendering.

**Usage:**
```bash
blender --background --python generate_3d.py -- design.json output_dir/
```

**Features:**
- Creates floor, walls, and ceiling from design JSON
- Applies PBR materials
- Sets up sun and area lighting
- Creates drone-style camera path
- Animates camera along bezier curve
- Exports GLTF for web viewer
- Renders MP4 animation

## Requirements

- Blender 3.6+
- Cycles render engine
- GPU (CUDA) recommended for faster rendering

## Output Files

- `model.gltf` - Interactive 3D model for web
- `animation.mp4` - Rendered fly-through video (1080p, 60fps)
