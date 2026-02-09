"""
Blender Automation Scripts for AutoArchitect AI
================================================
Python scripts for automated 3D model generation and animation.
"""

import bpy
import json
import os
import sys
import math
from mathutils import Vector


def clear_scene():
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def setup_scene():
    """Setup scene with proper settings."""
    scene = bpy.context.scene
    
    # Set units to meters
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    
    # Render settings
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = 60
    scene.cycles.samples = 128
    
    # Enable GPU if available
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True
        scene.cycles.device = 'GPU'
    except:
        pass


def create_floor(boundary, height=0):
    """Create floor from boundary points."""
    verts = [(p[0], p[1], height) for p in boundary]
    faces = [list(range(len(verts)))]
    
    mesh = bpy.data.meshes.new('Floor')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    obj = bpy.data.objects.new('Floor', mesh)
    bpy.context.collection.objects.link(obj)
    
    # Add floor material
    mat = bpy.data.materials.new('FloorMaterial')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.3
    obj.data.materials.append(mat)
    
    return obj


def create_walls(rooms, wall_height=2.7, wall_thickness=0.2):
    """Create walls for all rooms."""
    for room in rooms:
        x = room['x']
        y = room['y']
        w = room['width']
        h = room['height']
        
        # Create wall geometry
        create_room_walls(room['name'], x, y, w, h, wall_height, wall_thickness)


def create_room_walls(name, x, y, w, h, height, thickness):
    """Create walls for a single room."""
    t = thickness / 2
    
    # Four walls as separate objects
    walls = [
        ('South', [(x-t, y-t, 0), (x+w+t, y-t, 0), (x+w+t, y+t, 0), (x-t, y+t, 0)]),
        ('North', [(x-t, y+h-t, 0), (x+w+t, y+h-t, 0), (x+w+t, y+h+t, 0), (x-t, y+h+t, 0)]),
        ('West', [(x-t, y-t, 0), (x+t, y-t, 0), (x+t, y+h+t, 0), (x-t, y+h+t, 0)]),
        ('East', [(x+w-t, y-t, 0), (x+w+t, y-t, 0), (x+w+t, y+h+t, 0), (x+w-t, y+h+t, 0)])
    ]
    
    for wall_name, base_verts in walls:
        # Create top verts
        top_verts = [(v[0], v[1], height) for v in base_verts]
        all_verts = base_verts + top_verts
        
        # Define faces
        faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Front
            [2, 3, 7, 6],  # Back
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5]   # Right
        ]
        
        mesh = bpy.data.meshes.new(f'{name}_{wall_name}')
        mesh.from_pydata(all_verts, [], faces)
        mesh.update()
        
        obj = bpy.data.objects.new(f'{name}_{wall_name}', mesh)
        bpy.context.collection.objects.link(obj)
        
        # Add wall material
        add_wall_material(obj)


def add_wall_material(obj, color=(0.95, 0.95, 0.95, 1.0)):
    """Add material to wall object."""
    mat = bpy.data.materials.new('WallMaterial')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.5
    obj.data.materials.append(mat)


def create_ceiling(boundary, height=2.7):
    """Create ceiling."""
    verts = [(p[0], p[1], height) for p in boundary]
    faces = [list(range(len(verts)))]
    
    mesh = bpy.data.meshes.new('Ceiling')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    obj = bpy.data.objects.new('Ceiling', mesh)
    bpy.context.collection.objects.link(obj)
    
    # White ceiling material
    mat = bpy.data.materials.new('CeilingMaterial')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    obj.data.materials.append(mat)
    
    return obj


def setup_lighting():
    """Setup scene lighting with HDRI and area lights."""
    # Sun lamp
    sun_data = bpy.data.lights.new('Sun', 'SUN')
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new('Sun', sun_data)
    sun_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    bpy.context.collection.objects.link(sun_obj)
    
    # Area light for interior
    area_data = bpy.data.lights.new('AreaLight', 'AREA')
    area_data.energy = 500
    area_data.size = 2.0
    area_obj = bpy.data.objects.new('AreaLight', area_data)
    area_obj.location = (6, 5, 2.5)
    bpy.context.collection.objects.link(area_obj)


def create_camera():
    """Create and setup camera."""
    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = 24  # Wide angle for architecture
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    cam_obj.location = (15, -15, 10)
    cam_obj.rotation_euler = (math.radians(60), 0, math.radians(45))
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def create_camera_path(boundary, height=5.0, flight_height=8.0):
    """Create bezier curve camera path for drone animation."""
    # Calculate center
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    
    # Create path points (drone-style fly-around)
    radius = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.8
    
    curve_data = bpy.data.curves.new('CameraPath', 'CURVE')
    curve_data.dimensions = '3D'
    
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(7)  # 8 total points for circular path
    
    for i, point in enumerate(spline.bezier_points):
        angle = (i / 8) * 2 * math.pi
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        z = flight_height + 2 * math.sin(angle * 2)  # Vary height
        point.co = (x, y, z)
        point.handle_type = 'AUTO'
    
    spline.use_cyclic_u = True
    
    curve_obj = bpy.data.objects.new('CameraPath', curve_data)
    bpy.context.collection.objects.link(curve_obj)
    
    return curve_obj


def animate_camera(camera, path, duration=350):
    """Animate camera along path."""
    # Add follow path constraint
    constraint = camera.constraints.new('FOLLOW_PATH')
    constraint.target = path
    constraint.use_curve_follow = True
    constraint.forward_axis = 'FORWARD_Y'
    constraint.up_axis = 'UP_Z'
    
    # Animate offset
    camera.animation_data_create()
    action = bpy.data.actions.new('CameraAnimation')
    camera.animation_data.action = action
    
    # Keyframe the path offset
    path.data.path_duration = duration
    path.data.use_path = True
    
    path.data.eval_time = 0
    path.data.keyframe_insert('eval_time', frame=1)
    path.data.eval_time = duration
    path.data.keyframe_insert('eval_time', frame=duration)


def add_track_to_target(camera, target_location):
    """Make camera look at a target."""
    # Create empty target
    empty = bpy.data.objects.new('CameraTarget', None)
    empty.location = target_location
    bpy.context.collection.objects.link(empty)
    
    # Track constraint
    track = camera.constraints.new('TRACK_TO')
    track.target = empty
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'


def render_animation(output_path, start_frame=1, end_frame=350):
    """Render animation to video."""
    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    
    # Output settings
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.filepath = output_path
    
    # Render
    bpy.ops.render.render(animation=True)


def export_gltf(output_path):
    """Export scene as GLTF for web viewer."""
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLTF_SEPARATE')


def main(design_file, output_dir):
    """Main function to generate 3D model and animation."""
    # Load design
    with open(design_file, 'r') as f:
        design = json.load(f)
    
    # Setup
    clear_scene()
    setup_scene()
    
    # Create geometry
    boundary = design.get('boundary', [])
    rooms = design.get('rooms', [])
    
    create_floor(boundary)
    create_walls(rooms)
    create_ceiling(boundary)
    
    # Lighting
    setup_lighting()
    
    # Camera and animation
    camera = create_camera()
    path = create_camera_path(boundary)
    
    # Calculate center for tracking
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 1.5)
    add_track_to_target(camera, center)
    
    animate_camera(camera, path)
    
    # Export GLTF for web
    gltf_path = os.path.join(output_dir, 'model.gltf')
    export_gltf(gltf_path)
    
    # Render animation
    video_path = os.path.join(output_dir, 'animation.mp4')
    render_animation(video_path)
    
    print(f"3D generation complete. Files saved to {output_dir}")


if __name__ == '__main__':
    # Parse command line arguments
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
        if len(argv) >= 2:
            design_file = argv[0]
            output_dir = argv[1]
            main(design_file, output_dir)
        else:
            print("Usage: blender --python generate_3d.py -- design.json output_dir/")
