#!/usr/bin/env python3
"""
Diagnostic: check depth map values and occlusion decisions.
Spawns ego + NPCs, captures depth, and shows what the occlusion
check decides for each detected vehicle.
"""

import carla
import numpy as np
import math
import threading
import cv2

from collector.projection import CameraIntrinsics, Transform, BoundingBox3D, compute_2d_bbox

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town10HD_Opt')

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
world.tick()

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
world.set_weather(carla.WeatherParameters.ClearNoon)

# Spawn ego + NPCs
ego = world.spawn_actor(bp_lib.find('vehicle.mercedes.coupe'), spawn_points[0])
ego.set_autopilot(True)

npcs = []
for i in range(1, min(41, len(spawn_points))):
    try:
        npc = world.spawn_actor(bp_lib.find('vehicle.audi.a2'), spawn_points[i])
        npc.set_autopilot(True)
        npcs.append(npc)
    except:
        pass

print(f'Ego id={ego.id}, NPCs={len(npcs)}')

# Spawn RGB + Depth cameras
cam_transform = carla.Transform(
    carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-5.0)
)

rgb_bp = bp_lib.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', '800')
rgb_bp.set_attribute('image_size_y', '450')
rgb_bp.set_attribute('fov', '90')
rgb_cam = world.spawn_actor(rgb_bp, cam_transform, attach_to=ego)

depth_bp = bp_lib.find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x', '800')
depth_bp.set_attribute('image_size_y', '450')
depth_bp.set_attribute('fov', '90')
depth_cam = world.spawn_actor(depth_bp, cam_transform, attach_to=ego)

# Capture buffers
buffers = {}
events = {'rgb': threading.Event(), 'depth': threading.Event()}

def rgb_cb(data):
    arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))
    buffers['rgb'] = arr[:, :, :3][:, :, ::-1].copy()
    events['rgb'].set()

def depth_cb(data):
    arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))
    r = arr[:, :, 2].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 0].astype(np.float32)
    depth_meters = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0) * 1000.0
    buffers['depth'] = depth_meters
    events['depth'].set()

rgb_cam.listen(rgb_cb)
depth_cam.listen(depth_cb)

# Warm up
for _ in range(200):
    world.tick()

intrinsics = CameraIntrinsics(width=800, height=450, fov=90.0)

# Run 3 test ticks
for tick in range(3):
    for _ in range(40):
        world.tick()

    events['rgb'].clear()
    events['depth'].clear()
    world.tick()
    events['rgb'].wait(5.0)
    events['depth'].wait(5.0)

    depth_map = buffers['depth']
    rgb_img = buffers['rgb']
    cam_t = Transform.from_carla(rgb_cam.get_transform())
    ego_loc = ego.get_transform().location

    print(f'\n=== TICK {tick} ===')
    print(f'Depth map stats: min={depth_map.min():.2f}m, max={depth_map.max():.2f}m, '
          f'mean={depth_map.mean():.2f}m, median={np.median(depth_map):.2f}m')
    print(f'Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}')

    # Check center pixel depth
    cy, cx = depth_map.shape[0]//2, depth_map.shape[1]//2
    print(f'Center pixel depth: {depth_map[cy, cx]:.2f}m')

    vis_img = rgb_img.copy()

    for actor in world.get_actors():
        if not actor.type_id.startswith('vehicle.') or actor.id == ego.id:
            continue
        loc = actor.get_transform().location
        rot = actor.get_transform().rotation
        dist = loc.distance(ego_loc)
        if dist > 80:
            continue

        bbox = actor.bounding_box
        bbox_3d = BoundingBox3D(
            actor_location=np.array([loc.x, loc.y, loc.z]),
            actor_rotation_yaw=rot.yaw,
            extent_x=bbox.extent.x,
            extent_y=bbox.extent.y,
            extent_z=bbox.extent.z,
            center_offset=np.array([bbox.location.x, bbox.location.y, bbox.location.z]),
        )
        bbox_2d = compute_2d_bbox(bbox_3d, cam_t, intrinsics)

        if bbox_2d is None:
            continue

        x1, y1, x2, y2 = bbox_2d
        area = (x2 - x1) * (y2 - y1)
        if area < 100:
            continue

        # Check depth in bbox region
        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        sampled = roi[::3, ::3]
        roi_min = sampled.min()
        roi_max = sampled.max()
        roi_mean = sampled.mean()
        roi_median = np.median(sampled)

        # Occlusion check: what fraction of pixels show depth >= (actor_dist - tolerance)?
        tolerance = max(3.0, dist * 0.15)
        consistent = np.sum(sampled >= (dist - tolerance))
        visibility = float(consistent) / float(sampled.size)

        # Is it occluded?
        occluded = visibility < 0.15
        status = "OCCLUDED" if occluded else "VISIBLE"

        print(f'\n  Vehicle {actor.id} at {dist:.1f}m | bbox={bbox_2d} | area={area}px')
        print(f'    Depth in bbox: min={roi_min:.1f}m, max={roi_max:.1f}m, mean={roi_mean:.1f}m, median={roi_median:.1f}m')
        print(f'    Actor distance: {dist:.1f}m, tolerance: {tolerance:.1f}m')
        print(f'    Visibility: {visibility:.2f} → {status}')

        # Draw on image
        color = (0, 255, 0) if not occluded else (0, 0, 255)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label = f'{status} v={visibility:.2f} d={dist:.0f}m depth_med={roi_median:.0f}m'
        cv2.putText(vis_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Save annotated image
    out_path = f'debug_occlusion_tick{tick}.png'
    cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f'\n  Saved: {out_path}')

    # Also save depth as a visible grayscale image
    depth_vis = np.clip(depth_map / 100.0, 0, 1)  # normalize to 100m range
    depth_vis = (depth_vis * 255).astype(np.uint8)
    cv2.imwrite(f'debug_depth_tick{tick}.png', depth_vis)
    print(f'  Saved: debug_depth_tick{tick}.png')

# Cleanup
rgb_cam.stop(); rgb_cam.destroy()
depth_cam.stop(); depth_cam.destroy()
for n in npcs:
    try: n.destroy()
    except: pass
try: ego.destroy()
except: pass
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
print('\nDone. Check debug_occlusion_tick*.png and debug_depth_tick*.png')
