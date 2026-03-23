#!/usr/bin/env python3
"""Verify the fixed projection detects vehicles in FOV."""

import carla
import numpy as np
import math

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

ego = world.spawn_actor(bp_lib.find('vehicle.mercedes.coupe'), spawn_points[0])
ego.set_autopilot(True)

npcs = []
for i in range(1, min(31, len(spawn_points))):
    try:
        npc = world.spawn_actor(bp_lib.find('vehicle.audi.a2'), spawn_points[i])
        npc.set_autopilot(True)
        npcs.append(npc)
    except:
        pass

world.set_weather(carla.WeatherParameters.ClearNoon)

# Spawn camera
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '450')
cam_bp.set_attribute('fov', '90')
cam = world.spawn_actor(cam_bp, carla.Transform(
    carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-5.0)
), attach_to=ego)

print(f'Ego id={ego.id}, NPCs={len(npcs)}')

# Warm up
for _ in range(200):
    world.tick()

intrinsics = CameraIntrinsics(width=800, height=450, fov=90.0)

print('\n=== TESTING FIXED PROJECTION ===')
for tick in range(5):
    for _ in range(20):
        world.tick()

    cam_t = Transform.from_carla(cam.get_transform())
    ego_loc = ego.get_transform().location

    detected = []
    for actor in world.get_actors():
        if not actor.type_id.startswith('vehicle.') or actor.id == ego.id:
            continue
        loc = actor.get_transform().location
        rot = actor.get_transform().rotation
        dist = loc.distance(ego_loc)
        if dist > 100:
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
        if bbox_2d is not None:
            x1, y1, x2, y2 = bbox_2d
            area = (x2 - x1) * (y2 - y1)
            detected.append((actor.id, dist, bbox_2d, area))

    nearby = len([a for a in world.get_actors()
                  if a.type_id.startswith('vehicle.') and a.id != ego.id
                  and a.get_transform().location.distance(ego_loc) < 100])

    print(f'Tick {tick}: {nearby} nearby, {len(detected)} DETECTED in camera')
    for aid, d, bb, area in detected[:8]:
        print(f'  id={aid} dist={d:.1f}m bbox={bb} area={area}px')

# Cleanup
cam.stop()
cam.destroy()
for n in npcs:
    try: n.destroy()
    except: pass
try: ego.destroy()
except: pass
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
print('\nDone.')
