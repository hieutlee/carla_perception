#!/usr/bin/env python3
"""
CARLA 0.9.16 Perception Stack — Data Collection Script

Usage:
    # Start CARLA server first:
    ./CarlaUE4.sh                    (with spectator window — use during dev)
    ./CarlaUE4.sh -RenderOffScreen   (headless — use for big collection runs)

    # Then run collection:
    python collect.py
    python collect.py --map Town10HD_Opt --frames 10000
    python collect.py --runs 5
"""

import argparse
import logging
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import carla

from collector.projection import CameraIntrinsics, Transform
from collector.sensor_manager import SensorManager
from collector.actor_tracker import ActorTracker
from collector.world_manager import TrafficManager as NPCManager, WeatherManager
from collector.serializer import DataSerializer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect")

_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    logger.warning("Shutdown requested (Ctrl+C). Cleaning up...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _signal_handler)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def connect_to_carla(config: dict) -> tuple[carla.Client, carla.World]:
    carla_cfg = config["carla"]
    client = carla.Client(carla_cfg["host"], carla_cfg["port"])
    client.set_timeout(carla_cfg["timeout"])

    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = carla_cfg["fixed_delta_seconds"]
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    logger.info(
        f"Connected to CARLA at {carla_cfg['host']}:{carla_cfg['port']} | "
        f"Sync mode ON | dt={carla_cfg['fixed_delta_seconds']}s"
    )
    return client, world


def spawn_ego_vehicle(
    world: carla.World,
    config: dict,
    tm: carla.Client = None
) -> carla.Vehicle:
    ego_cfg = config["ego_vehicle"]
    bp_library = world.get_blueprint_library()
    bp = bp_library.find(ego_cfg["blueprint"])

    spawn_points = world.get_map().get_spawn_points()
    if ego_cfg.get("spawn_point") is not None:
        spawn_transform = spawn_points[ego_cfg["spawn_point"]]
    else:
        spawn_transform = random.choice(spawn_points)

    ego = world.spawn_actor(bp, spawn_transform)
    logger.info(f"Ego vehicle spawned: {ego_cfg['blueprint']} at {spawn_transform.location}")

    if ego_cfg.get("autopilot", True) and tm is not None:
        ego.set_autopilot(True, tm.get_port())
        speed_pct = ego_cfg.get("target_speed_pct", 70)
        tm.vehicle_percentage_speed_difference(ego, 100 - speed_pct)

    return ego


def update_spectator(world: carla.World, ego_vehicle: carla.Vehicle, config: dict):
    """
    Move the spectator camera to follow the ego vehicle.
    Gives you a third-person chase view in the CARLA window.
    """
    spec_cfg = config.get("spectator", {})
    if not spec_cfg.get("enabled", False):
        return

    ego_transform = ego_vehicle.get_transform()
    dist_behind = spec_cfg.get("distance_behind", 8.0)
    height = spec_cfg.get("height_above", 4.0)

    # Compute a position behind and above the ego vehicle
    yaw_rad = math.radians(ego_transform.rotation.yaw)
    spectator_loc = carla.Location(
        x=ego_transform.location.x - dist_behind * math.cos(yaw_rad),
        y=ego_transform.location.y - dist_behind * math.sin(yaw_rad),
        z=ego_transform.location.z + height,
    )

    # Look down toward the ego vehicle
    spectator_rot = carla.Rotation(
        pitch=-15.0,
        yaw=ego_transform.rotation.yaw,
        roll=0.0,
    )

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))


def run_collection(
    client: carla.Client,
    world: carla.World,
    config: dict,
    map_name: str = None,
    num_frames: int = None,
    run_name: str = None,
):
    global _shutdown_requested

    collection_cfg = config.get("collection", {})
    frames_target = num_frames or collection_cfg.get("frames_per_run", 5000)
    save_interval = collection_cfg.get("save_interval", 1)
    min_actors = collection_cfg.get("min_actors_to_save", 0)

    # Load map if specified
    if map_name:
        logger.info(f"Loading map: {map_name}")
        world = client.load_world(map_name)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config["carla"]["fixed_delta_seconds"]
        world.apply_settings(settings)
        world.tick()

    current_map = world.get_map().name

    ego_vehicle = None
    npc_manager = None
    sensor_manager = None

    try:
        # Traffic manager
        tm_port = config.get("ego_vehicle", {}).get("traffic_manager_port", 8000)
        tm = client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(random.randint(0, 10000))

        # Spawn ego
        ego_vehicle = spawn_ego_vehicle(world, config, tm)

        # Spawn NPCs
        npc_manager = NPCManager(client, world, config)
        num_vehicles = npc_manager.spawn_vehicles()
        num_walkers = npc_manager.spawn_walkers()

        # Weather
        weather_mgr = WeatherManager(world, config)
        weather_name = weather_mgr.sample_weather()

        # Sensors
        sensor_manager = SensorManager(world, ego_vehicle, config)
        sensor_manager.setup()

        # Camera intrinsics
        rgb_cfg = config["sensors"]["rgb"]
        intrinsics = CameraIntrinsics(
            width=rgb_cfg["image_size_x"],
            height=rgb_cfg["image_size_y"],
            fov=rgb_cfg["fov"],
        )

        # Actor tracker
        actor_tracker = ActorTracker(world, config)

        # Serializer
        serializer = DataSerializer(config, run_name=run_name)
        serializer.save_run_metadata(
            map_name=current_map,
            weather_name=weather_name,
            camera_intrinsics={
                "K": intrinsics.K.tolist(),
                "fov": intrinsics.fov,
                "width": intrinsics.width,
                "height": intrinsics.height,
                "focal_length": intrinsics.focal_length,
            },
            extra={
                "num_npc_vehicles": num_vehicles,
                "num_npc_walkers": num_walkers,
            },
        )

        # Warmup
        warmup_ticks = config["carla"].get("warmup_ticks", 100)
        logger.info(f"Warming up simulation ({warmup_ticks} ticks)...")
        for _ in range(warmup_ticks):
            world.tick()

        # Collection loop
        logger.info(
            f"Starting collection: {frames_target} frames | "
            f"map={current_map} | weather={weather_name} | "
            f"NPCs: {num_vehicles}v + {num_walkers}w"
        )

        saved_count = 0
        skipped_empty = 0
        tick_times = []

        for frame_idx in range(frames_target):
            if _shutdown_requested:
                logger.info("Shutdown requested, stopping collection.")
                break

            t0 = time.perf_counter()

            world.tick()

            # Update spectator to follow ego (chase cam view in CARLA window)
            update_spectator(world, ego_vehicle, config)

            # Get sensor data
            try:
                sensor_data = sensor_manager.get_data(timeout=10.0)
            except TimeoutError as e:
                logger.error(f"Frame {frame_idx}: {e}")
                continue

            # Get camera transform for projection
            cam_transform = sensor_manager.get_camera_transform()

            # Extract annotations
            annotation = actor_tracker.tick(
                frame=sensor_data.frame,
                timestamp=sensor_data.timestamp,
                ego_vehicle=ego_vehicle,
                camera_transform=cam_transform,
                intrinsics=intrinsics,
                depth_map=sensor_data.depth,
            )

            # Filter: skip frames with too few visible actors
            num_visible = len(annotation.actors)
            if num_visible < min_actors:
                skipped_empty += 1
            elif frame_idx % save_interval == 0:
                serializer.save_frame(saved_count, sensor_data, annotation)
                saved_count += 1

            dt = time.perf_counter() - t0
            tick_times.append(dt)

            # Progress logging
            if (frame_idx + 1) % 100 == 0:
                avg_dt = np.mean(tick_times[-100:])
                fps = 1.0 / avg_dt if avg_dt > 0 else 0
                eta_seconds = (frames_target - frame_idx - 1) * avg_dt
                eta_min = eta_seconds / 60

                logger.info(
                    f"Frame {frame_idx + 1}/{frames_target} | "
                    f"{fps:.1f} FPS | {num_visible} actors | "
                    f"saved: {saved_count} | skipped: {skipped_empty} | "
                    f"ETA: {eta_min:.1f}min"
                )

            if frame_idx % 500 == 0:
                actor_tracker.prune_dead_actors()

        # Save trajectories
        all_trajectories = actor_tracker.get_all_trajectories()
        serializer.save_trajectories(all_trajectories)

        # Summary
        total_time = sum(tick_times)
        avg_fps = len(tick_times) / total_time if total_time > 0 else 0

        logger.info("=" * 60)
        logger.info("Collection complete!")
        logger.info(f"  Frames saved: {saved_count}")
        logger.info(f"  Frames skipped (empty): {skipped_empty}")
        logger.info(f"  Trajectories: {len(all_trajectories)} actors")
        logger.info(f"  Avg FPS: {avg_fps:.1f}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Output: {serializer.run_dir}")
        logger.info("=" * 60)

    finally:
        # ---- Safe cleanup ----
        # The C++ runtime error on actor destruction kills the entire
        # Python process if not handled correctly. The key is:
        # 1. Switch to async mode FIRST (so CARLA doesn't block)
        # 2. Disable traffic manager sync
        # 3. Stop sensor listeners (before destroying anything)
        # 4. Give CARLA a moment to process
        # 5. Destroy actors

        # Step 1: Restore async mode
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass

        # Step 2: Disable traffic manager sync
        try:
            tm_port = config.get("ego_vehicle", {}).get("traffic_manager_port", 8000)
            client.get_trafficmanager(tm_port).set_synchronous_mode(False)
        except Exception:
            pass

        # Step 3: Stop sensor listeners (but don't destroy yet)
        if sensor_manager is not None:
            for name, sensor in sensor_manager.sensors.items():
                try:
                    sensor.stop()
                except Exception:
                    pass

        # Step 4: Let CARLA process the stops
        time.sleep(0.5)

        # Step 5: Destroy everything via batch command (faster and safer)
        all_actor_ids = []

        if sensor_manager is not None:
            for name, sensor in sensor_manager.sensors.items():
                try:
                    if sensor.is_alive:
                        all_actor_ids.append(sensor.id)
                except Exception:
                    pass
            sensor_manager.sensors.clear()
            sensor_manager._events.clear()
            logger.info("Sensors cleaned up")

        if npc_manager is not None:
            for v in npc_manager.vehicles:
                try:
                    if v is not None:
                        all_actor_ids.append(v.id)
                except Exception:
                    pass
            for w in npc_manager.walkers:
                try:
                    if w is not None:
                        all_actor_ids.append(w.id)
                except Exception:
                    pass
            for c in npc_manager.walker_controllers:
                try:
                    c.stop()
                except Exception:
                    pass
                try:
                    if c is not None:
                        all_actor_ids.append(c.id)
                except Exception:
                    pass
            npc_manager.vehicles.clear()
            npc_manager.walkers.clear()
            npc_manager.walker_controllers.clear()
            logger.info("NPCs cleaned up")

        if ego_vehicle is not None:
            try:
                if ego_vehicle.is_alive:
                    all_actor_ids.append(ego_vehicle.id)
            except Exception:
                pass

        # Single batch destroy — much less likely to trigger race conditions
        if all_actor_ids:
            try:
                client.apply_batch([carla.command.DestroyActor(x) for x in all_actor_ids])
                time.sleep(0.5)
                logger.info(f"Destroyed {len(all_actor_ids)} actors via batch")
            except Exception as e:
                logger.warning(f"Batch destroy error: {e}")


def main():
    parser = argparse.ArgumentParser(description="CARLA Perception Data Collector")
    parser.add_argument(
        "--config", type=str, default="config/collection_config.yaml",
        help="Path to collection config YAML"
    )
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    client, world = connect_to_carla(config)

    maps = [args.map] if args.map else config.get("collection", {}).get("maps", ["Town10HD_Opt"])

    for run_idx in range(args.runs):
        if _shutdown_requested:
            break

        map_name = maps[run_idx % len(maps)]
        run_name = f"run_{run_idx:03d}_{map_name}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Run {run_idx + 1}/{args.runs}: {map_name}")
        logger.info(f"{'='*60}\n")

        run_collection(
            client=client,
            world=world,
            config=config,
            map_name=map_name,
            num_frames=args.frames,
            run_name=run_name,
        )

        if run_idx < args.runs - 1:
            time.sleep(2.0)


if __name__ == "__main__":
    main()