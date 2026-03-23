"""
Actor Tracker for CARLA perception data collection.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import carla

from .projection import (
    Transform, BoundingBox3D, CameraIntrinsics,
    compute_2d_bbox
)


@dataclass
class ActorState:
    actor_id: int
    actor_type: str
    class_id: int
    class_name: str
    location: tuple[float, float, float]
    rotation: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    speed: float
    bbox_extent: tuple[float, float, float]
    bbox_location: tuple[float, float, float]
    bbox_2d: Optional[tuple[int, int, int, int]] = None
    visibility: float = 1.0
    distance_to_ego: float = 0.0


@dataclass
class FrameAnnotation:
    frame: int
    timestamp: float
    ego_transform: Transform
    ego_velocity: tuple[float, float, float]
    actors: list[ActorState] = field(default_factory=list)


class ActorTracker:

    def __init__(self, world: carla.World, config: dict, history_length: int = 200):
        self.world = world
        self.config = config
        self.label_mapping = config.get("label_mapping", {})
        self.bbox_filter = config.get("bbox_filter", {})
        self.history_length = history_length
        self.trajectory_history: dict[int, list[dict]] = defaultdict(list)
        self._active_actors: set[int] = set()

    def _get_semantic_tag(self, actor: carla.Actor) -> Optional[int]:
        type_id = actor.type_id
        if type_id.startswith("vehicle."):
            if "bicycle" in type_id or "crossbike" in type_id or "omafiets" in type_id or "century" in type_id:
                return 14
            elif "motorcycle" in type_id or "harley" in type_id or "kawasaki" in type_id or "yamaha" in type_id or "vespa" in type_id:
                return 15
            elif "bus" in type_id or "firetruck" in type_id or "ambulance" in type_id or "fusorosa" in type_id:
                return 16
            elif "european_hgv" in type_id:
                return 18
            else:
                return 10
        elif type_id.startswith("walker."):
            return 12
        return None

    def tick(
        self,
        frame: int,
        timestamp: float,
        ego_vehicle: carla.Vehicle,
        camera_transform: carla.Transform,
        intrinsics: CameraIntrinsics,
        depth_map: np.ndarray = None,
        **kwargs,
    ) -> FrameAnnotation:

        ego_transform = Transform.from_carla(ego_vehicle.get_transform())
        ego_vel = ego_vehicle.get_velocity()
        ego_loc = ego_vehicle.get_transform().location

        # Build camera transform with basis vectors
        cam_transform = Transform.from_carla(camera_transform)

        annotation = FrameAnnotation(
            frame=frame,
            timestamp=timestamp,
            ego_transform=ego_transform,
            ego_velocity=(ego_vel.x, ego_vel.y, ego_vel.z),
        )

        actors = self.world.get_actors()
        current_frame_actor_ids = set()

        for actor in actors:
            if actor.id == ego_vehicle.id:
                continue

            semantic_tag = self._get_semantic_tag(actor)
            if semantic_tag is None or semantic_tag not in self.label_mapping:
                continue

            label_info = self.label_mapping[semantic_tag]
            class_id = label_info["id"]
            class_name = label_info["name"]

            transform = actor.get_transform()
            velocity = actor.get_velocity()
            acceleration = actor.get_acceleration()
            angular_vel = actor.get_angular_velocity()

            loc = transform.location
            rot = transform.rotation

            distance = loc.distance(ego_loc)
            max_dist = self.bbox_filter.get("max_distance", 100.0)
            if distance > max_dist:
                if distance < max_dist * 1.5:
                    self._update_trajectory(actor.id, frame, loc, velocity, rot.yaw)
                    current_frame_actor_ids.add(actor.id)
                continue

            # Build 3D bounding box using new API
            bbox = actor.bounding_box
            bbox_3d = BoundingBox3D(
                actor_location=np.array([loc.x, loc.y, loc.z]),
                actor_rotation_yaw=rot.yaw,
                extent_x=bbox.extent.x,
                extent_y=bbox.extent.y,
                extent_z=bbox.extent.z,
                center_offset=np.array([bbox.location.x, bbox.location.y, bbox.location.z]),
            )

            # Project to 2D using dot-product method
            bbox_2d = compute_2d_bbox(bbox_3d, cam_transform, intrinsics)

            # Always update trajectory
            self._update_trajectory(actor.id, frame, loc, velocity, rot.yaw)
            current_frame_actor_ids.add(actor.id)

            if bbox_2d is None:
                continue

            # Dimension filter
            x_min, y_min, x_max, y_max = bbox_2d
            min_dim = self.bbox_filter.get("min_dimension", 10)
            if (x_max - x_min) < min_dim or (y_max - y_min) < min_dim:
                continue

            # Area filter
            bbox_area = (x_max - x_min) * (y_max - y_min)
            min_area = self.bbox_filter.get("min_area", 100)
            if bbox_area < min_area:
                continue

            # Depth-based occlusion check
            # If depth at the bbox center shows something much closer than
            # the actor, a wall/building is blocking the line of sight
            visibility = 1.0
            if depth_map is not None and depth_map.size > 0:
                visibility = self._check_depth_visibility(
                    depth_map, bbox_2d, distance
                )
                if visibility < 0.5:  # 0.0 = occluded, 1.0 = visible
                    continue

            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            state = ActorState(
                actor_id=actor.id,
                actor_type=actor.type_id,
                class_id=class_id,
                class_name=class_name,
                location=(loc.x, loc.y, loc.z),
                rotation=(rot.pitch, rot.yaw, rot.roll),
                velocity=(velocity.x, velocity.y, velocity.z),
                acceleration=(acceleration.x, acceleration.y, acceleration.z),
                angular_velocity=(angular_vel.x, angular_vel.y, angular_vel.z),
                speed=speed,
                bbox_extent=(bbox.extent.x, bbox.extent.y, bbox.extent.z),
                bbox_location=(bbox.location.x, bbox.location.y, bbox.location.z),
                bbox_2d=bbox_2d,
                visibility=visibility,
                distance_to_ego=distance,
            )

            annotation.actors.append(state)

        self._active_actors = current_frame_actor_ids
        return annotation

    def _check_depth_visibility(
        self,
        depth_map: np.ndarray,
        bbox_2d: tuple[int, int, int, int],
        actor_distance: float,
    ) -> float:
        """
        Determine if an actor is occluded using the depth map.

        Two-stage check:
        1. Center point check: if the depth at the bbox center is much
           closer than the actor, something is blocking the direct line of sight.
        2. Median check: the median depth in the bbox region should be
           roughly consistent with the actor distance. Median is robust
           to sky pixels (outliers at 1000m).

        A car at 40m behind a wall at 10m:
          - Center depth = 10m (wall) → 10 < 40*0.6 = 24 → OCCLUDED
          - Median depth ≈ 10m (wall dominates) → 10 < 24 → confirms occlusion

        A visible car at 40m:
          - Center depth ≈ 38m (car surface) → 38 > 24 → VISIBLE
          - Median depth ≈ 35m → confirms visibility

        Returns:
            1.0 if visible, 0.0 if occluded.
        """
        x_min, y_min, x_max, y_max = bbox_2d

        # Check 1: center point depth
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        h, w = depth_map.shape[:2]
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return 0.0

        center_depth = depth_map[cy, cx]

        # If center shows something much closer than the actor, it's occluded.
        # Use 60% of actor distance as threshold — generous enough for angled views
        # but catches walls that are clearly in front.
        occlusion_threshold = actor_distance * 0.6

        if center_depth < occlusion_threshold:
            # Center is blocked. Double-check with median of bbox region.
            roi = depth_map[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                return 0.0
            median_depth = np.median(roi)
            if median_depth < occlusion_threshold:
                return 0.0  # definitely occluded

        return 1.0

    def _update_trajectory(self, actor_id, frame, location, velocity, yaw):
        entry = {
            "frame": frame,
            "x": location.x,
            "y": location.y,
            "z": location.z,
            "vx": velocity.x,
            "vy": velocity.y,
            "vz": velocity.z,
            "yaw": yaw,
        }
        history = self.trajectory_history[actor_id]
        history.append(entry)
        if len(history) > self.history_length:
            self.trajectory_history[actor_id] = history[-self.history_length:]

    def get_trajectory(self, actor_id: int) -> list[dict]:
        return self.trajectory_history.get(actor_id, [])

    def get_all_trajectories(self) -> dict[int, list[dict]]:
        return dict(self.trajectory_history)

    def prune_dead_actors(self, keep_frames: int = 50):
        to_remove = [aid for aid in self.trajectory_history if aid not in self._active_actors]
        for aid in to_remove:
            del self.trajectory_history[aid]