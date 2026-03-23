"""
3D-to-2D projection utilities for CARLA sensor data.

CRITICAL NOTE ON COORDINATE SYSTEMS:
    CARLA / Unreal Engine uses a LEFT-HANDED coordinate system:
        X = forward, Y = right, Z = up
    
    Standard pinhole camera (OpenCV) uses:
        X = right, Y = down, Z = forward (depth)
    
    Previous versions of this module attempted to build rotation matrices
    from CARLA's (pitch, yaw, roll) angles, but got the left-handed
    rotation conventions wrong, causing ALL projections to fail.
    
    This version uses two approaches:
    1. LIVE MODE: Uses CARLA's native get_inverse_matrix() directly
       (called during data collection when carla.Transform is available)
    2. OFFLINE MODE: Uses stored camera basis vectors (forward, right, up)
       for projection during training/evaluation
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters derived from CARLA sensor config."""
    width: int
    height: int
    fov: float  # horizontal field of view in degrees

    @property
    def focal_length(self) -> float:
        """Focal length in pixels, derived from horizontal FOV."""
        return self.width / (2.0 * math.tan(math.radians(self.fov / 2.0)))

    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        f = self.focal_length
        cx = self.width / 2.0
        cy = self.height / 2.0
        return np.array([
            [f,   0.0, cx],
            [0.0, f,   cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


@dataclass
class Transform:
    """
    Wrapper around CARLA transform that stores the native matrix.
    
    Instead of reconstructing the rotation from angles (which broke
    due to left-handed coordinate system issues), we store CARLA's
    own matrix and basis vectors.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    # Camera basis vectors from CARLA (set during live collection)
    forward: Optional[np.ndarray] = None
    right: Optional[np.ndarray] = None
    up: Optional[np.ndarray] = None

    @staticmethod
    def from_carla(carla_transform) -> "Transform":
        """Construct from a carla.Transform object, capturing basis vectors."""
        loc = carla_transform.location
        rot = carla_transform.rotation
        fwd = carla_transform.get_forward_vector()
        rgt = carla_transform.get_right_vector()
        up = carla_transform.get_up_vector()
        return Transform(
            x=loc.x, y=loc.y, z=loc.z,
            pitch=rot.pitch, yaw=rot.yaw, roll=rot.roll,
            forward=np.array([fwd.x, fwd.y, fwd.z]),
            right=np.array([rgt.x, rgt.y, rgt.z]),
            up=np.array([up.x, up.y, up.z]),
        )


@dataclass
class BoundingBox3D:
    """
    3D bounding box of an actor in world coordinates.
    `extent` is the half-size along each axis (CARLA convention).
    `actor_transform` is the actor's world transform.
    """
    # Actor's world position and rotation
    actor_location: np.ndarray  # (3,) [x, y, z]
    actor_rotation_yaw: float   # degrees
    # Bounding box half-extents
    extent_x: float  # half-width (forward axis)
    extent_y: float  # half-width (right axis)
    extent_z: float  # half-height (up axis)
    # Bounding box center offset from actor origin
    center_offset: np.ndarray  # (3,) [x, y, z]

    def get_world_vertices(self) -> np.ndarray:
        """
        Compute the 8 corners of the 3D bbox in world coordinates.
        Returns: (8, 3) array of world-space vertices.
        """
        ex, ey, ez = self.extent_x, self.extent_y, self.extent_z

        # 8 corners in local bbox space
        local_verts = np.array([
            [-ex, -ey, -ez],
            [-ex, -ey,  ez],
            [-ex,  ey, -ez],
            [-ex,  ey,  ez],
            [ ex, -ey, -ez],
            [ ex, -ey,  ez],
            [ ex,  ey, -ez],
            [ ex,  ey,  ez],
        ], dtype=np.float64)

        # Rotate by actor's yaw (simplified: only yaw matters for ground vehicles)
        yaw_rad = math.radians(self.actor_rotation_yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        R = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1],
        ])

        # Bbox center in world
        center_world = self.actor_location + self.center_offset

        # Transform vertices to world
        world_verts = (R @ local_verts.T).T + center_world
        return world_verts


def project_point_to_image(
    point_world: np.ndarray,
    cam_location: np.ndarray,
    cam_forward: np.ndarray,
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[Optional[tuple[float, float]], float]:
    """
    Project a single 3D world point to 2D image coordinates
    using dot products with camera basis vectors.

    This method is immune to coordinate system handedness issues
    because it uses CARLA's own basis vectors directly.

    Returns:
        (u, v) pixel coordinates or None if behind camera.
        depth: signed depth (positive = in front of camera).
    """
    # Vector from camera to point
    dx = point_world[0] - cam_location[0]
    dy = point_world[1] - cam_location[1]
    dz = point_world[2] - cam_location[2]
    delta = np.array([dx, dy, dz])

    # Project onto camera basis
    depth = np.dot(delta, cam_forward)    # forward distance
    right = np.dot(delta, cam_right)      # lateral offset
    up = np.dot(delta, cam_up)            # vertical offset

    if depth <= 0.1:
        return None, depth

    # Perspective projection
    f = intrinsics.focal_length
    cx = intrinsics.width / 2.0
    cy = intrinsics.height / 2.0

    u = f * (right / depth) + cx
    v = f * (-up / depth) + cy  # negate up because image Y points down

    return (u, v), depth


def compute_2d_bbox(
    bbox_3d: BoundingBox3D,
    camera_transform: Transform,
    intrinsics: CameraIntrinsics,
) -> Optional[tuple[int, int, int, int]]:
    """
    Compute tight 2D bounding box from 3D bbox projection.

    Uses dot-product projection with CARLA's native basis vectors.

    Returns (x_min, y_min, x_max, y_max) in pixel coordinates,
    or None if the bbox is not visible.
    """
    if camera_transform.forward is None:
        return None

    vertices = bbox_3d.get_world_vertices()  # (8, 3)
    cam_loc = np.array([camera_transform.x, camera_transform.y, camera_transform.z])

    projected_points = []

    for vert in vertices:
        result, depth = project_point_to_image(
            vert, cam_loc,
            camera_transform.forward,
            camera_transform.right,
            camera_transform.up,
            intrinsics,
        )
        if result is not None:
            projected_points.append(result)

    if len(projected_points) == 0:
        return None

    points = np.array(projected_points)
    u_coords = points[:, 0]
    v_coords = points[:, 1]

    x_min = int(np.clip(np.floor(u_coords.min()), 0, intrinsics.width - 1))
    y_min = int(np.clip(np.floor(v_coords.min()), 0, intrinsics.height - 1))
    x_max = int(np.clip(np.ceil(u_coords.max()), 0, intrinsics.width - 1))
    y_max = int(np.clip(np.ceil(v_coords.max()), 0, intrinsics.height - 1))

    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)
