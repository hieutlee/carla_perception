"""
Data serializer for the CARLA perception dataset.

Output directory structure per run:
    dataset/
    └── run_XXX/
        ├── metadata.json
        ├── camera_intrinsics.json
        ├── rgb/              000000.png, 000001.png, ...
        ├── depth/            000000.npy, 000001.npy, ...  (float32 meters)
        ├── instance_seg/     000000.png, ...               (16-bit)
        ├── semantic_seg/     000000.png, ...               (8-bit)
        ├── annotations/      000000.json, ...
        └── trajectories/     trajectories.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .sensor_manager import SensorData
from .actor_tracker import FrameAnnotation, ActorState

logger = logging.getLogger(__name__)


class DataSerializer:
    """Handles all I/O for the collected dataset."""

    def __init__(self, config: dict, run_name: str = None):
        self.config = config
        output_root = config.get("collection", {}).get("output_dir", "./dataset")
        image_format = config.get("collection", {}).get("image_format", "png")

        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run_dir = Path(output_root) / run_name
        self.image_format = image_format

        # Create directory structure — including depth
        self.dirs = {
            "rgb": self.run_dir / "rgb",
            "depth": self.run_dir / "depth",
            "instance_seg": self.run_dir / "instance_seg",
            "semantic_seg": self.run_dir / "semantic_seg",
            "annotations": self.run_dir / "annotations",
            "trajectories": self.run_dir / "trajectories",
        }

        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self._frame_count = 0

    def save_run_metadata(
        self,
        map_name: str,
        weather_name: str,
        camera_intrinsics: dict,
        extra: dict = None,
    ):
        metadata = {
            "run_dir": str(self.run_dir),
            "map": map_name,
            "weather": weather_name,
            "config": {
                "fixed_delta_seconds": self.config.get("carla", {}).get("fixed_delta_seconds", 0.05),
                "image_width": self.config["sensors"]["rgb"]["image_size_x"],
                "image_height": self.config["sensors"]["rgb"]["image_size_y"],
                "fov": self.config["sensors"]["rgb"]["fov"],
            },
            "camera_intrinsics": camera_intrinsics,
            "created_at": datetime.now().isoformat(),
        }
        if extra:
            metadata.update(extra)

        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        with open(self.run_dir / "camera_intrinsics.json", "w") as f:
            json.dump(camera_intrinsics, f, indent=2)

    def save_frame(
        self,
        frame_idx: int,
        sensor_data: SensorData,
        annotation: FrameAnnotation,
    ):
        fname = f"{frame_idx:06d}"

        # RGB image
        if sensor_data.rgb.size > 0:
            rgb_path = self.dirs["rgb"] / f"{fname}.{self.image_format}"
            bgr = cv2.cvtColor(sensor_data.rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(rgb_path), bgr)

        # Depth as numpy float32 (meters)
        if sensor_data.depth.size > 0:
            depth_path = self.dirs["depth"] / f"{fname}.npy"
            np.save(str(depth_path), sensor_data.depth)

        # Instance segmentation as 16-bit PNG
        if sensor_data.instance_seg.size > 0:
            inst_path = self.dirs["instance_seg"] / f"{fname}.png"
            inst_16 = sensor_data.instance_seg.astype(np.uint16)
            cv2.imwrite(str(inst_path), inst_16)

        # Semantic segmentation as 8-bit PNG
        if sensor_data.semantic_seg.size > 0:
            sem_path = self.dirs["semantic_seg"] / f"{fname}.png"
            cv2.imwrite(str(sem_path), sensor_data.semantic_seg)

        # Annotation JSON
        ann_path = self.dirs["annotations"] / f"{fname}.json"
        ann_dict = self._annotation_to_dict(annotation)
        with open(str(ann_path), "w") as f:
            json.dump(ann_dict, f, separators=(",", ":"))

        self._frame_count += 1

    def save_trajectories(self, trajectories: dict[int, list[dict]]):
        traj_path = self.dirs["trajectories"] / "trajectories.json"

        serializable = {}
        for actor_id, history in trajectories.items():
            serializable[str(actor_id)] = history

        with open(str(traj_path), "w") as f:
            json.dump(serializable, f, separators=(",", ":"))

        logger.info(
            f"Saved trajectories for {len(trajectories)} actors "
            f"to {traj_path}"
        )

    @staticmethod
    def _annotation_to_dict(annotation: FrameAnnotation) -> dict:
        actors_list = []
        for a in annotation.actors:
            actor_dict = {
                "id": a.actor_id,
                "type": a.actor_type,
                "class_id": a.class_id,
                "class_name": a.class_name,
                "location": list(a.location),
                "rotation": list(a.rotation),
                "velocity": list(a.velocity),
                "acceleration": list(a.acceleration),
                "angular_velocity": list(a.angular_velocity),
                "speed": round(a.speed, 4),
                "bbox_3d": {
                    "extent": list(a.bbox_extent),
                    "center_offset": list(a.bbox_location),
                },
                "bbox_2d": list(a.bbox_2d) if a.bbox_2d else None,
                "visibility": round(a.visibility, 4),
                "distance": round(a.distance_to_ego, 4),
            }
            actors_list.append(actor_dict)

        return {
            "frame": annotation.frame,
            "timestamp": round(annotation.timestamp, 6),
            "ego": {
                "location": [
                    annotation.ego_transform.x,
                    annotation.ego_transform.y,
                    annotation.ego_transform.z,
                ],
                "rotation": [
                    annotation.ego_transform.pitch,
                    annotation.ego_transform.yaw,
                    annotation.ego_transform.roll,
                ],
                "velocity": list(annotation.ego_velocity),
            },
            "actors": actors_list,
        }

    @property
    def frame_count(self) -> int:
        return self._frame_count