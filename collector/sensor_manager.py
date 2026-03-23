"""
Sensor Manager for CARLA synchronous data collection.

Manages the lifecycle of all sensors attached to the ego vehicle,
handles per-tick data buffering via callbacks, and provides a
synchronous `get_data()` interface that blocks until all sensors
have reported for the current simulation frame.
"""

import threading
import numpy as np
import weakref
from dataclasses import dataclass, field
from typing import Any

import carla


@dataclass
class SensorData:
    """Container for one tick's worth of synchronized sensor output."""
    frame: int = -1
    timestamp: float = 0.0
    rgb: np.ndarray = field(default_factory=lambda: np.array([]))
    depth: np.ndarray = field(default_factory=lambda: np.array([]))
    instance_seg: np.ndarray = field(default_factory=lambda: np.array([]))
    semantic_seg: np.ndarray = field(default_factory=lambda: np.array([]))


class SensorManager:
    """
    Spawns and manages synchronized sensors on the ego vehicle.

    Usage:
        sm = SensorManager(world, ego_vehicle, config)
        sm.setup()

        # In simulation loop:
        world.tick()
        data = sm.get_data(timeout=10.0)
        # data.rgb, data.depth, etc. are numpy arrays for this frame

        # Cleanup:
        sm.destroy()
    """

    def __init__(self, world: carla.World, ego_vehicle: carla.Vehicle, config: dict):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.config = config
        self.sensors: dict[str, carla.Sensor] = {}
        self._data_buffer: dict[str, Any] = {}
        self._events: dict[str, threading.Event] = {}
        self._frame: int = -1
        self._lock = threading.Lock()

    def setup(self):
        """Spawn all configured sensors attached to the ego vehicle."""
        blueprint_library = self.world.get_blueprint_library()
        sensor_configs = self.config["sensors"]

        for sensor_name, sensor_cfg in sensor_configs.items():
            bp = blueprint_library.find(sensor_cfg["blueprint"])

            if "image_size_x" in sensor_cfg:
                bp.set_attribute("image_size_x", str(sensor_cfg["image_size_x"]))
            if "image_size_y" in sensor_cfg:
                bp.set_attribute("image_size_y", str(sensor_cfg["image_size_y"]))
            if "fov" in sensor_cfg:
                bp.set_attribute("fov", str(sensor_cfg["fov"]))

            t_cfg = sensor_cfg["transform"]
            loc = t_cfg["location"]
            rot = t_cfg["rotation"]
            transform = carla.Transform(
                carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
                carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
            )

            sensor = self.world.spawn_actor(bp, transform, attach_to=self.ego_vehicle)
            self.sensors[sensor_name] = sensor
            self._events[sensor_name] = threading.Event()

            weak_self = weakref.ref(self)
            sensor.listen(
                lambda data, name=sensor_name, ws=weak_self: SensorManager._on_data(ws, name, data)
            )

            print(f"  [Sensor] Spawned {sensor_name} ({sensor_cfg['blueprint']})")

    @staticmethod
    def _on_data(weak_self, sensor_name: str, data):
        """
        Callback invoked by CARLA when sensor data is ready.
        Runs in a CARLA worker thread — must be fast and thread-safe.
        """
        self = weak_self()
        if self is None:
            return

        with self._lock:
            frame = data.frame

            if sensor_name == "rgb":
                # BGRA uint8 → RGB uint8
                array = np.frombuffer(data.raw_data, dtype=np.uint8)
                array = array.reshape((data.height, data.width, 4))
                array = array[:, :, :3][:, :, ::-1].copy()  # BGRA→RGB
                self._data_buffer["rgb"] = array
                self._data_buffer["frame"] = frame
                self._data_buffer["timestamp"] = data.timestamp

            elif sensor_name == "depth":
                # CARLA encodes depth across RGB channels in BGRA format
                # Decode: depth_meters = (R + G*256 + B*256*256) / (256^3 - 1) * 1000
                array = np.frombuffer(data.raw_data, dtype=np.uint8)
                array = array.reshape((data.height, data.width, 4))
                # In BGRA layout: index 2=R, index 1=G, index 0=B
                r = array[:, :, 2].astype(np.float32)
                g = array[:, :, 1].astype(np.float32)
                b = array[:, :, 0].astype(np.float32)
                # Normalized depth [0, 1] then scaled to meters (max 1000m)
                depth_meters = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0) * 1000.0
                self._data_buffer["depth"] = depth_meters

            elif sensor_name == "instance_segmentation":
                # CARLA instance seg BGRA layout:
                #   B channel (index 0) = semantic tag
                #   G channel (index 1) = high byte of instance ID
                #   R channel (index 2) = low byte of instance ID
                array = np.frombuffer(data.raw_data, dtype=np.uint8)
                array = array.reshape((data.height, data.width, 4))
                instance_id = array[:, :, 2].astype(np.int32) + \
                              array[:, :, 1].astype(np.int32) * 256
                self._data_buffer["instance_seg"] = instance_id

            elif sensor_name == "semantic_segmentation":
                # Semantic tag is in the R channel (index 2 in BGRA)
                array = np.frombuffer(data.raw_data, dtype=np.uint8)
                array = array.reshape((data.height, data.width, 4))
                self._data_buffer["semantic_seg"] = array[:, :, 2].copy()

        self._events[sensor_name].set()

    def get_data(self, timeout: float = 10.0) -> SensorData:
        """
        Block until all sensors have reported data for the current frame.

        Returns:
            SensorData with all sensor outputs for this frame.

        Raises:
            TimeoutError if any sensor fails to report within timeout.
        """
        for name, event in self._events.items():
            if not event.wait(timeout):
                raise TimeoutError(
                    f"Sensor '{name}' did not deliver data within {timeout}s. "
                    f"Check if CARLA server is running in synchronous mode."
                )

        with self._lock:
            data = SensorData(
                frame=self._data_buffer.get("frame", -1),
                timestamp=self._data_buffer.get("timestamp", 0.0),
                rgb=self._data_buffer.get("rgb", np.array([])),
                depth=self._data_buffer.get("depth", np.array([])),
                instance_seg=self._data_buffer.get("instance_seg", np.array([])),
                semantic_seg=self._data_buffer.get("semantic_seg", np.array([])),
            )

        for event in self._events.values():
            event.clear()

        return data

    def get_camera_transform(self) -> carla.Transform:
        """Get the world-space transform of the RGB camera (used for projection)."""
        return self.sensors["rgb"].get_transform()

    def destroy(self):
        """Stop listening and destroy all sensors."""
        for name, sensor in self.sensors.items():
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
                print(f"  [Sensor] Destroyed {name}")
        self.sensors.clear()
        self._events.clear()