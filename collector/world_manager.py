"""
Traffic and Weather management for CARLA data collection.

Handles:
- Spawning NPC vehicles with traffic manager control
- Spawning NPC pedestrians with AI controllers
- Weather preset cycling for domain diversity
"""

import random
import logging
from typing import Optional

import carla

logger = logging.getLogger(__name__)


# Map config preset names to CARLA weather presets
WEATHER_PRESETS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "ClearNight": carla.WeatherParameters(
        sun_altitude_angle=-30.0,
        cloudiness=10.0,
        precipitation=0.0,
        fog_density=0.0,
    ),
    "CloudyNight": carla.WeatherParameters(
        sun_altitude_angle=-30.0,
        cloudiness=70.0,
        precipitation=0.0,
        fog_density=5.0,
    ),
}


class TrafficManager:
    """Spawns and manages NPC vehicles and pedestrians."""

    def __init__(self, client: carla.Client, world: carla.World, config: dict):
        self.client = client
        self.world = world
        self.config = config
        self.traffic_config = config.get("traffic", {})

        self.vehicles: list[carla.Vehicle] = []
        self.walkers: list[carla.Walker] = []
        self.walker_controllers: list[carla.WalkerAIController] = []

        # Get CARLA's traffic manager
        tm_port = config.get("ego_vehicle", {}).get("traffic_manager_port", 8000)
        self.tm = client.get_trafficmanager(tm_port)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(42)

    def spawn_vehicles(self) -> int:
        """Spawn NPC vehicles at random spawn points with traffic manager control."""
        num_vehicles = self.traffic_config.get("num_vehicles", 80)
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if len(spawn_points) < num_vehicles:
            logger.warning(
                f"Requested {num_vehicles} vehicles but only {len(spawn_points)} "
                f"spawn points available. Spawning {len(spawn_points)}."
            )
            num_vehicles = len(spawn_points)

        random.shuffle(spawn_points)

        # Filter to 4-wheeled vehicles only (no bikes for NPC, they behave erratically)
        vehicle_bps = []
        for bp in blueprint_library.filter("vehicle.*"):
            if bp.has_attribute("number_of_wheels"):
                if int(bp.get_attribute("number_of_wheels")) >= 4:
                    vehicle_bps.append(bp)
            else:
                # CARLA 0.10 may not expose this attribute — include all vehicles
                vehicle_bps.append(bp)

        batch = []
        for i in range(num_vehicles):
            bp = random.choice(vehicle_bps)
            # Randomize color if available
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            # Set driver_id for determinism
            if bp.has_attribute("driver_id"):
                driver_id = random.choice(bp.get_attribute("driver_id").recommended_values)
                bp.set_attribute("driver_id", driver_id)

            batch.append(
                carla.command.SpawnActor(bp, spawn_points[i])
                    .then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm.get_port()))
            )

        results = self.client.apply_batch_sync(batch, True)
        for result in results:
            if not result.error:
                vehicle = self.world.get_actor(result.actor_id)
                if vehicle is not None:
                    self.vehicles.append(vehicle)

        # Configure traffic behaviors
        ignore_lights_pct = self.traffic_config.get("ignore_lights_pct", 5)
        ignore_signs_pct = self.traffic_config.get("ignore_signs_pct", 5)

        for v in self.vehicles:
            self.tm.ignore_lights_percentage(v, ignore_lights_pct)
            self.tm.ignore_signs_percentage(v, ignore_signs_pct)
            # Add some speed variation for realism
            self.tm.vehicle_percentage_speed_difference(
                v, random.uniform(-20, 20)
            )

        logger.info(f"Spawned {len(self.vehicles)} NPC vehicles")
        return len(self.vehicles)

    def spawn_walkers(self) -> int:
        """Spawn NPC pedestrians with AI controllers."""
        num_walkers = self.traffic_config.get("num_walkers", 40)
        blueprint_library = self.world.get_blueprint_library()

        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        # Get random spawn points on sidewalks
        spawn_points = []
        for _ in range(num_walkers * 2):  # oversample to account for failures
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_points.append(carla.Transform(location=loc))
            if len(spawn_points) >= num_walkers:
                break

        # Spawn walkers
        batch = []
        for sp in spawn_points[:num_walkers]:
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            batch.append(carla.command.SpawnActor(bp, sp))

        walker_results = self.client.apply_batch_sync(batch, True)

        # Spawn AI controllers for each walker
        controller_batch = []
        valid_walkers = []
        for result in walker_results:
            if not result.error:
                walker = self.world.get_actor(result.actor_id)
                if walker is not None:
                    valid_walkers.append(walker)
                    controller_batch.append(
                        carla.command.SpawnActor(
                            controller_bp, carla.Transform(), walker
                        )
                    )

        controller_results = self.client.apply_batch_sync(controller_batch, True)

        # Start AI controllers
        self.world.tick()  # ensure controllers are spawned

        controllers = []
        for result in controller_results:
            if not result.error:
                controller = self.world.get_actor(result.actor_id)
                if controller is not None:
                    controllers.append(controller)

        for controller in controllers:
            # Set random destination
            dest = self.world.get_random_location_from_navigation()
            if dest is not None:
                controller.start()
                controller.go_to_location(dest)
                controller.set_max_speed(1.0 + random.random() * 1.5)

        self.walkers = valid_walkers
        self.walker_controllers = controllers

        logger.info(f"Spawned {len(self.walkers)} pedestrians with AI controllers")
        return len(self.walkers)

    def destroy(self):
        """Clean up all spawned NPCs."""
        # Stop walker controllers first
        for controller in self.walker_controllers:
            try:
                controller.stop()
            except Exception:
                pass

        # Destroy all actors
        all_actors = self.vehicles + self.walkers + self.walker_controllers
        if all_actors:
            ids = [a.id for a in all_actors if a is not None]
            self.client.apply_batch([carla.command.DestroyActor(x) for x in ids])

        self.vehicles.clear()
        self.walkers.clear()
        self.walker_controllers.clear()
        logger.info("Destroyed all NPC actors")


class WeatherManager:
    """Manages weather variation during data collection."""

    def __init__(self, world: carla.World, config: dict):
        self.world = world
        self.schedule = config.get("weather_schedule", {})
        self.presets = self.schedule.get("presets", [])

        # Build weighted distribution
        self.names = [p["name"] for p in self.presets]
        self.weights = [p["weight"] for p in self.presets]

    def sample_weather(self) -> str:
        """Randomly sample a weather preset according to configured weights."""
        name = random.choices(self.names, weights=self.weights, k=1)[0]
        self.set_weather(name)
        return name

    def set_weather(self, preset_name: str):
        """Apply a named weather preset to the world."""
        if preset_name in WEATHER_PRESETS:
            self.world.set_weather(WEATHER_PRESETS[preset_name])
            logger.info(f"Weather set to: {preset_name}")
        else:
            logger.warning(f"Unknown weather preset: {preset_name}")

    def interpolate_weather(self, from_name: str, to_name: str, alpha: float):
        """
        Smoothly interpolate between two weather presets.
        Useful for gradual transitions during long collection runs.
        """
        if from_name not in WEATHER_PRESETS or to_name not in WEATHER_PRESETS:
            return

        w1 = WEATHER_PRESETS[from_name]
        w2 = WEATHER_PRESETS[to_name]

        # Interpolate numeric parameters
        weather = carla.WeatherParameters(
            cloudiness=w1.cloudiness + alpha * (w2.cloudiness - w1.cloudiness),
            precipitation=w1.precipitation + alpha * (w2.precipitation - w1.precipitation),
            precipitation_deposits=w1.precipitation_deposits + alpha * (w2.precipitation_deposits - w1.precipitation_deposits),
            wind_intensity=w1.wind_intensity + alpha * (w2.wind_intensity - w1.wind_intensity),
            sun_azimuth_angle=w1.sun_azimuth_angle + alpha * (w2.sun_azimuth_angle - w1.sun_azimuth_angle),
            sun_altitude_angle=w1.sun_altitude_angle + alpha * (w2.sun_altitude_angle - w1.sun_altitude_angle),
            fog_density=w1.fog_density + alpha * (w2.fog_density - w1.fog_density),
            fog_distance=w1.fog_distance + alpha * (w2.fog_distance - w1.fog_distance),
            wetness=w1.wetness + alpha * (w2.wetness - w1.wetness),
        )
        self.world.set_weather(weather)
