import numpy as np
from .Fuel import Fuel
from VisionCore.config.VisionCoreConfig import VisionCoreConfig

# 0.3 means the tracked position moves 30% toward each new detectio, smooth but responsive.
_EMA_ALPHA = 0.3

class FuelTracker:
    def __init__(self, config: VisionCoreConfig):
        self.fuel_list: list[Fuel] = []

        raw_threshold = config["distance_threshold"]
        if raw_threshold is None or raw_threshold < 0:
            self.logger_warning = True
            self.distance_threshold = 0.5
        else:
            self.logger_warning = False
            self.distance_threshold = float(raw_threshold)

        self.stale_threshold = config.get("stale_threshold") or 1.0

        import logging
        self.logger = logging.getLogger(__name__)
        if self.logger_warning:
            self.logger.warning(
                "distance_threshold is negative or unset in config; "
                "defaulting to 0.5 m to prevent unbounded fuel list growth."
            )

    def update(
        self,
        new_fuel_list: list[Fuel],
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> list[Fuel]:
        for fuel in self.fuel_list:
            fuel.update()
        self.fuel_list = [f for f in self.fuel_list if not f.destroyed]

        for fuel in new_fuel_list:
            fuel.relative_to(robot_x, robot_y, robot_yaw)

        self._merge(new_fuel_list)
        return self.fuel_list

    def _merge(self, fuels: list[Fuel]):
        for fuel in fuels:
            if not self._already_exists(fuel):
                fuel.alive_time = self.stale_threshold
                self.fuel_list.append(fuel)

    def _already_exists(self, new_fuel: Fuel) -> bool:
        if not self.fuel_list:
            return False
        new_pos = np.array(new_fuel.get_position())
        for existing in self.fuel_list:
            if np.linalg.norm(new_pos - np.array(existing.get_position())) < self.distance_threshold:
                existing.reset_time()
                existing.x = existing.x + _EMA_ALPHA * (new_fuel.x - existing.x)
                existing.y = existing.y + _EMA_ALPHA * (new_fuel.y - existing.y)
                return True
        return False

    def get_fuel_list(self) -> list[Fuel]:
        return self.fuel_list