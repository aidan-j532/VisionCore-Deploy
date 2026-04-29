import numpy as np
from .CustomDBScan import CustomDBScan
from .Fuel import Fuel  # adjust import path as needed
from VisionCore.config.VisionCoreConfig import VisionCoreConfig

class PathPlanner:
    def __init__(self, config: VisionCoreConfig):
        self.epsilon = config["dbscan"]["elipson"]
        self.min_samples = config["dbscan"]["min_samples"]

        self.fuel_positions, self.noise_positions = self.dbscan([])

    def get_noise_positions(self) -> list[Fuel]:
        return self.noise_positions

    def update_fuel_positions(self, fuel_positions: list[Fuel]):
        self.fuel_positions, self.noise_positions = self.dbscan(fuel_positions)
        return self.noise_positions, self.fuel_positions

    def get_fuel_positions(self) -> list[Fuel]:
        return self.fuel_positions

    def dbscan(self, fuels: list[Fuel]) -> tuple[list[Fuel], list[Fuel]]:
        if len(fuels) == 0:
            return [], []

        # Extract positions as numpy array for DBSCAN
        points = np.array([f.get_position() for f in fuels])

        dbscan = CustomDBScan(points, eps=self.epsilon, samples=self.min_samples)
        labels = dbscan.get_dbscan()

        cleaned = [f for f, label in zip(fuels, labels) if label != -1]
        noise   = [f for f, label in zip(fuels, labels) if label == -1]

        return cleaned, noise