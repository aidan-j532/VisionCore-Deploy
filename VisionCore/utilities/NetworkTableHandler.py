import ntcore
import logging
import numpy as np
from VisionCore.trackers.Fuel import Fuel
import time
import dataclasses
import wpiutil.wpistruct
from ntcore import NetworkTableInstance
from wpimath.geometry import Pose2d, Rotation2d

@wpiutil.wpistruct.make_wpistruct(name="Fuel")
@dataclasses.dataclass
class FuelStruct:
    x: float
    y: float

class NetworkTableHandler:
    def __init__(self, ip: str):
        self.ip = ip
        self.logger = logging.getLogger(__name__)
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.inst.setServer(self.ip)
        self.inst.startClient4("CustomVisionStuff")

        for i in range(15):
            if self.inst.isConnected():
                break
            self.logger.warning("NetworkTables not connected, retrying… (%d/15)", i + 1)
            time.sleep(1)
        else:
            self.logger.error("NetworkTables could not connect after 15 s.")

        self._subscribers: dict = {}
        self._tables: dict = {}

    def isConnected(self) -> bool:
        return self.inst.isConnected()

    def _get_table(self, table_name: str):
        if table_name not in self._tables:
            self._tables[table_name] = self.inst.getTable(table_name)
        return self._tables[table_name]

    def get_robot_pose(self) -> Pose2d:
        try:
            if not self.isConnected():
                return Pose2d()

            table_name = "AdvantageKit/RealOutputs/Odometry"
            data_name  = "Robot"
            sub_key    = f"{table_name}/{data_name}"

            if sub_key not in self._subscribers:
                table = self._get_table(table_name)
                self._subscribers[sub_key] = table.getStructTopic(data_name, Pose2d).subscribe(Pose2d())

            return self._subscribers[sub_key].get()
        except Exception as e:
            self.logger.error("Failed to get robot pose: %s", e)
            return Pose2d()

    def send_fuel_list(
        self,
        fuels: list[Fuel],
        data_name: str = "fuel_data",
        table_name: str = "VisionData",
    ):
        try:
            if not self.isConnected():
                return

            table      = self._get_table(table_name)
            pub_key    = f"pub/{table_name}/{data_name}"
            struct_list = [FuelStruct(x=float(f.get_position_normally()[0]),
                                      y=float(f.get_position_normally()[1]))
                           for f in fuels]

            if pub_key not in self._subscribers:
                self._subscribers[pub_key] = (
                    table.getStructArrayTopic(data_name, FuelStruct).publish()
                )

            self._subscribers[pub_key].set(struct_list)
            table.putNumber("timestamp_ms", time.time() * 1000)
            self.inst.flush()
            self.logger.info("Sent %d fuels via StructArray", len(struct_list))
        except Exception as e:
            self.logger.error("Failed to send fuel structs: %s", e)

    def send_boolean(self, value: bool, data_name: str, table_name: str):
        try:
            if not self.isConnected():
                return
            pub_key = f"pub/{table_name}/{data_name}"
            if pub_key not in self._subscribers:
                self._subscribers[pub_key] = (
                    self._get_table(table_name).getBooleanTopic(data_name).publish()
                )
            self._subscribers[pub_key].set(value)
            self.inst.flush()
        except Exception as e:
            self.logger.error("Failed to send boolean: %s", e)

    def send_data(self, value: bool | int | float | str, data_name: str, table_name: str):
        try:
            if not self.isConnected():
                return

            table   = self._get_table(table_name)
            pub_key = f"pub/{table_name}/{data_name}"

            if pub_key not in self._subscribers:
                if isinstance(value, bool):
                    pub = table.getBooleanTopic(data_name).publish()
                elif isinstance(value, (int, float)):
                    pub = table.getDoubleTopic(data_name).publish()
                elif isinstance(value, str):
                    pub = table.getStringTopic(data_name).publish()
                else:
                    self.logger.error("Unsupported type for %s: %s", data_name, type(value))
                    return
                self._subscribers[pub_key] = pub

            self._subscribers[pub_key].set(value)
            self.inst.flush()
        except Exception as e:
            self.logger.error("Failed to send data: %s", e)

    def get_data(self, data_type, data_name: str, table_name: str):
        if not self.isConnected():
            return [0.0, 0.0]

        sub_key = f"{table_name}/{data_name}"
        if sub_key not in self._subscribers:
            table = self._get_table(table_name)
            if isinstance(data_type, (list, np.ndarray)):
                self._subscribers[sub_key] = table.getDoubleArrayTopic(data_name).subscribe([])
            elif isinstance(data_type, (int, float)):
                self._subscribers[sub_key] = table.getDoubleTopic(data_name).subscribe(0.0)
            elif isinstance(data_type, str):
                self._subscribers[sub_key] = table.getStringTopic(data_name).subscribe("")

        return self._subscribers[sub_key].get()