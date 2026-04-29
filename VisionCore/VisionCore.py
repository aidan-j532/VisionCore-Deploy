from VisionCore.utilities.MultipleCameraHandler import MultipleCameraHandler
from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera
from VisionCore.trackers.PathPlanner import PathPlanner
from VisionCore.utilities.NetworkTableHandler import NetworkTableHandler
import time
from VisionCore.web.CameraApp import CameraApp
import threading
import logging
import os
import numpy as np
from VisionCore.trackers.Fuel import Fuel
from VisionCore.trackers.FuelTracker import FuelTracker
from VisionCore.web.Metrics import Metrics
from VisionCore.web.healthReporter import HealthReporter
from VisionCore.utilities.VideoRecorder import VideoRecorder
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
import signal

try:
    from rknnlite.api import RKNNLite
    RKNN_FOUND = True
except ImportError:
    RKNN_FOUND = False

class VisionCore:
    def __init__(self, cameras: list[ObjectDetectionCamera], config: VisionCoreConfig):
        self.cameras = cameras
        self.config  = config
        self.shutdown_event = threading.Event()

        os.makedirs("Outputs", exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, config.get("log_level") or "INFO", logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filemode="w",
            filename=config.get("log_file") or "Outputs/log.txt",
        )
        self.logger = logging.getLogger(__name__)

        signal.signal(signal.SIGINT,  lambda *_: self.shutdown_event.set())
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown_event.set())

        self.metrics = Metrics() if config["metrics"] else None

        self.camera_app = CameraApp(cameras=cameras, config=config) if config["app_mode"] else None
        self.health     = HealthReporter(self.camera_app.app, config) if config["app_mode"] else None
        self.network_handler = (
            NetworkTableHandler(config["network_tables_ip"])
            if config["use_network_tables"] else None
        )

        if config["app_mode"]:
            threading.Thread(target=self.camera_app.run, daemon=True).start()
            if self.health and cameras:
                self.health.set_camera(cameras[0])
            if self.network_handler and self.health:
                self.health.set_network_handler(self.network_handler)

        if len(cameras) == 0:
            self.logger.warning("No cameras provided — vision will not run.")
            self.camera_handler = None
        elif len(cameras) == 1:
            self.logger.info("Single camera mode.")
            self.camera_handler = None
        else:
            self.logger.info("%d cameras — multi mode.", len(cameras))
            self.camera_handler = MultipleCameraHandler(cameras)

        self.planner      = PathPlanner(config)
        self.fuel_tracker = FuelTracker(config)
        self.recorder     = VideoRecorder(output_dir="VideoRecordings") if config["record_mode"] else None

    def get_default_config(self):
        return self.config.get_default_config()

    def _record_metrics(self, **kwargs):
        if self.metrics:
            self.metrics.record(**kwargs)

    def _tick_metrics(self):
        if self.metrics:
            self.metrics.tick()

    def _destroy_metrics(self):
        if self.metrics:
            self.metrics.destroy()

    def numpy_to_fuel_list(self, positions: np.ndarray) -> list[Fuel]:
        return [Fuel(float(p[0]), float(p[1])) for p in positions]

    def run_multi_vision(self, handler: MultipleCameraHandler):
        try:
            raw = handler.predict()
            return self.numpy_to_fuel_list(raw), handler.get_combined_frame()
        except Exception:
            self.logger.exception("Multi-vision exception")
            return [], None

    def run_solo_vision(self, camera: ObjectDetectionCamera):
        try:
            raw, frame = camera.run()
            return self.numpy_to_fuel_list(raw), frame
        except Exception:
            self.logger.exception("Solo-vision exception")
            return [], None

    def run(self, duration_s: float | None = None):
        if not self.cameras:
            self.logger.error("No cameras provided.")
            return

        if duration_s is not None:
            def _stop():
                time.sleep(duration_s)
                self.logger.info("Duration %.1fs reached — stopping.", duration_s)
                self.shutdown_event.set()
            threading.Thread(target=_stop, daemon=True).start()

        if len(self.cameras) == 1:
            self.run_solo_mode()
        else:
            self.run_multi_mode()

    def run_solo_mode(self):
        camera = self.cameras[0]
        try:
            if self.recorder:
                self.recorder.start(camera.input_size[0], camera.input_size[1])

            self.logger.info("Solo mode — warming up…")
            self.run_solo_vision(camera)
            self.logger.info("Warm-up complete.")

            while not self.shutdown_event.is_set():
                t0 = time.perf_counter()

                camera_lag_s = camera.get_frame_age()

                t_vis = time.perf_counter()
                fuel_list, annotated_frame = self.run_solo_vision(camera)
                vision_s = time.perf_counter() - t_vis

                if self.network_handler:
                    pose = self.network_handler.get_robot_pose()
                    fuel_list = self.fuel_tracker.update(
                        fuel_list, pose.X(), pose.Y(), pose.rotation().radians()
                    )
                else:
                    fuel_list = self.fuel_tracker.update(fuel_list, 0, 0, 0)

                flask_s = None
                if self.camera_app and annotated_frame is not None:
                    t_f = time.perf_counter()
                    cam_name = (camera.config.get("name", "Camera 1")
                                if hasattr(camera, "config") else "Camera 1")
                    self.camera_app.set_frame(annotated_frame, camera_name=cam_name)
                    flask_s = time.perf_counter() - t_f

                if self.recorder and annotated_frame is not None:
                    self.recorder.write(annotated_frame)

                loop_s = time.perf_counter() - t0

                if not fuel_list:
                    self._record_metrics(loop_s=loop_s, vision_s=vision_s,
                                         camera_lag_s=camera_lag_s, flask_s=flask_s)
                    self._tick_metrics()
                    print(f"\rFPS: {1/loop_s:.1f} (no detections)   ", end="")
                    continue

                _, fuel_list = self.planner.update_fuel_positions(fuel_list)

                network_s = None
                if self.network_handler:
                    t_n = time.perf_counter()
                    self.network_handler.send_fuel_list(fuel_list, "vision_data", "VisionData")
                    self.network_handler.send_data(1 / loop_s if loop_s > 0 else 0, "fps", "VisionData")
                    self.network_handler.send_data(len(fuel_list), "num_detections", "VisionData")
                    self.network_handler.send_data(camera_lag_s, "camera_lag", "VisionData")

                    hopper = camera.get_data_for_subsystem("hopper")
                    if hopper is not None:
                        self.network_handler.send_boolean(hopper, "hopper_sees_object", "VisionData")
                    network_s = time.perf_counter() - t_n

                loop_s = time.perf_counter() - t0

                health_s = None
                if self.health:
                    t_h = time.perf_counter()
                    self.health.tick(fps=1 / loop_s if loop_s > 0 else 0,
                                     vision_s=vision_s, detections=len(fuel_list))
                    health_s = time.perf_counter() - t_h

                self._record_metrics(loop_s=loop_s, vision_s=vision_s,
                                     camera_lag_s=camera_lag_s, flask_s=flask_s,
                                     network_s=network_s, health_s=health_s)
                self._tick_metrics()
                self.logger.debug("FPS: %.1f", 1 / loop_s)
                print(f"\rFPS: {1/loop_s:.1f}   ", end="")

        finally:
            camera.destroy()
            self._destroy_metrics()

    def run_multi_mode(self):
        handler = self.camera_handler
        if handler is None:
            self.logger.error("Multi-camera mode requested but camera handler failed to initialize.")
            return
        try:
            if self.recorder:
                h, w = handler.cameras[0].input_size[1], handler.cameras[0].input_size[0]
                self.recorder.start(w, h)

            self.logger.info("Multi mode. warming up…")
            self.run_multi_vision(handler)
            self.logger.info("Warm-up complete.")

            while not self.shutdown_event.is_set():
                t0 = time.perf_counter()

                ages = [cam.get_frame_age() for cam in handler.cameras]
                camera_lag_s = sum(ages) / len(ages) if ages else 0.0

                t_vis = time.perf_counter()
                fuel_list, combined_frame = self.run_multi_vision(handler)
                vision_s = time.perf_counter() - t_vis

                if self.network_handler:
                    pose = self.network_handler.get_robot_pose()
                    fuel_list = self.fuel_tracker.update(
                        fuel_list, pose.X(), pose.Y(), pose.rotation().radians()
                    )
                else:
                    fuel_list = self.fuel_tracker.update(fuel_list, 0, 0, 0)

                flask_s = None
                if self.camera_app and combined_frame is not None:
                    t_f = time.perf_counter()
                    # Set the combined frame (default feed) …
                    self.camera_app.set_frame(combined_frame)
                    # and set per-camera frames from the already-computed cache
                    # (MultipleCameraHandler stores the last frame per camera).
                    for i, cam in enumerate(handler.cameras):
                        cam_name = (cam.config.get("name", f"Camera {i+1}")
                                    if hasattr(cam, "config") else f"Camera {i+1}")
                        with handler._locks[i]:
                            cached_frame = handler._frames[i]
                        if cached_frame is not None:
                            self.camera_app.set_frame(cached_frame.copy(), camera_name=cam_name)
                    flask_s = time.perf_counter() - t_f

                loop_s = time.perf_counter() - t0

                if not fuel_list:
                    self._record_metrics(loop_s=loop_s, vision_s=vision_s,
                                         camera_lag_s=camera_lag_s, flask_s=flask_s)
                    self._tick_metrics()
                    print(f"\rFPS: {1/loop_s:.1f} (no detections)   ", end="")
                    continue

                _, fuel_list = self.planner.update_fuel_positions(fuel_list)

                network_s = None
                if self.network_handler:
                    t_n = time.perf_counter()
                    self.network_handler.send_fuel_list(fuel_list, "vision_data", "VisionData")
                    self.network_handler.send_data(1 / loop_s if loop_s > 0 else 0, "fps", "VisionData")
                    self.network_handler.send_data(len(fuel_list), "num_detections", "VisionData")
                    self.network_handler.send_data(camera_lag_s, "camera_lag", "VisionData")

                    for cam in handler.cameras:
                        hopper = cam.get_data_for_subsystem("hopper")
                        if hopper is not None:
                            self.network_handler.send_boolean(hopper, "hopper_sees_object", "VisionData")
                    network_s = time.perf_counter() - t_n

                loop_s = time.perf_counter() - t0

                health_s = None
                if self.health:
                    t_h = time.perf_counter()
                    self.health.tick(fps=1 / loop_s if loop_s > 0 else 0,
                                     vision_s=vision_s, detections=len(fuel_list))
                    health_s = time.perf_counter() - t_h

                self._record_metrics(loop_s=loop_s, vision_s=vision_s,
                                     camera_lag_s=camera_lag_s, flask_s=flask_s,
                                     network_s=network_s, health_s=health_s)
                self._tick_metrics()
                self.logger.debug("FPS: %.1f", 1 / loop_s)
                print(f"\rFPS: {1/loop_s:.1f}   ", end="")
        finally:
            handler.destroy()
            self._destroy_metrics()
