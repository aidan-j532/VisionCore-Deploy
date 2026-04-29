import json
import logging

class VisionCoreConfig:
    def __init__(self, file_path: str = None):
        self.logger = logging.getLogger(__name__)

        self.default_config = {
            "unit": "meter",
            "dbscan": {"elipson": 0, "min_samples": 0},
            "distance_threshold": 0.5,
            "vision_model_input_size": [640, 640],
            "vision_model_file_path": "model.pt",
            "network_tables_ip": "10.22.7.2",
            "use_network_tables": True,
            "app_mode": True,
            "debug_mode": False,
            "record_mode": True,
            "stale_threshold": 1.0,
            "log_level": "INFO",
            "log_file": "Outputs/log.txt",
            "metrics": False,
            "camera_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "dist_coeffs": [0, 0, 0, 0, 0],
            "camera_configs": {
                "default": {
                    "name": "default",
                    "x": 0, "y": 0, "height": 0, "pitch": 0, "yaw": 0,
                    "grayscale": False,
                    "fps_cap": -1,
                    "calibration": {"size": 0, "distance": 0, "game_piece_size": 0, "fov": 0},
                    "source": "/dev/video0",
                    "subsystem": "field",
                    "pipeline": "object",
                },
            },
            "vision_model": {
                "quantized": False,
                "file_path": "model.pt",
                "input_size": [640, 640],
            },
            "auto_opt": True,
        }
        self.config = json.loads(json.dumps(self.default_config))  # deep copy

        if file_path:
            self.load_from_file(file_path)

        self.camera_configs: dict[str, VisionCoreCameraConfig] = {
            name: VisionCoreCameraConfig(cam_cfg)
            for name, cam_cfg in self.config["camera_configs"].items()
        }

        self._check_config()

    def _check_config(self):
        if self.config == self.default_config:
            self.logger.warning(
                "Using default configuration. Load a config file for proper operation."
            )
        else:
            if self.config.get("vision_model") and self.config.get("april_tag"):
                self.logger.warning(
                    "Both vision_model and april_tag configs present — ensure this is intentional."
                )

    def get_default_config(self) -> dict:
        return self.default_config

    def camera_config(self, cam_name: str) -> "VisionCoreCameraConfig":
        cfg = self.camera_configs.get(cam_name)
        if cfg is None:
            raise KeyError(f"No camera config named '{cam_name}'. "
                           f"Available: {list(self.camera_configs)}")
        return cfg

    def load_from_file(self, file_path: str):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self._update_config(data)
        except Exception as e:
            self.logger.warning("Failed to load config from %s: %s", file_path, e)
            self.logger.info("Using default configuration.")

    def get(self, *keys):
        val = self.config
        try:
            for key in keys:
                val = val[key]
            return val
        except (KeyError, TypeError):
            self.logger.warning("Key path %s not found in config.", keys)
            return None

    def set(self, *keys_and_value):
        if len(keys_and_value) < 2:
            return
        *keys, value = keys_and_value
        target = self.config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def _update_config(self, data: dict, current_dict: dict = None):
        if current_dict is None:
            current_dict = self.config
        for key, value in data.items():
            if (
                isinstance(value, dict)
                and key in current_dict
                and isinstance(current_dict[key], dict)
            ):
                self._update_config(value, current_dict[key])
            else:
                current_dict[key] = value

    def __getitem__(self, args):
        if isinstance(args, tuple):
            return self.get(*args)
        return self.get(args)

    def __call__(self, *keys):
        return self.get(*keys)

    def __getattr__(self, item: str):
        if item.startswith("_") or item in {"config", "logger", "default_config", "camera_configs"}:
            raise AttributeError(item)
        val = self.get(item)
        if val is None:
            raise AttributeError(f"No config attribute or key named '{item}'")
        return val

class VisionCoreCameraConfig:
    DEFAULTS = {
        "name":       "default",
        "x":          0,
        "y":          0,
        "height":     0,
        "pitch":      0,
        "yaw":        0,
        "grayscale":  False,
        "fps_cap":    30,
        "calibration": {"size": 0, "distance": 0, "game_piece_size": 0, "fov": 0},
        "source":     "/dev/video0",
        "subsystem":  "field",
    }

    def __init__(self, config_dict: dict = None):
        import json
        self.data = json.loads(json.dumps(self.DEFAULTS)) # deep copy
        if config_dict:
            self.data.update(config_dict)

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __contains__(self, key):
        return key in self.data