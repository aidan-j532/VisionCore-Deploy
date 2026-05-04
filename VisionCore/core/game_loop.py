from pathlib import Path
import importlib.metadata
from VisionCore.VisionCore import VisionCore
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.validations.ez import unit_tests
import os

def main():
    config_path = (
        os.environ.get("VISIONCORE_CONFIG")
        or Path.cwd() / "config.json"
    )
    config = VisionCoreConfig(str(config_path))
    print("Loading configuration from:", config_path)

    # Load vision modules dynamically
    vision_entries = importlib.metadata.entry_points(group='visioncore_vision')
    vision_classes = {ep.name: ep.load() for ep in vision_entries}

    cameras = []
    for cam_name in config.camera_configs:
        cam_config = config.camera_config(cam_name)
        pipeline = cam_config.get('pipeline', 'object')
        if pipeline in vision_classes:
            vision_class = vision_classes[pipeline]
            camera = vision_class(cam_config, config)
            cameras.append(camera)
        else:
            config.logger.warning(f"Unknown vision pipeline: {pipeline}")

    vision = VisionCore(cameras, config)
    vision.run()

if __name__ == "__main__":
    if not unit_tests():
        raise SystemExit("Unit tests failed")
    main()