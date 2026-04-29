from pathlib import Path
from VisionCore import VisionCore
from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.validations.ez import unit_tests

def main():
    config_path = Path(__file__).parent / "config.json"
    config = VisionCoreConfig(str(config_path))

    cameras = [
        ObjectDetectionCamera(
            config.camera_config("Microsoft Cinema"),
            config
        )
    ]

    vision = VisionCore(cameras, config)
    vision.run()

if __name__ == "__main__":
    if not unit_tests():
        raise SystemExit("Unit tests failed")
    main()