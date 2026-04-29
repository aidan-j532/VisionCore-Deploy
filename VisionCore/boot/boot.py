import sys
import os
import logging
import subprocess
from pathlib import Path

_BOOT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BOOT_DIR.parents[1] # two levels up
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from VisionCore.config.AutoOpt import recommend_format
from VisionCore.validations.validate_system import validate_system
from VisionCore.config.VisionCoreConfig import VisionCoreConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORMAT_EXTENSIONS = {
    "onnx":      ".onnx",
    "openvino":  ".xml",
    "rknn":      ".rknn",
    "tflite":    ".tflite",
    "coreml":    ".mlpackage",
}

def search_for_config() -> str:
    config_dir = _REPO_ROOT / "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"config directory not found at {config_dir}")

    config_files = list(config_dir.rglob("*.json"))
    if not config_files:
        raise FileNotFoundError("No .json config files found in config/")

    chosen = str(config_files[0])
    logger.info("Found config files: %s  →  using %s", config_files, chosen)
    return chosen

def on_boot():
    logger.info("Starting VisionCore boot sequence…")

    # 1. Validate system
    if not validate_system():
        raise RuntimeError("System validation failed. Aborting boot.")
    logger.info("System validation passed.")

    # 2. Load config
    config_file = os.environ.get("VISIONCORE_CONFIG") or str(search_for_config())
    config = VisionCoreConfig(config_file)
    logger.info("Loaded config from %s", config_file)

    # 3. Auto-optimization
    if config.get("auto_opt"):
        best_format = recommend_format()
        logger.info("Auto-opt enabled. Recommended format: %s", best_format)

        extension = FORMAT_EXTENSIONS.get(best_format)
        if not extension:
            raise ValueError(f"No extension mapping for format: {best_format}")

        model_dir = _REPO_ROOT / "YoloModels"
        optimized = list(model_dir.rglob(f"*{extension}"))

        if optimized:
            chosen = str(optimized[0])
            logger.info("Found optimised model(s): %s  →  using %s",
                        [str(m) for m in optimized], chosen)
            config.set("model_path", chosen)
        else:
            logger.warning("No %s models found in YoloModels/. Falling back to config path.", best_format)
    else:
        logger.info("Auto-opt disabled.")

    model_path = config.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("Boot sequence complete. Model: %s", model_path)

    # 4. Install service using the same Python interpreter that launched boot.py
    install_script = str(_BOOT_DIR / "install.py")
    try:
        subprocess.run([sys.executable, install_script], check=True, cwd=str(_REPO_ROOT))
    except subprocess.CalledProcessError as e:
        logger.error("Failed to run install.py: %s", e)
        raise RuntimeError("Boot failed during service installation.")

if __name__ == "__main__":
    on_boot()