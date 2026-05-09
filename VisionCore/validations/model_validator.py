import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# File extensions by format
MODEL_FORMATS = {
    "pytorch": [".pt"],
    "openvino": [".xml", ".bin"],
    "rknn": [".rknn"],
    "onnx": [".onnx"],
    "tflite": [".tflite"],
    "coreml": [".mlpackage"],
}

# Flatten extensions for lookup
ALL_EXTENSIONS = {}
for fmt, exts in MODEL_FORMATS.items():
    for ext in exts:
        ALL_EXTENSIONS[ext.lower()] = fmt


class ModelValidationResult:
    def __init__(self):
        self.valid_organized_models: Dict[str, Dict] = {}  # Path -> details
        self.orphan_models: Dict[str, str] = {}  # Path -> reason
        self.config_mismatches: List[Tuple[str, str, str]] = []  # (config_path, actual_path, warning)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def is_valid(self) -> bool:
        return len(self.errors) == 0 and len(self.orphan_models) == 0

    def summary(self) -> str:
        lines = []
        
        if self.valid_organized_models:
            lines.append("\nValid Organized Models:")
            for path, details in self.valid_organized_models.items():
                lines.append(f"  [OK] {path}")
                lines.append(f"       Format: {details['format']}, Size: {details['size_mb']:.2f}MB")
        
        if self.orphan_models:
            lines.append("\nOrphan Models (NOT in organized structure):")
            for path, reason in self.orphan_models.items():
                lines.append(f"  [WARNING] {path}")
                lines.append(f"           Reason: {reason}")
        
        if self.config_mismatches:
            lines.append("\nConfig Path Mismatches:")
            for config_path, actual_path, warning in self.config_mismatches:
                lines.append(f"  [WARNING] Config specifies: {config_path}")
                lines.append(f"           {warning}")
                lines.append(f"           Using: {actual_path}")
        
        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  [ERROR] {error}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  [WARNING] {warning}")
        
        return "\n".join(lines) if lines else "No models found."


def validate_model_organization(repo_root: Path) -> ModelValidationResult:
    result = ModelValidationResult()
    yolo_dir = repo_root / "YoloModels"
    
    if not yolo_dir.exists():
        result.warnings.append(
            "YoloModels directory not found. Create it and add YOLO models."
        )
        return result
    
    # Find all model files
    all_model_files = {}  # extension -> [paths]
    for ext in ALL_EXTENSIONS.keys():
        all_model_files[ext] = list(yolo_dir.rglob(f"*{ext}"))
    
    # Check each model file
    for ext, model_paths in all_model_files.items():
        for model_path in model_paths:
            result_check = _validate_single_model(model_path, yolo_dir, repo_root)
            
            if result_check["valid"]:
                rel_path = str(model_path.relative_to(repo_root))
                result.valid_organized_models[rel_path] = {
                    "format": result_check["format"],
                    "size_mb": result_check["size_mb"],
                    "path": model_path,
                }
                logger.info(f"[OK] Valid model: {rel_path}")
            else:
                rel_path = str(model_path.relative_to(repo_root))
                result.orphan_models[rel_path] = result_check["reason"]
                logger.warning(f"[WARNING] Orphan model: {rel_path}")
                logger.warning(f"         {result_check['reason']}")
    
    # Check for standalone models in root or outside YoloModels
    _check_for_standalone_models(repo_root, result)
    
    return result


def _validate_single_model(model_path: Path, yolo_dir: Path, repo_root: Path) -> Dict:
    try:
        # Get extension
        ext = model_path.suffix.lower()
        fmt = ALL_EXTENSIONS.get(ext)
        
        if not fmt:
            return {
                "valid": False,
                "reason": f"Unknown model format: {ext}",
                "format": None,
                "size_mb": 0,
            }
        
        # Get file size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Check if file is at least 1MB (sanity check)
        if size_mb < 1:
            return {
                "valid": False,
                "reason": f"Model file too small ({size_mb:.2f}MB). Possibly corrupted.",
                "format": fmt,
                "size_mb": size_mb,
            }
        
        # Check structure: YoloModels/[format]/[size]/...
        parts = model_path.relative_to(yolo_dir).parts
        
        if len(parts) < 3:
            return {
                "valid": False,
                "reason": "Model not in YoloModels/[format]/[size]/ structure",
                "format": fmt,
                "size_mb": size_mb,
            }
        
        structure_format = parts[0]
        structure_size = parts[1]
        
        # Validate format level
        valid_formats = set(MODEL_FORMATS.keys())
        if structure_format not in valid_formats:
            return {
                "valid": False,
                "reason": f"Invalid format directory: {structure_format}. "
                         f"Must be one of: {', '.join(valid_formats)}",
                "format": fmt,
                "size_mb": size_mb,
            }
        
        # Validate size level (nano, small, medium, large, etc.)
        valid_sizes = {"nano", "small", "medium", "large", "xlarge", "2xlarge"}
        if structure_size not in valid_sizes:
            return {
                "valid": False,
                "reason": f"Invalid size directory: {structure_size}. "
                         f"Must be one of: {', '.join(valid_sizes)}",
                "format": fmt,
                "size_mb": size_mb,
            }
        
        # For OpenVINO, check for .xml and .bin pair
        if fmt == "openvino":
            xml_path = model_path.parent / model_path.stem / ".xml"
            bin_path = model_path.parent / model_path.stem / ".bin"
            
            if ext == ".xml":
                expected_bin = model_path.with_suffix(".bin")
                if not expected_bin.exists():
                    return {
                        "valid": False,
                        "reason": "OpenVINO model missing .bin file (found .xml only)",
                        "format": fmt,
                        "size_mb": size_mb,
                    }
            elif ext == ".bin":
                expected_xml = model_path.with_suffix(".xml")
                if not expected_xml.exists():
                    return {
                        "valid": False,
                        "reason": "OpenVINO model missing .xml file (found .bin only)",
                        "format": fmt,
                        "size_mb": size_mb,
                    }
        
        return {
            "valid": True,
            "reason": "OK",
            "format": fmt,
            "size_mb": size_mb,
        }
    
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Validation error: {str(e)}",
            "format": None,
            "size_mb": 0,
        }


def _check_for_standalone_models(repo_root: Path, result: ModelValidationResult) -> None:    
    # Check root directory
    root_models = []
    for ext in ALL_EXTENSIONS.keys():
        root_models.extend(repo_root.glob(f"*{ext}"))
    
    for model_path in root_models:
        if model_path.is_file():
            rel_path = str(model_path.relative_to(repo_root))
            result.orphan_models[rel_path] = (
                "cannot infer yolo parameters - model is standalone in repo root. "
                f"Move to YoloModels/[format]/[size]/ (e.g., YoloModels/pytorch/nano/)"
            )
            logger.warning(f"[STANDALONE] {rel_path}")
            logger.warning(f"            Cannot infer YOLO parameters for standalone model")
    
    # Check if there are model files in YoloModels root
    yolo_dir = repo_root / "YoloModels"
    if yolo_dir.exists():
        yolo_root_models = []
        for ext in ALL_EXTENSIONS.keys():
            yolo_root_models.extend(yolo_dir.glob(f"*{ext}"))
        
        for model_path in yolo_root_models:
            if model_path.is_file():
                rel_path = str(model_path.relative_to(repo_root))
                result.orphan_models[rel_path] = (
                    "cannot infer yolo parameters - model is in YoloModels root. "
                    f"Move to YoloModels/[format]/[size]/ (e.g., YoloModels/pytorch/nano/)"
                )
                logger.warning(f"[STANDALONE] {rel_path}")
                logger.warning(f"            Cannot infer YOLO parameters - move to organized directory")


def validate_config_model_paths(config: Dict, repo_root: Path, 
                               validation_result: ModelValidationResult) -> str:
    config_model_path = config.get("vision_model", {}).get("file_path", "")
    
    if not config_model_path:
        logger.warning("No model path in config.json - cannot validate")
        return None
    
    config_path = Path(config_model_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    
    # Check if config path exists
    if config_path.exists():
        # Check if it's in organized structure
        if _is_in_organized_structure(config_path, repo_root / "YoloModels"):
            logger.info(f"Config model path is valid and organized: {config_model_path}")
            return str(config_path.relative_to(repo_root))
        else:
            validation_result.warnings.append(
                f"Config model path exists but is not in organized structure: {config_model_path}"
            )
    
    # Try to find the model in organized structure
    model_filename = config_path.name
    yolo_dir = repo_root / "YoloModels"
    
    matches = list(yolo_dir.rglob(model_filename))
    
    if matches:
        # Use the first match (file system is source of truth)
        actual_path = matches[0]
        rel_path = str(actual_path.relative_to(repo_root))
        
        validation_result.config_mismatches.append((
            config_model_path,
            rel_path,
            f"Config path differs from filesystem. Using organized path."
        ))
        
        logger.warning(f"Config model path mismatch:")
        logger.warning(f"  Specified: {config_model_path}")
        logger.warning(f"  Found in organized structure: {rel_path}")
        logger.warning(f"  Using filesystem path (source of truth)")
        
        return rel_path
    else:
        validation_result.errors.append(
            f"Model file '{model_filename}' not found. Specified: {config_model_path}"
        )
        logger.error(f"Model not found: {config_model_path}")
        return None


def _is_in_organized_structure(model_path: Path, yolo_dir: Path) -> bool:
    try:
        rel = model_path.relative_to(yolo_dir)
        parts = rel.parts
        
        if len(parts) < 3:
            return False
        
        fmt = parts[0]
        size = parts[1]
        
        valid_formats = set(MODEL_FORMATS.keys())
        valid_sizes = {"nano", "small", "medium", "large", "xlarge", "2xlarge"}
        
        return fmt in valid_formats and size in valid_sizes
    
    except ValueError:
        return False


def enforce_model_organization(repo_root: Path, config: Dict) -> Tuple[bool, str]:
    # Validate filesystem organization
    validation_result = validate_model_organization(repo_root)
    
    # Log results
    summary = validation_result.summary()
    if summary:
        logger.info(summary)
    
    # Validate config paths
    corrected_path = validate_config_model_paths(config, repo_root, validation_result)
    
    # Report final status
    if validation_result.errors:
        logger.error("Model validation failed with errors")
        return False, None
    
    if corrected_path:
        logger.info(f"Using model: {corrected_path}")
        return True, corrected_path
    else:
        logger.error("No valid model path determined")
        return False, None
