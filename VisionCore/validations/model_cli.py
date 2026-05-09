import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def cmd_validate_models():
    from VisionCore.validations.model_validator import validate_model_organization
    
    repo_root = Path.cwd()
    result = validate_model_organization(repo_root)
    
    print("\n" + "="*70)
    print("YOLO Model Validation Report".center(70))
    print("="*70)
    print(result.summary())
    print("="*70 + "\n")
    
    if result.is_valid():
        logger.info("All models are properly organized.")
        return 0
    else:
        if result.orphan_models:
            logger.warning("Found orphan/standalone models - cannot infer YOLO parameters")
            logger.info("Move models to: YoloModels/[format]/[size]/")
        if result.errors:
            logger.error("Validation failed with errors")
            return 1
        return 0

def cmd_check_organization():
    from VisionCore.validations.model_validator import validate_model_organization
    
    repo_root = Path.cwd()
    result = validate_model_organization(repo_root)
    
    print("\nYoloModels Organization Status:")
    print("-" * 50)
    
    if result.valid_organized_models:
        print(f"Valid organized models: {len(result.valid_organized_models)}")
        for path in result.valid_organized_models:
            print(f"  + {path}")
    else:
        print("No valid organized models found")
    
    if result.orphan_models:
        print(f"\nOrphan models (not in organized structure): {len(result.orphan_models)}")
        for path, reason in result.orphan_models.items():
            print(f"  - {path}")
            print(f"    {reason}")
    
    print("\nCorrect Structure:")
    print("  YoloModels/[format]/[size]/model_file")
    print("  Examples:")
    print("    YoloModels/pytorch/nano/yolov8n.pt")
    print("    YoloModels/openvino/nano/yolov8n.xml")
    print("    YoloModels/rknn/nano/yolov8n.rknn")
    print()
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["validate-models", "check-org"]:
        if sys.argv[1] == "validate-models":
            sys.exit(cmd_validate_models())
        else:
            sys.exit(cmd_check_organization())
    else:
        print("Model Validation CLI")
        print("Usage: python -m VisionCore.validations.model_cli validate-models")
        print("       python -m VisionCore.validations.model_cli check-org")