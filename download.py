import os
from ultralytics import YOLO

def main():
    print("--- Step 1: Downloading standard PyTorch model (yolov8n.pt) ---")
    # This automatically downloads the pretrained yolov8n.pt file from Ultralytics
    model = YOLO("yolov8n.pt")
    
    print("\n--- Step 2: Exporting standard ONNX (yolov8n_nonms.onnx) ---")
    # Export without NMS gives the raw bounding box tensor (e.g., shape 1x84x8400)
    model.export(format="onnx", nms=False)
    # The default export saves it as 'yolov8n.onnx', so we rename it to match your test script
    if os.path.exists("yolov8n.onnx"):
        os.rename("yolov8n.onnx", "yolov8n_nonms.onnx")
        print("Created yolov8n_nonms.onnx")

    print("\n--- Step 3: Exporting ONNX with embedded NMS (yolov8n_nms.onnx) ---")
    # Setting nms=True bakes the Non-Maximum Suppression logic directly into the ONNX graph
    model.export(format="onnx", nms=True)
    if os.path.exists("yolov8n.onnx"):
        os.rename("yolov8n.onnx", "yolov8n_nms.onnx")
        print("Created yolov8n_nms.onnx")

    print("\nAll required model files have been successfully prepared!")

if __name__ == "__main__":
    main()