2026 Vision Testing

This repository contains vision processing code for 2026 FRC competition. The code is organized into simplified and complex implementations.

Simplified folder: detects fuel and sends positions over network tables. Good for competition and low-latency setups.

Complex folder: does path planning and plotting along with fuel detection. More features but slower.

Important Scripts

video_feed_broadcaster.py - Flask app that streams camera feed to a website
yolo_with_cam.py - Runs YOLO vision model on camera (default camera 0)
pt_to_wtv.py - Converts .pt files to other formats (onnx, xml, etc)
map_maker.py - Interactive window to create a fuel map
map_creator.py - Uses the vision model to create a birdseye view from images
camera_calibration.py - Calculates focal length constants for camera calibration
livestream_reader.py - Reads frames from a livestream using Camera class
onnx_to_rknn.py - Converts models to RKNN format

Folder Structure

COMPLEX - path planning and plotting with fuel detection
SIMPLIFIED - fuel detection and network tables sending
PLOTTERS - plotting and utility experiments
YOLO_MODELS - YOLO models and exports

Key Classes

Camera.py - camera configuration and capture
CustomDBScan.py - point cloud filtering with DBSCAN
NetworkTableHandler.py - network tables communication
PathPlanner.py - path planning logic
Fuel.py - fuel tracking object
FuelTracker.py - fuel tracking helper

Orange Pi 5 Device

IP: 10.22.7.200
Username: ubuntu
Password: 2207vision

To access the device, use SSH with the credentials above.
To clone the repo on the device: git clone git@github.com:FRC2207/2026-Vision-Testing.git

Common Commands

sudo systemctl restart vision.service - restart the vision service
tmux attach -t vision - attach to vision tmux session
git pull - update code from repository
git reset --hard HEAD - undo local changes
git clean -fd - remove untracked files

Model Performance

Model         Size    Latency      FPS
Yolov26       nano    99.5ms       10
Yolov11       nano    71.5ms       14
Yolov11       small   98.9ms       10
Yolov11       medium  235.3ms      4
Yolov8        nano    20-30ms      30-50
Yolov8        small   40-60ms      15-25
Yolov5        nano    15-25ms      40-60

Notes

The repository is functional but has rough edges and moved files during testing.
When converting models, double-check export settings to avoid losing work.
The simplified implementation is recommended for competition use.