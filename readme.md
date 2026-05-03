VisionCore-Deploy
=================

VisionCore is a comprehensive computer vision pipeline designed specifically for FIRST Robotics Competition (FRC) teams. It provides real-time object detection, tracking, and field mapping capabilities to enhance robot autonomy and driver assistance.

## What is VisionCore?

VisionCore is a modular, extensible computer vision system that processes camera feeds to detect and track objects on the FRC field. It can identify game pieces, obstacles, and field elements, providing real-time data to robot control systems via NetworkTables.

Key capabilities:
- Real-time object detection using YOLO models
- Multi-camera support with automatic stitching
- Object tracking and path planning
- NetworkTables integration for robot communication
- Web-based monitoring and configuration interface
- RKNN acceleration support for edge devices
- Plugin system for custom vision processing

## Features

### Core Vision Processing
- YOLO-based object detection (Ultralytics integration)
- Support for multiple model formats (PyTorch, ONNX, TFLite, RKNN)
- Real-time camera feed processing
- Multi-camera fusion and stitching
- Automatic model optimization based on hardware

### Tracking and Analysis
- Fuel/game piece tracking with Kalman filtering
- Path planning and trajectory prediction
- Distance estimation and 3D positioning
- Custom tracker plugins for specialized objects

### Robot Integration
- NetworkTables communication protocol
- Real-time data streaming to robot controllers
- Configurable data publishing
- Health monitoring and diagnostics

### Deployment Options
- Pip-installable Python package
- Docker container support
- Custom Orange Pi image building
- Systemd service integration
- Development and production configurations

### Extensibility
- Plugin architecture for custom components
- Entry point system for third-party extensions
- Configuration-driven component loading
- API for custom vision modules and trackers

## Prerequisites

### System Requirements
- Python 3.10 or newer
- Linux, macOS, or Windows (with WSL)
- For RKNN acceleration: Rockchip-based devices (Orange Pi, etc.)
- For development: Git, build tools

### Hardware Recommendations
- For development: Any modern computer with webcam
- For deployment: Orange Pi 5 or similar RKNN-capable device
- Camera: USB webcam or CSI camera (tested with Microsoft LifeCam)

### Software Dependencies
- pip for Python package management
- Git for repository cloning
- Docker (optional, for containerized deployment)
- Build tools (for custom image building)

## Installation

### Method 1: Pip Installation (Recommended)

#### Development Setup
For development machines (x86_64):
```bash
pip install visioncore-frc[dev]
```

#### Deployment Setup
For RKNN-capable devices (aarch64):
```bash
pip install visioncore-frc[deploy]
```

#### Full Installation
Install with all optional dependencies:
```bash
pip install visioncore-frc[dev,deploy,metrics]
```

### Method 2: Using Installation Scripts

#### Development Installation
```bash
# Make scripts executable
chmod +x install-dev.sh

# Run development installer
./install-dev.sh
```

#### Deployment Installation
```bash
# Make scripts executable
chmod +x install-deploy.sh

# Run deployment installer
./install-deploy.sh
```

### Method 3: From Source

Clone the repository:
```bash
git clone https://github.com/your-org/visioncore-deploy.git
cd visioncore-deploy
```

Install in development mode:
```bash
pip install -e .[dev]
```

### Method 4: Custom Image (Orange Pi)

Build a complete system image:
```bash
cd Image
./build-image.sh
```

Flash the resulting `orangepi.img` to an SD card and boot.

## Configuration

VisionCore uses JSON configuration files to control all aspects of operation.

### Basic Configuration

Create a `config.json` file:

```json
{
  "unit": "meter",
  "dbscan": {"eps": 0.3, "min_samples": 3},
  "distance_threshold": 0.5,
  "vision_model_input_size": [640, 640],
  "vision_model_file_path": "model.pt",
  "network_tables_ip": "10.22.7.2",
  "use_network_tables": true,
  "app_mode": true,
  "debug_mode": false,
  "record_mode": false,
  "stale_threshold": 1.0,
  "log_level": "INFO",
  "log_file": "Outputs/log.txt",
  "metrics": false,
  "camera_matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
  "dist_coeffs": [0, 0, 0, 0, 0],
  "vision_modules": ["object_detection"],
  "trackers": ["fuel", "path_planner"],
  "utilities": ["network_table", "video_recorder"],
  "camera_configs": {
    "front_camera": {
      "name": "front_camera",
      "x": 0, "y": 0, "height": 0.5,
      "pitch": 0, "yaw": 0,
      "grayscale": false,
      "fps_cap": 30,
      "calibration": {
        "size": [640, 480],
        "distance": 1.0,
        "game_piece_size": 0.2,
        "fov": 60
      },
      "source": "/dev/video0",
      "subsystem": "intake",
      "pipeline": "object_detection"
    }
  }
}
```

### Configuration Parameters

#### Core Settings
- `unit`: Distance unit ("meter", "inch", "foot")
- `dbscan`: Clustering parameters for object grouping
- `distance_threshold`: Maximum distance for object matching
- `vision_model_*`: Model configuration
- `network_tables_ip`: Robot NetworkTables server IP
- `use_network_tables`: Enable/disable robot communication
- `app_mode`: Enable web interface
- `debug_mode`: Enable debug logging
- `record_mode`: Enable video recording
- `stale_threshold`: Object timeout in seconds
- `log_level`: Logging verbosity ("DEBUG", "INFO", "WARNING", "ERROR")
- `log_file`: Log file path
- `metrics`: Enable performance metrics collection

#### Camera Calibration
- `camera_matrix`: Intrinsic camera parameters (3x3 matrix)
- `dist_coeffs`: Lens distortion coefficients
- `camera_configs`: Per-camera configuration objects

#### Modular Components
- `vision_modules`: List of vision processing modules to load
- `trackers`: List of object trackers to use
- `utilities`: List of utility components to enable

### Camera Configuration

Each camera config object contains:
- `name`: Unique camera identifier
- `x`, `y`, `height`: Camera position on robot (in specified units)
- `pitch`, `yaw`: Camera orientation angles (degrees)
- `grayscale`: Convert to grayscale for processing
- `fps_cap`: Maximum frame rate (0 = unlimited)
- `calibration`: Camera calibration data
- `source`: Camera device path or URL
- `subsystem`: Robot subsystem this camera serves
- `pipeline`: Vision processing pipeline to use

## Plugin System

VisionCore's plugin system allows teams to extend functionality without modifying core code.

### Plugin Types

#### Vision Modules (`visioncore_vision`)
Process camera frames and detect objects. Examples:
- `object_detection`: YOLO-based detection
- `generic_yolo`: Generic YOLO wrapper
- Custom: Team-specific detection algorithms

#### Trackers (`visioncore_trackers`)
Track detected objects across frames. Examples:
- `fuel`: Game piece tracking
- `path_planner`: Trajectory planning
- Custom: Specialized object tracking

#### Utilities (`visioncore_utilities`)
Supporting components. Examples:
- `network_table`: Robot communication
- `video_recorder`: Video capture
- Custom: Data logging, custom networking

### Creating Custom Plugins

#### Creating a Custom Tracker

**Step 1: Create the Tracker Class**

```python
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
import logging
import numpy as np

class MyCustomTracker:
    def __init__(self, config: VisionCoreConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tracked_objects = {}
        self.next_id = 0

    def process_detections(self, detections):
        """Process detections each frame - called after vision processing"""
        # detections is a list of Fuel objects or similar
        for detection in detections:
            # Your custom tracking logic here
            # Example: Simple ID assignment
            if not hasattr(detection, 'id'):
                detection.id = self.next_id
                self.next_id += 1
                self.logger.info(f"Assigned ID {detection.id} to new detection")

    def update(self, detections, robot_x, robot_y, robot_rotation):
        """Update tracking with robot pose - called in main loop"""
        # Implement Kalman filtering, prediction, etc.
        # This is where the main tracking logic goes
        pass

    def reset(self):
        """Reset tracker state"""
        self.tracked_objects = {}
        self.next_id = 0

    def get_status(self):
        """Return status for monitoring"""
        return {
            "tracked_objects": len(self.tracked_objects),
            "next_id": self.next_id
        }
```

**Step 2: Register the Tracker**

```toml
[project.entry-points.visioncore_trackers]
my_custom_tracker = "my_plugin.trackers:MyCustomTracker"
```

**Step 3: Configure Usage**

```json
{
  "trackers": ["fuel", "my_custom_tracker"]
}
```

#### Creating a Custom Vision Module

**Step 1: Create the Vision Class**

```python
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera
import cv2
import logging
import numpy as np

class MyCustomVision:
    def __init__(self, camera_config, global_config: VisionCoreConfig):
        self.camera_config = camera_config
        self.global_config = global_config
        self.logger = logging.getLogger(__name__)

        # Initialize camera capture
        self.cap = cv2.VideoCapture(camera_config.get("source", "/dev/video0"))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {camera_config.get('source')}")

        # Set camera properties
        fps_cap = camera_config.get("fps_cap", 30)
        if fps_cap > 0:
            self.cap.set(cv2.CAP_PROP_FPS, fps_cap)

        # Initialize your custom model/detector here
        self.detector = self._initialize_detector()

    def _initialize_detector(self):
        """Initialize your custom detection model"""
        # Example: Load a custom model
        # return YourCustomModel(model_path=self.global_config.get("custom_model_path"))
        return None  # Placeholder

    def run(self):
        """Main processing method - called each frame"""
        ret, frame = self.cap.read()
        if not ret:
            return [], None

        # Apply camera calibration if needed
        if self.camera_config.get("calibration"):
            # Undistort frame using camera matrix and distortion coefficients
            camera_matrix = np.array(self.global_config.get("camera_matrix"))
            dist_coeffs = np.array(self.global_config.get("dist_coeffs"))
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Convert to grayscale if specified
        if self.camera_config.get("grayscale", False):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back for annotation

        # Your custom detection logic here
        detections = self._detect_objects(frame)

        # Annotate frame with detections
        annotated_frame = self._annotate_frame(frame.copy(), detections)

        return detections, annotated_frame

    def _detect_objects(self, frame):
        """Implement your object detection algorithm"""
        # Example: Simple color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color range (example: red objects)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                # Create detection object (you can define your own format)
                detection = {
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.8,
                    'class': 'red_object',
                    'area': area
                }
                detections.append(detection)

        return detections

    def _annotate_frame(self, frame, detections):
        """Draw detections on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection['class']} {detection['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def destroy(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
```

**Step 2: Register the Vision Module**

```toml
[project.entry-points.visioncore_vision]
my_custom_vision = "my_plugin.vision:MyCustomVision"
```

**Step 3: Configure Camera to Use It**

```json
{
  "camera_configs": {
    "my_camera": {
      "name": "my_camera",
      "source": "/dev/video0",
      "pipeline": "my_custom_vision",
      "fps_cap": 30
    }
  }
}
```

#### Creating a Custom Utility

**Step 1: Create the Utility Class**

```python
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
import logging
import json
import time
from pathlib import Path

class MyCustomLogger:
    def __init__(self, config: VisionCoreConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create log directory
        self.log_dir = Path("custom_logs")
        self.log_dir.mkdir(exist_ok=True)

        # Initialize log file
        self.log_file = self.log_dir / f"vision_log_{int(time.time())}.json"
        self.log_data = []

    def log_detection(self, camera_name, detections, timestamp=None):
        """Log detection data"""
        if timestamp is None:
            timestamp = time.time()

        log_entry = {
            "timestamp": timestamp,
            "camera": camera_name,
            "num_detections": len(detections),
            "detections": [
                {
                    "class": getattr(det, 'class_name', 'unknown'),
                    "confidence": getattr(det, 'confidence', 0.0),
                    "position": getattr(det, 'position', [0, 0])
                } for det in detections
            ]
        }

        self.log_data.append(log_entry)

        # Write to file periodically (every 100 entries)
        if len(self.log_data) >= 100:
            self._flush_logs()

    def log_performance(self, fps, latency, memory_usage):
        """Log performance metrics"""
        log_entry = {
            "timestamp": time.time(),
            "type": "performance",
            "fps": fps,
            "latency_ms": latency * 1000,
            "memory_mb": memory_usage
        }

        self.log_data.append(log_entry)

    def _flush_logs(self):
        """Write accumulated logs to file"""
        try:
            with open(self.log_file, 'a') as f:
                for entry in self.log_data:
                    json.dump(entry, f)
                    f.write('\n')
            self.log_data = []
            self.logger.info(f"Flushed {len(self.log_data)} log entries")
        except Exception as e:
            self.logger.error(f"Failed to flush logs: {e}")

    def get_stats(self):
        """Get logging statistics"""
        return {
            "total_entries": len(self.log_data),
            "log_file": str(self.log_file),
            "log_file_size": self.log_file.stat().st_size if self.log_file.exists() else 0
        }

    def cleanup(self):
        """Clean up resources"""
        self._flush_logs()
        self.logger.info("Custom logger cleaned up")
```

**Step 2: Register the Utility**

```toml
[project.entry-points.visioncore_utilities]
my_custom_logger = "my_plugin.utilities:MyCustomLogger"
```

**Step 3: Configure Usage**

```json
{
  "utilities": ["network_table", "my_custom_logger"]
}
```

**Step 4: Integrate with VisionCore (Optional)**

If you want your utility to be called automatically, you may need to modify VisionCore.py to integrate it:

```python
# In VisionCore.__init__ after loading utilities
if 'my_custom_logger' in self.utilities:
    self.custom_logger = self.utilities['my_custom_logger']

# In the main loop, call your utility methods
if hasattr(self, 'custom_logger'):
    self.custom_logger.log_detection(camera_name, fuel_list)
    self.custom_logger.log_performance(1/loop_s, vision_s, memory_usage)
```

### Advanced Plugin Examples

#### Plugin with Configuration

```python
class ConfigurableTracker:
    def __init__(self, config: VisionCoreConfig):
        # Access custom config parameters
        self.max_objects = config.get("max_tracked_objects", 10)
        self.tracking_threshold = config.get("tracking_confidence_threshold", 0.5)

        # Plugin-specific config can be added to main config
        plugin_config = config.get("my_plugin_config", {})
        self.custom_param = plugin_config.get("custom_param", "default")
```

#### Plugin with Dependencies

```toml
[project]
name = "my-visioncore-plugin"
dependencies = [
    "visioncore-frc",
    "numpy>=1.21.0",
    "opencv-python>=4.5.0",
    "scipy>=1.7.0"
]
```

#### Plugin with Multiple Components

```toml
[project.entry-points.visioncore_trackers]
advanced_tracker = "my_plugin.trackers:AdvancedTracker"
simple_tracker = "my_plugin.trackers:SimpleTracker"

[project.entry-points.visioncore_vision]
thermal_vision = "my_plugin.vision:ThermalVision"
depth_vision = "my_plugin.vision:DepthVision"

[project.entry-points.visioncore_utilities]
data_exporter = "my_plugin.utilities:DataExporter"
alert_system = "my_plugin.utilities:AlertSystem"
```

### Plugin Interface Requirements

#### Vision Modules
Vision modules process camera frames and return detections. They must implement:

**Required Methods:**
- `__init__(self, camera_config, global_config)`: Initialize with camera and global config
- `run(self)`: Process one frame, return `(detections, annotated_frame)`
- `destroy(self)`: Clean up resources (cameras, models, etc.)

**Parameters:**
- `camera_config`: Dictionary with camera-specific settings (source, calibration, etc.)
- `global_config`: VisionCoreConfig object with global settings

**Return Values:**
- `detections`: List of detection objects (can be any format, but typically Fuel objects)
- `annotated_frame`: OpenCV image with detections drawn on it (or None)

**Optional Methods:**
- `get_frame_age(self)`: Return seconds since last frame capture
- `get_data_for_subsystem(self, subsystem)`: Return subsystem-specific data

#### Trackers
Trackers maintain object state across frames. They must implement:

**Required Methods:**
- `__init__(self, config)`: Initialize with global config

**Optional Methods (implement as needed):**
- `process_detections(self, detections)`: Process detections each frame (post-vision)
- `update(self, detections, robot_x, robot_y, robot_rotation)`: Update with robot pose
- `reset(self)`: Reset tracker state
- `get_status(self)`: Return status information for monitoring

**Integration Points:**
- `process_detections` called after vision processing, before network transmission
- `update` called in main loop with robot pose for motion compensation

#### Utilities
Utilities provide supporting functionality. Interface varies by purpose:

**Common Patterns:**
- `__init__(self, config)`: Standard initialization
- Custom methods based on utility purpose
- May need integration in VisionCore.py for automatic calling

**Examples:**
- Network handlers: `send_data(key, value)`, `get_robot_pose()`
- Recorders: `start(width, height)`, `write(frame)`, `stop()`
- Loggers: `log_event(data)`, `flush()`, `get_stats()`

**Integration:**
Utilities are loaded but may require manual integration in VisionCore.py for automatic operation.

### Testing and Debugging Plugins

#### Plugin Development Tips

1. **Start Simple**: Begin with basic functionality before adding complexity
2. **Use Logging**: Add comprehensive logging for debugging
3. **Handle Errors**: Wrap operations in try-catch blocks
4. **Resource Management**: Always clean up cameras, files, and connections

#### Testing Your Plugin

**Unit Testing:**
```python
import unittest
from my_plugin.trackers import MyCustomTracker
from VisionCore.config.VisionCoreConfig import VisionCoreConfig

class TestMyCustomTracker(unittest.TestCase):
    def setUp(self):
        config = VisionCoreConfig()
        self.tracker = MyCustomTracker(config)

    def test_initialization(self):
        self.assertIsNotNone(self.tracker)
        status = self.tracker.get_status()
        self.assertIn("tracked_objects", status)

    def test_process_detections(self):
        # Create mock detections
        mock_detections = [MockDetection() for _ in range(3)]
        self.tracker.process_detections(mock_detections)
        # Assert expected behavior
```

**Integration Testing:**
```python
# Test with actual VisionCore
config = VisionCoreConfig("test_config.json")
config.config["trackers"] = ["my_custom_tracker"]

# This will load your plugin
vision = VisionCore([], config)
# Verify your tracker is loaded
self.assertIn("my_custom_tracker", vision.trackers)
```

#### Debugging Common Issues

**Plugin Not Loading:**
- Check entry point name matches config exactly
- Verify package is installed: `pip list | grep my-plugin`
- Check for import errors: `python -c "from my_plugin.trackers import MyCustomTracker"`

**Configuration Errors:**
- Validate JSON syntax
- Check parameter names match what your plugin expects
- Use debug logging to see config values

**Runtime Errors:**
- Enable debug mode in config: `"debug_mode": true`
- Check logs: `tail -f Outputs/log.txt`
- Add try-catch blocks with logging in your plugin

**Performance Issues:**
- Profile your plugin code
- Check for memory leaks
- Optimize image processing operations
- Use appropriate data structures

#### Plugin Best Practices

**Code Organization:**
```
my_visioncore_plugin/
├── pyproject.toml
├── README.md
├── my_plugin/
│   ├── __init__.py
│   ├── trackers.py
│   ├── vision.py
│   └── utilities.py
└── tests/
    ├── test_trackers.py
    ├── test_vision.py
    └── test_utilities.py
```

**Version Management:**
- Use semantic versioning
- Document breaking changes
- Test against multiple VisionCore versions

**Documentation:**
- Include docstrings for all public methods
- Provide usage examples
- Document configuration parameters
- Create a README for your plugin

**Distribution:**
- Publish to PyPI for easy installation
- Include license information
- Provide issue tracker for support

### Built-in Plugins

VisionCore ships with these default plugins:

**Vision Modules:**
- `object_detection`: Full YOLO pipeline with camera calibration
- `generic_yolo`: Basic YOLO wrapper

**Trackers:**
- `fuel`: Game piece detection and tracking
- `path_planner`: Path planning and trajectory optimization

**Utilities:**
- `network_table`: NetworkTables communication
- `video_recorder`: Video file recording

### Real-World Plugin Examples

#### FRC Game-Specific Vision
```python
class PowerUpVision:
    """Detect 2018 Power Up game elements"""
    def run(self):
        # Detect switch panels, scale, and power cubes
        switch_detections = self.detect_switch_panels(frame)
        scale_detections = self.detect_scale(frame)
        cube_detections = self.detect_power_cubes(frame)
        return switch_detections + scale_detections + cube_detections, annotated_frame
```

#### Advanced Tracking
```python
class KalmanTracker:
    """Use Kalman filtering for robust object tracking"""
    def update(self, detections, robot_x, robot_y, robot_rotation):
        # Predict object positions
        # Update with measurements
        # Handle occlusions and re-identification
        pass
```

#### Custom Data Export
```python
class MatchDataExporter:
    """Export telemetry data for match analysis"""
    def log_match_data(self, match_time, robot_pose, detections):
        # Save to CSV/database for post-match analysis
        # Include timestamps, positions, success rates
        pass
```

#### Multi-Camera Fusion
```python
class StereoVision:
    """Combine two cameras for depth perception"""
    def __init__(self, left_config, right_config, global_config):
        self.left_cam = Camera(left_config)
        self.right_cam = Camera(right_config)
        # Calibrate stereo pair
        self.stereo_matcher = cv2.StereoBM_create()

    def run(self):
        left_frame = self.left_cam.capture()
        right_frame = self.right_cam.capture()
        # Compute disparity map
        # Triangulate 3D positions
        return detections_3d, annotated_frame
```

#### Machine Learning Integration
```python
class MLTracker:
    """Use ML model for object re-identification"""
    def __init__(self, config):
        self.reid_model = torch.load(config.get("reid_model_path"))
        self.feature_extractor = self.reid_model.feature_extractor

    def process_detections(self, detections):
        # Extract appearance features
        # Match detections to existing tracks using ML similarity
        pass
```

### Plugin Ecosystem

#### Sharing Plugins
- Create GitHub repositories for your plugins
- Use consistent naming: `visioncore-plugin-*`
- Tag releases with version numbers
- Provide installation instructions

#### Community Plugins
Teams can share plugins for common FRC tasks:
- Game-specific object detection
- Advanced tracking algorithms
- Custom data logging and analysis
- Specialized camera configurations
- Robot-specific vision requirements

#### Plugin Dependencies
Plugins can depend on each other:
```toml
[project]
dependencies = [
    "visioncore-frc",
    "visioncore-plugin-advanced-tracking",  # Depends on another plugin
    "numpy",
    "torch"
]
```

This creates a rich ecosystem where teams can build upon each other's work.

## Usage

### Basic Usage

#### Command Line
```bash
# Run with default config
visioncore-run

# Run with custom config
visioncore-run --config my_config.json

# Boot mode (initial setup)
visioncore-boot
```

#### Python API
```python
from VisionCore import VisionCore
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera

# Load configuration
config = VisionCoreConfig("config.json")

# Create cameras
cameras = [
    ObjectDetectionCamera(config.camera_config("front_camera"), config)
]

# Initialize VisionCore
vision = VisionCore(cameras, config)

# Run vision processing
vision.run()
```

### Web Interface

When `app_mode` is enabled, VisionCore provides a web interface at `http://localhost:5000`:

- Live camera feeds
- Real-time detection visualization
- Configuration management
- Performance metrics
- Health monitoring

### NetworkTables Integration

VisionCore publishes data to NetworkTables for robot consumption:

- `vision_data`: Array of detected objects
- `fps`: Processing frame rate
- `num_detections`: Number of objects detected
- `camera_lag`: Camera latency in seconds

### Video Recording

When `record_mode` is enabled, VisionCore saves video files to `VideoRecordings/` directory.

## Deployment

### Development Deployment

For testing on development machines:
1. Install with dev dependencies: `pip install visioncore-frc[dev]`
2. Configure cameras in `config.json`
3. Run: `visioncore-run`

### Production Deployment

For robot field deployment:
1. Install with deploy dependencies: `pip install visioncore-frc[deploy]`
2. Configure for robot network
3. Set up as system service (see below)

### System Service Setup

Create systemd service for automatic startup:

```bash
# Install service
sudo cp /opt/visioncore/visioncore.service /etc/systemd/system/
sudo systemctl enable visioncore
sudo systemctl start visioncore
```

### Custom Image Deployment

For Orange Pi or similar devices:
1. Build image: `cd Image && ./build-image.sh`
2. Flash to SD card
3. Boot device - VisionCore starts automatically

### Docker Deployment

Build and run container:
```bash
docker build -t visioncore .
docker run -p 5000:5000 --device=/dev/video0 visioncore
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/visioncore-deploy.git
cd visioncore-deploy

# Install in development mode
pip install -e .[dev]

# Run tests
python -m pytest VisionCore/validations/unit_tests.py

# Run with test config
visioncore-run --config VisionCore/core/config.json
```

### Code Structure

```
VisionCore/
├── __init__.py              # Package initialization
├── VisionCore.py            # Main vision processing loop
├── boot/                    # Boot and service management
├── config/                  # Configuration management
├── core/                    # Core processing logic
├── trackers/                # Object tracking components
├── utilities/               # Utility components
├── validations/             # Testing and validation
└── vision/                  # Vision processing modules
```

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add unit tests
4. Update documentation
5. Submit pull request

### Testing

Run the test suite:
```bash
python -m pytest VisionCore/validations/unit_tests.py -v
```

Run integration tests:
```bash
python VisionCore/validations/validate_system.py
```

## Troubleshooting

### Common Issues

#### Camera Not Detected
- Check device permissions: `ls -la /dev/video*`
- Verify camera compatibility
- Test with: `v4l2-ctl --list-devices`

#### NetworkTables Connection Failed
- Verify robot IP address in config
- Check network connectivity
- Confirm robot code is running

#### Model Loading Errors
- Ensure model file exists at specified path
- Check model format compatibility
- Verify RKNN runtime for RKNN models

#### Performance Issues
- Reduce camera resolution
- Lower FPS cap
- Enable hardware acceleration
- Check system resources

### Logs and Debugging

Enable debug logging in config:
```json
{
  "log_level": "DEBUG",
  "debug_mode": true
}
```

View logs:
```bash
tail -f Outputs/log.txt
```

System service logs:
```bash
journalctl -u visioncore -f
```

### Getting Help

- Check existing issues on GitHub
- Review configuration examples
- Test with minimal configuration
- Enable debug mode for detailed logs

## Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run tests: `python -m pytest`
5. Commit changes: `git commit -am "Add my feature"`
6. Push to branch: `git push origin feature/my-feature`
7. Create pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all public functions
- Include unit tests for new functionality
- Update documentation for API changes

### Plugin Contributions

- Create separate repositories for plugins
- Use clear naming conventions
- Provide comprehensive documentation
- Include example configurations

## License

This project is licensed under the GPL-3.0 License. See the LICENSE file for details.

## Support

For questions, issues, or contributions:
- GitHub Issues: Report bugs and request features
- GitHub Discussions: Ask questions and get help
- Documentation: Comprehensive guides and examples

---

VisionCore is designed to be the foundation for FRC robot vision systems, providing teams with powerful, extensible computer vision capabilities.