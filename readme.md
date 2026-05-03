VisionCore-Deploy
=================

This repository contains scripts, libraries, and assets for deploying the VisionCore system.

Prerequisites
-------------
- A Unix-like shell (Linux, macOS, or WSL on Windows)
- Python 3.10 or newer

Quick install
-------------
1. Make scripts executable if needed:

   chmod +x install-dev.sh install-deploy.sh

2. For development setup, run:

   ./install-dev.sh

3. For deployment setup, run:

   ./install-deploy.sh

Building an image
-----------------
To build a device image, run the image build script:

   cd Image
   ./build-image.sh

Project layout
--------------
- VisionCore/: main Python package with configuration and core modules
- RknnWheels/: prebuilt RKNN wheel files used for model conversion
- Image/: image build scripts and first-boot helpers
- LICENSE: project license file
- readme.md: legacy project readme (note: this file is distinct from README.md)

Usage
-----
- See VisionCore/example_usage.py for example code showing how to use the package.
- Configuration examples are in VisionCore/example_config.json

Contributing
------------
Contributions are welcome via pull requests. Please follow existing code style and add tests for new behavior.

License
-------
See the LICENSE file at the project root.