# 2026 Vision Testing

So everything needs a little cleanup and stuff but the simplfied folder should be the files that just sends the detected fuel data points over network tables. The complex folder does a lot more like plot the path and sends the path over network tables.

### Utilities

- **video_feed_broadcaster.py**: Sets up a custom Flask app to make a website with the camera livestream
- **yolo_with_cam.py**: Runs the set YOLO vision model on a camera (defualt id is 0)
- **pt_to_wtv.py**: Lets you convert a .pt file to other file types for optimization (.onnx, .xml, etc)
- **map_maker.py**: Opens a interactive window to let you create a fuel map
- **map_creator.py**: Uses the vision model and some math or wtv to make a birdseye veiw of the image file (in Images directory)
- **camera_calibration.py**: This file is used for calculating the constants needed for focal length smth. Run the file, enter the paramters, hold the fuel a certain distance away and enter the other paramters then it will give you some values. Paste that into focal_calibration_data.txt for averaging. Average the "known_calibration_pixel_height" value and make sure when creating the camera you set the correct distance
- **livestream_reader.py**: Create a Camera object and reads form a livestream
- **onnx_to_rknn.py**: This file almost made me break my comptuer in frustration. Took 2 hours to setup but finally I got a .rknn file from my linux server, I just hope I set the paramters correct.

## Folders

# 2026 Vision Testing

okay so this repo is kinda messy but it works. i'm leaving notes here so anyone (including future me) doesn't cry when stuff breaks.

basically: the "simplified" folder just finds fuel and sends positions over network tables. the "complex" one does that plus plotting, path stuff, and other nerdy things.

### what the important scripts do (quick and lazy version)

- video_feed_broadcaster.py — makes a tiny Flask site for the camera livestream
- yolo_with_cam.py — runs the chosen YOLO model on a camera (default id 0)
- pt_to_wtv.py — converts a .pt model to other formats (onnx, xml, etc)
- map_maker.py — interactive window to draw a fuel map
- map_creator.py — uses the model + math to make a birdseye view from images in `Images/`
- camera_calibration.py — run this to get focal length-ish constants. follow the prompts, measure stuff, then paste the results into `focal_calibration_data.txt`. average the "known_calibration_pixel_height" values.
- livestream_reader.py — reads frames from a livestream using the Camera class
- onnx_to_rknn.py — super annoying conversion script that finally made an `.rknn` file (took a while)

## folders

COMPLEX
- does path planning and plotting along with fuel detection. more features, more slow, more chaos.

SIMPLIFIED
- just detects fuel and sends positions over network tables. less lag, simpler, probably what you want for comp.

PLOTTERS
- experiments and plotting utilities (spline stuff, dbscan testing, etc)

YOLO_MODELS
- where all the YOLO models and exports live (.pt, .onnx, etc). mostly sorted but sometimes i'm lazy and dump files.

### notable files inside the folders

COMPLEX classes:
- Camera.py — camera class with lots of settings
- CustomDBScan.py — custom DBSCAN filtering for points
- NetworkTableHandler.py — send data over network tables
- PathPlanner.py — path planner for the complex flow

SIMPLIFIED classes:
- Camera.py — camera class (simpler usage)
- CustomDBScan.py — point filtering
- Fuel.py — fuel object
- FuelTracker.py — small helper for tracking fuel
- NetworkTableHandler.py — network tables helper
- PathPlanner.py — smaller path planner for simplified flow

Other small files:
- constants.py — repo constants
- game_loop.py / solo_game_loop.py — ways to run the camera loop (solo_game_loop usually works fine)


## Orange Pi 5 stuff

- IP: 10.22.7.200
- Username: ubuntu
- Password: 2207vision
- Auto-scheduler: run `bach crontab -e`
- To clone: `git clone git@github.com:FRC2207/2026-Vision-Testing.git`

I left the board creds here so it's easy to connect. yeah i know, maybe not the best security but it's the dev box lol.


# speed table (approx)
| Model   | Size   | Latency | Estimated FPS |
|:-------:|:------:|:-------:|--------------:|
| Yolov26 | nano   | ~99.5ms | ~10           |
| Yolov11 | nano   | 71.5ms  | ~14           |
| Yolov11 | small  | 98.9ms  | ~10           |
| Yolov11 | medium | 235.3ms | ~4            |
| Yolov8  | nano   | ~20-30ms| ~30-50        |
| Yolov8  | small  | ~40-60ms| ~15-25        |
| Yolov5  | nano   | ~15-25ms| ~40-60        |


## common commands
Use these when you're on the pi or dev box:

```powershell
sudo systemctl restart vision.service   # restart the vision process
tmux attach -t vision                   # attach to the vision tmux session
git pull                                # pull latest code
git reset --hard HEAD                   # undo local changes
git clean -fd                           # remove untracked files
```


## notes and warnings (real talk)
- repo is a little rough around the edges. files moved around a bunch while i was testing stuff.
- if you're converting models, double-check export settings — i've lost hours to tiny mistakes.
- simplified path is the one i'd use for comp or low-latency setups.