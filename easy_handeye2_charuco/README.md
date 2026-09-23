# easy_handeye2_charuco

ChArUco board detection for easy_handeye2. The package contains:

- `charuco_tf_publisher`: detects the board in the RealSense color image and
  publishes TF `camera_color_optical_frame → charuco`.
- Visual test tools: an RViz/image-view launch, a pose jitter monitor, and a
  board PNG generator.
- `eye_on_base_calib.launch.py`: a single launch for camera + detector +
  `handeye_server`. After it is up, run `handeye_auto_calibrate`
  (`easy_handeye2_franka_auto`).

End-to-end guide (concepts, frames, workflow, verification): [docs/franka_eye_on_base_calibration.md](../docs/franka_eye_on_base_calibration.md)

## 1. Check the board

```bash
# Printable board at true scale (geometry from config/charuco_board.yaml)
ros2 run easy_handeye2_charuco generate_board --out charuco_board.png --dpi 600

# Camera + detector + RViz (TF axes + debug image)
ros2 launch easy_handeye2_charuco charuco_view.launch.py
#   use_image_view:=true   also opens rqt_image_view on the debug image
#   use_camera:=false      if realsense2_camera is already running

# Detection rate and pose jitter (hold the board still)
ros2 run easy_handeye2_charuco charuco_pose_monitor
```

The debug image (`/charuco_tf_publisher/debug_image`) draws the detected
markers, the corners and the board axes, plus one status line:
`corners N | dist Z m | reproj E px` in green, or `NO POSE: <reason>` in red.

What to check:
- **Distance:** `dist` roughly matches a tape measure from the camera to the
  board origin (the first corner, where the red and green axes meet). If it is
  consistently off by a scale factor, `square_length` / `marker_length` are wrong.
- **Jitter:** with the board still, the monitor shows well under 1 mm and
  about 0.1 deg of jitter, and a detection rate near the camera fps.
- **Reprojection:** under about 0.5 px. The node drops poses above
  `max_reproj_err_px` (2.0) instead of publishing them.

## 2. Calibrate (eye-on-base)

```bash
# Terminal 1: camera + detector + handeye_server
#   debug:=true opens RViz (camera + charuco TF axes, debug image) and rqt_image_view
ros2 launch easy_handeye2_charuco eye_on_base_calib.launch.py name:=fr3_eob debug:=true

# Terminal 2: robot motion + sampling (franka_ros2 must NOT be running)
ros2 run easy_handeye2_franka_auto handeye_auto_calibrate \
  --robot-config $(ros2 pkg prefix easy_handeye2_franka_auto)/share/easy_handeye2_franka_auto/config/robot.yaml \
  --name fr3_eob --robot-base-frame fr3_link0 --robot-effector-frame handeye_ee

# Afterwards: publish the result
ros2 launch easy_handeye2 publish.launch.py name:=fr3_eob
```

- `handeye_server` advertises its services only once both `fr3_link0 →
  handeye_ee` (published by `handeye_auto_calibrate`) and `camera_link →
  charuco` are in TF. **The board must be visible when you start
  `handeye_auto_calibrate`**, or the script waits at "Waiting for sampling
  services".
- The robot is not started here. `handeye_auto_calibrate` drives it through
  pylibfranka, which has to be the only FCI client.
- The result is `fr3_link0 → camera_link`, the root of the RealSense TF tree,
  so `publish.launch.py` attaches the whole camera tree to the robot base. This
  launch deliberately leaves out the dummy base→camera static TF from
  `easy_handeye2/calibrate.launch.py`, because it would conflict with the
  published result.
- `use_rqt:=true` adds the easy_handeye2 GUI so you can watch the sample list.
  Don't click "take sample" during an automated run.

| Launch arg | Default | Must match |
|---|---|---|
| `name` | `fr3_eye_on_base` | `publish.launch.py name:=` |
| `debug` | `false` | (RViz + image view; `use_rviz` / `use_image_view` pick one) |
| `robot_base_frame` | `fr3_link0` | `--robot-base-frame` |
| `robot_effector_frame` | `handeye_ee` | `--robot-effector-frame` |
| `tracking_base_frame` | `camera_link` | |
| `tracking_marker_frame` | `charuco` | |
| `color_profile` | `1280,720,30` | |
| `board_config` | `config/charuco_board.yaml` | the printed board |

## Board configuration

`config/charuco_board.yaml` is the single source of truth for board geometry.
The detector, `generate_board` and the tests all read it. Measure the printed
squares with calipers and put the measured values in the yaml, because a scale
error goes directly into the calibrated translation. You can also pass
`board_config:=/path/to/other.yaml`.

## Requirements

- **OpenCV ≥ 4.7** (`cv2.aruco.CharucoDetector`). Humble's apt `python3-opencv`
  is 4.5, so install `opencv-python` with pip for the system `/usr/bin/python3`.
  Don't also install `opencv-contrib-python`.
- **Run the nodes with the system Python.** The installed scripts use
  `#!/usr/bin/python3`. If a conda env is active, `python3` from the shell can
  fail to load `cv_bridge` (`libgdal ... undefined symbol TIFFReadRGBATileExt`).
  Use `ros2 run` / `ros2 launch`, or call `/usr/bin/python3` explicitly.
- `realsense2_camera` (tested with its `rs_launch.py`), `easy_handeye2`.

## Tests

```bash
cd easy_handeye2_charuco
PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest test/ -v
```

`test_board_pose.py` renders the board at known poses and checks that
`estimate_board_pose` recovers them to within 1 mm / 0.5°. No camera is needed.
