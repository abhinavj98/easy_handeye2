# Franka FR3 eye-on-base calibration with ChArUco + RealSense

This is an end-to-end guide for calibrating a fixed RealSense camera against a
Franka FR3. The FR3 moves a ChArUco board held in the gripper, and the result is
the pose of the camera in the robot base frame, `fr3_link0 → camera_link`.

Three packages in this repo are involved:

| Package | Role |
|---|---|
| `easy_handeye2` (upstream) | `handeye_server`: takes samples from TF, solves AX = XB (OpenCV), saves and publishes the result |
| `easy_handeye2_charuco` | ChArUco detector (`charuco_tf_publisher`), board tools, bring-up launch files |
| `easy_handeye2_franka_auto` | `handeye_auto_calibrate`: moves the robot via pylibfranka, triggers sampling, prints a quality report. Also has verification tools |

Command references: `easy_handeye2_charuco/README.md`,
`easy_handeye2_franka_auto/README.md`.

---

## 1. How it fits together

```
RealSense ──image + camera_info──► charuco_tf_publisher ──TF camera_color_optical_frame → charuco──┐
                                                                                                    ├─► handeye_server
handeye_auto_calibrate                                                                              │   (take_sample: looks up
  ├─ FrankaInterface (pylibfranka, the only FCI client) → cached EE pose                            │    both transforms in TF)
  ├─ RobotTfBridge ──TF fr3_link0 → handeye_ee (30 Hz)──────────────────────────────────────────────┘
  └─ HandeyeClient ──services──► take_sample / compute_calibration / save_samples / save_calibration
```

- The robot and the calibration only talk to each other through **TF** and
  services. `handeye_auto_calibrate` reads the robot pose directly from
  pylibfranka and publishes it to TF itself. It does not use `franka_ros2`
  or `robot_state_publisher`.
- **`franka_ros2` / `franka_bringup` must not run during calibration.**
  pylibfranka needs the robot's control connection (FCI) to itself.
- `handeye_server` advertises its services only after both
  `fr3_link0 → handeye_ee` and `camera_link → charuco` exist in TF. **The
  board must be visible when you start the calibration script**, or the script
  waits at "Waiting for sampling services".

### Frames

| Frame | Published by | Notes |
|---|---|---|
| `fr3_link0` | `handeye_auto_calibrate` (TF bridge) | robot base; the result is expressed in it |
| `handeye_ee` | `handeye_auto_calibrate` (TF bridge) | pylibfranka `O_T_EE` (fingertip midpoint). A dedicated name, so it can't clash with `fr3_link8` from `robot_state_publisher` |
| `camera_link` | RealSense driver | root of the camera TF tree; used as `tracking_base_frame` |
| `camera_color_optical_frame` | RealSense driver | parent frame of the board pose (the image's `frame_id`) |
| `charuco` | `charuco_tf_publisher` | board origin: outer top-left corner as generated; x along `squares_x`, z into the board |

The calibration is saved as `fr3_link0 → camera_link`. Because that is the root
of the RealSense tree, `publish.launch.py` attaches the whole camera tree to the
robot and no frame ends up with two parents. For the same reason, the bring-up
launch leaves out the dummy base→camera static TF from upstream's
`calibrate.launch.py`.

### What is and isn't calibrated

| Quantity | Source |
|---|---|
| Intrinsics K, D | RealSense factory calibration, via `/camera/camera/color/camera_info`. Used by the detector, never estimated |
| `camera_link → camera_color_optical_frame` | RealSense factory calibration (static TF) |
| **`fr3_link0 → camera_link`** | **This calibration** |
| Board mount on the EE (T_ee←board) | Estimated as a side result (the report's "Board in EE" line) |

You do **not** need to measure where the board sits on the gripper. The
board-to-EE offset is an unknown that drops out of AX = XB. It only has to stay
rigid and constant during the run. For the same reason, the choice of EE frame
(`handeye_ee`, `fr3_link8`, flange) and the Desk TCP setting don't affect the
result, as long as they don't change mid-run.

---

## 2. The board

Board geometry lives in one file, `easy_handeye2_charuco/config/charuco_board.yaml`.
The detector, the board generator and the tests all read it:

```yaml
squares_x: 5              # layout: 10 markers with IDs 0-9, 12 inner corners with IDs 0-11
squares_y: 4
dictionary: DICT_5X5_100  # marker bit patterns
square_length: 0.0222     # m. Sets the metric scale: MEASURE ON THE PRINT
marker_length: 0.0162     # m. Only the ratio to square_length matters (for detection)
```

- **Marker IDs are never set by hand.** OpenCV places IDs 0…N−1 in the white
  squares based on the layout and the dictionary. The pose is solved from the
  chessboard corners, and `square_length` sets their 3D positions.
- **Print the board this package generates**, so it matches the detector by
  construction:
  ```bash
  ros2 run easy_handeye2_charuco generate_board --out board.png --dpi 600
  ```
  Print at 100% ("actual size"). Measure a square with calipers and write the
  measured value into the yaml. A 5% error in `square_length` becomes a 5% error
  in the calibrated translation.
- Boards from other generators must match the dictionary and grid, start at ID 0,
  and use the **new (non-legacy)** layout from OpenCV 4.6 and later. The legacy
  layout matters for an even `squares_y`.
- Mount the board rigidly and flat, at any position and angle on the gripper.
- **Bigger is better.** At about 1.3 m camera distance, the 11 × 9 cm board
  limits accuracy to roughly 4–5 mm / 0.5°. An A4/A3 board (e.g. 7×5 squares at
  35–40 mm) or a closer camera improves this considerably.

After changing the yaml, rebuild `easy_handeye2_charuco`. The launch files read
the installed copy.

---

## 3. Setup

```bash
cd ~/connor/franka_ros2_ws
colcon build --packages-select easy_handeye2 easy_handeye2_msgs easy_handeye2_charuco easy_handeye2_franka_auto
source install/setup.bash
```

- `easy_handeye2_franka_auto/config/robot.yaml`: `use_mock: false`, `ip: 192.168.1.11`.
- OpenCV ≥ 4.7 for the system Python (`cv2.aruco.CharucoDetector`), installed with
  pip. Humble's apt OpenCV is 4.5.
- **Use the system Python.** The installed scripts run `/usr/bin/python3`. With
  conda active, running scripts through conda's `python3` breaks `cv_bridge`
  (`libgdal … TIFFReadRGBATileExt`). Use `ros2 run` / `ros2 launch`, or `conda deactivate`.
- Don't run the old `src/charuco` package alongside this one. Both publish the
  `charuco` frame.

---

## 4. Check detection

```bash
ros2 launch easy_handeye2_charuco charuco_view.launch.py        # camera + detector + RViz
ros2 run easy_handeye2_charuco charuco_pose_monitor              # rate + jitter
```

The debug image (`/charuco_tf_publisher/debug_image`) shows markers, corners,
board axes and a status line:

- green `corners N | dist Z m | reproj E px`: pose published
- red `NO POSE: <reason>`: not published. Poses with reprojection error above
  `max_reproj_err_px` (2 px) are also dropped.

What good looks like: all 12 corners, reprojection under about 0.5 px, `dist`
agreeing with a tape measure, and, with the board held still, jitter well under
1 mm / 0.1° at a detection rate close to the camera fps. The jitter at the
calibration distance is the floor for calibration accuracy.

---

## 5. Calibrate

```bash
# Terminal 1: camera + detector + handeye_server   (debug:=true: RViz + image view)
ros2 launch easy_handeye2_charuco eye_on_base_calib.launch.py name:=fr3_eob debug:=true

# Terminal 2: robot (franka_ros2 must NOT be running)
ros2 run easy_handeye2_franka_auto handeye_auto_calibrate \
  --robot-config $(ros2 pkg prefix easy_handeye2_franka_auto)/share/easy_handeye2_franka_auto/config/robot.yaml \
  --name fr3_eob --robot-base-frame fr3_link0 --robot-effector-frame handeye_ee \
  --n-poses 15 --seed 0 --return-home
```

1. The script connects, publishes the robot TF, and waits for `handeye_server`
   (the board must be visible).
2. **Free-drive** the board into the image center, about 0.4–0.6 m from the camera
   if the setup allows, then press Enter.
3. For each pose from a random subset of 54 (9 cube positions × 6 tilts) it
   moves with the DLS controller, settles, and calls `take_sample`. Poses that
   fail to move, repeat an earlier pose, or have no detection are skipped and counted.
4. It saves the raw samples (`~/.ros2/easy_handeye2/samples/fr3_eob.samples`),
   computes (Tsai-Lenz), **prints the quality report**, and saves the result
   (`~/.ros2/easy_handeye2/calibrations/fr3_eob.calib`). It needs at least 5 samples.

The `name` launch argument sets the file names. The robot frames must match
between the launch and the script flags.

**First time:** add `--first-n 3` to check motion, detection and sampling (it
won't save). Then do the full run.

**Tuning:**

| Flag | Default | When to change |
|---|---|---|
| `--rotation-delta-degrees` | 25 | lower (15–20) if the board is lost at tilt extremes. Higher gives better conditioning |
| `--cube-half-size-meters` | 0.05 | 0.08–0.10 for more translation variety |
| `--n-poses` | 15 | 20–25 for more samples |
| `--min-samples` | 5 | minimum needed to compute |

---

## 6. Verify

### Quality report (printed automatically; re-run any time)

```bash
ros2 run easy_handeye2_franka_auto evaluate_calibration --name fr3_eob
```

The board is rigid on the EE, so `T_ee←board = robot_i · calib · tracking_i`
must be the same for every sample. The report measures how much it varies:

| Section | What to look for |
|---|---|
| Camera pose (t, rpy) | matches a tape measure from `fr3_link0` to the camera within a few cm |
| Board in EE | matches where the board origin physically sits relative to the fingertip midpoint |
| Rotation diversity | ≥ 20° max EE rotation between samples |
| Per-sample fit / LOO | error of the robot-predicted vs camera-measured board pose in the base frame. LOO re-solves without that sample. Look for outliers |
| Solver agreement | Tsai/Park/Horaud should agree to about 1 mm. Andreff/Daniilidis are more sensitive to noise |
| Verdict | GOOD ≤ 3 mm / 0.5°, BAD ≥ 10 mm / 2° (worst of fit and LOO RMS) |

Example (first real run: 12 samples, small board, camera about 1.3 m away):
fit 4.2 mm / 0.53°, LOO 4.9 mm / 0.62° → **MARGINAL**. Uniform residuals with no
outliers, meaning it's limited by detection noise. A bigger board fixes this.

### Live check (new poses, not the calibration samples)

Calibration is finished, so `franka_bringup` may run now:

```bash
ros2 launch franka_bringup franka.launch.py robot_ip:=192.168.1.11
ros2 launch easy_handeye2_charuco charuco_view.launch.py use_rviz:=false
ros2 launch easy_handeye2 publish.launch.py name:=fr3_eob
ros2 run easy_handeye2_franka_auto handeye_consistency_monitor
```

Hand-guide the robot to 6–8 different still poses. The monitor records the
board-in-EE pose at each and prints its spread, using the same metric as the
report's fit column.

### Visual check

With the robot model (`franka_bringup`), `publish.launch.py`, and the RealSense
with `enable_depth:=true pointcloud.enable:=true`: in RViz with fixed frame
`fr3_link0`, the point cloud of the arm should lie on the robot mesh.

---

## 7. Using the result

```bash
ros2 launch easy_handeye2 publish.launch.py name:=fr3_eob
```

This publishes `fr3_link0 → camera_link`. Together with the RealSense TF,
`fr3_link0 → camera_color_optical_frame` is then available. To map pixels
to and from the base frame, combine it with K from `camera_info`:

```
T_base←optical = T_base←camera_link (calibration) · T_camera_link←optical (RealSense)
pixel          = K · project( inv(T_base←optical) · p_base )
```

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Script stuck at "Waiting for sampling services" | a TF is missing: board not detected, or frame names differ between the launch and the script. Check `tf2_echo camera_link charuco` / `tf2_echo fr3_link0 handeye_ee` |
| pylibfranka can't connect / control conflicts | `franka_ros2` is running. Stop it |
| Red "insufficient corners" | wrong dictionary/grid, board too far away or too small, glare, blur |
| `dist` consistently off by a factor | `square_length` / `marker_length` don't match the print |
| Many `sample_failed` in the summary | board lost at tilt extremes: reduce `--rotation-delta-degrees`, check with `debug:=true` |
| Many `duplicate_skipped` | DLS stops short of targets near singularities or joint limits. Start from a more central home pose |
| Verdict BAD / camera pose implausible | board not rigid, wrong board size, frame mix-up. Look for single outlier samples in the report |
| `cv_bridge` ImportError (`libgdal`) | conda Python. Use `ros2 run` or `conda deactivate` |
| `charuco` frame jumps between two poses | two detectors running (the old `src/charuco` package) |

See also `docs/troubleshooting.md` (upstream, general).
