# easy_handeye2_franka_auto

Automated eye-on-base hand-eye sampling: free-drive the marker into camera
center, then run a random subset of cube-corner × EE-tilt poses via a
vendored pylibfranka `FrankaInterface`, publish robot `tf` from a cached
pose, and call `easy_handeye2` sample/compute/save services.

Design: `docs/superpowers/specs/2026-09-21-cube-subset-poses-design.md`  
(Package bring-up: `docs/superpowers/specs/2026-09-21-franka-auto-handeye-design.md`)

End-to-end guide (concepts, frames, workflow, verification): [docs/franka_eye_on_base_calibration.md](../docs/franka_eye_on_base_calibration.md)

## Pose set

From free-drive home:

- **Translations (base):** center + 8 cube corners at `(±d, ±d, ±d)`
- **Tilts (EE):** ±X, ±Y, ±Z at `--rotation-delta-degrees`
- **Pool:** 9 × 6 = 54 poses; default run samples **15** with `--seed`

Failed moves/samples are skipped (with `error_recovery` after a fault);
compute/save runs if enough samples remain. `--motion-mode` selects the
per-pose motion (default **`dls`**, a singularity-robust joint-space tracker
that stops at the closest safe pose instead of faulting; `cartesian` is the
original `reset_to_start_pose` streaming move). Either way the sample is
taken from the *measured* pose after the move, so a DLS move that stops
short still yields a correct sample — just possibly close to an earlier one.
`handeye_auto_calibrate` skips `take_sample` when the measured pose is
within `--min-translation-diff-m` **and** `--min-rotation-diff-deg` of a
pose already sampled, so short DLS stops don't add near-duplicate samples
(which add no rotation diversity to the AX=XB solve).

## Prerequisites

1. Start camera, ChArUco detector and `handeye_server` with
   `ros2 launch easy_handeye2_charuco eye_on_base_calib.launch.py name:=<name>`
   (see `easy_handeye2_charuco/README.md`), using the same `name` and robot
   frames as below, with the board visible to the camera.
2. Ensure **`franka_ros2` hardware control is not connected** — pylibfranka
   must be the sole FCI client.
3. Build and source this workspace.

## Motion-only smoke test (no camera / handeye)

Ensure `franka_ros2` control is off. Set `use_mock: false` and the robot IP in
`config/robot.yaml`, rebuild/source, then:

```bash
ros2 run easy_handeye2_franka_auto handeye_motion_test \
  --robot-config $(ros2 pkg prefix easy_handeye2_franka_auto)/share/easy_handeye2_franka_auto/config/robot.yaml \
  --first-n 3 \
  --rotation-delta-degrees 15 \
  --cube-half-size-meters 0.05 \
  --n-poses 15 \
  --seed 0
```

Free-drive to a safe pose, press Enter; it runs the first N selected poses and returns home.

## Full auto-calibrate

```bash
ros2 run easy_handeye2_franka_auto handeye_auto_calibrate \
  --robot-config $(ros2 pkg prefix easy_handeye2_franka_auto)/share/easy_handeye2_franka_auto/config/robot.yaml \
  --name my_eob_calib \
  --robot-base-frame fr3_link0 \
  --robot-effector-frame handeye_ee \
  --motion-mode dls \
  --cube-half-size-meters 0.05 \
  --n-poses 15 \
  --seed 0 \
  --return-home
```

Adjust frame names to your setup. `config/robot.yaml` defaults to
`use_mock: true`; set `use_mock: false` and the robot IP for hardware.

`--robot-effector-frame` must **not** be a frame `robot_state_publisher` (or
anything else) also publishes: `RobotTfBridge` publishes `O_T_EE`, the
fingertip-midpoint frame (`NE_T_EE` in `config/robot.yaml` is identity, while
Desk's `F_T_NE` already contains the 0.1034 m hand offset), not
`fr3_link8`/`panda_link8`. `handeye_ee` above is a dedicated name that avoids
the collision; any consistently-used EE frame gives a valid eye-on-base
result.

Useful flags:

- `--motion-mode dls` — default; falls back to `cartesian` for the original
  Cartesian streaming reset
- `--first-n 1` — single-pose bring-up (after subset selection)
- `--n-poses 15` / `--seed 0` — subset size and RNG seed
- `--cube-half-size-meters 0.05` — cube half-extent in the base frame
- `--tf-dwell-sec 0.6` — hold after refresh so TF covers the sampler's 0.2 s lookback
- `--min-translation-diff-m 0.005` / `--min-rotation-diff-deg 2.0` — how close a
  measured pose can be to an already-sampled one before it's skipped
- `--freedrive-poll-hz 0` — default; set `1` only after confirming hand-guiding still works
- `--keep-existing-samples` — do not clear samples already on the handeye server

Pre-existing samples are cleared at start unless `--keep-existing-samples`.

## Verify a calibration

Every run saves its raw samples to `~/.ros2/easy_handeye2/samples/<name>.samples`
(`<name>` is the `name` handeye_server was launched with). The board is rigid on
the EE, so the board pose `T_ee<-board = robot_i · calib · tracking_i` must be the
same for every sample. How much it varies is the calibration error.

**Quality report.** `handeye_auto_calibrate` prints it automatically right after
computing the calibration, before saving. A BAD verdict adds a warning, but the
result is still saved. To print it again later from the saved files (no robot needed):

```bash
ros2 run easy_handeye2_franka_auto evaluate_calibration --name fr3_eob
```

The report shows:
- the camera pose and board-in-EE pose, to check against a tape measure and
  the physical mount
- EE rotation diversity
- the base-frame board error for each sample under the saved calibration
  (fit), plus leave-one-out (each sample checked against a calibration solved
  without it)
- agreement between the five OpenCV solvers
- a verdict: GOOD ≤ 3 mm / 0.5°, BAD ≥ 10 mm / 2° (worst of fit and LOO RMS)

The exit code is 2 when the verdict is BAD. `--samples` / `--calibration`
evaluate arbitrary files.

**Live check** after publishing (moves nothing; hand-guide between still poses):

```bash
ros2 launch franka_bringup franka.launch.py robot_ip:=192.168.1.11   # live robot TF (after calibration!)
ros2 launch easy_handeye2_charuco charuco_view.launch.py use_rviz:=false
ros2 launch easy_handeye2 publish.launch.py name:=fr3_eob
ros2 run easy_handeye2_franka_auto handeye_consistency_monitor       # --effector fr3_link8 by default
```

It records the board-in-EE pose at each new still pose and prints its spread.
This uses the same metric as the offline fit, so aim for a few mm.

## Manual checklist

1. **Free-drive + state read:** hand-guide, Enter, confirm `refresh()` works and
   `ros2 run tf2_ros tf2_echo <base> <ee>` shows the pose. Optionally try
   `--freedrive-poll-hz 1`.
2. **Single pose:** `--first-n 1`; sample list length 1.
3. **Full run:** complete offsets → compute → save under
   `~/.ros2/easy_handeye2/calibrations/`.
4. **Publish:** `publish.launch.py` with the same `name`.

## Tests

```bash
source /opt/ros/humble/setup.bash
cd easy_handeye2_franka_auto
PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest test/ -v
```
