# easy_handeye2_franka_auto

Automated eye-on-base hand-eye sampling: free-drive the marker into camera
center, then run A-style EE offsets via a vendored pylibfranka
`FrankaInterface`, publish robot `tf` from a cached pose, and call
`easy_handeye2` sample/compute/save services.

Design: `docs/superpowers/specs/2026-09-21-franka-auto-handeye-design.md`

## Prerequisites

1. Start camera/marker publishers and `calibrate.launch.py` with
   `calibration_type:=eye_on_base`, matching `name` / robot frames, and
   freehand movement (default).
2. Ensure **`franka_ros2` hardware control is not connected** — pylibfranka
   must be the sole FCI client.
3. Build and source this workspace.

## Run

```bash
ros2 run easy_handeye2_franka_auto handeye_auto_calibrate \
  --robot-config $(ros2 pkg prefix easy_handeye2_franka_auto)/share/easy_handeye2_franka_auto/config/robot.yaml \
  --name my_eob_calib \
  --robot-base-frame fr3_link0 \
  --robot-effector-frame fr3_link8 \
  --return-home
```

Adjust frame names to your setup. `config/robot.yaml` defaults to
`use_mock: true`; set `use_mock: false` and the robot IP for hardware.

Useful flags:

- `--first-n 1` — single-pose bring-up
- `--tf-dwell-sec 0.6` — hold after refresh so TF covers the sampler's 0.2 s lookback
- `--freedrive-poll-hz 0` — default; set `1` only after confirming hand-guiding still works
- `--keep-existing-samples` — do not clear samples already on the handeye server

Pre-existing samples are cleared at start unless `--keep-existing-samples`.

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
