# Franka Auto Hand-Eye Calibration (pylibfranka) — Design

**Date:** 2026-09-21  
**Status:** Draft for review (rev 2: pose-source/TF-bridge split, sampler timing, corrected service usage)  
**Context:** Automate eye-on-base sampling after free-driving the marker into camera center, using A-style pose offsets and a vendored full `FrankaInterface` (Option 1).

## Problem

Manual freehand calibration with the `easy_handeye2` GUI works, but repeating offset poses by hand is slow and inconsistent. The camera location changes between sessions, so absolute taught poses are not reusable. The operator free-drives once so the EE-mounted marker is centered in the camera, then wants automated relative offsets and automatic samples.

## Goals

- Eye-on-base only (camera fixed; marker on EE).
- Free-drive to a “camera center” start pose, then run A-style rotation/translation offsets relative to that start.
- Move the arm with vendored `pro_robot_interface.FrankaInterface` (`reset_to_start_pose`).
- Automatically call `easy_handeye2` `take_sample` after each settle; then compute and save.
- Keep stock `easy_handeye2` sampling/calibration path; do not rely on MoveIt for this workflow.
- Deliver as a fork/self-contained package that vendors the full Franka PRO interface (Option 1).

## Non-goals (v1)

- Eye-in-hand automation.
- MoveIt / `franka_ros2` hardware control during the auto run.
- Slimming or rewriting the Franka interface (Option 2 deferred).
- Collision-aware planning or marker FOV vision checks.
- Changing upstream OpenCV calibration math.

## Architecture

```
[Operator free-drives marker to camera center]
                 │
                 ▼
┌────────────────────────────────────────────┐
│  New package (fork / sibling of handeye)   │
│  handeye_auto_calibrate.py                 │
│    ├─ FrankaInterface (vendored PRO)       │
│    │     refresh_state_snapshot / reset    │
│    ├─ handeye_offsets.py (A-style 4x4s)    │
│    ├─ robot_pose_source.py (lock + cache)  │
│    ├─ handeye_tf_bridge.py → /tf (cache)   │
│    └─ HandeyeClient → take_sample/etc.     │
└─────────────────┬──────────────────────────┘
                  │
     pylibfranka  │  ROS 2 services + tf
                  ▼
         easy_handeye2 server (unchanged)
         + camera/marker tf publishers
```

### Critical constraint: robot ownership and TF

- `easy_handeye2` samples **only from `tf`** (robot + tracking).
- Franka FCI allows **one** control client. During auto-cal: **pylibfranka owns the robot**; `franka_ros2` hardware/controllers must be **off**.
- Therefore the automator **must publish robot `tf`** from `FrankaInterface` state (bridge). Camera/marker stay on ROS 2 as today.

ROS 2 nodes and pylibfranka **do** run in parallel; they must not both command the arm.

### Verified facts that shape the design

- **The sampler reads TF at `now − 0.2 s`** (`handeye_sampler._get_transforms`) for both the robot and tracking transforms, and tf2 interpolates between stamps. Robot TF must therefore already hold the *new* pose for ≥ 0.2 s (plus margin) before `take_sample`, or the sample can lerp between old and new poses.
- **`FrankaInterface.refresh_state_snapshot()` is slow and exclusive.** It puts `("sample_state",)` on the comm-process command queue, opens a Cartesian control session, reads state, stops, and sleeps 0.25 s before replying. It shares `_cmd_queue` / `_response_queue` with `reset_to_start_pose()`. So it cannot run at 50 Hz, and it must never run concurrently with a move (responses would be consumed by the wrong caller).
- `take_sample` returns the sample list whether or not a sample was appended; success = list length grew.
- `compute_calibration` returns `.valid` and `.calibration`; `save` returns `.success`. Services live at the absolute `/easy_handeye2/calibration/*` topics (independent of calibration name).

## Package layout (fork)

Package name: `easy_handeye2_franka_auto`. Lives in the forked repo beside `easy_handeye2` / `easy_handeye2_msgs`.

### Vendored from Continuous_Force_RL (Option 1 — full copy)

| Source | Role |
|--------|------|
| `real_robot_exps/pro_robot_interface.py` | Process-based `FrankaInterface` |
| `real_robot_exps/robot_interface.py` | `StateSnapshot`, math helpers, limits, safety |
| `real_robot_exps/hybrid_controller.py` | Required import for PRO compute process |
| `real_robot_exps/mock_pylibfranka.py` | Optional mock mode |
| `real_robot_exps/config.yaml` (robot section) | IP, `use_mock`, `reset_duration_sec`, frames (`NE_T_EE`, etc.) |

Imports are rewritten from `real_robot_exps.*` to the new package module path.

### New modules

| Module | Responsibility |
|--------|----------------|
| `handeye_offsets.py` | Port of `handeye_robot.CalibrationMovements._compute_poses_around_state` to produce a list of 4×4 EE poses from a home pose + `rotation_delta` / `translation_delta` |
| `robot_pose_source.py` | Sole owner of robot calls: one `RLock` around `refresh` / `move_to` / optional poll; cached EE pose + `moving` flag |
| `handeye_tf_bridge.py` | ROS 2 node/timer only: publish the cached pose as `robot_base_frame` → `robot_effector_frame`; never calls the robot |
| `handeye_auto_calibrate.py` | CLI entry: init robot + ROS, HandeyeClient for sample/compute/save, free-drive confirm, offset loop |

### Unchanged

- `easy_handeye2` calibrate launch with `freehand_robot_movement:=true`.
- OpenCV backend, sample storage, publish/evaluate launches.

## Runtime loop

1. Start camera/marker + `calibrate.launch.py` (freehand, `eye_on_base`). Confirm `franka_ros2` control is not connected.
2. Start `handeye_auto_calibrate` with robot config + frame names matching the handeye launch.
3. Prompt: free-drive until marker is centered → Enter. (No robot TF is required for this; see “Free-drive” below.)
4. `pose_source.refresh()`; `home` = cached EE 4×4. TF bridge starts publishing.
5. `targets = offsets_around(home, angle_delta, translation_delta)` (optionally truncated by `--first-n`).
6. For each target `T_i`:
   - `pose_source.move_to(T_i)` (holds the robot lock; bridge stops publishing while moving).
   - Sleep `settle_sec` (default 1.0 s) for mechanical settling. No velocity gate in v1.
   - `pose_source.refresh()` — one measured-pose read, cache updated, bridge resumes at the new pose.
   - Sleep `tf_dwell_sec` (default 0.6 s, must be > 0.2 s sampler lookback + margin) so robot TF holds the new pose across the sampler's lookup time.
   - `n0 = len(get_sample_list())`; `take_sample()`; success iff `len(get_sample_list()) > n0`. Otherwise skip, log, continue.
7. If sample count ≥ `min_samples` (default 5, and compute needs > 2): `compute_calibration`; if `.valid` → `save`; else report and do not save.
8. Optionally return to `home`. Shut down robot interface cleanly.

### Free-drive (open question)

`refresh_state_snapshot()` opens a short Cartesian control session, and it is **unverified** whether that works while the arm is being hand-guided (FCI active vs. Desk guiding mode). v1 therefore assumes the operator guides the arm with whatever mechanism works on their setup, presses Enter, and only then does the first `refresh()`. Live robot TF during free-drive is **optional** (`--freedrive-poll-hz`, default 0 = off); if enabled it polls at ≤ 2 Hz on the pose source's lock and is stopped before any motion. This must be checked on hardware (manual test 1) before relying on it.

### Offset definition (A)

Same structure as existing MoveIt helper:

- ± rotation about X/Y/Z at `angle_delta` and at `angle_delta/2` (quaternion multiply on EE orientation).
- Small translations: ±X (`translation_delta/2`), ±Y (`translation_delta`), +Z (`translation_delta/3`).

Parameters: `rotation_delta_degrees` (default ~25), `translation_delta_meters` (default ~0.1).

### Robot pose source + TF bridge

Two objects, so that exactly one place ever talks to the robot:

- `RobotPoseSource` (wraps `FrankaInterface`): owns a single `RLock` around **every** robot call (`refresh`, `move_to`, optional poll). Keeps a cached EE transform and a `moving` flag. `latest()` returns the cached transform (or `None` before the first refresh / while moving) without touching the robot.
- `RobotTfBridge` (ROS 2 node): timer at `tf_rate_hz` (default 30) that **only reads the cache** and publishes `robot_base_frame` → `robot_effector_frame` stamped `now`. It never calls `FrankaInterface`. While `moving` or before the first refresh it publishes nothing.

Consequences:

- Robot is stationary whenever it is sampled, so republishing a cached pose is exact, not an approximation.
- TF is continuous for the sampler's 2 s buffer and 0.2 s lookback; the post-refresh `tf_dwell_sec` guarantees the lookup lands after the first new-pose stamp.
- No queue contention: the executor thread never blocks on or races the main thread's robot calls.
- Frames must match handeye launch `robot_base_frame` / `robot_effector_frame`. For `eye_on_base` the sampler looks up the inverse (effector→base); tf2 handles this.
- Tracking TF remains the responsibility of the existing marker stack.

## Configuration

CLI and/or YAML covering:

- Path to robot `config.yaml` (IP, mock, reset duration, `NE_T_EE`, …).
- `robot_base_frame`, `robot_effector_frame`, handeye calibration `name`.
- `rotation_delta_degrees`, `translation_delta_meters`, `settle_sec`, `tf_dwell_sec`, `tf_rate_hz`, `min_samples`.
- `first_n` (run only the first N offset poses; for single-pose bring-up), `freedrive_poll_hz` (default 0), `return_home`.

## Error handling

| Condition | Behavior |
|-----------|----------|
| `reset_to_start_pose` / state-refresh / libfranka fault | Abort loop; leave arm stopped; report failing index; **skip compute/save** (samples stay on the server for manual GUI compute) |
| Sample not appended (missing/extrapolating TF) | Skip pose; continue (detected by sample-list length not growing) |
| `compute_calibration` returns `valid=false` | Do not save; report sample count |
| Unreachable / rejected target pose | Same as motion fault: abort loop at that index, arm stopped. v1 has no reachability precheck (the MoveIt helper had one); use modest deltas and `first_n` to probe |
| Ctrl+C mid-run | `finally` shuts down robot interface; no save |
| Too few samples at end (< `min_samples`) | Do not save; print counts |
| Comm/compute process death | Surface via existing `SafetyViolation` / init errors |

No collision checking in v1 — operator must choose a safe free-drive center and modest deltas.

## Dependencies

- System/robot: `pylibfranka`, network to Franka.
- Python: `numpy`, `torch` (required by vendored PRO interface).
- ROS 2: `rclpy`, `tf2_ros`, `geometry_msgs`, `easy_handeye2`, `easy_handeye2_msgs`.

## Testing

Unit (no robot, no ROS graph): offset generation (17 poses, correct rotation/translation structure), pose helpers, pose-source cache/lock/moving semantics with a fake robot, snapshot→Transform conversion, save/skip gating.

Manual (operator, hardware):

1. **Free-drive + state read:** hand-guide, Enter, confirm `refresh()` works and `tf2_echo <base> <ee>` shows the pose. Also decide whether `--freedrive-poll-hz` is usable on this setup (open question above).
2. **Single pose:** `--first-n 1`; one move, one sample, sample list length = 1.
3. **Full run:** complete offset set → compute → save under `~/.ros2/easy_handeye2/calibrations/`.
4. **Publish:** `publish.launch.py` with the same `name` shows the calibrated transform.
5. **Mock (optional):** `use_mock: true` dry-run of offset generation, pose source and service calls if the mock supports Cartesian reset sufficiently.

## Alternatives considered

| Approach | Verdict |
|----------|---------|
| A: Script in Continuous_Force_RL calling handeye services | Rejected in favor of forked self-contained package |
| B: Replace MoveIt backend inside `easy_handeye2` | Rejected — packaging pain tying handeye to pylibfranka/torch |
| C: Standalone thin package depending on both repos | Superseded by fork + Option 1 vendor |
| Slim vendor (Option 2) | Deferred; user chose full copy (Option 1) |
| MoveIt / franka_ros2 for motion | Valid alternate (stock TF) but not chosen; conflicts with pylibfranka control |

## Success criteria

- Operator free-drives once, runs one command, gets a saved eye-on-base calibration without GUI sample clicks.
- Pre-existing samples on the server are cleared at start (unless `--keep-existing-samples`) so counts and results reflect this run only.
- Robot TF holds the measured pose across the sampler's lookback window for every accepted sample, while pylibfranka is the sole robot client and only one thread ever calls it.
- Vendored PRO interface runs without importing `Continuous_Force_RL`.
