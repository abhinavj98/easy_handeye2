# Franka Auto Hand-Eye Calibration (pylibfranka) — Design

**Date:** 2026-09-21  
**Status:** Draft for review  
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
│    ├─ handeye_tf_bridge.py → /tf           │
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
| `handeye_tf_bridge.py` | ROS 2 node/timer only: refresh robot state as needed and publish `robot_base_frame` → `robot_effector_frame` |
| `handeye_auto_calibrate.py` | CLI entry: init robot + ROS, HandeyeClient for sample/compute/save, free-drive confirm, offset loop |

### Unchanged

- `easy_handeye2` calibrate launch with `freehand_robot_movement:=true`.
- OpenCV backend, sample storage, publish/evaluate launches.

## Runtime loop

1. Start camera/marker + `calibrate.launch.py` (freehand). Confirm `franka_ros2` control is not connected.
2. Start `handeye_auto_calibrate` with robot config + frame names matching handeye launch.
3. Prompt: free-drive until marker is centered → Enter.
4. `refresh_state_snapshot()`; `home` = current EE 4×4.
5. `targets = offsets_around(home, angle_delta, translation_delta)`.
6. For each target `T_i`:
   - `reset_to_start_pose(T_i)`.
   - Wait fixed `settle_sec` (default 1.0 s). No velocity gate in v1.
   - If `get_current_transforms` missing robot or tracking → skip and log; else `take_sample()`.
7. If successful samples ≥ `min_samples` (default 5): `compute_calibration` → `save_calibration`; else abort save and report count.
8. Optionally return to `home`. Shutdown robot interface cleanly.

### Offset definition (A)

Same structure as existing MoveIt helper:

- ± rotation about X/Y/Z at `angle_delta` and at `angle_delta/2` (quaternion multiply on EE orientation).
- Small translations: ±X (`translation_delta/2`), ±Y (`translation_delta`), +Z (`translation_delta/3`).

Parameters: `rotation_delta_degrees` (default ~25), `translation_delta_meters` (default ~0.1).

### TF bridge

- Rate: configurable, default 50 Hz.
- Each timer tick: `refresh_state_snapshot()` (or equivalent non-moving state sample), then publish `ee_pos`+`ee_quat` as `robot_base_frame` → `robot_effector_frame`.
- This keeps robot `tf` live during free-drive wait and between offset moves (PRO does not stream state without a sample/control session).
- Frames must match handeye launch `robot_base_frame` / `robot_effector_frame`.
- Tracking TF remains the responsibility of the existing marker stack.
- Note: frequent `refresh_state_snapshot` starts short Cartesian sessions; if that proves too heavy, v1.1 can batch refresh (e.g. only after each `reset_to_start_pose` plus a slower free-drive poll). For v1, prefer correct live TF over minimizing FCI chatter.

## Configuration

CLI and/or YAML covering:

- Path to robot `config.yaml` (IP, mock, reset duration, `NE_T_EE`, …).
- `robot_base_frame`, `robot_effector_frame`, handeye calibration `name`.
- `rotation_delta_degrees`, `translation_delta_meters`, `settle_sec`, `tf_rate_hz`, `min_samples`.

## Error handling

| Condition | Behavior |
|-----------|----------|
| `reset_to_start_pose` / libfranka fault | Abort loop; leave arm stopped; report failing index |
| Missing TF at sample time | Skip pose; continue |
| Too few samples at end | Do not save; print counts |
| Comm/compute process death | Surface via existing `SafetyViolation` / init errors |

No collision checking in v1 — operator must choose a safe free-drive center and modest deltas.

## Dependencies

- System/robot: `pylibfranka`, network to Franka.
- Python: `numpy`, `torch` (required by vendored PRO interface).
- ROS 2: `rclpy`, `tf2_ros`, `geometry_msgs`, `easy_handeye2`, `easy_handeye2_msgs`.

## Testing (manual)

1. **TF only:** free-drive / idle with bridge publishing; `tf2_echo` / RViz shows robot EE updating without `franka_ros2` control.
2. **Single pose:** one offset move + one successful `take_sample`.
3. **Full run:** complete offset set → compute → save → `publish.launch.py` loads result.
4. **Mock (optional):** `use_mock: true` dry-run of offset generation + service calls if mock supports Cartesian reset sufficiently.

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
- Robot TF remains valid for every accepted sample while pylibfranka is the sole motion client.
- Vendored PRO interface runs without importing `Continuous_Force_RL`.
