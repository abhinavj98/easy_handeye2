# Franka Auto Hand-Eye Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add package `easy_handeye2_franka_auto` that free-drive-captures a start pose, moves through A-style EE offsets via vendored pylibfranka `FrankaInterface`, publishes robot `tf`, and auto-samples/saves an eye-on-base calibration through `easy_handeye2`.

**Architecture:** Sibling ROS 2 Python package vendors the full Continuous_Force_RL PRO robot stack (Option 1). New modules generate 4×4 offset poses, bridge EE pose to `/tf`, and drive the sample/compute/save loop. Stock `easy_handeye2` stays freehand; `franka_ros2` hardware must be off while pylibfranka owns the arm.

**Tech Stack:** ROS 2 (`rclpy`, `tf2_ros`), `easy_handeye2` / `easy_handeye2_msgs`, `pylibfranka`, `numpy`, `torch`, `pytest`, `PyYAML`

**Spec:** `docs/superpowers/specs/2026-09-21-franka-auto-handeye-design.md`

## Global Constraints

- Eye-on-base only in v1.
- Do not run `franka_ros2` hardware control in parallel with pylibfranka.
- Do not slim/rewrite `FrankaInterface` (full vendor copy).
- Fixed settle wait only (default `settle_sec=1.0`); no velocity gate in v1.
- Default `min_samples=5`; do not save if fewer successful samples.
- TF bridge frame names must match handeye launch `robot_base_frame` / `robot_effector_frame`.
- Package name: `easy_handeye2_franka_auto`.
- Vendor source: `/home/skand/connor/Continuous_Force_RL/real_robot_exps/`.

## File Structure

```
easy_handeye2_franka_auto/
  package.xml
  setup.py
  setup.cfg
  resource/easy_handeye2_franka_auto
  config/robot.yaml
  README.md
  easy_handeye2_franka_auto/
    __init__.py
    robot_interface.py
    pro_robot_interface.py
    hybrid_controller.py
    mock_pylibfranka.py
    handeye_offsets.py
    handeye_tf_bridge.py
    handeye_auto_calibrate.py
  test/
    __init__.py
    test_handeye_offsets.py
    test_pose_helpers.py
```

---

### Task 1: Scaffold package + vendor PRO interface

**Files:**
- Create: `easy_handeye2_franka_auto/package.xml`
- Create: `easy_handeye2_franka_auto/setup.py`
- Create: `easy_handeye2_franka_auto/setup.cfg`
- Create: `easy_handeye2_franka_auto/resource/easy_handeye2_franka_auto`
- Create: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/__init__.py`
- Create: `easy_handeye2_franka_auto/config/robot.yaml`
- Create: `easy_handeye2_franka_auto/README.md`
- Copy + rewrite imports:
  - `easy_handeye2_franka_auto/easy_handeye2_franka_auto/robot_interface.py`
  - `easy_handeye2_franka_auto/easy_handeye2_franka_auto/pro_robot_interface.py`
  - `easy_handeye2_franka_auto/easy_handeye2_franka_auto/hybrid_controller.py`
  - `easy_handeye2_franka_auto/easy_handeye2_franka_auto/mock_pylibfranka.py`
- Test: `python3 -c "from easy_handeye2_franka_auto.pro_robot_interface import FrankaInterface"`

**Interfaces:**
- Consumes: Continuous_Force_RL source files listed above
- Produces: importable `easy_handeye2_franka_auto.pro_robot_interface.FrankaInterface`

- [ ] **Step 1: Create directories and empty package marker**

```bash
cd /home/skand/connor/franka_ros2_ws/src/easy_handeye2
mkdir -p easy_handeye2_franka_auto/easy_handeye2_franka_auto \
         easy_handeye2_franka_auto/resource \
         easy_handeye2_franka_auto/config \
         easy_handeye2_franka_auto/test
touch easy_handeye2_franka_auto/resource/easy_handeye2_franka_auto
touch easy_handeye2_franka_auto/easy_handeye2_franka_auto/__init__.py
touch easy_handeye2_franka_auto/test/__init__.py
```

- [ ] **Step 2: Write packaging files**

`setup.cfg`:

```ini
[develop]
script_dir=$base/lib/easy_handeye2_franka_auto
[install]
install_scripts=$base/lib/easy_handeye2_franka_auto
```

`setup.py`:

```python
import os
from glob import glob
from setuptools import setup

package_name = 'easy_handeye2_franka_auto'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools', 'numpy', 'torch', 'PyYAML'],
    zip_safe=True,
    maintainer='Local',
    maintainer_email='local@example.com',
    description='Automated eye-on-base hand-eye sampling via pylibfranka',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'handeye_auto_calibrate = easy_handeye2_franka_auto.handeye_auto_calibrate:main',
        ],
    },
)
```

`package.xml` — copy structure from `easy_handeye2/package.xml`, then set:

- package name `easy_handeye2_franka_auto`, version `0.1.0`
- description: automated eye-on-base hand-eye sampling via pylibfranka
- `exec_depend`: `rclpy`, `tf2_ros`, `geometry_msgs`, `easy_handeye2`, `easy_handeye2_msgs`
- `test_depend`: `python3-pytest`
- `export` → `build_type` = `ament_python`

- [ ] **Step 3: Copy vendor files**

```bash
SRC=/home/skand/connor/Continuous_Force_RL/real_robot_exps
DST=/home/skand/connor/franka_ros2_ws/src/easy_handeye2/easy_handeye2_franka_auto/easy_handeye2_franka_auto
cp "$SRC/robot_interface.py" "$DST/"
cp "$SRC/pro_robot_interface.py" "$DST/"
cp "$SRC/hybrid_controller.py" "$DST/"
cp "$SRC/mock_pylibfranka.py" "$DST/"
```

Copy robot section into `easy_handeye2_franka_auto/config/robot.yaml` from `$SRC/config.yaml` (keep `robot:` keys: `ip`, `use_mock`, `reset_duration_sec`, `NE_T_EE`, `EE_T_K`, etc.). Set `use_mock: true` in the committed default for safer first bring-up; operators override IP/`use_mock` locally.

- [ ] **Step 4: Rewrite imports in vendored `pro_robot_interface.py`**

Replace every `real_robot_exps.` import with `easy_handeye2_franka_auto.`:

| Old | New |
|-----|-----|
| `from real_robot_exps.robot_interface import` | `from easy_handeye2_franka_auto.robot_interface import` |
| `from real_robot_exps.hybrid_controller import` | `from easy_handeye2_franka_auto.hybrid_controller import` |
| `import real_robot_exps.mock_pylibfranka as plf` | `import easy_handeye2_franka_auto.mock_pylibfranka as plf` |

Apply in both top-level and nested `from ... import` inside `_comm_process_fn` / `_compute_process_fn` / gripper helpers (search the file for `real_robot_exps`).

- [ ] **Step 5: Verify import without robot**

```bash
cd /home/skand/connor/franka_ros2_ws
# Prefer colcon if env is ready; otherwise PYTHONPATH smoke test:
export PYTHONPATH=/home/skand/connor/franka_ros2_ws/src/easy_handeye2/easy_handeye2_franka_auto:$PYTHONPATH
python3 -c "from easy_handeye2_franka_auto.pro_robot_interface import FrankaInterface; print(FrankaInterface)"
```

Expected: prints class object; no `ModuleNotFoundError` for `real_robot_exps`.

- [ ] **Step 6: Commit**

```bash
cd /home/skand/connor/franka_ros2_ws/src/easy_handeye2
git add easy_handeye2_franka_auto
git commit -m "$(cat <<'EOF'
Add easy_handeye2_franka_auto package with vendored Franka PRO interface.

EOF
)"
```

---

### Task 2: A-style offset generation (`handeye_offsets.py`)

**Files:**
- Create: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_offsets.py`
- Test: `easy_handeye2_franka_auto/test/test_handeye_offsets.py`
- Test: `easy_handeye2_franka_auto/test/test_pose_helpers.py`

**Interfaces:**
- Consumes: none (pure numpy)
- Produces:
  - `snapshot_to_pose_4x4(ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray) -> np.ndarray` shape `(4,4)`
  - `compute_poses_around_state(home_pose_4x4: np.ndarray, angle_delta_rad: float, translation_delta_m: float) -> list[np.ndarray]`
  - Expected length: **17** poses (12 rotations + 5 translations), matching `handeye_robot._compute_poses_around_state`

- [ ] **Step 1: Write failing tests**

`test_pose_helpers.py`:

```python
import numpy as np
from easy_handeye2_franka_auto.handeye_offsets import snapshot_to_pose_4x4


def test_snapshot_to_pose_identity_rotation():
    pos = np.array([0.1, 0.2, 0.3])
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    T = snapshot_to_pose_4x4(pos, quat_wxyz)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, 3], pos)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])
```

`test_handeye_offsets.py`:

```python
import math
import numpy as np
from easy_handeye2_franka_auto.handeye_offsets import compute_poses_around_state


def _home():
    T = np.eye(4)
    T[:3, 3] = [0.4, 0.0, 0.4]
    return T


def test_offset_count_is_seventeen():
    poses = compute_poses_around_state(_home(), math.radians(25), 0.1)
    assert len(poses) == 17


def test_first_rotation_changes_orientation_not_translation():
    home = _home()
    poses = compute_poses_around_state(home, math.radians(25), 0.1)
    # First 12 are rotations about home position
    for T in poses[:12]:
        np.testing.assert_allclose(T[:3, 3], home[:3, 3], atol=1e-9)
        assert not np.allclose(T[:3, :3], home[:3, :3], atol=1e-6)


def test_translation_offsets_match_deltas():
    home = _home()
    d = 0.1
    poses = compute_poses_around_state(home, math.radians(25), d)
    translations = poses[12:]
    expected = [
        home[:3, 3] + np.array([d / 2, 0, 0]),
        home[:3, 3] + np.array([-d / 2, 0, 0]),
        home[:3, 3] + np.array([0, d, 0]),
        home[:3, 3] + np.array([0, -d, 0]),
        home[:3, 3] + np.array([0, 0, d / 3]),
    ]
    for T, exp in zip(translations, expected):
        np.testing.assert_allclose(T[:3, 3], exp, atol=1e-9)
        np.testing.assert_allclose(T[:3, :3], home[:3, :3], atol=1e-9)
```

- [ ] **Step 2: Run tests — expect fail**

```bash
cd /home/skand/connor/franka_ros2_ws/src/easy_handeye2/easy_handeye2_franka_auto
PYTHONPATH=$PWD pytest test/test_handeye_offsets.py test/test_pose_helpers.py -v
```

Expected: `ModuleNotFoundError` or import error for `handeye_offsets`.

- [ ] **Step 3: Implement `handeye_offsets.py`**

Port math from `easy_handeye2/easy_handeye2/handeye_robot.py` `_compute_poses_around_state` to 4×4 matrices (no `geometry_msgs`):

```python
"""A-style hand-eye pose offsets around a free-driven home EE pose."""
from __future__ import annotations

from itertools import chain
from typing import List

import numpy as np


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    # Shepperd's method (same idea as robot_interface helper)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def _quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def _quat_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Intrinsic XYZ euler to wxyz (matches transforms3d/handeye_robot usage)."""
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return np.array([w, x, y, z], dtype=float)


def snapshot_to_pose_4x4(ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_wxyz_to_R(np.asarray(ee_quat_wxyz, dtype=float))
    T[:3, 3] = np.asarray(ee_pos, dtype=float).reshape(3)
    return T


def compute_poses_around_state(
    home_pose_4x4: np.ndarray,
    angle_delta_rad: float,
    translation_delta_m: float,
) -> List[np.ndarray]:
    home = np.asarray(home_pose_4x4, dtype=float).copy()
    basis = np.eye(3)
    home_q = _R_to_quat_wxyz(home[:3, :3])

    final_rots = []
    for scale in (1.0, 0.5):
        pos_deltas = [_quat_from_euler_xyz(*(axis * angle_delta_rad * scale)) for axis in basis]
        neg_deltas = [_quat_from_euler_xyz(*(axis * (-angle_delta_rad * scale))) for axis in basis]
        final_rots.extend(chain.from_iterable(zip(pos_deltas, neg_deltas)))

    poses: List[np.ndarray] = []
    for qd in final_rots:
        q = _quat_multiply_wxyz(home_q, qd)
        T = home.copy()
        T[:3, :3] = _quat_wxyz_to_R(q)
        poses.append(T)

    for delta in (
        np.array([translation_delta_m / 2, 0.0, 0.0]),
        np.array([-translation_delta_m / 2, 0.0, 0.0]),
        np.array([0.0, translation_delta_m, 0.0]),
        np.array([0.0, -translation_delta_m, 0.0]),
        np.array([0.0, 0.0, translation_delta_m / 3]),
    ):
        T = home.copy()
        T[:3, 3] = home[:3, 3] + delta
        poses.append(T)

    return poses
```

- [ ] **Step 4: Run tests — expect pass**

```bash
cd /home/skand/connor/franka_ros2_ws/src/easy_handeye2/easy_handeye2_franka_auto
PYTHONPATH=$PWD pytest test/test_handeye_offsets.py test/test_pose_helpers.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_offsets.py \
        easy_handeye2_franka_auto/test/test_handeye_offsets.py \
        easy_handeye2_franka_auto/test/test_pose_helpers.py
git commit -m "$(cat <<'EOF'
Add A-style hand-eye EE offset generation with unit tests.

EOF
)"
```

---

### Task 3: TF bridge node

**Files:**
- Create: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_tf_bridge.py`
- Test: `easy_handeye2_franka_auto/test/test_tf_bridge_pose.py` (pure helper; no live robot)

**Interfaces:**
- Consumes: `FrankaInterface.refresh_state_snapshot()`, `get_state_snapshot()` → `ee_pos`, `ee_quat` (wxyz torch tensors)
- Produces:
  - `class RobotTfBridge(Node)` with constructor `(robot, base_frame: str, ee_frame: str, rate_hz: float = 50.0)`
  - Timer callback refreshes state and publishes `TransformStamped` parent=`base_frame`, child=`ee_frame`
  - `pose_msg_from_snapshot(snapshot) -> geometry_msgs.msg.Transform` helper for tests

- [ ] **Step 1: Write failing helper test**

```python
import numpy as np
from types import SimpleNamespace
import torch
from easy_handeye2_franka_auto.handeye_tf_bridge import transform_from_snapshot


def test_transform_from_snapshot_translation():
    snap = SimpleNamespace(
        ee_pos=torch.tensor([0.1, 0.2, 0.3]),
        ee_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),  # wxyz
    )
    t = transform_from_snapshot(snap)
    assert abs(t.translation.x - 0.1) < 1e-6
    assert abs(t.translation.y - 0.2) < 1e-6
    assert abs(t.translation.z - 0.3) < 1e-6
    assert abs(t.rotation.w - 1.0) < 1e-6
```

- [ ] **Step 2: Run test — expect fail**

```bash
PYTHONPATH=$PWD pytest test/test_tf_bridge_pose.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement `handeye_tf_bridge.py`**

```python
"""Publish robot base→EE tf from FrankaInterface snapshots."""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Transform, TransformStamped
from tf2_ros import TransformBroadcaster


def transform_from_snapshot(snapshot) -> Transform:
    pos = snapshot.ee_pos.detach().cpu().numpy().reshape(3)
    quat = snapshot.ee_quat.detach().cpu().numpy().reshape(4)  # wxyz
    t = Transform()
    t.translation.x = float(pos[0])
    t.translation.y = float(pos[1])
    t.translation.z = float(pos[2])
    t.rotation.w = float(quat[0])
    t.rotation.x = float(quat[1])
    t.rotation.y = float(quat[2])
    t.rotation.z = float(quat[3])
    return t


class RobotTfBridge(Node):
    def __init__(self, robot, base_frame: str, ee_frame: str, rate_hz: float = 50.0):
        super().__init__('handeye_robot_tf_bridge')
        self._robot = robot
        self._base_frame = base_frame
        self._ee_frame = ee_frame
        self._br = TransformBroadcaster(self)
        period = 1.0 / float(rate_hz)
        self._timer = self.create_timer(period, self._on_timer)

    def _on_timer(self):
        try:
            self._robot.refresh_state_snapshot()
            snap = self._robot.get_state_snapshot()
        except Exception as exc:  # noqa: BLE001 — keep bridge alive during free-drive
            self.get_logger().warn(f'TF bridge state refresh failed: {exc}')
            return
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.child_frame_id = self._ee_frame
        msg.transform = transform_from_snapshot(snap)
        self._br.sendTransform(msg)
```

- [ ] **Step 4: Run helper test — expect pass**

```bash
PYTHONPATH=$PWD pytest test/test_tf_bridge_pose.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_tf_bridge.py \
        easy_handeye2_franka_auto/test/test_tf_bridge_pose.py
git commit -m "$(cat <<'EOF'
Add Franka snapshot to robot TF bridge for hand-eye sampling.

EOF
)"
```

---

### Task 4: Auto-calibrate CLI loop

**Files:**
- Create: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_auto_calibrate.py`
- Modify: `easy_handeye2_franka_auto/README.md` (usage)
- Test: `easy_handeye2_franka_auto/test/test_auto_calibrate_logic.py` (pure helpers: skip/save gating)

**Interfaces:**
- Consumes:
  - `FrankaInterface(config, device="cpu")`
  - `compute_poses_around_state`, `snapshot_to_pose_4x4`
  - `RobotTfBridge`
  - `easy_handeye2.handeye_client.HandeyeClient`
  - `easy_handeye2_msgs.msg.HandeyeCalibrationParameters`
- Produces: console script `handeye_auto_calibrate` implementing the design runtime loop
- Helpers for tests:
  - `should_save(num_success: int, min_samples: int) -> bool`
  - `sample_ok(current_transforms) -> bool` — True iff transforms object is not None

- [ ] **Step 1: Write failing gating tests**

```python
from easy_handeye2_franka_auto.handeye_auto_calibrate import should_save, sample_ok


def test_should_save_requires_min_samples():
    assert should_save(5, 5) is True
    assert should_save(4, 5) is False


def test_sample_ok_none_is_false():
    assert sample_ok(None) is False
    assert sample_ok(object()) is True
```

- [ ] **Step 2: Run — expect fail**

```bash
PYTHONPATH=$PWD pytest test/test_auto_calibrate_logic.py -v
```

- [ ] **Step 3: Implement `handeye_auto_calibrate.py`**

Core structure (full file in package; keep helpers at module top):

```python
"""CLI: free-drive home → A-style offsets → take_sample → compute/save."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from easy_handeye2.handeye_client import HandeyeClient
from easy_handeye2_msgs.msg import HandeyeCalibrationParameters
from rclpy.executors import MultiThreadedExecutor

from easy_handeye2_franka_auto.handeye_offsets import (
    compute_poses_around_state,
    snapshot_to_pose_4x4,
)
from easy_handeye2_franka_auto.handeye_tf_bridge import RobotTfBridge
from easy_handeye2_franka_auto.pro_robot_interface import FrankaInterface


def should_save(num_success: int, min_samples: int) -> bool:
    return num_success >= min_samples


def sample_ok(current_transforms) -> bool:
    return current_transforms is not None


def _parse_args():
    p = argparse.ArgumentParser(description='Automated eye-on-base hand-eye sampling')
    p.add_argument('--robot-config', type=Path, required=True)
    p.add_argument('--name', required=True, help='easy_handeye2 calibration name')
    p.add_argument('--robot-base-frame', required=True)
    p.add_argument('--robot-effector-frame', required=True)
    p.add_argument('--rotation-delta-degrees', type=float, default=25.0)
    p.add_argument('--translation-delta-meters', type=float, default=0.1)
    p.add_argument('--settle-sec', type=float, default=1.0)
    p.add_argument('--tf-rate-hz', type=float, default=50.0)
    p.add_argument('--min-samples', type=int, default=5)
    p.add_argument('--return-home', action='store_true')
    return p.parse_args()


def main(args=None):
    cli = _parse_args()
    with open(cli.robot_config) as f:
        robot_cfg = yaml.safe_load(f)

    rclpy.init(args=args)
    robot = FrankaInterface(robot_cfg, device='cpu')

    node = rclpy.create_node('handeye_auto_calibrate')
    params = HandeyeCalibrationParameters(
        name=cli.name,
        calibration_type='eye_on_base',
        robot_base_frame=cli.robot_base_frame,
        robot_effector_frame=cli.robot_effector_frame,
        tracking_base_frame='',
        tracking_marker_frame='',
        freehand_robot_movement=True,
    )
    client = HandeyeClient(node, params)
    bridge = RobotTfBridge(
        robot,
        base_frame=cli.robot_base_frame,
        ee_frame=cli.robot_effector_frame,
        rate_hz=cli.tf_rate_hz,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(bridge)

    # Spin executor in background thread so service calls + tf timer work
    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    success = 0
    try:
        input('Free-drive marker into camera center, then press Enter...')
        robot.refresh_state_snapshot()
        snap = robot.get_state_snapshot()
        home = snapshot_to_pose_4x4(
            snap.ee_pos.detach().cpu().numpy(),
            snap.ee_quat.detach().cpu().numpy(),
        )
        targets = compute_poses_around_state(
            home,
            math.radians(cli.rotation_delta_degrees),
            cli.translation_delta_meters,
        )
        node.get_logger().info(f'Home captured; running {len(targets)} offset poses')

        for i, T in enumerate(targets):
            node.get_logger().info(f'Moving to pose {i}/{len(targets) - 1}')
            try:
                robot.reset_to_start_pose(T)
            except Exception as exc:
                node.get_logger().error(f'Motion failed at pose {i}: {exc}')
                break
            time.sleep(cli.settle_sec)
            # Ensure TF is fresh for sampler
            robot.refresh_state_snapshot()
            current = client.get_current_transforms()
            if not sample_ok(current):
                node.get_logger().warn(f'Skipping pose {i}: missing transforms')
                continue
            client.take_sample()
            success += 1
            node.get_logger().info(f'Sample ok ({success} total)')

        if should_save(success, cli.min_samples):
            result = client.compute_calibration()
            if result.valid:
                client.save()
                node.get_logger().info(f'Saved calibration ({success} samples)')
            else:
                node.get_logger().error('compute_calibration returned invalid; not saving')
        else:
            node.get_logger().error(
                f'Not saving: only {success} samples (need >= {cli.min_samples})'
            )

        if cli.return_home:
            robot.reset_to_start_pose(home)
    finally:
        robot.shutdown()
        executor.shutdown()
        bridge.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**Important:** `compute_calibration()` returns a response with `.valid` (see `handeye_rqt_calibrator_widget.handle_compute_calibration`). Only call `save()` when `result.valid` is True.

- [ ] **Step 4: Run gating tests — expect pass**

```bash
PYTHONPATH=$PWD pytest test/test_auto_calibrate_logic.py -v
```

- [ ] **Step 5: Write README usage**

Document:

1. Start camera/marker + `calibrate.launch.py` with matching frames and `name`, freehand movement.
2. Ensure `franka_ros2` control is **not** connected.
3. Build/source workspace.
4. Run:

```bash
ros2 run easy_handeye2_franka_auto handeye_auto_calibrate -- \
  --robot-config $(ros2 pkg prefix easy_handeye2_franka_auto)/share/easy_handeye2_franka_auto/config/robot.yaml \
  --name my_eob_calib \
  --robot-base-frame fr3_link0 \
  --robot-effector-frame fr3_hand \
  --return-home
```

(Adjust frame names to the operator’s setup.)

- [ ] **Step 6: Commit**

```bash
git add easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_auto_calibrate.py \
        easy_handeye2_franka_auto/test/test_auto_calibrate_logic.py \
        easy_handeye2_franka_auto/README.md
git commit -m "$(cat <<'EOF'
Add automated hand-eye calibrate loop using pylibfranka and TF bridge.

EOF
)"
```

---

### Task 5: Build + manual integration checklist

**Files:**
- Modify: `easy_handeye2_franka_auto/README.md` (checklist section)
- No new production code unless build reveals import gaps

**Interfaces:**
- Consumes: Tasks 1–4 deliverables
- Produces: verified install + documented manual test results (operator-run)

- [ ] **Step 1: Build package**

```bash
cd /home/skand/connor/franka_ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select easy_handeye2_franka_auto easy_handeye2 easy_handeye2_msgs
source install/setup.bash
ros2 pkg executables easy_handeye2_franka_auto
```

Expected: lists `handeye_auto_calibrate`.

- [ ] **Step 2: Run all unit tests**

```bash
cd /home/skand/connor/franka_ros2_ws/src/easy_handeye2/easy_handeye2_franka_auto
PYTHONPATH=$PWD pytest test/ -v
```

Expected: all PASS.

- [ ] **Step 3: Manual checklist (operator on robot)**

Append to README and execute when hardware available:

1. **TF only:** `use_mock: false`, start bridge via auto script up to free-drive prompt; `ros2 run tf2_ros tf2_echo <base> <ee>` updates while jogging in freedrive / after Enter refresh.
2. **Single pose:** temporarily limit targets to first pose (or Ctrl+C after one) and confirm one sample appears in handeye sample list / logs.
3. **Full run:** complete offsets → compute → save under `~/.ros2/easy_handeye2/calibrations/`.
4. **Publish:** `publish.launch.py` with same `name` shows calibrated transform.

- [ ] **Step 4: Commit README checklist updates**

```bash
git add easy_handeye2_franka_auto/README.md
git commit -m "$(cat <<'EOF'
Document manual bring-up checklist for Franka auto hand-eye.

EOF
)"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|------------------|------|
| Eye-on-base, free-drive then A-style offsets | Task 2 + 4 |
| Vendored full PRO interface (Option 1) | Task 1 |
| TF bridge while pylibfranka owns robot | Task 3 + 4 |
| Auto take_sample / compute / save | Task 4 |
| Skip missing TF; abort on motion fault; min_samples gate | Task 4 |
| No MoveIt / no franka_ros2 control during run | README + Global Constraints |
| Package `easy_handeye2_franka_auto` | Task 1 |
| Manual TF / single / full / publish tests | Task 5 |

## Placeholder / consistency notes

- `HandeyeCalibrationParameters` unused tracking frames are left empty; sampler uses whatever the running `handeye_server` was launched with — **operator must launch calibrate with the same `name` and robot frames as CLI**.
- Quaternion convention throughout: **wxyz** from `FrankaInterface`; TF `geometry_msgs` uses **xyzw** fields filled from wxyz components as shown in Task 3.
- Euler convention for offset deltas matches `handeye_robot` (XYZ half-angles via quat multiply on the right of home orientation).
