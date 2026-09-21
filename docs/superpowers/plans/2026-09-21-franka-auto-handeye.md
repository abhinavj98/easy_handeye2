# Franka Auto Hand-Eye Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add package `easy_handeye2_franka_auto` that free-drive-captures a start pose, moves through A-style EE offsets via vendored pylibfranka `FrankaInterface`, publishes robot `tf`, and auto-samples/saves an eye-on-base calibration through `easy_handeye2`.

**Architecture:** Sibling ROS 2 Python package vendors the full Continuous_Force_RL PRO robot stack (Option 1). New modules generate 4×4 offset poses, wrap the robot in a lock-guarded pose source, publish its *cached* EE pose to `/tf`, and drive the sample/compute/save loop. Stock `easy_handeye2` stays freehand; `franka_ros2` hardware must be off while pylibfranka owns the arm.

**Tech Stack:** ROS 2 (`rclpy`, `tf2_ros`), `easy_handeye2` / `easy_handeye2_msgs`, `pylibfranka`, `numpy`, `torch`, `pytest`, `PyYAML`

**Spec:** `docs/superpowers/specs/2026-09-21-franka-auto-handeye-design.md`

## Global Constraints

- Eye-on-base only in v1.
- Do not run `franka_ros2` hardware control in parallel with pylibfranka.
- Do not slim/rewrite `FrankaInterface` (full vendor copy).
- Fixed settle wait only (default `settle_sec=1.0`); no velocity gate in v1.
- Default `min_samples=5`; do not save if fewer successful samples.
- TF bridge frame names must match handeye launch `robot_base_frame` / `robot_effector_frame`.
- **Only `RobotPoseSource` may call `FrankaInterface`, always under its lock.** The TF bridge timer reads the cache only. (`refresh_state_snapshot` shares queues with `reset_to_start_pose`, takes ≥ 0.25 s, and cannot run at 50 Hz.)
- The handeye sampler reads TF at `now − 0.2 s`: after each move, refresh once and hold `tf_dwell_sec` (default 0.6) before `take_sample`.
- Success of `take_sample` = sample-list length grew (the service returns the list either way).
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
    robot_pose_source.py
    handeye_tf_bridge.py
    handeye_auto_calibrate.py
  test/
    __init__.py
    test_handeye_offsets.py
    test_pose_helpers.py
    test_robot_pose_source.py
    test_tf_bridge_pose.py
    test_auto_calibrate_logic.py
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

- [ ] **Step 4: Rewrite `real_robot_exps` references in all vendored files**

The old module path appears at top level and inside nested imports (in `pro_robot_interface.py`: lines ~42, 196, 207, 820, 823, 1043, 1305; other files may have more). Do a global replace and verify none remain:

```bash
cd "$DST"
sed -i 's/real_robot_exps\./easy_handeye2_franka_auto./g' robot_interface.py pro_robot_interface.py hybrid_controller.py mock_pylibfranka.py
grep -n "real_robot_exps" *.py && echo "LEFTOVERS - fix manually" || echo "clean"
```

Expected: `clean`. Also check for path-based references (`Path(__file__)`, `sys.path` hacks, `config.yaml` relative loads) and fix them:

```bash
grep -n "__file__\|sys.path\|config.yaml" *.py
```

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

### Task 3: Robot pose source (lock + cache)

**Files:**
- Create: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/robot_pose_source.py`
- Test: `easy_handeye2_franka_auto/test/test_robot_pose_source.py`

**Interfaces:**
- Consumes: `FrankaInterface.refresh_state_snapshot()`, `get_state_snapshot()` (→ `ee_pos`, `ee_quat` wxyz; torch tensors or arrays), `reset_to_start_pose(T)`
- Produces:
  - `class RobotPoseSource(robot)`
  - `refresh() -> None` — under robot lock: refresh state, cache `(pos[3], quat_wxyz[4])` as numpy; on failure invalidate cache and re-raise
  - `move_to(T_4x4) -> None` — under robot lock: **invalidate cache first**, then `reset_to_start_pose`; cache stays empty until the next `refresh()`
  - `latest() -> Optional[tuple[np.ndarray, np.ndarray]]` — cached copy or `None`; uses a *separate* small cache lock so it never blocks behind a slow robot call
  - `start_polling(hz)` / `stop_polling()` — optional background `refresh()` loop (free-drive TF); `stop_polling` joins the thread
- Must not import `torch`, `rclpy`, or `pylibfranka` (unit-testable anywhere).

- [ ] **Step 1: Write failing tests**

```python
import threading
import time
import numpy as np
from types import SimpleNamespace
from easy_handeye2_franka_auto.robot_pose_source import RobotPoseSource


class FakeRobot:
    def __init__(self, pos=(0.1, 0.2, 0.3), quat=(1.0, 0.0, 0.0, 0.0), delay=0.0):
        self.pos, self.quat, self.delay = np.array(pos), np.array(quat), delay
        self.calls = []
        self._active = 0
        self.max_concurrent = 0
        self._g = threading.Lock()

    def _enter(self, name):
        with self._g:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.calls.append(name)
        time.sleep(self.delay)
        with self._g:
            self._active -= 1

    def refresh_state_snapshot(self):
        self._enter("refresh")

    def get_state_snapshot(self):
        return SimpleNamespace(ee_pos=self.pos, ee_quat=self.quat)

    def reset_to_start_pose(self, T):
        self._enter("reset")
        self.pos = np.asarray(T)[:3, 3].copy()


def test_latest_none_before_refresh():
    assert RobotPoseSource(FakeRobot()).latest() is None


def test_refresh_populates_cache():
    src = RobotPoseSource(FakeRobot())
    src.refresh()
    pos, quat = src.latest()
    np.testing.assert_allclose(pos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(quat, [1, 0, 0, 0])


def test_move_invalidates_cache_until_refresh():
    robot = FakeRobot()
    src = RobotPoseSource(robot)
    src.refresh()
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.4]
    src.move_to(T)
    assert src.latest() is None
    src.refresh()
    np.testing.assert_allclose(src.latest()[0], [0.5, 0.0, 0.4])


def test_failed_refresh_invalidates_and_raises():
    robot = FakeRobot()
    src = RobotPoseSource(robot)
    src.refresh()

    def boom():
        raise RuntimeError("boom")

    robot.refresh_state_snapshot = boom
    try:
        src.refresh()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert src.latest() is None


def test_robot_calls_never_overlap_across_threads():
    robot = FakeRobot(delay=0.02)
    src = RobotPoseSource(robot)
    T = np.eye(4)
    threads = [threading.Thread(target=src.refresh) for _ in range(4)]
    threads += [threading.Thread(target=src.move_to, args=(T,)) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert robot.max_concurrent == 1


def test_latest_does_not_block_behind_robot_call():
    robot = FakeRobot(delay=0.3)
    src = RobotPoseSource(robot)
    src.refresh()
    t = threading.Thread(target=src.refresh)
    t.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    src.latest()
    assert time.monotonic() - t0 < 0.1
    t.join()


def test_polling_refreshes_and_stops():
    robot = FakeRobot()
    src = RobotPoseSource(robot)
    src.start_polling(50.0)
    time.sleep(0.15)
    src.stop_polling()
    n = robot.calls.count("refresh")
    assert n >= 2
    time.sleep(0.1)
    assert robot.calls.count("refresh") == n
```

- [ ] **Step 2: Run tests — expect fail (ModuleNotFoundError)**

```bash
cd /home/skand/connor/franka_ros2_ws/src/easy_handeye2/easy_handeye2_franka_auto
PYTHONPATH=$PWD pytest test/test_robot_pose_source.py -v
```

- [ ] **Step 3: Implement `robot_pose_source.py`**

```python
"""Single owner of FrankaInterface calls, plus a cached EE pose for TF."""
from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).copy()


class RobotPoseSource:
    def __init__(self, robot):
        self._robot = robot
        self._robot_lock = threading.RLock()   # serializes ALL robot calls
        self._cache_lock = threading.Lock()    # guards only the cache; never held during robot calls
        self._latest: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()

    def _set_cache(self, value):
        with self._cache_lock:
            self._latest = value

    def latest(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self._cache_lock:
            if self._latest is None:
                return None
            pos, quat = self._latest
            return pos.copy(), quat.copy()

    def refresh(self) -> None:
        with self._robot_lock:
            try:
                self._robot.refresh_state_snapshot()
                snap = self._robot.get_state_snapshot()
                pos = _to_numpy(snap.ee_pos).reshape(3)
                quat = _to_numpy(snap.ee_quat).reshape(4)  # wxyz
            except Exception:
                self._set_cache(None)
                raise
            self._set_cache((pos, quat))

    def move_to(self, target_pose_4x4: np.ndarray) -> None:
        with self._robot_lock:
            self._set_cache(None)  # pose is unknown until the next refresh()
            self._robot.reset_to_start_pose(np.asarray(target_pose_4x4, dtype=float))

    def start_polling(self, hz: float) -> None:
        if self._poll_thread is not None or hz <= 0:
            return
        self._poll_stop.clear()

        def _loop():
            while not self._poll_stop.wait(1.0 / hz):
                try:
                    self.refresh()
                except Exception as exc:  # noqa: BLE001 — keep polling during free-drive
                    _log.warning("pose poll failed: %s", exc)

        self._poll_thread = threading.Thread(target=_loop, daemon=True)
        self._poll_thread.start()

    def stop_polling(self) -> None:
        if self._poll_thread is None:
            return
        self._poll_stop.set()
        self._poll_thread.join()
        self._poll_thread = None
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=$PWD pytest test/test_robot_pose_source.py -v
```

- [ ] **Step 5: Commit**

```bash
git add easy_handeye2_franka_auto/easy_handeye2_franka_auto/robot_pose_source.py \
        easy_handeye2_franka_auto/test/test_robot_pose_source.py
git commit -m "$(cat <<'EOF'
Add lock-guarded robot pose source with cached EE pose.

EOF
)"
```

---

### Task 4: TF bridge node (cache-only)

**Files:**
- Create: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_tf_bridge.py`
- Test: `easy_handeye2_franka_auto/test/test_tf_bridge_pose.py`

**Interfaces:**
- Consumes: `RobotPoseSource.latest()`
- Produces:
  - `transform_from_pose(pos, quat_wxyz) -> geometry_msgs.msg.Transform`
  - `class RobotTfBridge(Node)` with constructor `(pose_source, base_frame: str, ee_frame: str, rate_hz: float = 30.0)`; timer publishes `base_frame → ee_frame` stamped `now` **only if** `latest()` is not `None`; **never calls the robot**

Requires a sourced ROS 2 environment (`rclpy`, `geometry_msgs`, `tf2_ros`).

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from easy_handeye2_franka_auto.handeye_tf_bridge import transform_from_pose


def test_transform_from_pose_maps_wxyz_to_xyzw_fields():
    t = transform_from_pose(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.1, 0.2, 0.3]))
    assert abs(t.translation.x - 0.1) < 1e-9
    assert abs(t.translation.y - 0.2) < 1e-9
    assert abs(t.translation.z - 0.3) < 1e-9
    assert abs(t.rotation.w - 0.5) < 1e-9
    assert abs(t.rotation.x - 0.1) < 1e-9
    assert abs(t.rotation.y - 0.2) < 1e-9
    assert abs(t.rotation.z - 0.3) < 1e-9
```

- [ ] **Step 2: Run test — expect fail**

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
PYTHONPATH=$PWD pytest test/test_tf_bridge_pose.py -v
```

- [ ] **Step 3: Implement `handeye_tf_bridge.py`**

```python
"""Publish the cached robot base→EE pose to /tf. Never touches the robot."""
from __future__ import annotations

from geometry_msgs.msg import Transform, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def transform_from_pose(pos, quat_wxyz) -> Transform:
    t = Transform()
    t.translation.x, t.translation.y, t.translation.z = (float(v) for v in pos)
    t.rotation.w = float(quat_wxyz[0])
    t.rotation.x = float(quat_wxyz[1])
    t.rotation.y = float(quat_wxyz[2])
    t.rotation.z = float(quat_wxyz[3])
    return t


class RobotTfBridge(Node):
    def __init__(self, pose_source, base_frame: str, ee_frame: str, rate_hz: float = 30.0):
        super().__init__('handeye_robot_tf_bridge')
        self._source = pose_source
        self._base_frame = base_frame
        self._ee_frame = ee_frame
        self._br = TransformBroadcaster(self)
        self._timer = self.create_timer(1.0 / float(rate_hz), self._on_timer)

    def _on_timer(self):
        latest = self._source.latest()
        if latest is None:  # before first refresh, or while moving: publish nothing
            return
        pos, quat = latest
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.child_frame_id = self._ee_frame
        msg.transform = transform_from_pose(pos, quat)
        self._br.sendTransform(msg)
```

- [ ] **Step 4: Run test — expect pass**

```bash
PYTHONPATH=$PWD pytest test/test_tf_bridge_pose.py -v
```

- [ ] **Step 5: Commit**

```bash
git add easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_tf_bridge.py \
        easy_handeye2_franka_auto/test/test_tf_bridge_pose.py
git commit -m "$(cat <<'EOF'
Add cache-only robot TF bridge for hand-eye sampling.

EOF
)"
```

---

### Task 5: Auto-calibrate CLI loop

**Files:**
- Create: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_auto_calibrate.py`
- Modify: `easy_handeye2_franka_auto/README.md` (usage)
- Test: `easy_handeye2_franka_auto/test/test_auto_calibrate_logic.py`

**Interfaces:**
- Consumes: `FrankaInterface(config, device="cpu")` (imported lazily inside `main`), `RobotPoseSource`, `RobotTfBridge`, `compute_poses_around_state`, `snapshot_to_pose_4x4`, `easy_handeye2.handeye_client.HandeyeClient`, `easy_handeye2_msgs.msg.HandeyeCalibrationParameters` (valid import; `easy_handeye2.handeye_calibration` re-exports it)
- Produces: console script `handeye_auto_calibrate`
- Pure helpers (unit-tested; module top level imports stdlib only, so tests need no ROS/robot stack):
  - `should_save(num_samples: int, min_samples: int) -> bool` — `num_samples >= max(min_samples, 3)` (compute needs > 2 samples)
  - `sample_added(n_before: int, n_after: int) -> bool`

Service facts (verified against the repo): `take_sample` returns the sample list whether or not a sample was appended; `compute_calibration` returns `.valid` / `.calibration`; `save` returns `.success`; the client needs the `handeye_server` from `calibrate.launch.py` already running (its constructor blocks on `wait_for_service`).

- [ ] **Step 1: Write failing tests**

```python
from easy_handeye2_franka_auto.handeye_auto_calibrate import should_save, sample_added


def test_should_save_requires_min_samples():
    assert should_save(5, 5) is True
    assert should_save(4, 5) is False


def test_should_save_never_below_three():
    assert should_save(2, 1) is False
    assert should_save(3, 1) is True


def test_sample_added_detects_list_growth():
    assert sample_added(0, 1) is True
    assert sample_added(3, 3) is False
```

- [ ] **Step 2: Run — expect fail**

```bash
PYTHONPATH=$PWD pytest test/test_auto_calibrate_logic.py -v
```

- [ ] **Step 3: Implement `handeye_auto_calibrate.py`**

```python
"""CLI: free-drive home → A-style offsets → take_sample → compute/save."""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path


def should_save(num_samples: int, min_samples: int) -> bool:
    return num_samples >= max(min_samples, 3)


def sample_added(n_before: int, n_after: int) -> bool:
    return n_after > n_before


def _parse_args(argv):
    p = argparse.ArgumentParser(description='Automated eye-on-base hand-eye sampling')
    p.add_argument('--robot-config', type=Path, required=True)
    p.add_argument('--name', required=True, help='easy_handeye2 calibration name (must match calibrate launch)')
    p.add_argument('--robot-base-frame', required=True)
    p.add_argument('--robot-effector-frame', required=True)
    p.add_argument('--rotation-delta-degrees', type=float, default=25.0)
    p.add_argument('--translation-delta-meters', type=float, default=0.1)
    p.add_argument('--settle-sec', type=float, default=1.0)
    p.add_argument('--tf-dwell-sec', type=float, default=0.6,
                   help='hold after refresh so TF covers the sampler 0.2 s lookback')
    p.add_argument('--tf-rate-hz', type=float, default=30.0)
    p.add_argument('--freedrive-poll-hz', type=float, default=0.0,
                   help='>0 polls robot state during free-drive (unverified on hardware; keep <=2)')
    p.add_argument('--min-samples', type=int, default=5)
    p.add_argument('--first-n', type=int, default=0, help='only run the first N offset poses (0 = all)')
    p.add_argument('--keep-existing-samples', action='store_true')
    p.add_argument('--return-home', action='store_true')
    return p.parse_args(argv)


def main(args=None):
    import rclpy
    import yaml
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.utilities import remove_ros_args

    from easy_handeye2.handeye_client import HandeyeClient
    from easy_handeye2_msgs.msg import HandeyeCalibrationParameters
    from easy_handeye2_franka_auto.handeye_offsets import (
        compute_poses_around_state,
        snapshot_to_pose_4x4,
    )
    from easy_handeye2_franka_auto.handeye_tf_bridge import RobotTfBridge
    from easy_handeye2_franka_auto.robot_pose_source import RobotPoseSource

    cli = _parse_args(remove_ros_args(args=sys.argv)[1:])
    with open(cli.robot_config) as f:
        robot_cfg = yaml.safe_load(f)

    rclpy.init(args=args)
    node = rclpy.create_node('handeye_auto_calibrate')
    log = node.get_logger()
    robot = source = bridge = executor = None
    try:
        from easy_handeye2_franka_auto.pro_robot_interface import FrankaInterface  # heavy: torch, pylibfranka

        robot = FrankaInterface(robot_cfg, device='cpu')
        source = RobotPoseSource(robot)
        bridge = RobotTfBridge(source, cli.robot_base_frame, cli.robot_effector_frame, cli.tf_rate_hz)

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.add_node(bridge)
        threading.Thread(target=executor.spin, daemon=True).start()  # needed for sync service calls

        params = HandeyeCalibrationParameters(
            name=cli.name,
            calibration_type='eye_on_base',
            robot_base_frame=cli.robot_base_frame,
            robot_effector_frame=cli.robot_effector_frame,
            freehand_robot_movement=True,
        )
        client = HandeyeClient(node, params)  # blocks until handeye_server services are up

        def n_samples() -> int:
            return len(client.get_sample_list().samples)

        existing = n_samples()
        if existing and not cli.keep_existing_samples:
            log.warn(f'Removing {existing} pre-existing samples (use --keep-existing-samples to keep)')
            while n_samples():
                client.remove_sample(0)

        source.start_polling(cli.freedrive_poll_hz)
        input('Free-drive marker into camera center, then press Enter...')
        source.stop_polling()

        source.refresh()
        pos, quat = source.latest()
        home = snapshot_to_pose_4x4(pos, quat)
        targets = compute_poses_around_state(
            home, math.radians(cli.rotation_delta_degrees), cli.translation_delta_meters)
        if cli.first_n > 0:
            targets = targets[:cli.first_n]
        log.info(f'Home captured; running {len(targets)} offset poses')

        motion_fault = False
        for i, T in enumerate(targets):
            log.info(f'Pose {i + 1}/{len(targets)}')
            try:
                source.move_to(T)
                time.sleep(cli.settle_sec)
                source.refresh()               # measured pose; bridge resumes publishing it
                time.sleep(cli.tf_dwell_sec)   # TF must hold the new pose across the sampler lookback
            except Exception as exc:  # noqa: BLE001
                log.error(f'Motion/state failure at pose index {i}: {exc}')
                motion_fault = True
                break
            before = n_samples()
            client.take_sample()
            if sample_added(before, n_samples()):
                log.info(f'Sample ok ({n_samples()} total)')
            else:
                log.warn(f'Skipping pose {i}: sample not recorded (missing/extrapolating TF?)')

        total = n_samples()
        if motion_fault:
            log.error(f'Aborted after motion fault; NOT computing/saving. {total} samples remain on the server.')
        elif should_save(total, cli.min_samples):
            result = client.compute_calibration()
            if result.valid:
                saved = client.save()
                log.info(f'Saved calibration ({total} samples): success={saved.success}')
            else:
                log.error('compute_calibration returned valid=false; not saving')
        else:
            log.error(f'Not saving: only {total} samples (need >= {max(cli.min_samples, 3)})')

        if cli.return_home and not motion_fault:
            source.move_to(home)
    finally:
        if source is not None:
            source.stop_polling()
        if robot is not None:
            robot.shutdown()
        if executor is not None:
            executor.shutdown()
        if bridge is not None:
            bridge.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run gating tests — expect pass**

```bash
PYTHONPATH=$PWD pytest test/test_auto_calibrate_logic.py -v
```

- [ ] **Step 5: Write README usage**

Document:

1. Start camera/marker + `calibrate.launch.py` with `calibration_type:=eye_on_base`, `freehand_robot_movement:=true`, and robot/tracking frames + `name` matching the CLI.
2. Ensure `franka_ros2` control is **not** connected (pylibfranka is the sole robot client).
3. Build/source workspace.
4. Run (no `--` separator; `ros2 run` forwards args directly):

```bash
ros2 run easy_handeye2_franka_auto handeye_auto_calibrate \
  --robot-config $(ros2 pkg prefix easy_handeye2_franka_auto)/share/easy_handeye2_franka_auto/config/robot.yaml \
  --name my_eob_calib \
  --robot-base-frame fr3_link0 \
  --robot-effector-frame fr3_hand \
  --return-home
```

(Adjust frame names to the operator's setup.) Explain `--first-n 1` for first bring-up, `--tf-dwell-sec`, and that pre-existing samples are cleared unless `--keep-existing-samples`.

- [ ] **Step 6: Commit**

```bash
git add easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_auto_calibrate.py \
        easy_handeye2_franka_auto/test/test_auto_calibrate_logic.py \
        easy_handeye2_franka_auto/README.md
git commit -m "$(cat <<'EOF'
Add automated hand-eye calibrate loop using pylibfranka and cached TF bridge.

EOF
)"
```

---

### Task 6: Build + manual integration checklist

**Files:**
- Modify: `easy_handeye2_franka_auto/README.md` (checklist section)
- No new production code unless build reveals import gaps

**Interfaces:**
- Consumes: Tasks 1–5 deliverables
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

1. **Free-drive + state read:** `use_mock: false`; run the script to the free-drive prompt. Hand-guide, press Enter, confirm `refresh()` succeeds and `ros2 run tf2_ros tf2_echo <base> <ee>` shows the pose. Separately try `--freedrive-poll-hz 1` and record whether hand-guiding still works while polling (open question in the spec); if not, leave it at 0 and note it in the README.
2. **Single pose:** `--first-n 1`; confirm one move, sample-list length 1, and no "sample not recorded" warning. Also try `--tf-dwell-sec 0.1` once to confirm the dwell matters (expect an occasional skipped sample or a visibly off pose), then restore the default.
3. **Full run:** complete offsets → compute → save under `~/.ros2/easy_handeye2/calibrations/`.
4. **Publish:** `publish.launch.py` with the same `name` shows the calibrated transform.

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
| Vendored full PRO interface (Option 1) | Task 1 |
| Eye-on-base, free-drive then A-style offsets | Task 2 + 5 |
| Single robot owner + lock + cached pose | Task 3 |
| TF bridge while pylibfranka owns robot (cache-only) | Task 4 |
| Post-refresh TF dwell for sampler 0.2 s lookback | Task 5 |
| Auto take_sample (success = list growth) / compute (`.valid`) / save | Task 5 |
| Skip missing TF; abort on motion fault; min_samples gate; clear stale samples | Task 5 |
| No MoveIt / no franka_ros2 control during run | README + Global Constraints |
| Package `easy_handeye2_franka_auto` | Task 1 |
| Manual free-drive/state, single, full, publish tests | Task 6 |

## Notes and known gaps

- `HandeyeCalibrationParameters` tracking frames are left empty in the client; the sampler uses whatever the running `handeye_server` was launched with — **operator must launch calibrate with the same robot frames and `eye_on_base`**. (`name` only affects sample/calibration file naming on the server; service topics are absolute `/easy_handeye2/calibration/*`.)
- Quaternion convention: **wxyz** from `FrankaInterface`; `geometry_msgs` uses xyzw fields filled from wxyz components in `transform_from_pose`.
- Offset math verified against `handeye_robot._compute_poses_around_state`: 12 rotations (± about X/Y/Z at `angle_delta`, then `angle_delta/2`, interleaved) then 5 translations; delta quaternion is right-multiplied onto the home orientation (rotation in the EE frame). Single-axis rotations make the euler-order convention irrelevant.
- No reachability precheck (the MoveIt helper had `_check_target_poses`). Unreachable targets surface as a `reset_to_start_pose` failure; use `--first-n` and modest deltas.
- Whether `refresh_state_snapshot()` works during hand-guiding is unverified; polling is opt-in until manual test 1 passes.
