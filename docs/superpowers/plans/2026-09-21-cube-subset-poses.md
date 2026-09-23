# Cube-Subset Pose Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace A-style hand-eye offsets with a base-frame cube (center+8 corners) × EE ±X/Y/Z tilt pool, randomly subsampled to 15 poses by default.

**Architecture:** Pure math in `handeye_offsets.compute_cube_poses`; CLIs call it with `--cube-half-size-meters`, `--n-poses`, `--seed`. Auto-calibrate skips failed poses instead of aborting.

**Tech Stack:** Python 3, NumPy, pytest, ROS 2 argparse CLIs (existing package).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-21-cube-subset-poses-design.md`
- Translations in **base** frame; tilts in **EE** frame (right-multiply).
- Pool size fixed at **54**; default subset **15**; seed default **0**.
- Remove A-style API and `--translation-delta-meters` / `--translations-only`.
- Motion fault → skip and continue (no abort).

---

### Task 1: `compute_cube_poses` + unit tests

**Files:**
- Modify: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_offsets.py`
- Modify: `easy_handeye2_franka_auto/test/test_handeye_offsets.py`
- Keep: `snapshot_to_pose_4x4` and quaternion helpers in `handeye_offsets.py`

**Interfaces:**
- Produces: `compute_cube_poses(home_pose_4x4: np.ndarray, angle_delta_rad: float, cube_half_size_m: float, n_poses: int = 15, seed: int = 0) -> List[np.ndarray]`
- Removes: `compute_poses_around_state`

- [ ] **Step 1: Rewrite failing tests**

Replace `test_handeye_offsets.py` with:

```python
import math
import numpy as np
from easy_handeye2_franka_auto.handeye_offsets import compute_cube_poses


def _home():
    T = np.eye(4)
    T[:3, 3] = [0.4, 0.0, 0.4]
    return T


def test_full_pool_when_n_poses_54():
    poses = compute_cube_poses(_home(), math.radians(25), 0.05, n_poses=54, seed=0)
    assert len(poses) == 54


def test_default_subset_size_15():
    poses = compute_cube_poses(_home(), math.radians(25), 0.05)
    assert len(poses) == 15


def test_same_seed_reproducible():
    a = compute_cube_poses(_home(), math.radians(25), 0.05, n_poses=15, seed=7)
    b = compute_cube_poses(_home(), math.radians(25), 0.05, n_poses=15, seed=7)
    assert len(a) == 15
    for Ta, Tb in zip(a, b):
        np.testing.assert_allclose(Ta, Tb)


def test_different_seed_usually_differs():
    a = compute_cube_poses(_home(), math.radians(25), 0.05, n_poses=15, seed=1)
    b = compute_cube_poses(_home(), math.radians(25), 0.05, n_poses=15, seed=2)
    assert any(not np.allclose(Ta, Tb) for Ta, Tb in zip(a, b))


def test_corner_translations_are_pm_d():
    home = _home()
    d = 0.05
    poses = compute_cube_poses(home, math.radians(15), d, n_poses=54, seed=0)
    offsets = {tuple(np.round(T[:3, 3] - home[:3, 3], 9)) for T in poses}
    expected = {(0.0, 0.0, 0.0)}
    for sx in (-d, d):
        for sy in (-d, d):
            for sz in (-d, d):
                expected.add((sx, sy, sz))
    assert offsets == expected


def test_each_pose_has_tilted_orientation():
    home = _home()
    poses = compute_cube_poses(home, math.radians(25), 0.05, n_poses=54, seed=0)
    for T in poses:
        assert not np.allclose(T[:3, :3], home[:3, :3], atol=1e-6)


def test_invalid_n_poses_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_cube_poses(_home(), 0.1, 0.05, n_poses=0)
    with pytest.raises(ValueError):
        compute_cube_poses(_home(), 0.1, 0.05, n_poses=55)
```

- [ ] **Step 2: Run tests — expect fail**

```bash
cd easy_handeye2_franka_auto && PYTHONPATH=$PWD python3 -m pytest test/test_handeye_offsets.py -v
```

Expected: import/attribute errors for `compute_cube_poses`.

- [ ] **Step 3: Implement `compute_cube_poses`**

In `handeye_offsets.py`, remove `compute_poses_around_state`. Add:

```python
import random
from itertools import product

def compute_cube_poses(
    home_pose_4x4: np.ndarray,
    angle_delta_rad: float,
    cube_half_size_m: float,
    n_poses: int = 15,
    seed: int = 0,
) -> List[np.ndarray]:
    home = np.asarray(home_pose_4x4, dtype=float).copy()
    home_q = _R_to_quat_wxyz(home[:3, :3])
    d = float(cube_half_size_m)

    translations = [np.zeros(3)]
    for sx, sy, sz in product((-d, d), repeat=3):
        translations.append(np.array([sx, sy, sz], dtype=float))

    basis = np.eye(3)
    rot_deltas = []
    for axis in basis:
        rot_deltas.append(_quat_from_euler_xyz(*(axis * angle_delta_rad)))
        rot_deltas.append(_quat_from_euler_xyz(*(axis * (-angle_delta_rad))))

    pool: List[np.ndarray] = []
    for t in translations:
        for qd in rot_deltas:
            T = home.copy()
            T[:3, 3] = home[:3, 3] + t
            T[:3, :3] = _quat_wxyz_to_R(_quat_multiply_wxyz(home_q, qd))
            pool.append(T)

    assert len(pool) == 54
    if n_poses < 1 or n_poses > len(pool):
        raise ValueError(f'n_poses must be in 1..{len(pool)}, got {n_poses}')
    rng = random.Random(seed)
    return rng.sample(pool, n_poses)
```

Keep existing helpers + `snapshot_to_pose_4x4`.

- [ ] **Step 4: Run tests — expect pass**

```bash
cd easy_handeye2_franka_auto && PYTHONPATH=$PWD python3 -m pytest test/test_handeye_offsets.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit** (only if user requested commits)

---

### Task 2: Wire CLIs + skip-on-failure + README

**Files:**
- Modify: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_auto_calibrate.py`
- Modify: `easy_handeye2_franka_auto/easy_handeye2_franka_auto/handeye_motion_test.py`
- Modify: `easy_handeye2_franka_auto/README.md`

**Interfaces:**
- Consumes: `compute_cube_poses(...)` from Task 1

- [ ] **Step 1: Update `handeye_auto_calibrate.py`**

- Replace import `compute_poses_around_state` → `compute_cube_poses`.
- CLI: remove `--translation-delta-meters`; add `--cube-half-size-meters` (default 0.05), `--n-poses` (15), `--seed` (0).
- Build targets:

```python
targets = compute_cube_poses(
    home,
    math.radians(cli.rotation_delta_degrees),
    cli.cube_half_size_meters,
    n_poses=cli.n_poses,
    seed=cli.seed,
)
if cli.first_n > 0:
    targets = targets[:cli.first_n]
```

- Motion loop: on exception, log and **continue** (remove `motion_fault` abort / break). Track nothing that blocks compute; use final `n_samples()` vs `min_samples`.
- `--return-home`: always attempt if flag set (after loop), not gated on motion_fault.

- [ ] **Step 2: Update `handeye_motion_test.py`**

- Same CLI flag swap; remove `--translations-only` and rotation-delta-0 special case.
- Call `compute_cube_poses`; on motion failure **continue** (print and skip), do not `return`.
- Keep `--first-n`, `--return-home`.

- [ ] **Step 3: Update README**

Document cube subset, `--cube-half-size-meters`, `--n-poses`, `--seed`; update example commands; note skip-on-failure.

- [ ] **Step 4: Run full package tests**

```bash
source /opt/ros/humble/setup.bash
cd easy_handeye2_franka_auto
PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest test/ -v
```

Expected: all PASS.

- [ ] **Step 5: Commit** (only if user requested commits)

---

## Spec coverage

| Spec requirement | Task |
|------------------|------|
| 9×6=54 pool, base translate, EE tilt | 1 |
| Random subset n_poses + seed | 1 |
| Replace A-style | 1–2 |
| CLI flags | 2 |
| Skip on failure | 2 |
| README | 2 |
