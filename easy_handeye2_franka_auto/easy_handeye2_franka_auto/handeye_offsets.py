"""Cube corner+center × EE-tilt poses around a free-driven home EE pose."""
from __future__ import annotations

import random
from itertools import product
from typing import List, Sequence, Tuple

import numpy as np


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
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


def compute_cube_poses(
    home_pose_4x4: np.ndarray,
    angle_delta_rad: float,
    cube_half_size_m: float,
    n_poses: int = 15,
    seed: int = 0,
) -> List[np.ndarray]:
    """Base-frame cube (center+8 corners) × EE ±X/Y/Z tilts; random subset.

    Pool size is always 54. Returns ``n_poses`` poses sampled without
    replacement using ``seed``. Raises ValueError if n_poses not in 1..54.
    """
    home = np.asarray(home_pose_4x4, dtype=float).copy()
    home_q = _R_to_quat_wxyz(home[:3, :3])
    d = float(cube_half_size_m)

    translations = [np.zeros(3, dtype=float)]
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

    if len(pool) != 54:
        raise RuntimeError(f'expected pool size 54, got {len(pool)}')
    if n_poses < 1 or n_poses > len(pool):
        raise ValueError(f'n_poses must be in 1..{len(pool)}, got {n_poses}')
    rng = random.Random(seed)
    return rng.sample(pool, n_poses)


def pose_distance(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[float, float]:
    """(translation_m, rotation_rad) between two 4x4 poses.

    Rotation is the angle of R_a^T R_b (axis-angle magnitude), via the trace
    identity: trace(R) = 1 + 2*cos(theta).
    """
    T_a = np.asarray(T_a, dtype=float)
    T_b = np.asarray(T_b, dtype=float)
    trans = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    R_rel = T_a[:3, :3].T @ T_b[:3, :3]
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    rot = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return trans, rot


def is_distinct(
    T: np.ndarray,
    prior: Sequence[np.ndarray],
    min_trans_m: float,
    min_rot_rad: float,
) -> bool:
    """True if ``T`` differs from every pose in ``prior`` by at least
    ``min_trans_m`` in translation OR ``min_rot_rad`` in rotation.

    An empty ``prior`` is always distinct.
    """
    for other in prior:
        trans, rot = pose_distance(T, other)
        if trans < min_trans_m and rot < min_rot_rad:
            return False
    return True
