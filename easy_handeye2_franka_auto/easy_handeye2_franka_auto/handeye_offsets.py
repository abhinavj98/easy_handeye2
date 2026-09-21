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
