"""Eye-on-base calibration quality metrics. Pure numpy/OpenCV, no ROS.

Conventions (match easy_handeye2's sampler/backend for eye_on_base):
  robot[i]    = T_ee<-base    (sample.robot: lookup(effector, base))
  tracking[i] = T_cam<-board  (sample.tracking: lookup(camera, marker))
  X           = T_base<-cam   (the saved calibration, robot_base_frame -> tracking_base_frame)
The board is rigid on the EE, so Y_i = T_ee<-board = robot[i] @ X @ tracking[i]
must be the same for every sample; its spread is the calibration error.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

from easy_handeye2_franka_auto.handeye_offsets import (
    _quat_wxyz_to_R,
    _R_to_quat_wxyz,
    pose_distance,
    snapshot_to_pose_4x4,
)

METHODS: Dict[str, int] = {
    'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
    'Park': cv2.CALIB_HAND_EYE_PARK,
    'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def transform_dict_to_4x4(d: dict) -> np.ndarray:
    """geometry_msgs/Transform as a yaml dict -> 4x4."""
    t, q = d['translation'], d['rotation']
    return snapshot_to_pose_4x4([t['x'], t['y'], t['z']], [q['w'], q['x'], q['y'], q['z']])


def transform_msg_to_4x4(t) -> np.ndarray:
    """geometry_msgs/Transform message -> 4x4."""
    return snapshot_to_pose_4x4([t.translation.x, t.translation.y, t.translation.z],
                                [t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z])


def load_samples(path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Read an easy_handeye2 .samples file -> (robot, tracking) lists of 4x4."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    samples = data.get('samples') or []
    return ([transform_dict_to_4x4(s['robot']) for s in samples],
            [transform_dict_to_4x4(s['tracking']) for s in samples])


def load_calibration(path: Path) -> Tuple[np.ndarray, dict]:
    """Read an easy_handeye2 .calib file -> (X = T_base<-cam, parameters dict)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return transform_dict_to_4x4(data['transform']), data['parameters']


def solve_calibration(robot: Sequence[np.ndarray], tracking: Sequence[np.ndarray],
                      method: str = 'Tsai-Lenz') -> np.ndarray:
    """Same call as easy_handeye2's OpenCV backend -> X = T_base<-cam."""
    R, t = cv2.calibrateHandEye(
        [T[:3, :3] for T in robot], [T[:3, 3] for T in robot],
        [T[:3, :3] for T in tracking], [T[:3, 3] for T in tracking],
        method=METHODS[method])
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = np.asarray(t).reshape(3)
    return X


def board_in_ee(robot: Sequence[np.ndarray], tracking: Sequence[np.ndarray],
                X: np.ndarray) -> List[np.ndarray]:
    """Per-sample T_ee<-board implied by the calibration."""
    return [Ti @ X @ Ci for Ti, Ci in zip(robot, tracking)]


def mean_pose(Ts: Sequence[np.ndarray]) -> np.ndarray:
    """Mean translation + sign-aligned quaternion average."""
    qs = np.array([_R_to_quat_wxyz(T[:3, :3]) for T in Ts])
    qs = np.where((qs @ qs[0])[:, None] < 0.0, -qs, qs)
    q = qs.mean(axis=0)
    M = np.eye(4)
    M[:3, :3] = _quat_wxyz_to_R(q / np.linalg.norm(q))
    M[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return M


def residuals(robot: Sequence[np.ndarray], tracking: Sequence[np.ndarray], X: np.ndarray,
              Y: np.ndarray | None = None) -> List[Tuple[float, float]]:
    """Per-sample (trans_m, rot_rad) between the board pose in the base frame as
    predicted by the robot (inv(robot_i) @ Y) and as seen by the camera (X @ tracking_i).
    Y (T_ee<-board) defaults to the mean over the same samples."""
    if Y is None:
        Y = mean_pose(board_in_ee(robot, tracking, X))
    return [pose_distance(np.linalg.inv(Ti) @ Y, X @ Ci) for Ti, Ci in zip(robot, tracking)]


def leave_one_out(robot: Sequence[np.ndarray], tracking: Sequence[np.ndarray],
                  method: str = 'Tsai-Lenz') -> List[Tuple[float, float]]:
    """Residual of each sample under a calibration (and board offset) fit without it."""
    out = []
    for i in range(len(robot)):
        r = [T for j, T in enumerate(robot) if j != i]
        c = [T for j, T in enumerate(tracking) if j != i]
        X = solve_calibration(r, c, method)
        Y = mean_pose(board_in_ee(r, c, X))
        out.append(residuals([robot[i]], [tracking[i]], X, Y)[0])
    return out


def max_pairwise_rotation_deg(robot: Sequence[np.ndarray]) -> float:
    """Largest EE rotation between any two samples (rotation diversity)."""
    best = 0.0
    for i in range(len(robot)):
        for j in range(i + 1, len(robot)):
            best = max(best, pose_distance(robot[i], robot[j])[1])
    return float(np.degrees(best))


def rms(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float('nan')


def rpy_deg(R: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (fixed XYZ / ROS convention), degrees."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])
