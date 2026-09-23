"""ChArUco board construction and pose estimation. Pure OpenCV/numpy, no ROS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def make_board(squares_x: int, squares_y: int, square_length: float,
               marker_length: float, dictionary: str):
    """Return (CharucoBoard, CharucoDetector). Needs OpenCV >= 4.7."""
    if not hasattr(cv2.aruco, dictionary):
        raise ValueError(f'Unknown ArUco dictionary: {dictionary}')
    if not hasattr(cv2.aruco, 'CharucoDetector'):
        raise RuntimeError(
            f'OpenCV {cv2.__version__} has no cv2.aruco.CharucoDetector; need >= 4.7')
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, d)
    return board, cv2.aruco.CharucoDetector(board)


@dataclass
class BoardPose:
    ok: bool
    reason: str = ''
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    n_corners: int = 0
    reproj_err_px: float = float('nan')
    charuco_corners: Optional[np.ndarray] = None
    charuco_ids: Optional[np.ndarray] = None
    marker_corners: tuple = ()
    marker_ids: Optional[np.ndarray] = None


def estimate_board_pose(gray: np.ndarray, board, detector, camera_matrix: np.ndarray,
                        dist_coeffs: np.ndarray, min_corners: int = 4) -> BoardPose:
    """Detect the board in a grayscale image and solve PnP for camera->board."""
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    res = BoardPose(ok=False, charuco_corners=charuco_corners, charuco_ids=charuco_ids,
                    marker_corners=marker_corners, marker_ids=marker_ids)
    res.n_corners = 0 if charuco_ids is None else len(charuco_ids)
    if res.n_corners < min_corners:
        res.reason = f'insufficient corners ({res.n_corners} < {min_corners})'
        return res

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or img_points is None or len(obj_points) < min_corners:
        res.reason = 'failed to match image points'
        return res

    valid, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    if not valid:
        res.reason = 'solvePnP failed'
        return res

    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    err = projected.reshape(-1, 2) - img_points.reshape(-1, 2)
    res.reproj_err_px = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    res.ok, res.rvec, res.tvec = True, rvec, tvec
    return res


def rotation_matrix_to_quaternion(rmat: np.ndarray) -> np.ndarray:
    """Return quaternion [x, y, z, w] from a 3x3 rotation matrix."""
    m = rmat
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)
