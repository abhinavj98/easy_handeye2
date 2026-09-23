"""Synthetic render of the board at a known pose -> estimate_board_pose recovers it."""
import cv2
import numpy as np
import pytest

from easy_handeye2_charuco.board import (
    estimate_board_pose,
    make_board,
    rotation_matrix_to_quaternion,
)

SQUARES_X, SQUARES_Y, SQUARE, MARKER, DICT = 5, 4, 0.0222, 0.0162, 'DICT_5X5_100'
K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
IMG_SIZE = (1280, 720)


def _render(board, rvec, tvec):
    """Warp a flat board image into a camera image at pose (rvec, tvec)."""
    px_per_m = 20000.0
    w, h = SQUARES_X * SQUARE, SQUARES_Y * SQUARE
    flat = board.generateImage((round(w * px_per_m), round(h * px_per_m)), marginSize=0)
    # generateImage's pixel (0, 0) is board point (0, 0, 0), x right / y down, so map the
    # 4 outer corners of the flat image onto their projections.
    obj = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)
    src = np.array([[0, 0], [flat.shape[1], 0], [flat.shape[1], flat.shape[0]], [0, flat.shape[0]]],
                   dtype=np.float32)
    dst, _ = cv2.projectPoints(obj, rvec, tvec, K, DIST)
    H = cv2.getPerspectiveTransform(src, dst.reshape(-1, 2).astype(np.float32))
    return cv2.warpPerspective(flat, H, IMG_SIZE, borderValue=255)


@pytest.fixture(scope='module')
def board_and_detector():
    return make_board(SQUARES_X, SQUARES_Y, SQUARE, MARKER, DICT)


@pytest.mark.parametrize('rvec_deg,tvec', [
    ((0, 0, 0), (-0.05, -0.04, 0.40)),
    ((20, -15, 10), (-0.04, -0.03, 0.45)),
    ((-25, 20, -30), (-0.03, -0.05, 0.50)),
])
def test_recovers_known_pose(board_and_detector, rvec_deg, tvec):
    board, detector = board_and_detector
    R_true, _ = cv2.Rodrigues(np.radians(np.array(rvec_deg, dtype=np.float64)))
    rvec = cv2.Rodrigues(R_true)[0]
    tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)

    img = _render(board, rvec, tvec)
    res = estimate_board_pose(img, board, detector, K, DIST, min_corners=4)

    assert res.ok, res.reason
    assert res.n_corners == (SQUARES_X - 1) * (SQUARES_Y - 1)
    assert res.reproj_err_px < 1.0
    assert np.linalg.norm(res.tvec - tvec) < 1e-3  # < 1 mm
    R_est, _ = cv2.Rodrigues(res.rvec)
    angle = np.degrees(np.arccos(np.clip((np.trace(R_true.T @ R_est) - 1) / 2, -1, 1)))
    assert angle < 0.5


def test_no_board_reports_reason(board_and_detector):
    board, detector = board_and_detector
    blank = np.full((IMG_SIZE[1], IMG_SIZE[0]), 255, np.uint8)
    res = estimate_board_pose(blank, board, detector, K, DIST, min_corners=4)
    assert not res.ok
    assert 'insufficient corners' in res.reason


def test_unknown_dictionary_raises():
    with pytest.raises(ValueError):
        make_board(5, 4, 0.02, 0.015, 'DICT_NOPE')


def test_quaternion_matches_rotation():
    R, _ = cv2.Rodrigues(np.radians(np.array([30.0, -40.0, 100.0])))
    x, y, z, w = rotation_matrix_to_quaternion(R)
    R_q = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(R, R_q, atol=1e-9)
