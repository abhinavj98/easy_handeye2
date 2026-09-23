"""Synthetic eye-on-base data with known camera (X) and board mount (Y)."""
import math

import numpy as np
import pytest
import yaml

from easy_handeye2_franka_auto.handeye_eval import (
    METHODS,
    board_in_ee,
    leave_one_out,
    load_calibration,
    load_samples,
    max_pairwise_rotation_deg,
    mean_pose,
    pose_distance,
    residuals,
    solve_calibration,
)
from easy_handeye2_franka_auto.handeye_offsets import _R_to_quat_wxyz, _quat_from_euler_xyz, snapshot_to_pose_4x4


def _pose(t, rpy_deg):
    return snapshot_to_pose_4x4(t, _quat_from_euler_xyz(*np.radians(rpy_deg)))


X_TRUE = _pose([1.1, -0.3, 0.6], [-100, 5, 95])     # T_base<-cam
Y_TRUE = _pose([0.02, -0.01, 0.06], [180, 0, 45])    # T_ee<-board


def _make_samples(n=15, noise_mm=0.0, noise_deg=0.0, seed=0):
    rng = np.random.default_rng(seed)
    robot, tracking = [], []
    for _ in range(n):
        base_ee = _pose(rng.uniform([0.35, -0.2, 0.3], [0.6, 0.2, 0.6]), rng.uniform(-30, 30, 3) + [180, 0, 0])
        cam_board = np.linalg.inv(X_TRUE) @ base_ee @ Y_TRUE
        noise = _pose(rng.normal(0, noise_mm / 1000, 3), rng.normal(0, noise_deg, 3))
        robot.append(np.linalg.inv(base_ee))          # sample.robot = T_ee<-base
        tracking.append(cam_board @ noise)            # sample.tracking = T_cam<-board
    return robot, tracking


def test_solver_recovers_camera_pose():
    robot, tracking = _make_samples()
    X = solve_calibration(robot, tracking)
    dt, dr = pose_distance(X, X_TRUE)
    assert dt < 1e-6 and dr < 1e-6


def test_board_in_ee_is_constant_for_true_calibration():
    robot, tracking = _make_samples()
    for Y in board_in_ee(robot, tracking, X_TRUE):
        dt, dr = pose_distance(Y, Y_TRUE)
        assert dt < 1e-9 and dr < 1e-6  # arccos near 0 limits rotation precision


def test_residuals_zero_for_perfect_data():
    robot, tracking = _make_samples()
    assert max(t for t, _ in residuals(robot, tracking, X_TRUE)) < 1e-9
    assert max(t for t, _ in leave_one_out(robot, tracking)) < 1e-6


def test_residuals_flag_wrong_calibration():
    robot, tracking = _make_samples()
    X_bad = X_TRUE @ _pose([0.02, 0.0, 0.0], [0, 2, 0])  # 2 cm / 2 deg off
    rms_mm = 1000 * math.sqrt(np.mean([t ** 2 for t, _ in residuals(robot, tracking, X_bad)]))
    assert rms_mm > 5.0


def test_noise_shows_up_in_residuals():
    robot, tracking = _make_samples(noise_mm=1.0, noise_deg=0.1)
    X = solve_calibration(robot, tracking)
    fit = [t for t, _ in residuals(robot, tracking, X)]
    loo = [t for t, _ in leave_one_out(robot, tracking)]
    assert 0.3e-3 < np.sqrt(np.mean(np.square(fit))) < 5e-3
    assert np.mean(loo) >= np.mean(fit)  # held-out error is never better on average


def test_all_methods_agree_on_clean_data():
    robot, tracking = _make_samples()
    for name in METHODS:
        dt, dr = pose_distance(solve_calibration(robot, tracking, name), X_TRUE)
        assert dt < 1e-3 and dr < math.radians(0.1), name


def test_mean_pose_and_rotation_span():
    assert pose_distance(mean_pose([Y_TRUE, Y_TRUE]), Y_TRUE)[0] < 1e-12
    a, b = _pose([0, 0, 0], [0, 0, 0]), _pose([0, 0, 0], [0, 0, 30])
    assert abs(max_pairwise_rotation_deg([a, b]) - 30.0) < 1e-6


def _tf_dict(T):
    w, x, y, z = _R_to_quat_wxyz(T[:3, :3])
    return {'translation': dict(zip('xyz', map(float, T[:3, 3]))),
            'rotation': {'x': float(x), 'y': float(y), 'z': float(z), 'w': float(w)}}


def _write_files(tmp_path, robot, tracking, X):
    samples = tmp_path / 'c.samples'
    calib = tmp_path / 'c.calib'
    samples.write_text(yaml.safe_dump(
        {'samples': [{'robot': _tf_dict(r), 'tracking': _tf_dict(c)} for r, c in zip(robot, tracking)]}))
    calib.write_text(yaml.safe_dump({
        'parameters': {'name': 'c', 'calibration_type': 'eye_on_base', 'robot_base_frame': 'fr3_link0',
                       'robot_effector_frame': 'handeye_ee', 'tracking_base_frame': 'camera_link',
                       'tracking_marker_frame': 'charuco', 'freehand_robot_movement': True},
        'transform': _tf_dict(X)}))
    return samples, calib


def test_file_roundtrip(tmp_path):
    robot, tracking = _make_samples(n=4)
    samples, calib = _write_files(tmp_path, robot, tracking, X_TRUE)
    r2, c2 = load_samples(samples)
    X2, params = load_calibration(calib)
    assert params['tracking_base_frame'] == 'camera_link'
    assert pose_distance(X2, X_TRUE)[0] < 1e-9
    assert all(pose_distance(a, b)[0] < 1e-9 for a, b in zip(r2 + c2, robot + tracking))


@pytest.mark.parametrize('X_offset,expected_rc,verdict', [
    (np.eye(4), 0, 'GOOD'),
    (_pose([0.03, 0, 0], [0, 3, 0]), 2, 'BAD'),
])
def test_cli_verdict(tmp_path, capsys, X_offset, expected_rc, verdict):
    from easy_handeye2_franka_auto.evaluate_calibration import main
    robot, tracking = _make_samples(noise_mm=0.5, noise_deg=0.05)
    samples, calib = _write_files(tmp_path, robot, tracking, X_TRUE @ X_offset)
    assert main(['--samples', str(samples), '--calibration', str(calib)]) == expected_rc
    assert f'Verdict: {verdict}' in capsys.readouterr().out


def _tf_msg(T):
    from types import SimpleNamespace as NS
    d = _tf_dict(T)
    return NS(translation=NS(**d['translation']), rotation=NS(**d['rotation']))


def test_auto_calibrate_prints_report_from_messages(capsys):
    from types import SimpleNamespace as NS
    from easy_handeye2_franka_auto.handeye_auto_calibrate import _print_quality_report
    robot, tracking = _make_samples(noise_mm=0.5, noise_deg=0.05)
    samples = [NS(robot=_tf_msg(r), tracking=_tf_msg(c)) for r, c in zip(robot, tracking)]
    calib = NS(transform=_tf_msg(X_TRUE),
               parameters=NS(robot_base_frame='fr3_link0', tracking_base_frame='camera_link'))
    errors = []
    log = NS(warn=errors.append, error=errors.append)
    _print_quality_report(samples, calib, log)
    out = capsys.readouterr().out
    assert 'Calibration (new): fr3_link0 -> camera_link' in out
    assert 'Verdict: GOOD' in out
    assert errors == []
