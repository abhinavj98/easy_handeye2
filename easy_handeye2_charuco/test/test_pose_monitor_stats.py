import numpy as np

from easy_handeye2_charuco.charuco_pose_monitor import quat_mean_and_rms_deg, translation_jitter_mm


def test_identical_poses_have_zero_jitter():
    q = np.tile([0.0, 0.0, 0.0, 1.0], (10, 1))
    _, rms = quat_mean_and_rms_deg(q)
    assert rms < 1e-6
    assert translation_jitter_mm(np.tile([0.1, 0.2, 0.5], (10, 1))) < 1e-9


def test_sign_flipped_quaternions_are_same_rotation():
    q = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])
    _, rms = quat_mean_and_rms_deg(q)
    assert rms < 1e-6


def test_rotation_jitter_degrees():
    half = np.radians(1.0) / 2  # +-1 deg about z
    q = np.array([[0, 0, np.sin(half), np.cos(half)], [0, 0, -np.sin(half), np.cos(half)]])
    _, rms = quat_mean_and_rms_deg(q)
    assert abs(rms - 1.0) < 1e-6


def test_translation_jitter_mm():
    t = np.array([[0.0, 0.0, 0.5], [0.002, 0.0, 0.5]])  # +-1 mm about the mean
    assert abs(translation_jitter_mm(t) - 1.0) < 1e-9
