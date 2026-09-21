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
