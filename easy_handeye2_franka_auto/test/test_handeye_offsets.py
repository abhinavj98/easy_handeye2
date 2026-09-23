import math

import numpy as np
import pytest
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
    with pytest.raises(ValueError):
        compute_cube_poses(_home(), 0.1, 0.05, n_poses=0)
    with pytest.raises(ValueError):
        compute_cube_poses(_home(), 0.1, 0.05, n_poses=55)
