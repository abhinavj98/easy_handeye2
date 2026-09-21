import numpy as np
from easy_handeye2_franka_auto.handeye_offsets import snapshot_to_pose_4x4


def test_snapshot_to_pose_identity_rotation():
    pos = np.array([0.1, 0.2, 0.3])
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    T = snapshot_to_pose_4x4(pos, quat_wxyz)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, 3], pos)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])
