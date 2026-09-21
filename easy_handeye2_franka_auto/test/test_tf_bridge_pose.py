import numpy as np
from easy_handeye2_franka_auto.handeye_tf_bridge import transform_from_pose


def test_transform_from_pose_maps_wxyz_to_xyzw_fields():
    t = transform_from_pose(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.1, 0.2, 0.3]))
    assert abs(t.translation.x - 0.1) < 1e-9
    assert abs(t.translation.y - 0.2) < 1e-9
    assert abs(t.translation.z - 0.3) < 1e-9
    assert abs(t.rotation.w - 0.5) < 1e-9
    assert abs(t.rotation.x - 0.1) < 1e-9
    assert abs(t.rotation.y - 0.2) < 1e-9
    assert abs(t.rotation.z - 0.3) < 1e-9
