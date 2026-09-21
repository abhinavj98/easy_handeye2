"""Publish the cached robot base→EE pose to /tf. Never touches the robot."""
from __future__ import annotations

from geometry_msgs.msg import Transform, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def transform_from_pose(pos, quat_wxyz) -> Transform:
    t = Transform()
    t.translation.x, t.translation.y, t.translation.z = (float(v) for v in pos)
    t.rotation.w = float(quat_wxyz[0])
    t.rotation.x = float(quat_wxyz[1])
    t.rotation.y = float(quat_wxyz[2])
    t.rotation.z = float(quat_wxyz[3])
    return t


class RobotTfBridge(Node):
    def __init__(self, pose_source, base_frame: str, ee_frame: str, rate_hz: float = 30.0):
        super().__init__('handeye_robot_tf_bridge')
        self._source = pose_source
        self._base_frame = base_frame
        self._ee_frame = ee_frame
        self._br = TransformBroadcaster(self)
        self._timer = self.create_timer(1.0 / float(rate_hz), self._on_timer)

    def _on_timer(self):
        latest = self._source.latest()
        if latest is None:
            return
        pos, quat = latest
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.child_frame_id = self._ee_frame
        msg.transform = transform_from_pose(pos, quat)
        self._br.sendTransform(msg)
