#!/usr/bin/env python3
"""Detect a ChArUco board and publish its pose as TF for easy_handeye2."""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from easy_handeye2_charuco.board import (
    estimate_board_pose,
    make_board,
    rotation_matrix_to_quaternion,
)

_GREEN = (0, 200, 0)
_RED = (0, 0, 255)


class CharucoTFPublisher(Node):
    def __init__(self) -> None:
        super().__init__('charuco_tf_publisher')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('target_frame', 'charuco')
        self.declare_parameter('squares_x', 5)
        self.declare_parameter('squares_y', 4)
        self.declare_parameter('square_length', 0.0222)
        self.declare_parameter('marker_length', 0.0162)
        self.declare_parameter('dictionary', 'DICT_5X5_100')
        self.declare_parameter('min_charuco_corners', 4)
        self.declare_parameter('max_reproj_err_px', 2.0)
        self.declare_parameter('axis_length', 0.05)

        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.target_frame = str(self.get_parameter('target_frame').value)
        squares_x = int(self.get_parameter('squares_x').value)
        squares_y = int(self.get_parameter('squares_y').value)
        square_length = float(self.get_parameter('square_length').value)
        marker_length = float(self.get_parameter('marker_length').value)
        dictionary_name = str(self.get_parameter('dictionary').value)
        self.min_corners = int(self.get_parameter('min_charuco_corners').value)
        self.max_reproj_err = float(self.get_parameter('max_reproj_err_px').value)
        self.axis_length = float(self.get_parameter('axis_length').value)

        self.board, self.detector = make_board(
            squares_x, squares_y, square_length, marker_length, dictionary_name)

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        # '~/' so the topic is /charuco_tf_publisher/debug_image, not /debug_image
        self.debug_pub = self.create_publisher(Image, '~/debug_image', 10)
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None

        self.create_subscription(CameraInfo, camera_info_topic, self.info_callback, 10)
        self.create_subscription(Image, image_topic, self.image_callback, 10)

        self.get_logger().info(
            f'ChArUco TF publisher ready: {squares_x}x{squares_y}, '
            f'square={square_length} m, marker={marker_length} m, '
            f'dict={dictionary_name}, image={image_topic}, '
            f'debug={self.debug_pub.topic_name}'
        )

    def info_callback(self, msg: CameraInfo) -> None:
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.get_logger().info('Received camera intrinsics')

    def image_callback(self, msg: Image) -> None:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug = frame.copy()

        try:
            res = estimate_board_pose(gray, self.board, self.detector,
                                      self.camera_matrix, self.dist_coeffs, self.min_corners)
            if res.marker_ids is not None and len(res.marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(debug, res.marker_corners, res.marker_ids)
            if res.charuco_ids is not None and len(res.charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(debug, res.charuco_corners, res.charuco_ids)

            if res.ok and res.reproj_err_px > self.max_reproj_err:
                res.ok = False
                res.reason = f'reprojection error {res.reproj_err_px:.2f} px > {self.max_reproj_err}'

            if not res.ok:
                self.get_logger().warn(f'ChArUco pose unavailable: {res.reason}',
                                       throttle_duration_sec=2.0)
                status, color = f'NO POSE: {res.reason}', _RED
            else:
                cv2.drawFrameAxes(debug, self.camera_matrix, self.dist_coeffs,
                                  res.rvec, res.tvec, self.axis_length)
                self._publish_tf(msg, res.rvec, res.tvec)
                dist = float(np.linalg.norm(res.tvec))
                status = (f'corners {res.n_corners} | dist {dist:.3f} m | '
                          f'reproj {res.reproj_err_px:.2f} px')
                color = _GREEN
            cv2.putText(debug, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except cv2.error as exc:
            self.get_logger().warn(f'ChArUco pose unavailable: {exc}', throttle_duration_sec=2.0)
        except Exception as exc:  # noqa: BLE001 - keep node alive on bad frames
            self.get_logger().warn(f'ChArUco pose unavailable: unexpected error: {exc}',
                                   throttle_duration_sec=2.0)

        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

    def _publish_tf(self, msg: Image, rvec: np.ndarray, tvec: np.ndarray) -> None:
        rmat, _ = cv2.Rodrigues(rvec)
        q = rotation_matrix_to_quaternion(rmat)
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id
        t.child_frame_id = self.target_frame
        t.transform.translation.x = float(tvec[0][0])
        t.transform.translation.y = float(tvec[1][0])
        t.transform.translation.z = float(tvec[2][0])
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CharucoTFPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
