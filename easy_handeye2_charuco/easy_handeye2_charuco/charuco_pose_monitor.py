#!/usr/bin/env python3
"""Print ChArUco detection rate and pose jitter from TF (test tool, never moves anything).

Hold the board still: translation jitter should be well under 1 mm and rotation
jitter under ~0.1 deg. Move/tilt it to see where detection drops out.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque

import numpy as np


def quat_mean_and_rms_deg(quats_xyzw: np.ndarray) -> tuple[np.ndarray, float]:
    """Mean quaternion (sign-aligned average) and RMS angular deviation from it, in degrees."""
    q = np.asarray(quats_xyzw, dtype=np.float64)
    q = np.where((q @ q[0])[:, None] < 0.0, -q, q)
    mean = q.mean(axis=0)
    mean /= np.linalg.norm(mean)
    dots = np.clip(np.abs(q @ mean), 0.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return mean, float(np.degrees(np.sqrt(np.mean(angles ** 2))))


def translation_jitter_mm(translations: np.ndarray) -> float:
    """RMS distance of translations from their mean, in millimetres."""
    t = np.asarray(translations, dtype=np.float64)
    return float(1000.0 * np.sqrt(np.mean(np.sum((t - t.mean(axis=0)) ** 2, axis=1))))


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', default='camera_link', help='camera frame (tracking_base_frame)')
    p.add_argument('--child', default='charuco', help='marker frame (tracking_marker_frame)')
    p.add_argument('--rate-hz', type=float, default=60.0,
                   help='TF poll rate (caps the measured detection rate)')
    p.add_argument('--window-sec', type=float, default=2.0, help='stats window / print period')
    return p.parse_args(argv)


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.time import Time
    from rclpy.utilities import remove_ros_args
    from tf2_ros import Buffer, TransformException, TransformListener

    cli = _parse_args(remove_ros_args(args=sys.argv)[1:])
    rclpy.init(args=args)
    node = rclpy.create_node('charuco_pose_monitor')
    buf = Buffer()
    TransformListener(buf, node)

    window = deque()  # (stamp_sec, translation[3], quat_xyzw[4])
    state = {'last_stamp': None, 'polls': 0, 'fresh': 0}

    def poll():
        state['polls'] += 1
        try:
            tf = buf.lookup_transform(cli.parent, cli.child, Time())
        except TransformException:
            return
        stamp = Time.from_msg(tf.header.stamp).nanoseconds * 1e-9
        if stamp == state['last_stamp']:
            return
        state['last_stamp'] = stamp
        now = node.get_clock().now().nanoseconds * 1e-9
        if now - stamp > 0.5:  # stale: board not currently detected
            return
        state['fresh'] += 1
        tr, rot = tf.transform.translation, tf.transform.rotation
        window.append((stamp, np.array([tr.x, tr.y, tr.z]), np.array([rot.x, rot.y, rot.z, rot.w])))

    def report():
        now = node.get_clock().now().nanoseconds * 1e-9
        while window and now - window[0][0] > cli.window_sec:
            window.popleft()
        polls, fresh = state['polls'], state['fresh']
        state['polls'] = state['fresh'] = 0
        if not window:
            node.get_logger().warn(
                f'{cli.parent} -> {cli.child}: no fresh detections in the last {cli.window_sec:.1f} s')
            return
        ts = np.array([w[1] for w in window])
        qs = np.array([w[2] for w in window])
        _, rot_rms = quat_mean_and_rms_deg(qs)
        mean_t = ts.mean(axis=0)
        node.get_logger().info(
            f'det {fresh / cli.window_sec:5.1f} Hz ({fresh}/{polls} polls) | '
            f'dist {np.linalg.norm(mean_t):.3f} m | '
            f't=({mean_t[0]:+.3f}, {mean_t[1]:+.3f}, {mean_t[2]:+.3f}) | '
            f'jitter {translation_jitter_mm(ts):.2f} mm, {rot_rms:.3f} deg (n={len(window)})'
        )

    node.create_timer(1.0 / cli.rate_hz, poll)
    node.create_timer(cli.window_sec, report)
    node.get_logger().info(f'Monitoring {cli.parent} -> {cli.child} (Ctrl-C to stop)')
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
