"""Live check of a published eye-on-base calibration (never moves the robot).

With the calibration published (publish.launch.py), the ChArUco detector running
and a live robot TF, the board pose in the EE frame, looked up through the
calibration, must stay constant while you move the robot. This node records it at
each new still pose and prints how much it varies.

    ros2 launch franka_bringup franka.launch.py robot_ip:=...      # live robot TF
    ros2 launch easy_handeye2_charuco charuco_view.launch.py use_rviz:=false
    ros2 launch easy_handeye2 publish.launch.py name:=fr3_eob
    ros2 run easy_handeye2_franka_auto handeye_consistency_monitor

Any frame rigid to the EE works as --effector (fr3_link8 is published by
robot_state_publisher; handeye_ee only exists while handeye_auto_calibrate runs).
Hold still at each pose: only still, freshly detected poses are recorded.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', default='fr3_link0', help='robot base frame')
    p.add_argument('--effector', default='fr3_link8', help='any frame rigid to the EE')
    p.add_argument('--marker', default='charuco', help='board frame')
    p.add_argument('--rate-hz', type=float, default=10.0)
    p.add_argument('--still-mm', type=float, default=1.0,
                   help='robot counts as still if the EE moved less than this since the last tick')
    p.add_argument('--new-pose-mm', type=float, default=20.0,
                   help='record a pose only if the EE is this far from all recorded poses ...')
    p.add_argument('--new-pose-deg', type=float, default=5.0, help='... or rotated this much')
    p.add_argument('--max-age-sec', type=float, default=0.3, help='ignore board detections older than this')
    return p.parse_args(argv)


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.time import Time
    from rclpy.utilities import remove_ros_args
    from tf2_ros import Buffer, TransformException, TransformListener

    from easy_handeye2_franka_auto.handeye_eval import mean_pose, pose_distance, rms
    from easy_handeye2_franka_auto.handeye_offsets import is_distinct

    cli = _parse_args(remove_ros_args(args=sys.argv)[1:])
    rclpy.init(args=args)
    node = rclpy.create_node('handeye_consistency_monitor')
    log = node.get_logger()
    buf = Buffer()
    TransformListener(buf, node)

    def to_4x4(tf) -> np.ndarray:
        t, q = tf.transform.translation, tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        T = np.eye(4)
        T[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    state = {'last_ee': None, 'warned': False}
    ee_poses, boards = [], []

    def tick():
        try:
            ee = to_4x4(buf.lookup_transform(cli.base, cli.effector, Time()))
            board_tf = buf.lookup_transform(cli.effector, cli.marker, Time())
        except TransformException as exc:
            if not state['warned']:
                log.warn(f'waiting for TF ({exc}); need robot TF, publish.launch.py and a visible board')
                state['warned'] = True
            return
        state['warned'] = False
        last, state['last_ee'] = state['last_ee'], ee
        if last is None or pose_distance(ee, last)[0] * 1000 > cli.still_mm:
            return  # moving
        # board_tf is stamped at the latest time common to robot + board transforms
        age = (node.get_clock().now() - Time.from_msg(board_tf.header.stamp)).nanoseconds * 1e-9
        if age > cli.max_age_sec:
            return  # board not currently detected
        if not is_distinct(ee, ee_poses, cli.new_pose_mm / 1000, math.radians(cli.new_pose_deg)):
            return  # already recorded here

        ee_poses.append(ee)
        boards.append(to_4x4(board_tf))
        if len(boards) < 2:
            log.info(f'pose 1 recorded; move to a different pose (>{cli.new_pose_mm} mm or '
                     f'>{cli.new_pose_deg} deg) and hold still')
            return
        ref = mean_pose(boards)
        errs = [pose_distance(B, ref) for B in boards]
        t_mm = [e[0] * 1000 for e in errs]
        r_deg = [math.degrees(e[1]) for e in errs]
        log.info(
            f'pose {len(boards)}: this {t_mm[-1]:.2f} mm / {r_deg[-1]:.3f} deg from mean | '
            f'all: RMS {rms(t_mm):.2f} mm / {rms(r_deg):.3f} deg, max {max(t_mm):.2f} mm / {max(r_deg):.3f} deg'
        )

    node.create_timer(1.0 / cli.rate_hz, tick)
    log.info(f'Monitoring {cli.effector} -> {cli.marker} through the calibration '
             f'(base {cli.base}); move the robot between still poses, Ctrl-C to stop')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if len(boards) >= 2:
            ref = mean_pose(boards)
            errs = [pose_distance(B, ref) for B in boards]
            print(f'\n{len(boards)} poses: board-in-EE spread RMS '
                  f'{rms([e[0] for e in errs]) * 1000:.2f} mm / {math.degrees(rms([e[1] for e in errs])):.3f} deg')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
