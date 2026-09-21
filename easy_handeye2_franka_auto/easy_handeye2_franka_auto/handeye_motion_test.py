"""CLI: free-drive home → A-style offsets → move only (no handeye / TF)."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description='Motion-only smoke test: free-drive home, then run A-style EE offsets')
    p.add_argument('--robot-config', type=Path, required=True,
                   help='YAML with robot: {ip, use_mock, reset_duration_sec, ...}')
    p.add_argument('--rotation-delta-degrees', type=float, default=15.0,
                   help='rotation magnitude for offsets (default 15, smaller than full cal)')
    p.add_argument('--translation-delta-meters', type=float, default=0.05,
                   help='translation magnitude for offsets (default 0.05)')
    p.add_argument('--settle-sec', type=float, default=1.0)
    p.add_argument('--first-n', type=int, default=3,
                   help='only run the first N offset poses (0 = all 17; default 3)')
    p.add_argument('--return-home', action='store_true', default=True,
                   help='return to free-drive home after offsets (default on)')
    p.add_argument('--no-return-home', action='store_false', dest='return_home')
    return p.parse_args(argv)


def main(args=None):
    import yaml
    try:
        from rclpy.utilities import remove_ros_args
        argv = remove_ros_args(args=sys.argv)[1:]
    except Exception:
        argv = sys.argv[1:]

    cli = _parse_args(argv)
    with open(cli.robot_config) as f:
        robot_cfg = yaml.safe_load(f)

    from easy_handeye2_franka_auto.handeye_offsets import (
        compute_poses_around_state,
        snapshot_to_pose_4x4,
    )
    from easy_handeye2_franka_auto.pro_robot_interface import FrankaInterface
    from easy_handeye2_franka_auto.robot_pose_source import RobotPoseSource

    robot = FrankaInterface(robot_cfg, device='cpu')
    source = RobotPoseSource(robot)

    def _move(label: str, T) -> None:
        try:
            source.move_to(T)
        except RuntimeError as exc:
            if 'reflex' not in str(exc).lower():
                raise
            print(f'{label}: reflex — running error_recovery and retrying once')
            robot.error_recovery()
            time.sleep(0.5)
            source.move_to(T)

    try:
        input(
            'Free-drive to a safe start pose, RELEASE guiding/freedrive, '
            'let the arm settle, then press Enter...'
        )
        try:
            robot.error_recovery()
        except Exception as exc:  # noqa: BLE001
            print(f'error_recovery before motion (ok to ignore if clean): {exc}')
        time.sleep(0.5)

        source.refresh()
        latest = source.latest()
        if latest is None:
            raise RuntimeError('No EE pose after free-drive refresh')
        pos, quat = latest
        home = snapshot_to_pose_4x4(pos, quat)

        targets = compute_poses_around_state(
            home,
            math.radians(cli.rotation_delta_degrees),
            cli.translation_delta_meters,
        )
        if cli.first_n > 0:
            targets = targets[:cli.first_n]
        print(f'Home captured; running {len(targets)} offset poses '
              f'(rot={cli.rotation_delta_degrees} deg, '
              f'trans={cli.translation_delta_meters} m)')

        for i, T in enumerate(targets):
            print(f'Pose {i + 1}/{len(targets)}')
            _move(f'Pose {i + 1}', T)
            time.sleep(cli.settle_sec)

        if cli.return_home:
            print('Returning home')
            _move('Return home', home)
        print('Motion test done')
    finally:
        robot.shutdown()


if __name__ == '__main__':
    main()
