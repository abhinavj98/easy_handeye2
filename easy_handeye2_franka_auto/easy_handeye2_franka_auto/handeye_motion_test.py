"""CLI: free-drive home → A-style offsets → move only (no handeye / TF)."""
from __future__ import annotations

import argparse
import math
import sys
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
    p.add_argument('--tf-dwell-sec', type=float, default=0.6,
                   help='hold after refresh (same as handeye_auto_calibrate)')
    p.add_argument('--first-n', type=int, default=3,
                   help='only run the first N offset poses (0 = all; default 3). '
                        'Note: poses 1-12 are EE rotations; 13-17 are base translations.')
    p.add_argument('--translations-only', action='store_true',
                   help='skip EE rotations; only run base-frame translation offsets')
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
    try:
        input('Free-drive to a safe start pose, then press Enter...')
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
        # Pose layout: [12 EE rotations] + [5 base translations].
        # With rotation_delta=0 the first 12 are identical to home — skip them
        # unless the user explicitly wants that no-op list.
        if cli.translations_only or abs(cli.rotation_delta_degrees) < 1e-9:
            targets = targets[12:]
            print('Using translation offsets only (base ±X/±Y/+Z)')
        if cli.first_n > 0:
            targets = targets[:cli.first_n]
        print(f'Home captured; running {len(targets)} offset poses '
              f'(rot={cli.rotation_delta_degrees} deg, '
              f'trans={cli.translation_delta_meters} m)')

        for i, T in enumerate(targets):
            print(f'Pose {i + 1}/{len(targets)}')
            try:
                source.go_to(T, cli.settle_sec, cli.tf_dwell_sec)
            except Exception as exc:  # noqa: BLE001
                print(f'Motion/state failure at pose index {i}: {exc}')
                return

        if cli.return_home:
            print('Returning home')
            source.go_to(home, cli.settle_sec, cli.tf_dwell_sec)
        print('Motion test done')
    finally:
        robot.shutdown()


if __name__ == '__main__':
    main()
