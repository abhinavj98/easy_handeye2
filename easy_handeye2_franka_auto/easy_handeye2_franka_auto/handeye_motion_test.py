"""CLI: free-drive home → cube-subset offsets → move only (no handeye / TF)."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description='Motion-only smoke test: free-drive home, then cube-subset EE offsets')
    p.add_argument('--robot-config', type=Path, required=True,
                   help='YAML with robot: {ip, use_mock, reset_duration_sec, ...}')
    p.add_argument('--rotation-delta-degrees', type=float, default=15.0,
                   help='EE-frame tilt magnitude about ±X/Y/Z (default 15)')
    p.add_argument('--cube-half-size-meters', type=float, default=0.05,
                   help='base-frame cube half-size; corners at (±d,±d,±d)')
    p.add_argument('--n-poses', type=int, default=15,
                   help='random subset size from the 54-pose pool (default 15)')
    p.add_argument('--seed', type=int, default=0, help='RNG seed for pose subset')
    p.add_argument('--settle-sec', type=float, default=1.0)
    p.add_argument('--tf-dwell-sec', type=float, default=0.6,
                   help='hold after refresh (same as handeye_auto_calibrate)')
    p.add_argument('--motion-mode', choices=('cartesian', 'dls'), default='cartesian',
                   help="'cartesian' = original Cartesian streaming reset; "
                        "'dls' = singularity-robust joint-space tracker that gets "
                        "as close as possible and stops instead of faulting")
    p.add_argument('--first-n', type=int, default=3,
                   help='only run the first N selected poses (0 = all; default 3)')
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
        compute_cube_poses,
        snapshot_to_pose_4x4,
    )
    from easy_handeye2_franka_auto.pro_robot_interface import FrankaInterface
    from easy_handeye2_franka_auto.robot_pose_source import RobotPoseSource

    robot = FrankaInterface(robot_cfg, device='cpu')
    source = RobotPoseSource(robot)
    home = None
    try:
        input('Free-drive to a safe start pose, then press Enter...')
        source.refresh()
        latest = source.latest()
        if latest is None:
            raise RuntimeError('No EE pose after free-drive refresh')
        pos, quat = latest
        home = snapshot_to_pose_4x4(pos, quat)

        targets = compute_cube_poses(
            home,
            math.radians(cli.rotation_delta_degrees),
            cli.cube_half_size_meters,
            n_poses=cli.n_poses,
            seed=cli.seed,
        )
        if cli.first_n > 0:
            targets = targets[:cli.first_n]
        print(
            f'Home captured; running {len(targets)} cube-subset poses '
            f'(pool=54, n_poses={cli.n_poses}, seed={cli.seed}, '
            f'rot={cli.rotation_delta_degrees} deg, d={cli.cube_half_size_meters} m)'
        )

        succeeded = 0
        failed_indices = []
        for i, T in enumerate(targets):
            print(f'Pose {i + 1}/{len(targets)}')
            try:
                residual = source.go_to(T, cli.settle_sec, cli.tf_dwell_sec, motion=cli.motion_mode)
            except Exception as exc:  # noqa: BLE001
                print(f'Motion/state failure at pose index {i}: {exc}; skipping')
                failed_indices.append(i)
                continue
            succeeded += 1
            if residual is not None:
                print(f'  DLS residual pose error: {residual:.4f}')

        if cli.return_home and home is not None:
            print('Returning home')
            try:
                source.go_to(home, cli.settle_sec, cli.tf_dwell_sec, motion=cli.motion_mode)
            except Exception as exc:  # noqa: BLE001
                print(f'Return-home failed: {exc}')
        print(
            f'Motion test done ({cli.motion_mode}): {succeeded}/{len(targets)} poses ok'
            + (f'; faulted at indices {failed_indices}' if failed_indices else '')
        )
    finally:
        robot.shutdown()


if __name__ == '__main__':
    main()
