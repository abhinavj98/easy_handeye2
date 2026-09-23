"""CLI: free-drive home → cube-subset offsets → take_sample → compute/save."""
from __future__ import annotations

import argparse
import math
import sys
import threading
from pathlib import Path


def should_save(num_samples: int, min_samples: int) -> bool:
    return num_samples >= max(min_samples, 3)


def sample_added(n_before: int, n_after: int) -> bool:
    return n_after > n_before


def _parse_args(argv):
    p = argparse.ArgumentParser(description='Automated eye-on-base hand-eye sampling')
    p.add_argument('--robot-config', type=Path, required=True)
    p.add_argument('--name', required=True, help='easy_handeye2 calibration name (must match calibrate launch)')
    p.add_argument('--robot-base-frame', required=True)
    p.add_argument('--robot-effector-frame', required=True)
    p.add_argument('--rotation-delta-degrees', type=float, default=25.0,
                   help='EE-frame tilt magnitude about ±X/Y/Z (degrees)')
    p.add_argument('--cube-half-size-meters', type=float, default=0.05,
                   help='base-frame cube half-size; corners at (±d,±d,±d)')
    p.add_argument('--n-poses', type=int, default=15,
                   help='random subset size from the 54-pose cube×tilt pool')
    p.add_argument('--seed', type=int, default=0, help='RNG seed for pose subset')
    p.add_argument('--settle-sec', type=float, default=1.0)
    p.add_argument('--tf-dwell-sec', type=float, default=0.6,
                   help='hold after refresh so TF covers the sampler 0.2 s lookback')
    p.add_argument('--tf-rate-hz', type=float, default=30.0)
    p.add_argument('--freedrive-poll-hz', type=float, default=0.0,
                   help='>0 polls robot state during free-drive (unverified on hardware; keep <=2)')
    p.add_argument('--min-samples', type=int, default=5)
    p.add_argument('--first-n', type=int, default=0, help='only run the first N selected poses (0 = all)')
    p.add_argument('--keep-existing-samples', action='store_true')
    p.add_argument('--return-home', action='store_true')
    return p.parse_args(argv)


def main(args=None):
    import rclpy
    import yaml
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.utilities import remove_ros_args

    from easy_handeye2.handeye_client import HandeyeClient
    from easy_handeye2_msgs.msg import HandeyeCalibrationParameters
    from easy_handeye2_franka_auto.handeye_offsets import (
        compute_cube_poses,
        snapshot_to_pose_4x4,
    )
    from easy_handeye2_franka_auto.handeye_tf_bridge import RobotTfBridge
    from easy_handeye2_franka_auto.robot_pose_source import RobotPoseSource

    cli = _parse_args(remove_ros_args(args=sys.argv)[1:])
    with open(cli.robot_config) as f:
        robot_cfg = yaml.safe_load(f)

    rclpy.init(args=args)
    node = rclpy.create_node('handeye_auto_calibrate')
    log = node.get_logger()
    robot = source = bridge = executor = None
    home = None
    try:
        from easy_handeye2_franka_auto.pro_robot_interface import FrankaInterface

        robot = FrankaInterface(robot_cfg, device='cpu')
        source = RobotPoseSource(robot)
        bridge = RobotTfBridge(source, cli.robot_base_frame, cli.robot_effector_frame, cli.tf_rate_hz)

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.add_node(bridge)
        threading.Thread(target=executor.spin, daemon=True).start()

        params = HandeyeCalibrationParameters(
            name=cli.name,
            calibration_type='eye_on_base',
            robot_base_frame=cli.robot_base_frame,
            robot_effector_frame=cli.robot_effector_frame,
            freehand_robot_movement=True,
        )
        client = HandeyeClient(node, params)

        def n_samples() -> int:
            return len(client.get_sample_list().samples)

        existing = n_samples()
        if existing and not cli.keep_existing_samples:
            log.warn(f'Removing {existing} pre-existing samples (use --keep-existing-samples to keep)')
            while n_samples():
                client.remove_sample(0)

        source.start_polling(cli.freedrive_poll_hz)
        input('Free-drive marker into camera center, then press Enter...')
        source.stop_polling()

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
        log.info(
            f'Home captured; running {len(targets)} cube-subset poses '
            f'(pool=54, n_poses={cli.n_poses}, seed={cli.seed}, '
            f'rot={cli.rotation_delta_degrees} deg, d={cli.cube_half_size_meters} m)'
        )

        for i, T in enumerate(targets):
            log.info(f'Pose {i + 1}/{len(targets)}')
            try:
                source.go_to(T, cli.settle_sec, cli.tf_dwell_sec)
            except Exception as exc:  # noqa: BLE001
                log.error(f'Motion/state failure at pose index {i}: {exc}; skipping')
                continue
            before = n_samples()
            client.take_sample()
            if sample_added(before, n_samples()):
                log.info(f'Sample ok ({n_samples()} total)')
            else:
                log.warn(f'Skipping pose {i}: sample not recorded (missing/extrapolating TF?)')

        total = n_samples()
        if should_save(total, cli.min_samples):
            result = client.compute_calibration()
            if result.valid:
                saved = client.save()
                log.info(f'Saved calibration ({total} samples): success={saved.success}')
            else:
                log.error('compute_calibration returned valid=false; not saving')
        else:
            log.error(f'Not saving: only {total} samples (need >= {max(cli.min_samples, 3)})')

        if cli.return_home and home is not None:
            try:
                source.go_to(home, cli.settle_sec, cli.tf_dwell_sec)
            except Exception as exc:  # noqa: BLE001
                log.error(f'Return-home failed: {exc}')
    finally:
        if source is not None:
            source.stop_polling()
        if robot is not None:
            robot.shutdown()
        if executor is not None:
            executor.shutdown()
        if bridge is not None:
            bridge.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
