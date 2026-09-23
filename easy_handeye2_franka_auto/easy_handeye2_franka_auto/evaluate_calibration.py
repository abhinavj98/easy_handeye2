"""Quality report for an eye-on-base calibration from its samples.

handeye_auto_calibrate prints it right after computing (print_report); this CLI
re-runs it later from the saved files:

    ros2 run easy_handeye2_franka_auto evaluate_calibration --name fr3_eob

Reads ~/.ros2/easy_handeye2/samples/<name>.samples (written by handeye_auto_calibrate)
and ~/.ros2/easy_handeye2/calibrations/<name>.calib. No robot/ROS graph needed.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

EASY_HANDEYE2_DIR = Path('~/.ros2/easy_handeye2').expanduser()

# Rough pass/fail bands for a RealSense at ~0.5 m (base-frame board residual, RMS)
GOOD_MM, GOOD_DEG = 3.0, 0.5
BAD_MM, BAD_DEG = 10.0, 2.0


def _parse_args(argv):
    p = argparse.ArgumentParser(description='Offline eye-on-base calibration quality report')
    p.add_argument('--name', help='calibration name (sets default --samples / --calibration paths)')
    p.add_argument('--samples', type=Path, help='.samples file (default: <dir>/samples/<name>.samples)')
    p.add_argument('--calibration', type=Path,
                   help='.calib file (default: <dir>/calibrations/<name>.calib); '
                        'omit the file to evaluate a fresh Tsai-Lenz solve')
    p.add_argument('--method', default='Tsai-Lenz', help='solver used for leave-one-out')
    args = p.parse_args(argv)
    if args.samples is None:
        if not args.name:
            p.error('give --name or --samples')
        args.samples = EASY_HANDEYE2_DIR / 'samples' / f'{args.name}.samples'
    if args.calibration is None and args.name:
        args.calibration = EASY_HANDEYE2_DIR / 'calibrations' / f'{args.name}.calib'
    return args


def _fmt_t(t) -> str:
    return '(' + ', '.join(f'{v:+.4f}' for v in t) + ') m'


def _grade(mm: float, deg: float) -> str:
    if mm <= GOOD_MM and deg <= GOOD_DEG:
        return 'GOOD'
    if mm >= BAD_MM or deg >= BAD_DEG:
        return 'BAD'
    return 'MARGINAL'


def main(argv=None) -> int:
    from easy_handeye2_franka_auto.handeye_eval import load_calibration, load_samples, solve_calibration

    cli = _parse_args(sys.argv[1:] if argv is None else argv)
    if not cli.samples.exists():
        print(f'No samples file at {cli.samples}. Samples are only written by handeye_auto_calibrate '
              f'(or the save_samples service); re-run the calibration to get one.')
        return 1
    robot, tracking = load_samples(cli.samples)
    print(f'Samples: {len(robot)} from {cli.samples}')
    if len(robot) < 3:
        print('Need at least 3 samples to evaluate.')
        return 1

    if cli.calibration is not None and cli.calibration.exists():
        X, params = load_calibration(cli.calibration)
        header = (f'Calibration: {cli.calibration}\n'
                  f'  {params["robot_base_frame"]} -> {params["tracking_base_frame"]}')
    else:
        X = solve_calibration(robot, tracking, cli.method)
        header = f'Calibration: fresh {cli.method} solve (no .calib file)'
    grade = print_report(robot, tracking, X, header, cli.method)
    return 0 if grade != 'BAD' else 2


def print_report(robot, tracking, X, header: str = 'Calibration:', method: str = 'Tsai-Lenz') -> str:
    """Print the quality report for calibration X over the samples; return GOOD/MARGINAL/BAD."""
    from easy_handeye2_franka_auto.handeye_eval import (
        METHODS,
        board_in_ee,
        leave_one_out,
        max_pairwise_rotation_deg,
        mean_pose,
        pose_distance,
        residuals,
        rms,
        rpy_deg,
        solve_calibration,
    )

    n = len(robot)
    print(header)
    print(f'  t = {_fmt_t(X[:3, 3])}  |t| = {np.linalg.norm(X[:3, 3]):.4f} m')
    print(f'  rpy = ({", ".join(f"{v:+.2f}" for v in rpy_deg(X[:3, :3]))}) deg')
    print('  -> compare against a tape measure from the robot base to the camera')

    Y = mean_pose(board_in_ee(robot, tracking, X))
    print(f'\nBoard in EE (mean T_ee<-board): t = {_fmt_t(Y[:3, 3])}, '
          f'rpy = ({", ".join(f"{v:+.2f}" for v in rpy_deg(Y[:3, :3]))}) deg')
    print('  -> should match how the board is physically mounted on the gripper')

    span = max_pairwise_rotation_deg(robot)
    print(f'\nRotation diversity: max EE rotation between samples = {span:.1f} deg'
          + ('  (low: < 20 deg makes the solve ill-conditioned)' if span < 20.0 else ''))

    res = residuals(robot, tracking, X, Y)
    loo = leave_one_out(robot, tracking, method) if n >= 4 else []
    print('\nPer-sample board pose error in base frame (robot-predicted vs camera-measured):')
    print('    #   fit mm  fit deg' + ('   LOO mm  LOO deg' if loo else ''))
    for i, (t, r) in enumerate(res):
        line = f'  {i:3d}  {t * 1000:7.2f}  {math.degrees(r):7.3f}'
        if loo:
            line += f'  {loo[i][0] * 1000:7.2f}  {math.degrees(loo[i][1]):7.3f}'
        print(line)
    fit_mm, fit_deg = rms([t for t, _ in res]) * 1000, math.degrees(rms([r for _, r in res]))
    print(f'  RMS  {fit_mm:7.2f}  {fit_deg:7.3f}', end='')
    if loo:
        loo_mm, loo_deg = rms([t for t, _ in loo]) * 1000, math.degrees(rms([r for _, r in loo]))
        print(f'  {loo_mm:7.2f}  {loo_deg:7.3f}')
    else:
        loo_mm, loo_deg = fit_mm, fit_deg
        print('   (leave-one-out needs >= 4 samples)')
    worst = int(np.argmax([t for t, _ in (loo or res)]))
    print(f'  worst sample: #{worst}')

    print('\nSolver agreement (vs evaluated calibration):')
    print('  method        dt mm   drot deg   fit RMS mm')
    for name in METHODS:
        try:
            Xm = solve_calibration(robot, tracking, name)
        except Exception as exc:  # noqa: BLE001 - one solver failing shouldn't stop the report
            print(f'  {name:<12}  failed: {exc}')
            continue
        dt, dr = pose_distance(X, Xm)
        fit = rms([t for t, _ in residuals(robot, tracking, Xm)]) * 1000
        print(f'  {name:<12} {dt * 1000:7.2f}  {math.degrees(dr):8.3f}   {fit:9.2f}')
    print('  -> methods should agree to a few mm / tenths of a degree on good data')

    # Fit RMS judges the evaluated calibration itself; LOO (which re-solves) judges how well
    # the data generalises. A wrong saved .calib only shows up in the fit, so grade on both.
    worst_mm, worst_deg = max(fit_mm, loo_mm), max(fit_deg, loo_deg)
    grade = _grade(worst_mm, worst_deg)
    print(f'\nVerdict: {grade} (worst of fit / LOO RMS: {worst_mm:.2f} mm / {worst_deg:.3f} deg; '
          f'good <= {GOOD_MM} mm / {GOOD_DEG} deg, bad >= {BAD_MM} mm / {BAD_DEG} deg)')
    return grade


if __name__ == '__main__':
    sys.exit(main())
