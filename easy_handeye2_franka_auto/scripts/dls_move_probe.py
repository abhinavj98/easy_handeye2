#!/usr/bin/env python3
"""Hardware probe: run the production DLS move standalone, outside the comm process.

Same code path as ``--motion-mode dls`` (``run_continuous_dls_move`` with the
``robot:`` section of robot.yaml), but in a plain process with nothing else
running, logging every cycle (sent vs robot-reported q_d, measured q,
control_command_success_rate) to CSV. If this moves cleanly while the motion
test faults, the difference is the comm-process environment, not the DLS
command stream.

Target: the current EE pose shifted by --dx/--dy/--dz (m, base frame) and rotated
by --rot-deg about --rot-axis (EE frame). Defaults: 2 cm in +x, no rotation.

  python3 dls_move_probe.py --config ../config/robot.yaml --csv /tmp/dls_probe.csv
"""
import argparse
import csv
import gc
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from easy_handeye2_franka_auto.dls_motion import (  # noqa: E402
    DlsParams,
    JointPositionStreamer,
    limit_blas_threads,
    otee_flat_to_Rt,
    reshape_jacobian_colmajor,
    run_continuous_dls_move,
)
from easy_handeye2_franka_auto.robot_interface import (  # noqa: E402
    _FR3_JOINT_ACCEL_LIMITS,
    _FR3_JOINT_JERK_LIMITS,
    _FR3_JOINT_POS_LIMITS_URDF,
    _FR3_JOINT_VEL_LIMITS,
)


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    ap.add_argument("--rot-deg", type=float, default=0.0)
    ap.add_argument("--rot-axis", choices=("x", "y", "z"), default="x")
    ap.add_argument("--hold-steps", type=int, default=None,
                    help="override dls_hold_steps_first (cycles held before moving)")
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()
    _blas_limit = limit_blas_threads()  # noqa: F841 (RT thread must not wait on BLAS helpers)
    if max(abs(a.dx), abs(a.dy), abs(a.dz)) > 0.05 or abs(a.rot_deg) > 30:
        sys.exit("keep the probe move small: |d*| <= 0.05 m, |rot-deg| <= 30")

    cfg = yaml.safe_load(open(a.config))["robot"]
    if a.hold_steps is not None:
        cfg["dls_hold_steps_first"] = a.hold_steps
    p = DlsParams.from_config(cfg, _FR3_JOINT_VEL_LIMITS, _FR3_JOINT_ACCEL_LIMITS,
                              _FR3_JOINT_JERK_LIMITS, _FR3_JOINT_POS_LIMITS_URDF)

    import pylibfranka as plf
    robot = plf.Robot(cfg["ip"])
    robot.set_EE(cfg.get("NE_T_EE", [0.7071, -0.7071, 0, 0, 0.7071, 0.7071, 0, 0, 0, 0, 1, 0, 0, 0, 0.1034, 1]))
    robot.set_K(cfg.get("EE_T_K", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
    robot.set_collision_behavior([100.0] * 7, [100.0] * 7, [100.0] * 6, [100.0] * 6)
    model = robot.load_model()

    state0 = robot.read_once()
    R0, t0 = otee_flat_to_Rt(state0.O_T_EE)
    t_des = t0 + np.array([a.dx, a.dy, a.dz])
    R_des = R0 @ _axis_rotation(a.rot_axis, math.radians(a.rot_deg))

    input(f"Will move EE by ({a.dx:+.3f}, {a.dy:+.3f}, {a.dz:+.3f}) m, {a.rot_deg:+.1f} deg about "
          f"EE {a.rot_axis}, hold_steps={p.hold_first_steps}. Press Enter to start...")

    rows = []
    stream_ref = [None]

    def read_pose_and_jacobian(state):
        R, t = otee_flat_to_Rt(state.O_T_EE)
        return R, t, reshape_jacobian_colmajor(model.zero_jacobian(state))

    def on_send(cycle, v_prev, v_cmd):
        s = stream_ref[0].state
        sent = stream_ref[0].last_command
        rows.append([-1 if cycle is None else cycle, float(s.control_command_success_rate)]
                    + sent.tolist() + list(s.q_d) + list(s.q))

    log, error, result = [], None, None
    gc.disable()
    try:
        ctrl = robot.start_joint_position_control(plf.ControllerMode.JointImpedance)
        stream = JointPositionStreamer(ctrl, plf.JointPositions, p.rate_vel, p.rate_acc, p.rate_jerk)
        stream_ref[0] = stream
        result = run_continuous_dls_move(stream, read_pose_and_jacobian, R_des, t_des, p,
                                         log=log, on_send=on_send)
    except Exception as e:  # noqa: BLE001
        error = str(e)
    finally:
        gc.enable()
    stream_ref[0] = stream = ctrl = None
    gc.collect()
    try:
        robot.stop()
    except Exception:  # noqa: BLE001
        pass
    if error is not None:
        try:
            robot.automatic_error_recovery()
        except Exception:  # noqa: BLE001
            pass

    for line in log:
        print("DLS", line)
    print(f"result={result} error={error}")
    if a.csv is not None and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cycle", "success_rate"] + [f"sent{i}" for i in range(1, 8)]
                       + [f"q_d{i}" for i in range(1, 8)] + [f"q{i}" for i in range(1, 8)])
            w.writerows(rows)
        print(f"wrote {len(rows)} cycles to {a.csv}")
    if rows:
        r = np.asarray(rows)
        sent, q_d = r[:, 2:9], r[:, 9:16]
        # Logged right after each write: the state is the tick that write answers, so
        # a healthy robot reports q_d == the previous command.
        lag = np.max(np.abs(q_d[1:] - sent[:-1]), axis=1)
        if lag.size:
            print(f"  |q_d - previous sent| per cycle (healthy ~1e-8): median {np.median(lag):.2e}, "
                  f"max {lag.max():.2e}; first 10: {' '.join(f'{x:.1e}' for x in lag[:10])}")
        print(f"  success_rate: min {r[:, 1].min():.3f}")


if __name__ == "__main__":
    main()
