#!/usr/bin/env python3
"""Hardware probe: does the FR3 accept externally streamed joint-position commands?

Moves ONE joint by a small amount (default joint 7, 0.05 rad ≈ 3°, over 3 s, cosine
profile, then holds) through pylibfranka's startJointPositionControl +
readOnce/writeOnce, exactly the API the DLS mode uses, and logs every cycle:
the robot-reported q_d/dq_d vs what was sent, the readOnce period, and
control_command_success_rate.

--ref picks the rate-limiting reference:
  robot  franka::limitRate against the robot's own q_d/dq_d/ddq_d each cycle, as in
         libfranka's examples/generate_joint_position_motion_external_control_loop.cpp
  own    against our own last command (what JointPositionStreamer does)
  none   no rate limiting (what the working "move_joints"/Cartesian reset paths do)

Usage (robot in FCI mode, user stop at hand, nothing else controlling the arm):
  python3 joint_stream_probe.py 192.168.1.11 --ref robot --csv probe_robot.csv
"""
import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from easy_handeye2_franka_auto.dls_motion import limit_rate_joint_positions  # noqa: E402
from easy_handeye2_franka_auto.robot_interface import (  # noqa: E402
    _FR3_JOINT_ACCEL_LIMITS,
    _FR3_JOINT_JERK_LIMITS,
    _FR3_JOINT_VEL_LIMITS,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ip")
    ap.add_argument("--ref", choices=("robot", "own", "none"), default="robot")
    ap.add_argument("--joint", type=int, default=7, help="1-indexed joint to move (default 7)")
    ap.add_argument("--amplitude", type=float, default=0.05, help="rad (default 0.05)")
    ap.add_argument("--duration", type=float, default=3.0, help="s for the move (default 3)")
    ap.add_argument("--hold", type=float, default=0.5, help="s to hold before and after")
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()
    if not (0.0 < abs(a.amplitude) <= 0.2):
        sys.exit("--amplitude must be in (0, 0.2] rad")

    import pylibfranka as plf

    robot = plf.Robot(a.ip)
    robot.set_collision_behavior([100.0] * 7, [100.0] * 7, [100.0] * 6, [100.0] * 6)
    j = a.joint - 1
    frac = 0.9
    vel, acc, jerk = _FR3_JOINT_VEL_LIMITS * frac, _FR3_JOINT_ACCEL_LIMITS * frac, _FR3_JOINT_JERK_LIMITS * frac

    input(f"Will move joint {a.joint} by {a.amplitude:+.3f} rad over {a.duration:.1f}s "
          f"(ref={a.ref}). Press Enter to start...")
    ctrl = robot.start_joint_position_control(plf.ControllerMode.JointImpedance)
    state, _ = ctrl.readOnce()
    q0 = np.asarray(state.q, dtype=float).copy()
    last_q, last_dq, last_ddq = q0.copy(), np.zeros(7), np.zeros(7)
    n_hold = int(a.hold * 1000)
    n_move = int(a.duration * 1000)
    rows = []
    error = None
    gc.disable()
    try:
        for k in range(n_hold + n_move + n_hold):
            i = k - n_hold
            alpha = 0.0 if i < 0 else 0.5 * (1.0 - math.cos(math.pi * min(i + 1, n_move) / n_move))
            target = q0.copy()
            target[j] += alpha * a.amplitude
            if a.ref == "robot":
                ref_q = q0 if k == 0 else np.asarray(state.q_d, dtype=float)
                cmd = limit_rate_joint_positions(
                    target, ref_q, np.asarray(state.dq_d, dtype=float),
                    np.asarray(state.ddq_d, dtype=float), vel, acc, jerk)
            elif a.ref == "own":
                cmd = limit_rate_joint_positions(target, last_q, last_dq, last_ddq, vel, acc, jerk)
            else:
                cmd = target
            new_dq = (cmd - last_q) / 1e-3
            last_ddq, last_dq, last_q = (new_dq - last_dq) / 1e-3, new_dq, cmd
            ctrl.writeOnce(plf.JointPositions(cmd.tolist()))
            t0 = time.perf_counter()
            state, period = ctrl.readOnce()
            rows.append((
                k, period.to_sec() * 1e3, (time.perf_counter() - t0) * 1e3,
                cmd[j], state.q_d[j], state.q[j], state.dq_d[j], state.ddq_d[j],
                float(state.control_command_success_rate)))
        cmd_obj = plf.JointPositions(last_q.tolist())
        cmd_obj.motion_finished = True
        ctrl.writeOnce(cmd_obj)
    except Exception as e:  # noqa: BLE001
        error = str(e)
    finally:
        gc.enable()
        ctrl = None
        try:
            robot.stop()
        except Exception:  # noqa: BLE001
            pass
        if error is not None:
            try:
                robot.automatic_error_recovery()
            except Exception:  # noqa: BLE001
                pass

    hdr = ("cycle", "period_ms", "readOnce_wait_ms", "sent", "q_d", "q", "dq_d", "ddq_d", "success_rate")
    if a.csv is not None:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(rows)
        print(f"wrote {len(rows)} cycles to {a.csv}")

    r = np.asarray(rows, dtype=float) if rows else np.zeros((0, len(hdr)))
    print(f"ref={a.ref} joint={a.joint} cycles={len(rows)} error={error}")
    if len(r) > 1:
        moved_cmd = r[-1, 3] - r[0, 3]
        moved_qd = r[-1, 4] - r[0, 4]
        moved_q = r[-1, 5] - r[0, 5]
        lag0 = np.abs(r[1:, 4] - r[1:, 3])      # q_d vs command just sent
        lag1 = np.abs(r[1:, 4] - r[:-1, 3])     # q_d vs previous command
        print(f"  commanded moved {moved_cmd:+.5f} rad, q_d moved {moved_qd:+.5f}, measured q moved {moved_q:+.5f}")
        print(f"  median |q_d - sent| {np.median(lag0):.2e}, median |q_d - prev sent| {np.median(lag1):.2e}")
        print(f"  period_ms: mean {r[:, 1].mean():.3f} max {r[:, 1].max():.3f} n>1ms {int(np.sum(r[:, 1] > 1.0))}")
        print(f"  success_rate: min {r[:, 8].min():.3f} last {r[-1, 8]:.3f}")
        print("  last 5 cycles (cycle, period_ms, wait_ms, sent, q_d, q, dq_d, ddq_d, success):")
        for row in rows[-5:]:
            print("   ", " ".join(f"{x:.6g}" for x in row))


if __name__ == "__main__":
    main()
