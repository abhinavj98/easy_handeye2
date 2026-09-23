import math
from types import SimpleNamespace

import numpy as np
import pytest

from easy_handeye2_franka_auto.dls_motion import (
    DT,
    FR3_READY_Q,
    DlsParams,
    JointPositionStreamer,
    clamp_step_for_duration,
    clamp_task_error,
    continuous_dls_step,
    contract_jacobian_derivative,
    dls_joint_step,
    dls_step_with_joint_limits,
    format_singularity_report,
    limit_rate_joint_positions,
    jacobian_derivative,
    log_manipulability_gradient,
    manipulability,
    min_singular_value,
    min_singular_value_gradient,
    nearest_singular_direction_joint_report,
    otee_flat_to_Rt,
    plan_dls_hop,
    plan_nullspace_escape_hop,
    pose_error_6d,
    quintic,
    quintic_duration,
    reshape_jacobian_colmajor,
    rotation_error_axis_angle,
    run_continuous_dls_move,
    singularity_barrier_filter,
    singularity_robust_lambda,
    slew_limit_velocity,
    task_velocity,
)
from easy_handeye2_franka_auto.robot_interface import (
    _FR3_JOINT_ACCEL_LIMITS,
    _FR3_JOINT_JERK_LIMITS,
    _FR3_JOINT_POS_LIMITS_URDF,
    _FR3_JOINT_VEL_LIMITS,
)

# A comfortable, non-singular FR3 configuration.
Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])

# ∂J/∂q for the linear task models (x = A q) used below: exactly zero.
ZERO_DJ = np.zeros((7, 6, 7))


def _params(**overrides) -> DlsParams:
    return DlsParams.from_config(
        overrides, _FR3_JOINT_VEL_LIMITS, _FR3_JOINT_ACCEL_LIMITS,
        _FR3_JOINT_JERK_LIMITS, _FR3_JOINT_POS_LIMITS_URDF)


class StrictFakeFci:
    """Stand-in for an FR3 joint position session that faults like the real robot:
    implied acceleration > kMaxJointAcceleration → velocity_discontinuity, implied jerk
    > kMaxJointJerk → acceleration_discontinuity, joint velocity over limit, joint
    position outside limits, and motion_finished while still moving. Tracks perfectly
    (q = q_d), so it isolates the command stream."""

    def __init__(self, q0):
        self.q_d = np.asarray(q0, dtype=float).copy()
        self.dq_d = np.zeros(7)
        self.ddq_d = np.zeros(7)
        self.finished = False
        self.cycles = 0
        self.peak = {"vel": 0.0, "acc": 0.0, "jerk": 0.0}

    def readOnce(self):
        s = SimpleNamespace(
            q=self.q_d.copy(), q_d=self.q_d.copy(), dq_d=self.dq_d.copy(),
            ddq_d=self.ddq_d.copy(), dq=self.dq_d.copy())
        return s, None

    def writeOnce(self, cmd):
        assert not self.finished, "command after motion_finished"
        q = np.asarray(cmd.q, dtype=float)
        v = (q - self.q_d) / DT
        a = (v - self.dq_d) / DT
        j = (a - self.ddq_d) / DT
        self.peak["vel"] = max(self.peak["vel"], float(np.max(np.abs(v) / _FR3_JOINT_VEL_LIMITS)))
        self.peak["acc"] = max(self.peak["acc"], float(np.max(np.abs(a) / _FR3_JOINT_ACCEL_LIMITS)))
        self.peak["jerk"] = max(self.peak["jerk"], float(np.max(np.abs(j) / _FR3_JOINT_JERK_LIMITS)))
        if np.any(np.abs(a) > _FR3_JOINT_ACCEL_LIMITS):
            raise RuntimeError(f"joint_motion_generator_velocity_discontinuity at cycle {self.cycles}")
        if np.any(np.abs(j) > _FR3_JOINT_JERK_LIMITS):
            raise RuntimeError(f"joint_motion_generator_acceleration_discontinuity at cycle {self.cycles}")
        if np.any(np.abs(v) > _FR3_JOINT_VEL_LIMITS):
            raise RuntimeError(f"joint_velocity_violation at cycle {self.cycles}")
        lo, hi = _FR3_JOINT_POS_LIMITS_URDF[:, 0], _FR3_JOINT_POS_LIMITS_URDF[:, 1]
        if np.any(q < lo) or np.any(q > hi):
            raise RuntimeError(f"joint_position_limits_violation at cycle {self.cycles}")
        self.q_d, self.dq_d, self.ddq_d = q, v, a
        self.cycles += 1
        if getattr(cmd, "motion_finished", False):
            if np.max(np.abs(v)) > 1e-3 or np.max(np.abs(a)) > 5e-2:
                raise RuntimeError("Motion finished commanded, but the robot is still moving!")
            self.finished = True


class LaggyFakeFci(StrictFakeFci):
    """Like StrictFakeFci, but ``readOnce()`` reports state that lags one
    ``writeOnce()`` behind — approximating the round-trip latency between a
    real robot's control loop and our external readOnce/writeOnce driver
    (StrictFakeFci's synchronous q==q_d tracking is what let the windup and
    discontinuity bugs through unit tests but not real hardware). Every
    ``writeOnce()`` still validates against the *true* last-applied state, so
    a client that computes its next command off this stale reference still
    gets caught here if that produces a genuine discontinuity."""

    def __init__(self, q0):
        super().__init__(q0)
        self._queue = [self._snapshot()]

    def _snapshot(self):
        return SimpleNamespace(
            q=self.q_d.copy(), q_d=self.q_d.copy(), dq_d=self.dq_d.copy(),
            ddq_d=self.ddq_d.copy(), dq=self.dq_d.copy())

    def readOnce(self):
        s = self._queue.pop(0)
        if not self._queue:
            self._queue.append(self._snapshot())
        return s, None

    def writeOnce(self, cmd):
        super().writeOnce(cmd)
        self._queue.append(self._snapshot())


def _make_cmd(q):
    return SimpleNamespace(q=q, motion_finished=False)


def _streamer(fci, p):
    return JointPositionStreamer(fci, _make_cmd, p.rate_vel, p.rate_acc, p.rate_jerk)


def _execute_hop(stream, q_plan, dq, T, settle_steps):
    n = max(int(round(T / DT)), 1)
    for k in range(1, n + 1):
        stream.send(q_plan + quintic(k / n) * dq)
    q_plan = q_plan + dq
    stream.hold(q_plan, settle_steps)
    return q_plan


# --- basic kinematics helpers ------------------------------------------------

def test_rotation_error_zero_for_identity():
    R = np.eye(3)
    np.testing.assert_allclose(rotation_error_axis_angle(R, R), 0.0, atol=1e-9)


def test_pose_error_translation_only():
    R = np.eye(3)
    e = pose_error_6d(R, np.zeros(3), R, np.array([0.1, -0.2, 0.3]))
    np.testing.assert_allclose(e[:3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(e[3:], 0.0, atol=1e-9)


def test_reshape_jacobian_colmajor_shape():
    assert reshape_jacobian_colmajor(np.arange(42, dtype=float)).shape == (6, 7)


def test_dls_finite_near_singularity():
    J = np.zeros((6, 7))
    J[0, 0] = 1e-6
    dq = dls_joint_step(J, np.ones(6), damping=0.1)
    assert np.all(np.isfinite(dq)) and np.linalg.norm(dq) < 50.0


def test_manipulability_values():
    J = np.zeros((6, 7))
    J[:, :6] = np.eye(6)
    assert abs(manipulability(J) - 1.0) < 1e-9
    assert manipulability(np.zeros((6, 7))) == 0.0


def test_singularity_robust_lambda_schedule():
    assert abs(singularity_robust_lambda(1.0, 0.05, 0.03, 0.3) - 0.05) < 1e-12
    lams = [singularity_robust_lambda(m, 0.05, 0.03, 0.3) for m in np.linspace(0, 0.06, 13)]
    assert all(a >= b for a, b in zip(lams, lams[1:]))
    assert abs(lams[0] - math.hypot(0.05, 0.3)) < 1e-12


def test_clamp_task_error_caps_each_part():
    e = clamp_task_error(np.array([0.3, 0.4, 0.0, 1.0, 0.0, 0.0]), 0.05, 0.25)
    assert abs(np.linalg.norm(e[:3]) - 0.05) < 1e-12
    assert abs(np.linalg.norm(e[3:]) - 0.25) < 1e-12
    small = np.array([0.01, 0, 0, 0.01, 0, 0])
    np.testing.assert_allclose(clamp_task_error(small, 0.05, 0.25), small)


# --- singularity diagnosis ----------------------------------------------------

def test_singularity_report_identifies_the_weak_joint():
    J = np.zeros((6, 7))
    J[:5, :5] = np.eye(5)
    J[5, 5] = 1e-5  # joint6 (0-indexed 5) barely couples to the last task direction
    # column 6 (joint7) is left all-zero: that's the arm's ordinary redundancy,
    # not a singularity, and must not be reported as the "weak" joint.
    report = nearest_singular_direction_joint_report(J, top_k=1)
    assert report[0][0] == 5
    assert report[0][1] > 0.9  # near-exclusively joint6
    assert format_singularity_report(J, top_k=1) == f"joint6={report[0][1]:.2f}"


def test_singularity_report_weights_sum_to_one():
    J = np.zeros((6, 7))
    J[:5, :5] = np.eye(5)
    J[5, 5] = 1e-5
    full = nearest_singular_direction_joint_report(J, top_k=7)
    assert len(full) == 7
    assert abs(sum(w for _, w in full) - 1.0) < 1e-9


# --- nullspace escape ----------------------------------------------------------

def test_nullspace_escape_is_task_neutral_and_within_step_cap():
    p = _params()
    rng = np.random.default_rng(1)
    J = rng.normal(size=(6, 7))  # full row rank almost surely
    dq, T = plan_nullspace_escape_hop(Q_HOME, J, p)
    assert np.all(np.isfinite(dq))
    assert np.linalg.norm(J @ dq) < 1e-8  # doesn't disturb the commanded pose
    assert 0.0 < np.linalg.norm(dq) <= p.escape_step_rad + 1e-9
    assert p.min_sec <= T <= p.max_sec


def test_nullspace_escape_is_zero_when_already_at_preferred_posture():
    p = _params()
    J = np.eye(6, 7)
    dq, T = plan_nullspace_escape_hop(p.q_preferred, J, p)
    np.testing.assert_allclose(dq, 0.0, atol=1e-12)
    assert T == 0.0


def test_nullspace_escape_respects_joint_limits():
    p = _params()
    J = np.zeros((6, 7))
    J[:6, :6] = np.eye(6)  # column 6 (joint7) is the only nullspace direction
    q_start = Q_HOME.copy()
    q_start[6] = p.upper[6] - 1e-4  # joint7 right at its margined upper limit
    p.q_preferred = p.q_preferred.copy()
    p.q_preferred[6] = p.upper[6] + 5.0  # pulls joint7 further past the limit
    dq, T = plan_nullspace_escape_hop(q_start, J, p)
    np.testing.assert_allclose(dq[:6], 0.0, atol=1e-9)
    assert dq[6] > 0.0
    assert q_start[6] + dq[6] <= p.upper[6] + 1e-9


# --- trajectory shaping ------------------------------------------------------

def test_quintic_endpoints_and_flat_ends():
    assert quintic(0.0) == 0.0 and quintic(1.0) == 1.0
    n = 1000
    s = np.array([quintic(k / n) for k in range(n + 1)])
    # zero velocity and acceleration at both ends: first/last increments are O(1/n^3)
    assert s[1] < 1e-7 and 1.0 - s[-2] < 1e-7


@pytest.mark.parametrize("dq_mag,binding", [(1.5, "vel"), (0.2, "acc"), (1e-4, "jerk")])
def test_quintic_duration_hits_the_binding_cap(dq_mag, binding):
    p = _params()
    dq = np.zeros(7)
    dq[0] = dq_mag
    T = quintic_duration(dq, p.plan_vel, p.plan_acc, p.plan_jerk, 1e-4, 100.0)
    peaks = {
        "vel": 1.875 * dq_mag / T / p.plan_vel[0],
        "acc": (10 / math.sqrt(3)) * dq_mag / T ** 2 / p.plan_acc[0],
        "jerk": 60.0 * dq_mag / T ** 3 / p.plan_jerk[0],
    }
    assert abs(peaks[binding] - 1.0) < 1e-9
    assert all(v <= 1.0 + 1e-9 for v in peaks.values())


def test_clamp_step_is_noop_when_duration_not_capped_and_shrinks_when_capped():
    p = _params()
    dq = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    T = quintic_duration(dq, p.plan_vel, p.plan_acc, p.plan_jerk, 0.2, 5.0)
    np.testing.assert_allclose(clamp_step_for_duration(dq, p.plan_vel, p.plan_acc, p.plan_jerk, T), dq)
    big = dq * 50
    T_cap = quintic_duration(big, p.plan_vel, p.plan_acc, p.plan_jerk, 0.2, 5.0)
    assert T_cap == 5.0
    out = clamp_step_for_duration(big, p.plan_vel, p.plan_acc, p.plan_jerk, T_cap)
    assert np.linalg.norm(out) < np.linalg.norm(big)
    np.testing.assert_allclose(out / np.linalg.norm(out), big / np.linalg.norm(big))


# --- rate limiter + streaming against a strict fake robot -------------------

def test_rate_limited_step_target_never_faults():
    """Even a discontinuous target (0.5 rad step on every joint) is made
    feasible by the limiter: this is the guarantee against *_discontinuity."""
    p = _params()
    fci = StrictFakeFci(Q_HOME)
    stream = _streamer(fci, p)
    target = Q_HOME + np.array([0.5, 0.3, -0.5, 0.3, 0.5, -0.3, 0.5])
    for _ in range(3000):
        stream.send(target)
    assert fci.peak["acc"] <= 0.9 + 1e-9 and fci.peak["jerk"] <= 0.9 + 1e-9


def test_limit_rate_passes_through_feasible_commands():
    p = _params()
    last_q, last_dq, last_ddq = Q_HOME, np.full(7, 0.1), np.full(7, 0.5)
    target = last_q + (last_dq + last_ddq * DT) * DT  # exactly continues current motion
    out = limit_rate_joint_positions(target, last_q, last_dq, last_ddq, p.rate_vel, p.rate_acc, p.rate_jerk)
    np.testing.assert_allclose(out, target, atol=1e-12)


def test_planned_quintic_hop_is_untouched_by_limiter_and_finishes_cleanly():
    p = _params()
    fci = StrictFakeFci(Q_HOME)
    stream = _streamer(fci, p)
    q_plan = stream.measured_q()
    stream.hold(q_plan, p.hold_first_steps)
    dq = np.array([0.4, -0.3, 0.2, 0.3, -0.2, 0.25, -0.4])
    T = quintic_duration(dq, p.plan_vel, p.plan_acc, p.plan_jerk, p.min_sec, p.max_sec)
    n = max(int(round(T / DT)), 1)
    for k in range(1, n + 1):
        target = q_plan + quintic(k / n) * dq
        sent = stream.send(target)
        np.testing.assert_allclose(sent, target, atol=1e-9)  # limiter is only a safety net
    q_plan = q_plan + dq
    stream.finish(q_plan, p.finish_max_steps)
    assert fci.finished
    np.testing.assert_allclose(fci.q_d, q_plan, atol=1e-9)
    assert fci.peak["acc"] <= 0.5 + 1e-6 and fci.peak["vel"] <= 0.5 + 1e-6


# --- planning ----------------------------------------------------------------

def test_joint_limit_aware_step_locks_joint_and_uses_the_others():
    p = _params()
    q = Q_HOME.copy()
    q[0] = p.upper[0] - 1e-3  # joint 0 right at its (margined) upper limit
    J = np.zeros((6, 7))
    J[0, 0] = 1.0
    J[0, 2] = 1.0  # joint 2 can also produce task x
    J[1, 1] = J[2, 3] = J[3, 4] = J[4, 5] = J[5, 6] = 1.0
    err = np.array([0.2, 0, 0, 0, 0, 0])
    dq, locked = dls_step_with_joint_limits(J, err, 0.05, q, p.lower, p.upper)
    assert locked[0]
    assert q[0] + dq[0] <= p.upper[0] + 1e-12
    assert dq[2] > 0.1  # redundancy used instead of stalling


def test_plan_near_singularity_stays_finite_and_within_caps():
    p = _params()
    J = np.zeros((6, 7))
    J[:5, :5] = np.eye(5)
    J[5, 5] = 1e-5  # nearly lost one task direction
    dq, T, sigma, lam = plan_dls_hop(Q_HOME, J, np.full(6, 0.2), p)
    assert sigma < p.sigma_avoid and lam > p.lambda_min
    assert np.all(np.isfinite(dq)) and p.min_sec <= T <= p.max_sec
    np.testing.assert_allclose(clamp_step_for_duration(dq, p.plan_vel, p.plan_acc, p.plan_jerk, T), dq)


def test_end_to_end_converges_on_linear_model_without_faults():
    """Full executor loop (plan → quintic hop → settle → re-measure) on a linear
    task model x = A q, streamed through the strict fake robot."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 7)) * 0.5
    p = _params()
    q_goal = Q_HOME + np.array([0.3, -0.2, 0.25, 0.2, -0.3, 0.2, -0.4])
    x_goal = A @ q_goal

    fci = StrictFakeFci(Q_HOME)
    stream = _streamer(fci, p)
    q_plan = stream.measured_q()
    stream.hold(q_plan, p.hold_first_steps)
    err_norm = None
    for _ in range(p.max_segments):
        err = x_goal - A @ stream.measured_q()
        err_norm = float(np.linalg.norm(err))
        if err_norm < 1e-3:
            break
        dq, T, _, _ = plan_dls_hop(q_plan, A, err, p)
        q_plan = _execute_hop(stream, q_plan, dq, T, p.settle_steps)
    stream.finish(q_plan, p.finish_max_steps)
    assert err_norm < 1e-3
    assert fci.finished


def test_end_to_end_unreachable_target_stops_at_closest_without_faults():
    """Target outside joint limits: the arm should approach, hit the margined
    limit, stall and stop — never violating limits or rates."""
    p = _params()
    A = np.zeros((6, 7))
    A[:6, :6] = np.eye(6)
    x_goal = A @ Q_HOME
    x_goal[0] += 10.0  # joint 0 would need to go far past its limit

    fci = StrictFakeFci(Q_HOME)
    stream = _streamer(fci, p)
    q_plan = stream.measured_q()
    stream.hold(q_plan, p.hold_first_steps)
    best, stalls = float("inf"), 0
    for _ in range(200):
        err = x_goal - A @ stream.measured_q()
        n = float(np.linalg.norm(err))
        if n < best - p.min_improve:
            best, stalls = n, 0
        else:
            stalls += 1
            if stalls >= p.stall_segments:
                break
        dq, T, _, _ = plan_dls_hop(q_plan, A, err, p)
        if np.max(np.abs(dq)) < 1e-6:
            break
        q_plan = _execute_hop(stream, q_plan, dq, T, p.settle_steps)
    stream.finish(q_plan, p.finish_max_steps)
    assert fci.finished
    assert abs(fci.q_d[0] - p.upper[0]) < 1e-3  # got as close as the limit allows


# --- continuous (non-stopping) tracker ----------------------------------------

def _run_continuous(stream, q_plan, A, x_goal, p, max_cycles, tol=1e-3, smooth=True):
    """Same loop shape as ``move_pose_dls`` in pro_robot_interface.py, but on
    the linear task model ``x = A q`` instead of O_T_EE/Jacobian, so it can
    run against the strict fake robot like the hop-based tests above.

    q_plan/v_cmd are tracked from ``send()``'s own return value — the exact
    q_cmd it just validated and wrote — never re-derived from a subsequent
    ``readOnce()``. (JointPositionStreamer's internal rate-limiting reference
    now works the same way; this mirrors that at the caller level too.)

    ``smooth=True`` (the production behavior) slew-limits the commanded joint
    velocity each cycle instead of commanding ``continuous_dls_step``'s raw
    output directly; ``smooth=False`` reproduces the pre-fix behavior, for
    tests that need to show the difference it makes."""
    best_err, best_cycle = float("inf"), 0
    stall_cycles = max(int(round(p.stall_window_sec / DT)), 1)
    v_cmd = np.zeros(7)
    for cycle in range(max_cycles):
        err = x_goal - A @ stream.measured_q()
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            return q_plan, cycle, "converged"
        if cycle - best_cycle >= stall_cycles:
            if best_err - err_norm < p.min_improve:
                return q_plan, cycle, "stalled"
            best_err, best_cycle = err_norm, cycle
        dq_step, _, _ = continuous_dls_step(q_plan, A, err, p, dJ_dq=ZERO_DJ)
        if float(np.max(np.abs(dq_step))) < 1e-9:
            return q_plan, cycle, "no_step"
        if smooth:
            v_cmd = slew_limit_velocity(v_cmd, dq_step / DT, p.plan_acc, DT)
            sent = stream.send(q_plan + v_cmd * DT)
            v_cmd = (sent - q_plan) / DT
            q_plan = sent
        else:
            q_plan = q_plan + dq_step
            stream.send(q_plan)
    return q_plan, max_cycles, "timeout"


def test_continuous_step_matches_pure_task_term_away_from_singularities():
    """Well away from any singularity the posture pull is exactly zero, and on a
    linear model (∂J/∂q = 0) so are the manipulability gradient and the barrier,
    so the step is just the task-velocity-clamped DLS correction."""
    p = _params()
    J = np.zeros((6, 7))
    J[:, :6] = np.eye(6)  # sigma_min 1.0, far above sigma_avoid
    err = np.array([0.2, -0.1, 0.05, 0.3, -0.2, 0.1])
    dq_step, sigma, lam = continuous_dls_step(Q_HOME, J, err, p, dJ_dq=ZERO_DJ)
    assert sigma > p.sigma_avoid and lam == p.lambda_min

    v = task_velocity(err, p)
    expected, _ = dls_step_with_joint_limits(J, v * DT, lam, Q_HOME, p.lower, p.upper)
    np.testing.assert_allclose(dq_step, expected, atol=1e-12)


def test_continuous_step_nullspace_bias_is_task_neutral_near_singularity():
    """At zero task error but low manipulability, the whole step comes from
    the nullspace posture bias, which must not disturb the (already-zero)
    task error to first order."""
    p = _params()
    J = np.zeros((6, 7))
    J[:5, :5] = np.eye(5)
    J[5, 5] = 1e-5  # near-singular: sigma_min << sigma_floor
    q_start = Q_HOME.copy()
    dq_step, sigma, _ = continuous_dls_step(q_start, J, np.zeros(6), p, dJ_dq=ZERO_DJ)
    assert sigma < p.sigma_floor
    assert np.linalg.norm(dq_step) > 0.0
    assert np.linalg.norm(J @ dq_step) < 1e-6


def test_continuous_step_never_crosses_joint_limits():
    p = _params()
    q = Q_HOME.copy()
    q[0] = p.upper[0] - 1e-4  # joint 0 right at its (margined) upper limit
    J = np.zeros((6, 7))
    J[0, 0] = 1.0
    J[1, 1] = J[2, 2] = J[3, 3] = J[4, 4] = J[5, 5] = 1.0
    err = np.array([1.0, 0, 0, 0, 0, 0])  # pushes joint 0 further past its limit
    for _ in range(50):
        dq_step, _, _ = continuous_dls_step(q, J, err, p, dJ_dq=ZERO_DJ)
        q = q + dq_step
        assert q[0] <= p.upper[0] + 1e-9


def test_continuous_tracker_converges_without_faults():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 7)) * 0.5
    p = _params()
    q_goal = Q_HOME + np.array([0.3, -0.2, 0.25, 0.2, -0.3, 0.2, -0.4])
    x_goal = A @ q_goal

    fci = StrictFakeFci(Q_HOME)
    stream = _streamer(fci, p)
    q_plan = stream.measured_q()
    stream.hold(q_plan, p.hold_first_steps)
    q_plan, cycles, reason = _run_continuous(stream, q_plan, A, x_goal, p, max_cycles=30_000)
    stream.finish(q_plan, p.finish_max_steps)

    assert reason == "converged"
    assert fci.finished
    assert fci.peak["acc"] <= 0.9 + 1e-9 and fci.peak["jerk"] <= 0.9 + 1e-9


def test_continuous_tracker_is_smooth_no_long_holds():
    """The whole point: unlike the hop-based executor (which deliberately
    commands zero velocity for ``settle_steps`` consecutive cycles after every
    hop, via ``stream.hold``), the continuous tracker must never sit still for
    anything like that long while it's still correcting toward the target.
    Uses a task error/Jacobian at a realistic pose-error scale (a few cm /
    tens of degrees, like a cube-subset hand-eye offset) so the default
    kp/task-velocity gains are exercised the way they are in production."""
    p = _params()
    J = np.eye(6, 7)  # constant well-conditioned task Jacobian
    q_goal = Q_HOME + np.array([0.05, -0.04, 0.03, 0.06, -0.05, 0.04, 0.0])
    x_goal = J @ q_goal

    fci = StrictFakeFci(Q_HOME)
    stream = _streamer(fci, p)
    q_plan = stream.measured_q()
    stream.hold(q_plan, p.hold_first_steps)

    vel_eps = 1e-4  # rad/s: far below any meaningful joint motion
    max_stall_run = 0
    stall_run = 0
    prev_q = stream.measured_q()
    converged = False
    v_cmd = np.zeros(7)
    for _ in range(20_000):
        err = x_goal - J @ stream.measured_q()
        if float(np.linalg.norm(err)) < 1e-3:
            converged = True
            break
        dq_step, _, _ = continuous_dls_step(q_plan, J, err, p, dJ_dq=ZERO_DJ)
        v_cmd = slew_limit_velocity(v_cmd, dq_step / DT, p.plan_acc, DT)
        sent = stream.send(q_plan + v_cmd * DT)
        v_cmd = (sent - q_plan) / DT
        q_plan = sent
        q_now = stream.measured_q()
        v = float(np.max(np.abs(q_now - prev_q))) / DT
        prev_q = q_now
        if v < vel_eps:
            stall_run += 1
            max_stall_run = max(max_stall_run, stall_run)
        else:
            stall_run = 0

    assert converged
    # settle_steps (150 by default) is the hop-based executor's dead time after
    # every hop; the continuous tracker should never come close to that.
    assert max_stall_run < 20


# --- velocity slew limiting (the fix for the real-hardware discontinuity fault) ---

def test_slew_limit_velocity_caps_the_change():
    acc = np.full(7, 5.0)
    v_prev = np.zeros(7)
    v_target = np.full(7, 100.0)  # far beyond what one cycle can reach
    v_next = slew_limit_velocity(v_prev, v_target, acc, DT)
    np.testing.assert_allclose(v_next, acc * DT)


def test_slew_limit_velocity_passes_through_small_changes():
    acc = np.full(7, 5.0)
    v_prev = np.full(7, 0.1)
    v_target = v_prev + acc * DT * 0.5  # well within one cycle's budget
    v_next = slew_limit_velocity(v_prev, v_target, acc, DT)
    np.testing.assert_allclose(v_next, v_target, atol=1e-12)


def test_slew_limit_velocity_converges_monotonically_to_a_fixed_target():
    p = _params()
    v = np.zeros(7)
    target = p.plan_vel.copy()  # the largest a raw continuous_dls_step output can ask for
    prev_err = None
    for _ in range(5000):
        v = slew_limit_velocity(v, target, p.plan_acc, DT)
        err = float(np.max(np.abs(target - v)))
        if prev_err is not None:
            assert err <= prev_err + 1e-12  # never overshoots, error shrinks monotonically
        prev_err = err
    np.testing.assert_allclose(v, target, atol=1e-9)


@pytest.mark.parametrize("smooth", [True, False])
def test_continuous_tracker_survives_state_report_latency(smooth):
    """Regression test for the real-hardware fault: with the original
    JointPositionStreamer (which re-derived its rate-limiting reference from
    a subsequent readOnce() every cycle), LaggyFakeFci's one-cycle report lag
    was enough to make the loop command a real discontinuity, independent of
    whether the caller's own steps were smoothed. Now that the streamer
    tracks its own last-commanded q/dq/ddq directly from what it just wrote
    (never re-derived from a read), this scenario completes cleanly either
    way — see JointPositionStreamer's docstring for why that's the layer
    that actually has to be lag-immune."""
    p = _params()
    J = np.eye(6, 7)
    q_goal = Q_HOME + np.array([0.05, -0.04, 0.03, 0.06, -0.05, 0.04, 0.0])
    x_goal = J @ q_goal

    fci = LaggyFakeFci(Q_HOME)
    stream = _streamer(fci, p)
    q_plan = stream.measured_q()
    stream.hold(q_plan, p.hold_first_steps)
    # _run_continuous itself raising is the failure this test targets — a real
    # joint_motion_generator_*_discontinuity fault from LaggyFakeFci's strict
    # validation. Reaching *any* clean stop reason (converged, or stalled from
    # delayed-feedback oscillation when unsmoothed) demonstrates that.
    q_plan, _, reason = _run_continuous(
        stream, q_plan, J, x_goal, p, max_cycles=20_000, smooth=smooth)
    assert reason in ("converged", "stalled")

    if smooth:
        # Only the smoothed (production) path is expected to actually reach
        # the target and decelerate to zero cleanly; finish()'s "hold until
        # quiet" is a separate concern from the discontinuity-fault question
        # this test targets, and the unsmoothed comparison path was never
        # meant to finish cleanly (or even necessarily converge, under
        # delayed feedback with no damping on the raw target).
        assert reason == "converged"
        stream.finish(q_plan, p.finish_max_steps)
        assert fci.finished


def test_raw_continuous_step_output_can_jump_between_cycles_unlike_smoothed():
    """continuous_dls_step re-solves from scratch every call, so nothing
    guarantees consecutive raw outputs are acceleration-bounded the way a
    single quintic hop is by construction: two calls reacting to genuinely
    different live errors (as consecutive 1 kHz cycles would whenever, say,
    a task-velocity clamp component saturates or desaturates) can imply an
    acceleration far beyond one cycle's budget. This is the property
    slew_limit_velocity exists to fix — independent of the state-report
    latency issue above, and why it's worth keeping even though that issue
    is now fixed at the streamer layer too (defense in depth: the streamer
    fix keeps the *validated* command safe; this keeps the *reference*
    smooth by construction, matching the design's own stated intent for
    the rate limiter to be a safety net rather than the primary smoother)."""
    p = _params()
    J = np.eye(6, 7)
    dq_a, _, _ = continuous_dls_step(Q_HOME, J, np.array([0.2, 0, 0, 0, 0, 0]), p, dJ_dq=ZERO_DJ)
    dq_b, _, _ = continuous_dls_step(Q_HOME, J, np.array([-0.2, 0, 0, 0, 0, 0]), p, dJ_dq=ZERO_DJ)
    v_a, v_b = dq_a / DT, dq_b / DT  # dq_step is a per-cycle position delta; v = dq_step/dt
    raw_implied_acc = float(np.max(np.abs(v_b - v_a))) / DT
    assert raw_implied_acc > float(np.max(p.plan_acc))

    v_smoothed = slew_limit_velocity(v_a, v_b, p.plan_acc, DT)
    smoothed_implied_acc = float(np.max(np.abs(v_smoothed - v_a))) / DT
    assert smoothed_implied_acc <= float(np.max(p.plan_acc)) + 1e-9


# --- singularity avoidance on real FR3 kinematics ------------------------------
#
# Linear task models (x = A q) have a constant Jacobian, so they can't exercise
# anything singularity-related. These tests use FR3 forward kinematics (Franka's
# published modified-DH parameters, flange + the configured F_T_EE: Rz(-45°),
# z +0.1034 m) and drive the production loop, run_continuous_dls_move, against
# the strict fake robot.

_DH_A = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0]
_DH_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0, 0.107]
_DH_ALPHA = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0.0]
_F_T_EE = np.array([
    [math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0.0, 0.0],
    [math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.1034],
    [0.0, 0.0, 0.0, 1.0],
])


def _dh(a, d, alpha, theta):
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fr3_fk(q):
    """(O_T_EE 4x4, base-frame geometric Jacobian 6x7), like libfranka's O_T_EE /
    zero_jacobian."""
    T = np.eye(4)
    frames = []
    for i in range(7):
        T = T @ _dh(_DH_A[i], _DH_D[i], _DH_ALPHA[i], q[i])
        frames.append(T)
    T = T @ _dh(_DH_A[7], _DH_D[7], _DH_ALPHA[7], 0.0) @ _F_T_EE
    J = np.zeros((6, 7))
    for i, F in enumerate(frames):
        z = F[:3, 2]
        J[:3, i] = np.cross(z, T[:3, 3] - F[:3, 3])
        J[3:, i] = z
    return T, J


class KinematicFakeFci(StrictFakeFci):
    """StrictFakeFci that also reports O_T_EE and records σ_min of every state it
    reports, i.e. every configuration the arm actually passes through."""

    def __init__(self, q0):
        super().__init__(q0)
        self.sigmas = []

    def readOnce(self):
        s, _ = super().readOnce()
        T, J = fr3_fk(s.q)
        s.O_T_EE = T.T.reshape(16)  # column-major, like libfranka
        self.sigmas.append(min_singular_value(J))
        return s, None


def _fr3_read(state):
    R, t = otee_flat_to_Rt(state.O_T_EE)
    return R, t, fr3_fk(state.q)[1]


def _fr3_move(q_start, T_target, **overrides):
    p = _params(**overrides)
    fci = KinematicFakeFci(q_start)
    stream = _streamer(fci, p)
    result = run_continuous_dls_move(stream, _fr3_read, T_target[:3, :3], T_target[:3, 3], p)
    assert fci.finished
    return result, fci, p


def test_jacobian_derivative_matches_finite_differences_on_fr3():
    rng = np.random.default_rng(0)
    p = _params()
    eps = 1e-6
    for _ in range(5):
        q = rng.uniform(p.lower, p.upper)
        H = jacobian_derivative(fr3_fk(q)[1])
        H_fd = np.stack([
            (fr3_fk(q + eps * e)[1] - fr3_fk(q - eps * e)[1]) / (2 * eps) for e in np.eye(7)])
        np.testing.assert_allclose(H, H_fd, atol=1e-7)


def test_fast_contraction_matches_full_jacobian_derivative():
    rng = np.random.default_rng(4)
    p = _params()
    for _ in range(5):
        J = fr3_fk(rng.uniform(p.lower, p.upper))[1]
        W = rng.normal(size=(6, 7))
        np.testing.assert_allclose(
            contract_jacobian_derivative(J, W),
            np.einsum("ri,kri->k", W, jacobian_derivative(J)), atol=1e-12)


def test_singularity_gradients_match_finite_differences_on_fr3():
    p = _params()
    q = np.random.default_rng(1).uniform(p.lower, p.upper)
    J = fr3_fk(q)[1]
    dJ = jacobian_derivative(J)

    def logw(qq):
        Jq = fr3_fk(qq)[1]
        return 0.5 * np.linalg.slogdet(Jq @ Jq.T + 1e-6 * np.eye(6))[1]

    eps = 1e-6
    fd_logw = np.array([(logw(q + eps * e) - logw(q - eps * e)) / (2 * eps) for e in np.eye(7)])
    fd_sigma = np.array([
        (min_singular_value(fr3_fk(q + eps * e)[1]) - min_singular_value(fr3_fk(q - eps * e)[1]))
        / (2 * eps) for e in np.eye(7)])
    for dJ_arg in (dJ, None):  # explicit tensor, and the fast analytic default
        np.testing.assert_allclose(log_manipulability_gradient(J, dJ_arg), fd_logw, atol=1e-5)
        sigma, grad = min_singular_value_gradient(J, dJ_arg)
        assert sigma == pytest.approx(min_singular_value(J))
        np.testing.assert_allclose(grad, fd_sigma, atol=1e-5)


def test_min_singular_value_flags_what_yoshikawa_misses():
    """Why the measure changed: bending the elbow toward straight cuts σ_min to a
    third while Yoshikawa still reads 'healthy' (above the old 0.03 threshold)."""
    q = FR3_READY_Q.copy()
    q[3] = -1.0
    J = fr3_fk(q)[1]
    assert manipulability(J) > 0.03
    assert min_singular_value(J) < 0.35 * min_singular_value(fr3_fk(FR3_READY_Q)[1])


def test_barrier_filter_passes_safe_steps_and_minimally_corrects_unsafe_ones():
    rng = np.random.default_rng(3)
    floor, gain = 0.05, 2.0
    for _ in range(200):
        g = rng.normal(size=7)
        dq = rng.normal(size=7) * 1e-3
        sigma = rng.uniform(0.0, 0.2)
        bound = -gain * (sigma - floor) * DT
        out = singularity_barrier_filter(dq, g, sigma, floor, gain, DT)
        assert g @ out >= bound - 1e-12
        if g @ dq >= bound:
            np.testing.assert_array_equal(out, dq)
        else:
            delta = out - dq  # minimal correction: purely along g
            np.testing.assert_allclose(delta - (delta @ g) / (g @ g) * g, 0.0, atol=1e-12)


def test_barrier_filter_leaves_pinned_joints_alone():
    g = np.array([1.0, 1.0, 0, 0, 0, 0, 0])
    dq = np.array([-1e-3, 0, 0, 0, 0, 0, 0])
    free = np.array([False, True, True, True, True, True, True])
    out = singularity_barrier_filter(dq, g, 0.06, 0.05, 2.0, DT, free=free)
    assert out[0] == dq[0] and out[1] > 0.0
    assert g @ out >= -2.0 * 0.01 * DT - 1e-12


def test_barrier_filter_without_gradient_never_makes_sigma_worse():
    dq = np.full(7, 1e-3)
    out = singularity_barrier_filter(dq, np.zeros(7), 0.01, 0.05, 2.0, DT)
    np.testing.assert_array_equal(out, dq)  # Δσ = 0 ≥ min(bound, 0)


def test_task_velocity_clips_by_norm_preserving_direction():
    p = _params()
    err = np.array([1.0, 0.1, 0.0, 0.0, 0.5, 0.5])
    v = task_velocity(err, p)
    assert np.linalg.norm(v[:3]) == pytest.approx(p.task_vel_pos_m_s)
    assert np.linalg.norm(v[3:]) == pytest.approx(p.task_vel_rot_rad_s)
    np.testing.assert_allclose(np.cross(v[:3], err[:3]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(v[3:], err[3:]), 0.0, atol=1e-12)


def test_joint_limit_locking_damps_on_the_reduced_jacobian():
    """Locking a joint can leave the rest near-singular though the full arm isn't;
    damping must follow the reduced Jacobian."""
    p = _params()
    J = np.zeros((6, 7))
    J[:5, :5] = np.eye(5)
    J[5, 5] = 1e-3  # without joint 6, row 5 is nearly unreachable...
    J[5, 6] = 1.0   # ...and joint 6 is the one serving it
    q = Q_HOME.copy()
    q[6] = p.upper[6]  # pinned at its limit; the error pushes it further
    err = np.zeros(6)
    err[5] = 0.01
    _, sigma, lam = continuous_dls_step(q, J, err, p, dJ_dq=ZERO_DJ)
    assert sigma > p.sigma_avoid
    assert lam > 0.95 * math.hypot(p.lambda_min, p.lambda_max)


def _rotvec_matrix(rvec):
    angle = float(np.linalg.norm(rvec))
    if angle == 0.0:
        return np.eye(3)
    k = np.asarray(rvec) / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


@pytest.mark.parametrize("offset", [
    (0.05, 0.05, 0.05, 0.0, 0.0, 0.0),
    (-0.05, 0.05, -0.05, 0.26, 0.0, 0.0),
    (0.05, -0.05, 0.05, 0.0, -0.26, 0.0),
    (0.0, 0.0, -0.05, 0.0, 0.0, 0.26),
])
def test_fr3_cube_offsets_converge_without_faults(offset):
    """Normal hand-eye targets (5 cm cube corners, 15° EE-frame tilts) from the
    ready pose still converge; the barrier and gradient terms don't get in the way."""
    T0 = fr3_fk(FR3_READY_Q)[0]
    T = T0.copy()
    T[:3, 3] += offset[:3]
    T[:3, :3] = T0[:3, :3] @ _rotvec_matrix(offset[3:])
    result, fci, p = _fr3_move(FR3_READY_Q, T)
    assert result.stop_reason == "converged"
    assert min(fci.sigmas) > p.sigma_floor
    assert fci.peak["acc"] <= 0.9 + 1e-9 and fci.peak["jerk"] <= 0.9 + 1e-9


def _stretched_target():
    """Pose of the ready configuration with the elbow almost straight (joint4 = -0.3):
    σ_min ≈ 0.024, below the default floor, reachable only through near-singular
    configurations."""
    q = FR3_READY_Q.copy()
    q[3] = -0.3
    T, J = fr3_fk(q)
    assert min_singular_value(J) < 0.03
    return T


def test_fr3_barrier_never_lets_sigma_cross_the_floor():
    result, fci, p = _fr3_move(FR3_READY_Q, _stretched_target(), dls_timeout_sec=15.0)
    assert min(fci.sigmas) >= p.sigma_floor - 1e-3
    assert result.stop_reason.startswith("stalled")  # closest safe pose, not a fault
    assert fci.peak["acc"] <= 0.9 + 1e-9 and fci.peak["jerk"] <= 0.9 + 1e-9


def test_fr3_stretched_target_really_is_singular_without_the_barrier():
    """Control for the test above: with the barrier and hard stop disabled the same
    move goes well below the floor, so it's the barrier keeping σ_min up."""
    _, fci, _ = _fr3_move(
        FR3_READY_Q, _stretched_target(), dls_timeout_sec=15.0,
        dls_sigma_floor=-1.0, dls_sigma_stop=0.0)
    assert min(fci.sigmas) < _params().sigma_floor - 0.01


def test_hard_stop_below_sigma_stop_finishes_cleanly():
    """Starting already below dls_sigma_stop: no motion toward anything, clean finish."""
    q = FR3_READY_Q.copy()
    q[3] = -0.3
    T0 = fr3_fk(q)[0]
    T = T0.copy()
    T[:3, 3] += (0.05, 0.0, 0.0)
    result, fci, _ = _fr3_move(q, T)
    assert result.stop_reason.startswith("near singularity")
    np.testing.assert_allclose(fci.q_d, q, atol=1e-9)


def test_renamed_manipulability_keys_are_rejected():
    with pytest.raises(ValueError, match="dls_sigma_avoid"):
        _params(dls_manip_threshold=0.03)


def test_rotation_error_is_finite_and_correct_near_pi():
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    for angle in (math.pi - 1e-3, math.pi - 1e-6, math.pi):
        R = _rotvec_matrix(axis * angle)
        e = rotation_error_axis_angle(np.eye(3), R)
        assert np.all(np.isfinite(e))
        assert np.linalg.norm(e) == pytest.approx(angle, abs=1e-6)
        assert abs(abs(e @ axis) / np.linalg.norm(e) - 1.0) < 1e-6


class DeadlineFakeFci(KinematicFakeFci):
    """Like the real FR3: a command only counts if it answers the state just read
    without slow work in between. Planning (reading the pose/Jacobian) between
    readOnce and writeOnce makes the command late, so it's dropped: q_d stays put,
    exactly what the hardware logs showed."""

    def __init__(self, q0):
        super().__init__(q0)
        self.planned_since_read = False
        self.dropped = 0

    def readOnce(self):
        self.planned_since_read = False
        return super().readOnce()

    def writeOnce(self, cmd):
        if self.planned_since_read and not getattr(cmd, "motion_finished", False):
            self.dropped += 1
            return
        super().writeOnce(cmd)


def test_fr3_move_never_plans_between_read_and_write():
    """Regression for the hardware fault: every DLS command was dropped because the
    loop planned (FK, Jacobian, SVDs) between readOnce and writeOnce."""
    T0 = fr3_fk(FR3_READY_Q)[0]
    T = T0.copy()
    T[:3, 3] += (0.02, 0.0, 0.0)
    p = _params()
    fci = DeadlineFakeFci(FR3_READY_Q)
    stream = _streamer(fci, p)

    def read(state):
        fci.planned_since_read = True
        return _fr3_read(state)

    result = run_continuous_dls_move(stream, read, T[:3, :3], T[:3, 3], p)
    assert fci.dropped == 0
    assert result.stop_reason == "converged"
    assert fci.finished


class StictionFakeFci(KinematicFakeFci):
    """Measured joints stick until the command pulls more than ``deadband`` away,
    like joint friction under impedance control: the arm sits a little behind the
    command. A planner that closes the loop through the measured pose hunts here."""

    def __init__(self, q0, deadband=3e-4):
        super().__init__(q0)
        self.q_meas = np.asarray(q0, dtype=float).copy()
        self.deadband = deadband
        self.q_d_history = []

    def writeOnce(self, cmd):
        super().writeOnce(cmd)
        self.q_d_history.append(self.q_d.copy())
        gap = self.q_d - self.q_meas
        slip = np.abs(gap) > self.deadband
        self.q_meas[slip] = self.q_d[slip] - np.sign(gap[slip]) * self.deadband

    def readOnce(self):
        s, _ = super().readOnce()
        T, _ = fr3_fk(self.q_meas)
        s.q = self.q_meas.copy()
        s.O_T_EE = T.T.reshape(16)
        return s, None


def test_fr3_move_does_not_hunt_against_joint_friction():
    """Regression for the buzzing on the real FR3: steering the measured pose through
    a friction deadband made the command flip acceleration nearly every cycle near
    the target and never converge. Steering the commanded pose converges cleanly."""
    T0 = fr3_fk(FR3_READY_Q)[0]
    T = T0.copy()
    T[:3, 3] += (0.02, 0.0, 0.0)
    p = _params()
    fci = StictionFakeFci(FR3_READY_Q)
    stream = _streamer(fci, p)
    result = run_continuous_dls_move(stream, _fr3_read, T[:3, :3], T[:3, 3], p)
    assert result.stop_reason == "converged"
    assert result.pos_err_m < 0.006  # measured: the fake deadband alone costs a few mm
    a = np.diff(np.asarray(fci.q_d_history), n=2, axis=0) / DT ** 2
    tail = a[len(a) // 2:]
    big = np.abs(tail) > 0.1
    flips = sum(int(np.sum(np.diff(np.sign(tail[big[:, k], k])) != 0)) for k in range(7))
    assert flips < 20, f"{flips} acceleration sign flips in the second half: hunting"


def test_fr3_move_is_jerk_smooth():
    """Regression for the buzzing on the real FR3: the command stream must not step
    its acceleration. Before the velocity smoother, moves started with a full
    plan_acc step (jerk at the rate limiter's cap) and near-target target changes
    became ±plan_acc bang-bang; now jerk stays a small fraction of the limit."""
    T0 = fr3_fk(FR3_READY_Q)[0]
    T = T0.copy()
    T[:3, 3] += (0.02, -0.02, 0.02)
    T[:3, :3] = T0[:3, :3] @ _rotvec_matrix((0.26, 0.0, 0.0))
    result, fci, _ = _fr3_move(FR3_READY_Q, T)
    assert result.stop_reason == "converged"
    assert fci.peak["jerk"] < 0.05, fci.peak   # < 250 rad/s^3 (limit 5000; was 3500 before)
    assert fci.peak["acc"] <= 0.5 + 1e-9, fci.peak  # at most the planning accel cap


def test_velocity_smoother_turns_a_one_cycle_spike_into_a_small_bump():
    from easy_handeye2_franka_auto.dls_motion import VelocitySmoother
    sm = VelocitySmoother(0.02)
    out = [sm.step(np.full(7, 0.01)) for _ in range(300)]
    out.append(sm.step(np.full(7, 0.01 + 0.005)))  # one-cycle spike, like the hardware kicks
    out += [sm.step(np.full(7, 0.01)) for _ in range(100)]
    v = np.asarray(out)[:, 0]
    a = np.diff(v) / DT
    assert np.abs(a).max() < 0.5          # vs 2.5 rad/s^2 bang-bang before
    assert abs(v[0]) < 0.01 * 0.01        # starts from rest, no velocity step
