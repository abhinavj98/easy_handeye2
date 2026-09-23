"""Damped least-squares Cartesian→joint tracking for the FR3 joint position interface.

Design (mirrors libfranka examples/generate_joint_position_motion_external_control_loop.cpp,
which uses the same readOnce/writeOnce API):

* The robot validates every 1 kHz joint command against its own last desired state
  (q_d, dq_d, ddq_d). Implied acceleration above kMaxJointAcceleration reports
  ``joint_motion_generator_velocity_discontinuity``; implied jerk above kMaxJointJerk
  reports ``joint_motion_generator_acceleration_discontinuity``. The external-loop API
  does *not* rate-limit for you, so every command goes through ``limit_rate_joint_positions``
  (a port of ``franka::limitRate``) against the robot-reported q_d/dq_d/ddq_d.
* Each DLS hop is a quintic (minimum-jerk) blend, zero velocity and acceleration at both
  ends, with duration sized so its peak velocity/acceleration/jerk stay well inside the
  limits. The rate limiter is then only a safety net.
* Hops are planned with singularity-robust damping, a per-hop task-space error clamp (keeps
  the linearization valid), and joint-limit-aware solving, so the arm gets as close to the
  commanded pose as it safely can and stops instead of faulting.
* When manipulability drops too low to keep making progress, ``nearest_singular_direction_joint_report``
  identifies which joints are implicated (for logging), and ``plan_nullspace_escape_hop`` takes
  self-motion steps toward a comfortable reference posture through the current Jacobian's
  nullspace — reconfiguring the arm without disturbing the commanded end-effector pose — before
  giving up.

``continuous_dls_step`` is the smooth alternative that actually drives ``move_pose_dls``:
instead of planning a whole hop from a static Jacobian and then stopping to re-measure, it
recomputes the DLS step from the live Jacobian/error every 1 kHz cycle and streams
continuously. The task error is turned into a velocity command (proportional gain, clamped to
a max task speed) rather than a position-clamped full step, so there's no hop boundary to stop
at; the nullspace posture bias (the same self-motion idea as ``plan_nullspace_escape_hop``) is
blended in continuously, strengthening smoothly as manipulability drops, instead of firing as a
separate discrete escape stage after a stall. The quintic-hop functions above remain as
lower-level building blocks (``dls_step_with_joint_limits``, ``singularity_robust_lambda``, ...)
that ``continuous_dls_step`` itself reuses, and as an alternate/reference planner.

A single quintic hop is smooth *by construction* — the rate limiter (``limit_rate_joint_positions``)
only ever has to catch a rare excursion. ``continuous_dls_step`` re-solves from scratch every
cycle, so its raw output has no such guarantee (a task-velocity component saturating or
desaturating, damping changing as manipulability shifts, joint-limit locking flipping a column
on or off — any of these can change the solution meaningfully cycle to cycle). Commanding that
raw sequence directly puts the *entire* smoothing burden on the rate limiter, continuously, for
the whole move, instead of the occasional safety net it's meant to be — on real hardware
(latency and tracking dynamics the tests' zero-latency fake robot doesn't have) that showed up
as an actual ``joint_motion_generator_*_discontinuity`` fault. ``slew_limit_velocity`` fixes
this: callers keep a running commanded joint-velocity state across cycles and slew it toward
each cycle's raw target at a bounded acceleration, so the *reference itself* is
acceleration-smooth before it ever reaches the rate limiter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

DT = 1e-3  # FCI control period

# Peaks of the quintic s(τ)=10τ³−15τ⁴+6τ⁵ for a unit step over unit time.
QUINTIC_PEAK_VEL = 1.875
QUINTIC_PEAK_ACC = 10.0 / math.sqrt(3.0)
QUINTIC_PEAK_JERK = 60.0


def otee_flat_to_Rt(flat) -> tuple[np.ndarray, np.ndarray]:
    """Column-major O_T_EE (16,) → rotation R (3,3) and translation t (3,)."""
    flat = np.asarray(flat, dtype=float).reshape(16)
    R = np.array([
        [flat[0], flat[4], flat[8]],
        [flat[1], flat[5], flat[9]],
        [flat[2], flat[6], flat[10]],
    ], dtype=float)
    t = flat[12:15].copy()
    return R, t


def rotation_error_axis_angle(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Base-frame rotation error as axis-angle (matches space Jacobian)."""
    R_err = R_des @ R_cur.T
    cos_angle = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return np.zeros(3, dtype=float)
    if angle > math.pi - 1e-4:
        # sin(angle) → 0: the skew part no longer carries the axis. Recover it from
        # the symmetric part, R ≈ 2 a aᵀ − I, choosing the sign that matches the
        # (tiny) skew part so the result stays continuous as angle → π.
        B = 0.5 * (R_err + np.eye(3))
        i = int(np.argmax(np.diag(B)))
        axis = B[:, i] / math.sqrt(max(B[i, i], 1e-12))
        skew = np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
        if float(axis @ skew) < 0.0:
            axis = -axis
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ], dtype=float) / (2.0 * np.sin(angle))
    return axis * angle


def pose_error_6d(
    R_cur: np.ndarray,
    t_cur: np.ndarray,
    R_des: np.ndarray,
    t_des: np.ndarray,
) -> np.ndarray:
    e_pos = np.asarray(t_des, dtype=float).reshape(3) - np.asarray(t_cur, dtype=float).reshape(3)
    e_rot = rotation_error_axis_angle(R_cur, R_des)
    return np.concatenate([e_pos, e_rot])


def dls_joint_step(J_6x7: np.ndarray, err_6: np.ndarray, damping: float) -> np.ndarray:
    """dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e."""
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    e = np.asarray(err_6, dtype=float).reshape(6)
    lam2 = float(damping) ** 2
    A = J @ J.T + lam2 * np.eye(6)
    return J.T @ np.linalg.solve(A, e)


def reshape_jacobian_colmajor(jac_flat) -> np.ndarray:
    """libfranka zero_jacobian: 42 column-major → (6, 7)."""
    return np.asarray(jac_flat, dtype=float).reshape(7, 6).T


def manipulability(J_6x7: np.ndarray) -> float:
    """Yoshikawa manipulability index sqrt(det(J Jᵀ)); 0 exactly at a singularity."""
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    det = float(np.linalg.det(J @ J.T))
    return math.sqrt(det) if det > 0.0 else 0.0


def min_singular_value(J_6x7: np.ndarray) -> float:
    """Smallest of the 6 singular values of J: the distance-to-singularity measure
    used for damping, the nullspace ramp, the barrier and the hard stop.

    Unlike ``manipulability`` (the product of all six, mixing m/rad and rad/rad
    rows), this doesn't let large well-conditioned directions mask one that is
    collapsing. On the FR3 it's ~0.22 at the ready pose and ~0.07 with the elbow
    at joint4 = -1.0, where the Yoshikawa index still looks healthy (0.047).
    """
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def jacobian_derivative(J_6x7: np.ndarray) -> np.ndarray:
    """∂J/∂q_k for a *geometric* base-frame Jacobian of a serial revolute arm,
    computed from J alone (no kinematic model). Returns shape (7, 6, 7), indexed
    ``[k, row, column]``.

    Column i is [z_i × (p − p_i); z_i]. Rotating joint k ≤ i turns everything
    distal of k rigidly, so the column rotates: ∂J_i/∂q_k = [z_k × Jv_i; z_k × z_i].
    Joint k > i only moves the end point p by Jv_k: ∂J_i/∂q_k = [z_i × Jv_k; 0].
    Matches libfranka's ``zero_jacobian`` (base frame, EE point).
    """
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    Jv, Jw = J[:3], J[3:]
    # Batched cross products as skew-matrix products (np.cross is ~20x slower at
    # this size, and this runs every 1 kHz cycle). [k, :, i] layout throughout.
    Sw, Sv = _skew_columns(Jw), _skew_columns(Jv)
    wk_vi = Sw @ Jv                # z_k × Jv_i
    wk_wi = Sw @ Jw                # z_k × z_i
    wi_vk = -(Sv @ Jw)             # z_i × Jv_k = −(Jv_k × z_i)
    dv = np.where(_K_LE_I, wk_vi, wi_vk)
    dw = np.where(_K_LE_I, wk_wi, 0.0)
    return np.concatenate([dv, dw], axis=1)


_K_LE_I = (np.arange(7)[:, None] <= np.arange(7)[None, :])[:, None, :]  # [k, 1, i]


def _skew_columns(V_3x7: np.ndarray) -> np.ndarray:
    """(7, 3, 3) stack of skew matrices [v_k]× for each column v_k."""
    x, y, z = V_3x7
    o = np.zeros(7)
    return np.stack([
        np.stack([o, -z, y], axis=1),
        np.stack([z, o, -x], axis=1),
        np.stack([-y, x, o], axis=1),
    ], axis=1)


def _cross_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise cross product of two (3, n) arrays."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def contract_jacobian_derivative(J_6x7: np.ndarray, W_6x7: np.ndarray) -> np.ndarray:
    """``g_k = Σ_{r,i} W[r,i] · ∂J[r,i]/∂q_k`` for a geometric Jacobian, in O(7),
    without building the (7, 6, 7) tensor. Same result as
    ``np.einsum('ri,kri->k', W, jacobian_derivative(J))``.

    Using the column-derivative rules in ``jacobian_derivative`` and the scalar
    triple product: k ≤ i gives z_k·(Jv_i × wv_i + z_i × ww_i); k > i gives
    Jv_k·(wv_i × z_i). Those are a reverse and an exclusive cumulative sum over i.
    """
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    W = np.asarray(W_6x7, dtype=float).reshape(6, 7)
    Jv, Jw, Wv, Ww = J[:3], J[3:], W[:3], W[3:]
    a = _cross_cols(Jv, Wv) + _cross_cols(Jw, Ww)
    b = _cross_cols(Wv, Jw)
    A = np.cumsum(a[:, ::-1], axis=1)[:, ::-1]                 # Σ_{i ≥ k}
    B = np.cumsum(b, axis=1) - b                               # Σ_{i < k}
    return np.sum(Jw * A, axis=0) + np.sum(Jv * B, axis=0)


def _contract(J: np.ndarray, W: np.ndarray, dJ_dq: np.ndarray | None) -> np.ndarray:
    if dJ_dq is None:
        return contract_jacobian_derivative(J, W)
    return np.einsum("ri,kri->k", W, np.asarray(dJ_dq, dtype=float).reshape(7, 6, 7))


def log_manipulability_gradient(
    J_6x7: np.ndarray, dJ_dq: np.ndarray | None = None, eps: float = 1e-6
) -> np.ndarray:
    """∇_q ½·log det(J Jᵀ + εI).

    Ascending this pushes every singular value up with weight 1/σ_i, so the
    smallest dominates as it shrinks. It's smooth (unlike ∇σ_min, which jumps when
    two singular values cross) and invariant to scaling the task rows, so mixing
    m and rad doesn't bias its direction. ε keeps it finite at a singularity.

    ``dJ_dq`` defaults to the analytic geometric-Jacobian derivative (the real
    arm); pass an explicit ∂J/∂q (e.g. zeros) for anything else.
    """
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    M = np.linalg.solve(J @ J.T + eps * np.eye(6), J)
    return _contract(J, M, dJ_dq)


def min_singular_value_gradient(
    J_6x7: np.ndarray, dJ_dq: np.ndarray | None = None
) -> Tuple[float, np.ndarray]:
    """``(σ_min, ∇_q σ_min)`` with ∂σ/∂q_k = uᵀ (∂J/∂q_k) v for the smallest
    singular pair (u, v). ``dJ_dq`` as in ``log_manipulability_gradient``."""
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    U, S, Vt = np.linalg.svd(J, full_matrices=True)
    return float(S[5]), _contract(J, np.outer(U[:, 5], Vt[5]), dJ_dq)


def singularity_ramp(sigma: float, sigma_floor: float, sigma_avoid: float) -> float:
    """0 when ``sigma >= sigma_avoid``, rising along the Chiaverini sqrt profile to
    1 at ``sigma <= sigma_floor``."""
    span = sigma_avoid - sigma_floor
    if span <= 0.0:
        return 0.0 if sigma >= sigma_avoid else 1.0
    r = min(max((sigma - sigma_floor) / span, 0.0), 1.0)
    return math.sqrt(1.0 - r * r)


def singularity_barrier_filter(
    dq: np.ndarray,
    grad_sigma: np.ndarray,
    sigma: float,
    sigma_floor: float,
    gain: float,
    dt: float = DT,
    free: np.ndarray | None = None,
) -> np.ndarray:
    """Discrete control-barrier filter on σ_min.

    Enforces, to first order, ``Δσ ≈ ∇σ·dq ≥ −gain·(σ − σ_floor)·dt``: σ may
    decay toward ``sigma_floor`` at most exponentially and never cross it. When
    already below the floor the bound turns positive and the step is pushed back
    out. A violating step is minimally corrected along ``∇σ`` (restricted to the
    ``free`` joints, i.e. not pinned at a joint limit), so the rest of the motion
    goes on and the arm slides along the barrier. The cost is task tracking error.
    Scaling the result by any α ∈ [0, 1] keeps it feasible while the bound is ≤ 0.
    """
    dq = np.asarray(dq, dtype=float).reshape(7)
    g = np.asarray(grad_sigma, dtype=float).reshape(7)
    bound = -gain * (sigma - sigma_floor) * dt
    rate = float(g @ dq)
    if rate >= bound:
        return dq
    g_free = g if free is None else np.where(free, g, 0.0)
    gg = float(g @ g_free)
    if gg > 1e-12:
        return dq + ((bound - rate) / gg) * g_free
    # No usable direction to raise σ: at least never make it worse.
    if rate >= min(bound, 0.0):
        return dq
    if bound < 0.0:
        return dq * (bound / rate)
    return np.zeros(7)


def clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale ``x`` down (direction preserved) so ``‖x‖ <= max_norm``."""
    x = np.asarray(x, dtype=float)
    n = float(np.linalg.norm(x))
    return x * (max_norm / n) if n > max_norm > 0.0 else x


def scale_to_joint_limits(
    dq: np.ndarray, q_start: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Uniformly shrink ``dq`` so no joint moving *toward* a limit crosses it.
    (Joints already past a limit may still move back inside.)"""
    q_new = q_start + dq
    over = (dq > 0.0) & (q_new > upper)
    under = (dq < 0.0) & (q_new < lower)
    if not (over.any() or under.any()):
        return dq
    room = np.where(over, upper - q_start, lower - q_start)
    ratios = np.maximum(room[over | under] / dq[over | under], 0.0)
    return dq * min(1.0, float(ratios.min()))


def singularity_robust_lambda(
    sigma: float,
    lambda_min: float,
    sigma_threshold: float,
    lambda_max: float,
) -> float:
    """Damping (Chiaverini SR-inverse schedule) that rises as σ_min drops.

    ``lambda_min`` when ``sigma >= sigma_threshold``; blends in up to ``lambda_max``
    (in quadrature) as ``sigma`` falls to 0, so steps toward a singularity shrink
    instead of blowing up.
    """
    if sigma_threshold <= 0.0:
        return lambda_min
    ratio = min(max(sigma, 0.0) / sigma_threshold, 1.0)
    extra = lambda_max * math.sqrt(max(1.0 - ratio * ratio, 0.0))
    return math.sqrt(lambda_min ** 2 + extra ** 2)


# Standard Franka "ready" joint configuration (0, -45, 0, -135, 0, 90, 45 deg):
# a comfortable, high-manipulability posture used as the nullspace-escape target
# when a hop stalls near a singularity.
FR3_READY_Q = np.array([
    0.0, -0.7853981633974483, 0.0, -2.356194490192345,
    0.0, 1.5707963267948966, 0.7853981633974483,
])


def nearest_singular_direction_joint_report(J_6x7: np.ndarray, top_k: int = 3) -> list[Tuple[int, float]]:
    """Which joints are driving the current singularity.

    SVD of J; the right-singular vector paired with the *smallest* of the 6
    principal singular values is the joint-velocity direction that produces the
    least task-space motion per unit joint motion — i.e. the direction that's
    becoming ineffective. Its largest-magnitude components are the joints
    responsible (e.g. joints 4 and 6 both large ~ wrist singularity; joint 4
    alone ~ elbow singularity). Returns up to ``top_k`` ``(joint_index, weight)``
    pairs, 0-indexed, weights normalized to sum to 1 over all 7 joints, most-
    implicated first.

    The always-present 7th singular direction (the arm's basic kinematic
    redundancy for a 6D task) is excluded on purpose — it isn't a singularity,
    it's exactly what ``plan_nullspace_escape_hop`` uses to get out of one.
    """
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    _, S, Vt = np.linalg.svd(J, full_matrices=True)
    idx = int(np.argmin(S))
    v = np.abs(Vt[idx])
    total = float(v.sum())
    if total > 0.0:
        v = v / total
    order = np.argsort(v)[::-1][:max(int(top_k), 0)]
    return [(int(i), float(v[i])) for i in order]


def format_singularity_report(J_6x7: np.ndarray, top_k: int = 3) -> str:
    """Human-readable joint report, e.g. ``'joint4=0.58 joint6=0.31 joint2=0.06'``
    (1-indexed to match the robot's own joint numbering)."""
    return " ".join(
        f"joint{i + 1}={w:.2f}"
        for i, w in nearest_singular_direction_joint_report(J_6x7, top_k))


def clamp_task_error(err_6: np.ndarray, max_pos_m: float, max_rot_rad: float) -> np.ndarray:
    """Cap the translation/rotation parts of a 6D error so one hop stays in the
    region where the Jacobian linearization is accurate."""
    e = np.asarray(err_6, dtype=float).reshape(6).copy()
    pn = float(np.linalg.norm(e[:3]))
    if pn > max_pos_m > 0.0:
        e[:3] *= max_pos_m / pn
    rn = float(np.linalg.norm(e[3:]))
    if rn > max_rot_rad > 0.0:
        e[3:] *= max_rot_rad / rn
    return e


def dls_step_with_joint_limits(
    J_6x7: np.ndarray,
    err_6: np.ndarray,
    damping: float | Callable[[np.ndarray], float],
    q_start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """DLS step that never drives a joint past ``[lower, upper]``.

    Joints whose step would cross a limit are locked (their Jacobian column zeroed,
    which makes their DLS component exactly 0) and the remaining joints re-solve for
    the error, so the arm still makes progress with the joints it has left. Any
    residual crossing is removed by uniformly scaling the step.

    ``damping`` may be a callable of the (column-reduced) Jacobian. Locking a joint
    can leave the remaining joints near-singular even when the full arm isn't, so
    the damping has to be re-evaluated on the reduced Jacobian every time.

    Returns ``(dq, locked_mask)``.
    """
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7).copy()
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    lower = np.asarray(lower, dtype=float).reshape(7)
    upper = np.asarray(upper, dtype=float).reshape(7)
    damping_for = damping if callable(damping) else (lambda _J, _d=float(damping): _d)
    locked = np.zeros(7, dtype=bool)
    dq = np.zeros(7)
    for _ in range(7):
        dq = dls_joint_step(J, err_6, damping_for(J))
        q_new = q_start + dq
        crossing = ~locked & (((q_new > upper) & (dq > 0.0)) | ((q_new < lower) & (dq < 0.0)))
        if not crossing.any():
            break
        locked |= crossing
        J[:, crossing] = 0.0
    dq[locked] = 0.0
    return scale_to_joint_limits(dq, q_start, lower, upper), locked


def quintic(tau: float) -> float:
    """Minimum-jerk blend 10τ³−15τ⁴+6τ⁵: zero velocity and acceleration at τ=0 and τ=1."""
    t = min(max(float(tau), 0.0), 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def quintic_duration(
    dq: np.ndarray,
    vel_caps: np.ndarray,
    acc_caps: np.ndarray,
    jerk_caps: np.ndarray,
    min_sec: float,
    max_sec: float,
) -> float:
    """Shortest quintic duration keeping every joint's peak velocity, acceleration
    and jerk within the given caps, clipped to ``[min_sec, max_sec]``."""
    a = np.abs(np.asarray(dq, dtype=float).reshape(7))
    t_vel = QUINTIC_PEAK_VEL * a / np.maximum(vel_caps, 1e-9)
    t_acc = np.sqrt(QUINTIC_PEAK_ACC * a / np.maximum(acc_caps, 1e-9))
    t_jerk = np.cbrt(QUINTIC_PEAK_JERK * a / np.maximum(jerk_caps, 1e-9))
    T = float(max(t_vel.max(), t_acc.max(), t_jerk.max()))
    return float(np.clip(T, min_sec, max_sec))


def clamp_step_for_duration(
    dq: np.ndarray,
    vel_caps: np.ndarray,
    acc_caps: np.ndarray,
    jerk_caps: np.ndarray,
    T: float,
) -> np.ndarray:
    """Uniformly shrink ``dq`` (direction preserved) so a quintic of duration ``T``
    respects all three caps. No-op when ``T`` came from ``quintic_duration`` and
    wasn't clipped at ``max_sec``."""
    dq = np.asarray(dq, dtype=float).reshape(7)
    max_step = np.minimum.reduce([
        vel_caps * T / QUINTIC_PEAK_VEL,
        acc_caps * T ** 2 / QUINTIC_PEAK_ACC,
        jerk_caps * T ** 3 / QUINTIC_PEAK_JERK,
    ])
    worst = float(np.max(np.abs(dq) / np.maximum(max_step, 1e-12)))
    return dq if worst <= 1.0 else dq / worst


def limit_rate_joint_positions(
    target_q: np.ndarray,
    last_q: np.ndarray,
    last_dq: np.ndarray,
    last_ddq: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    jerk_limits: np.ndarray,
    dt: float = DT,
) -> np.ndarray:
    """Vectorized port of libfranka ``franka::limitRate`` for joint positions
    (src/rate_limiting.cpp), symmetric velocity limits."""
    target_q = np.asarray(target_q, dtype=float).reshape(7)
    if not np.all(np.isfinite(target_q)):
        raise ValueError("commanded_position is infinite or NaN.")
    last_q = np.asarray(last_q, dtype=float).reshape(7)
    last_dq = np.asarray(last_dq, dtype=float).reshape(7)
    last_ddq = np.asarray(last_ddq, dtype=float).reshape(7)

    cmd_vel = (target_q - last_q) / dt
    cmd_jerk = ((cmd_vel - last_dq) / dt - last_ddq) / dt
    cmd_acc = last_ddq + np.clip(cmd_jerk, -jerk_limits, jerk_limits) * dt
    safe_max_acc = np.minimum((jerk_limits / acc_limits) * (vel_limits - last_dq), acc_limits)
    safe_min_acc = np.maximum((jerk_limits / acc_limits) * (-vel_limits - last_dq), -acc_limits)
    acc = np.maximum(np.minimum(cmd_acc, safe_max_acc), safe_min_acc)
    return last_q + (last_dq + acc * dt) * dt


class JointPositionStreamer:
    """Owns one active joint position control session: every command is rate
    limited against *our own* last-commanded q/dq/ddq (the first cycle uses
    the measured q with zero velocity/acceleration, exactly like libfranka's
    example) — tracked directly from what ``send()`` itself just computed and
    wrote, never re-derived from a subsequent ``readOnce()``.

    This matters: ``readOnce()``'s reported ``q_d/dq_d/ddq_d`` reflects
    whatever timing/buffering the underlying control loop actually has, which
    isn't guaranteed to be perfectly synchronous with the ``writeOnce()`` that
    immediately preceded it. Rate-limiting against a reference that could be
    even one cycle stale relative to what ``writeOnce()`` actually validates
    against is exactly the kind of gap that produces a real
    ``joint_motion_generator_*_discontinuity`` fault — the client and the
    robot's own validator would be reasoning from different "last state"
    values. Tracking our own last-sent q_cmd (which is, by construction,
    exactly what the immediately following ``writeOnce()`` validates against)
    removes that gap entirely, matching libfranka's own external-control-loop
    examples, which keep this state client-side rather than re-reading it."""

    def __init__(
        self,
        ctrl,
        make_command: Callable,
        vel_limits: np.ndarray,
        acc_limits: np.ndarray,
        jerk_limits: np.ndarray,
    ):
        self._ctrl = ctrl
        self._make = make_command
        self._vel = np.asarray(vel_limits, dtype=float)
        self._acc = np.asarray(acc_limits, dtype=float)
        self._jerk = np.asarray(jerk_limits, dtype=float)
        self.state, _ = ctrl.readOnce()
        self._last_q = self.measured_q()
        self._last_dq = np.zeros(7)
        self._last_ddq = np.zeros(7)

    def measured_q(self) -> np.ndarray:
        return np.asarray(self.state.q, dtype=float).copy()

    def send(self, target_q: np.ndarray, finished: bool = False) -> np.ndarray:
        """``write`` then (unless finished) ``read``: for callers whose per-cycle
        work is trivial. Anything slow must go between ``write`` and ``read``."""
        q_cmd = self.write(target_q, finished)
        if not finished:
            self.read()
        return q_cmd

    def write(self, target_q: np.ndarray, finished: bool = False) -> np.ndarray:
        """Rate-limit and write one command. Call this right after ``read``: the
        robot only accepts a command that arrives within the same 1 kHz tick as the
        state it answers. A late one is dropped (control_command_success_rate falls,
        q_d stays put), and the next accepted command then looks like a jump:
        joint_motion_generator_*_discontinuity. On the FR3 a few hundred µs of
        planning between read and write was enough to drop every command."""
        q_cmd = limit_rate_joint_positions(
            target_q, self._last_q, self._last_dq, self._last_ddq,
            self._vel, self._acc, self._jerk)
        cmd = self._make(q_cmd.tolist())
        if finished:
            cmd.motion_finished = True
        self._ctrl.writeOnce(cmd)
        new_dq = (q_cmd - self._last_q) / DT
        new_ddq = (new_dq - self._last_dq) / DT
        self._last_q, self._last_dq, self._last_ddq = q_cmd, new_dq, new_ddq
        return q_cmd

    def read(self):
        """Block until the next robot state (the next 1 kHz tick)."""
        self.state, _ = self._ctrl.readOnce()
        return self.state

    @property
    def last_command(self) -> np.ndarray:
        """The q_cmd most recently written (what the robot should now hold as q_d)."""
        return self._last_q.copy()

    def hold(self, target_q: np.ndarray, steps: int) -> None:
        for _ in range(max(int(steps), 0)):
            self.send(target_q)

    def _quiet(self, target_q: np.ndarray) -> bool:
        s = self.state
        q_d = np.asarray(getattr(s, "q_d", s.q), dtype=float)
        dq_d = np.asarray(getattr(s, "dq_d", np.zeros(7)), dtype=float)
        ddq_d = np.asarray(getattr(s, "ddq_d", np.zeros(7)), dtype=float)
        dq = np.asarray(getattr(s, "dq", np.zeros(7)), dtype=float)
        return (
            float(np.max(np.abs(np.asarray(target_q) - q_d))) < 1e-6
            and float(np.max(np.abs(dq_d))) < 1e-3
            and float(np.max(np.abs(ddq_d))) < 5e-2
            and float(np.max(np.abs(dq))) < 2e-2
        )

    def finish(self, target_q: np.ndarray, max_steps: int) -> None:
        """Hold ``target_q`` until the commanded and measured motion has died out
        (avoids "Motion finished commanded, but the robot is still moving!"),
        then send motion_finished."""
        for _ in range(max(int(max_steps), 0)):
            if self._quiet(target_q):
                break
            self.send(target_q)
        self.send(target_q, finished=True)


@dataclass
class DlsParams:
    plan_vel: np.ndarray
    plan_acc: np.ndarray
    plan_jerk: np.ndarray
    rate_vel: np.ndarray
    rate_acc: np.ndarray
    rate_jerk: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lambda_min: float
    lambda_max: float
    sigma_avoid: float
    sigma_floor: float
    sigma_stop: float
    barrier_gain: float
    max_task_pos_m: float
    max_task_rot_rad: float
    min_sec: float
    max_sec: float
    max_segments: int
    pos_tol: float
    rot_tol: float
    min_improve: float
    stall_segments: int
    hold_first_steps: int
    settle_steps: int
    finish_max_steps: int
    q_preferred: np.ndarray
    escape_step_rad: float
    escape_max_hops: int
    kp_pos: float
    kp_rot: float
    task_vel_pos_m_s: float
    task_vel_rot_rad_s: float
    null_gain_max: float
    null_grad_gain_min: float
    null_grad_gain_max: float
    null_vel_max: float
    stall_window_sec: float
    timeout_sec: float
    log_interval_sec: float
    vel_smoothing_sec: float

    @classmethod
    def from_config(cls, cfg: dict, vel_limits, acc_limits, jerk_limits, pos_limits) -> "DlsParams":
        vel = np.asarray(vel_limits, dtype=float)
        acc = np.asarray(acc_limits, dtype=float)
        jerk = np.asarray(jerk_limits, dtype=float)
        pos = np.asarray(pos_limits, dtype=float)
        renamed = [k for k in ("dls_manip_threshold", "dls_manip_stop") if k in cfg]
        if renamed:
            raise ValueError(
                f"{renamed} no longer exist: singularity distance is now the smallest "
                "Jacobian singular value (dls_sigma_avoid / dls_sigma_floor / dls_sigma_stop), "
                "on a different scale than the Yoshikawa index. See config/robot.yaml.")
        plan_frac = float(cfg.get("dls_plan_limit_frac", 0.5))
        rate_frac = float(cfg.get("dls_rate_limit_frac", 0.9))
        margin = float(cfg.get("dls_joint_limit_margin_rad", 0.1))
        q_preferred = np.asarray(cfg.get("dls_escape_q_preferred", FR3_READY_Q), dtype=float)
        return cls(
            plan_vel=vel * plan_frac,
            plan_acc=acc * plan_frac,
            plan_jerk=jerk * plan_frac,
            rate_vel=vel * rate_frac,
            rate_acc=acc * rate_frac,
            rate_jerk=jerk * rate_frac,
            lower=pos[:, 0] + margin,
            upper=pos[:, 1] - margin,
            lambda_min=float(cfg.get("dls_lambda_min", cfg.get("dls_lambda", 0.05))),
            lambda_max=float(cfg.get("dls_lambda_max", 0.15)),
            sigma_avoid=float(cfg.get("dls_sigma_avoid", 0.10)),
            sigma_floor=float(cfg.get("dls_sigma_floor", 0.05)),
            sigma_stop=float(cfg.get("dls_sigma_stop", 0.03)),
            barrier_gain=float(cfg.get("dls_barrier_gain", 2.0)),
            max_task_pos_m=float(cfg.get("dls_max_task_step_m", 0.05)),
            max_task_rot_rad=float(cfg.get("dls_max_task_step_rad", 0.25)),
            min_sec=float(cfg.get("dls_min_segment_sec", 0.2)),
            max_sec=float(cfg.get("dls_segment_duration_sec", 5.0)),
            max_segments=int(cfg.get("dls_max_segments", 20)),
            pos_tol=float(cfg.get("dls_pos_tol_m", 0.003)),
            rot_tol=float(cfg.get("dls_rot_tol_rad", 0.02)),
            min_improve=float(cfg.get("dls_min_improve", 1e-4)),
            stall_segments=int(cfg.get("dls_stall_segments", 3)),
            hold_first_steps=int(cfg.get("dls_hold_steps_first", 50)),
            settle_steps=int(cfg.get("dls_settle_steps", 150)),
            finish_max_steps=int(cfg.get("dls_finish_max_steps", 1000)),
            q_preferred=q_preferred,
            escape_step_rad=float(cfg.get("dls_escape_step_rad", 0.15)),
            escape_max_hops=int(cfg.get("dls_escape_max_hops", 4)),
            kp_pos=float(cfg.get("dls_kp_pos", 4.0)),
            kp_rot=float(cfg.get("dls_kp_rot", 4.0)),
            task_vel_pos_m_s=float(cfg.get("dls_task_vel_pos_m_s", 0.08)),
            task_vel_rot_rad_s=float(cfg.get("dls_task_vel_rot_rad_s", 0.4)),
            null_gain_max=float(cfg.get("dls_null_gain_max", 0.5)),
            null_grad_gain_min=float(cfg.get("dls_null_grad_gain_min", 0.02)),
            null_grad_gain_max=float(cfg.get("dls_null_grad_gain_max", 0.2)),
            null_vel_max=float(cfg.get("dls_null_vel_max", 0.2)),
            stall_window_sec=float(cfg.get("dls_stall_window_sec", 1.5)),
            timeout_sec=float(cfg.get("dls_timeout_sec", 20.0)),
            log_interval_sec=float(cfg.get("dls_log_interval_sec", 0.5)),
            vel_smoothing_sec=float(cfg.get("dls_vel_smoothing_sec", 0.05)),
        )


def plan_dls_hop(
    q_start: np.ndarray,
    J_6x7: np.ndarray,
    err_6: np.ndarray,
    p: DlsParams,
) -> Tuple[np.ndarray, float, float, float]:
    """Plan one hop from the commanded joint setpoint ``q_start``.

    Returns ``(dq, duration_sec, sigma_min, damping)``. ``dq`` respects joint
    limits (with margin) and, executed as a quintic over ``duration_sec``, stays
    within the planning velocity/acceleration/jerk caps.
    """
    sigma = min_singular_value(J_6x7)
    lam = singularity_robust_lambda(sigma, p.lambda_min, p.sigma_avoid, p.lambda_max)
    e = clamp_task_error(err_6, p.max_task_pos_m, p.max_task_rot_rad)
    dq, _ = dls_step_with_joint_limits(J_6x7, e, lam, q_start, p.lower, p.upper)
    T = quintic_duration(dq, p.plan_vel, p.plan_acc, p.plan_jerk, p.min_sec, p.max_sec)
    dq = clamp_step_for_duration(dq, p.plan_vel, p.plan_acc, p.plan_jerk, T)
    return dq, T, sigma, lam


def plan_nullspace_escape_hop(
    q_start: np.ndarray,
    J_6x7: np.ndarray,
    p: DlsParams,
) -> Tuple[np.ndarray, float]:
    """Self-motion step toward ``p.q_preferred`` (a comfortable, high-manipulability
    posture) through the nullspace of the *current* Jacobian:
    ``dq = (I - J+ J) @ (q_preferred - q_start)``, scaled to at most
    ``p.escape_step_rad`` and clipped so it never crosses the (margined) joint
    limits.

    This reconfigures the arm (typically the elbow) without commanding any
    end-effector motion — the nullspace projector guarantees that to first
    order, using whatever redundancy the *current*, possibly near-singular,
    Jacobian still has. It doesn't need a kinematic model or a hypothetical
    Jacobian elsewhere in joint space, so it's safe to compute from a single
    live reading.

    Returns ``(dq, duration_sec)``; ``dq`` is exactly zero when the nullspace
    has nothing useful to offer here (already at ``q_preferred``, no usable
    redundancy left, or the only nullspace motion available is blocked by a
    joint limit).
    """
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    J_pinv = np.linalg.pinv(J)
    N = np.eye(7) - J_pinv @ J
    dq_null = N @ (p.q_preferred - q_start)
    norm = float(np.linalg.norm(dq_null))
    if norm < 1e-9:
        return np.zeros(7), 0.0
    dq = dq_null * (min(p.escape_step_rad, norm) / norm)
    dq = scale_to_joint_limits(dq, q_start, p.lower, p.upper)
    if float(np.max(np.abs(dq))) < 1e-9:
        return np.zeros(7), 0.0

    T = quintic_duration(dq, p.plan_vel, p.plan_acc, p.plan_jerk, p.min_sec, p.max_sec)
    dq = clamp_step_for_duration(dq, p.plan_vel, p.plan_acc, p.plan_jerk, T)
    return dq, T


def task_velocity(err_6: np.ndarray, p: DlsParams) -> np.ndarray:
    """Proportional task-space velocity, with translation and rotation each
    clipped by *norm* so the commanded direction is the error direction
    (per-axis clipping would bend it)."""
    e = np.asarray(err_6, dtype=float).reshape(6)
    return np.concatenate([
        clip_norm(p.kp_pos * e[:3], p.task_vel_pos_m_s),
        clip_norm(p.kp_rot * e[3:], p.task_vel_rot_rad_s),
    ])


def continuous_dls_step(
    q_plan: np.ndarray,
    J_6x7: np.ndarray,
    err_6: np.ndarray,
    p: DlsParams,
    dt: float = DT,
    dJ_dq: np.ndarray | None = None,
) -> Tuple[np.ndarray, float, float]:
    """One 1 kHz control-cycle increment of a continuous, never-stopping DLS tracker.

    ``J_6x7`` is the base-frame geometric Jacobian; the singularity gradients use
    its analytic derivative (``contract_jacobian_derivative``). Pass an explicit
    ``dJ_dq`` (zeros for a linear test model) when J isn't a geometric Jacobian.
    Layers, in order:

    1. **Task term.** The live error becomes a task *velocity* (proportional,
       norm-clipped), solved with singularity-robust damped least squares with
       joint-limit locking. Damping follows σ_min of the Jacobian that's left
       after locking.
    2. **Nullspace term** (task-neutral to first order, via the 1-D nullspace of
       the 6×7 Jacobian). *Always on*, it ascends log-manipulability so the arm
       keeps drifting away from singular configurations even while
       well-conditioned. As σ_min falls below ``sigma_avoid``, the gradient gain
       and a posture pull toward ``q_preferred`` ramp up. Its speed is capped at
       ``null_vel_max``.
    3. **Singularity barrier.** ``singularity_barrier_filter`` on σ_min: the step
       may lower σ_min at most exponentially toward ``sigma_floor``, and never
       below it. It gives up task tracking, never distance to the singularity.
    4. Uniform scaling (direction preserved, barrier preserved) to joint limits
       and to the per-cycle planning velocity cap.

    Returns ``(dq_step, sigma_min, damping)``. ``q_plan + dq_step`` is the next
    commanded joint position.
    """
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    q_plan = np.asarray(q_plan, dtype=float).reshape(7)
    U, S, Vt = np.linalg.svd(J, full_matrices=True)
    sigma = float(S[5])

    zero_cols = ~J.any(axis=0)

    def damping_for(J_reduced: np.ndarray) -> float:
        # Reuse σ_min of J unless joint-limit locking actually removed a column.
        s = sigma if np.array_equal(~J_reduced.any(axis=0), zero_cols) else min_singular_value(J_reduced)
        return singularity_robust_lambda(s, p.lambda_min, p.sigma_avoid, p.lambda_max)

    v = task_velocity(err_6, p)
    dq_task, locked = dls_step_with_joint_limits(J, v * dt, damping_for, q_plan, p.lower, p.upper)
    J_reduced = J.copy()
    J_reduced[:, locked] = 0.0
    lam = damping_for(J_reduced)

    ramp = singularity_ramp(sigma, p.sigma_floor, p.sigma_avoid)
    grad_logw = log_manipulability_gradient(J, dJ_dq)  # O(7) analytic when dJ_dq is None
    k_grad = p.null_grad_gain_min + (p.null_grad_gain_max - p.null_grad_gain_min) * ramp
    qdot_pref = k_grad * grad_logw + p.null_gain_max * ramp * (p.q_preferred - q_plan)
    n = Vt[6]  # unit basis of the (1-D) nullspace
    qdot_null = clip_norm(n * float(n @ qdot_pref), p.null_vel_max)
    dq_null = qdot_null * dt
    q_next = q_plan + dq_task + dq_null
    blocked = ((q_next > p.upper) & (dq_null > 0.0)) | ((q_next < p.lower) & (dq_null < 0.0))
    dq_null[blocked] = 0.0
    dq = dq_task + dq_null

    grad_sigma = _contract(J, np.outer(U[:, 5], Vt[5]), dJ_dq)
    at_limit = (q_plan >= p.upper - 1e-6) | (q_plan <= p.lower + 1e-6)
    dq = singularity_barrier_filter(
        dq, grad_sigma, sigma, p.sigma_floor, p.barrier_gain, dt, free=~(locked | at_limit))

    dq = scale_to_joint_limits(dq, q_plan, p.lower, p.upper)
    worst = float(np.max(np.abs(dq) / np.maximum(p.plan_vel * dt, 1e-12)))
    if worst > 1.0:
        dq = dq / worst
    return dq, sigma, lam


def commanded_pose(
    R_meas: np.ndarray, t_meas: np.ndarray, J_6x7: np.ndarray, dq_track: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """EE pose at the commanded joints, from the measured pose/Jacobian and the
    tracking offset ``dq_track = q_cmd − q_meas`` (first order: the offset is
    ~1e-4 rad, so the error is ~1e-8, and encoder noise in q_meas cancels)."""
    J = np.asarray(J_6x7, dtype=float).reshape(6, 7)
    d = J @ np.asarray(dq_track, dtype=float).reshape(7)
    w = d[3:]
    angle = float(np.linalg.norm(w))
    R = np.asarray(R_meas, dtype=float)
    if angle > 1e-12:
        k = w / angle
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        R = (np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * K @ K) @ R
    return R, np.asarray(t_meas, dtype=float).reshape(3) + d[:3]


def limit_blas_threads(n: int = 1):
    """Pin numpy's BLAS/LAPACK to ``n`` threads for the rest of the process.

    Call in any process that runs the 1 kHz loop. libfranka puts the thread that
    creates the Robot at SCHED_FIFO; with multithreaded OpenBLAS, each SVD/solve
    hands work to normal-priority helper threads the RT thread then waits on. On
    this machine that gave planner stalls up to 40 ms (12 over 5 ms per 30 s) vs a
    1.6 ms worst case single-threaded, and on the FR3 a single ~30 ms stall caused
    communication_constraints_violation. The matrices here are 6×7, so threads
    never help. Returns the threadpoolctl limiter (keep a reference) or None.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    return threadpool_limits(limits=n, user_api="blas")


class VelocitySmoother:
    """Two cascaded first-order low-pass stages (critically damped, ``tau`` each) on
    the target joint velocity.

    ``slew_limit_velocity`` alone bounds acceleration but not jerk: any cycle-to-cycle
    change in the planner's target became a full ±plan_acc acceleration step, and on
    the FR3 that rang audibly (at move start, and as a periodic ~68 Hz kick near the
    target). After two stages the acceleration starts from zero and a one-cycle
    spike becomes a small smooth bump. Costs ~2·tau of lag (100 ms by default).
    """

    def __init__(self, tau: float, dt: float = DT):
        self._k = 1.0 if tau <= 0.0 else dt / (tau + dt)
        self._s1 = np.zeros(7)
        self._s2 = np.zeros(7)

    def step(self, v_target: np.ndarray) -> np.ndarray:
        self._s1 = self._s1 + self._k * (np.asarray(v_target, dtype=float).reshape(7) - self._s1)
        self._s2 = self._s2 + self._k * (self._s1 - self._s2)
        return self._s2.copy()

    def peak(self) -> float:
        return float(max(np.max(np.abs(self._s1)), np.max(np.abs(self._s2))))


def slew_limit_velocity(
    v_prev: np.ndarray,
    v_target: np.ndarray,
    acc_caps: np.ndarray,
    dt: float = DT,
) -> np.ndarray:
    """One acceleration-limited step of a joint-velocity command toward ``v_target``.

    Caps ``|v_next - v_prev|`` at ``acc_caps * dt`` per joint. Callers of
    ``continuous_dls_step`` should keep a running ``v_cmd`` across cycles and
    pass ``v_target = dq_step / dt`` (that cycle's raw DLS output) through
    this before commanding ``q_plan + v_cmd * dt`` — see the module
    docstring for why the raw per-cycle output needs this and the rate
    limiter alone isn't enough on real hardware.
    """
    v_prev = np.asarray(v_prev, dtype=float).reshape(7)
    v_target = np.asarray(v_target, dtype=float).reshape(7)
    acc_caps = np.asarray(acc_caps, dtype=float).reshape(7)
    dv = np.clip(v_target - v_prev, -acc_caps * dt, acc_caps * dt)
    return v_prev + dv


@dataclass
class DlsMoveResult:
    stop_reason: str
    pos_err_m: float
    rot_err_rad: float
    sigma_min: float

    @property
    def residual(self) -> float:
        """Combined 6D error norm (m and rad mixed): kept for callers that log one number."""
        return math.hypot(self.pos_err_m, self.rot_err_rad)


def run_continuous_dls_move(
    stream: JointPositionStreamer,
    read_pose_and_jacobian: Callable[[object], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    R_des: np.ndarray,
    t_des: np.ndarray,
    p: DlsParams,
    log: list | None = None,
    on_send: Callable[[int | None, np.ndarray, np.ndarray], None] | None = None,
) -> DlsMoveResult:
    """Drive one continuous DLS move on an open joint-position session.

    ``read_pose_and_jacobian(state) -> (R, t, J)`` gives the measured EE pose and
    base-frame geometric Jacobian for a robot state (libfranka: ``O_T_EE`` and
    ``model.zero_jacobian``). The loop holds the start pose, tracks with
    ``continuous_dls_step`` under a velocity slew limit, decelerates, and finishes the
    session cleanly. It ends on convergence, stall, timeout, no feasible step, or
    σ_min < ``sigma_stop`` (a backstop: the barrier keeps σ_min above
    ``sigma_floor`` > ``sigma_stop``). Progress lines go to ``log``, which the caller
    keeps even if this raises. ``on_send(cycle, v_prev, v_cmd)`` runs right after every
    command written while tracking (``cycle``) or decelerating (``None``), off the
    critical read→write path. ``stream.state`` is then the tick the command answers, so
    its q_d reflects the *previous* command.
    """
    log = [] if log is None else log
    log_every = max(int(round(p.log_interval_sec / DT)), 1)
    stall_cycles = max(int(round(p.stall_window_sec / DT)), 1)
    max_cycles = max(int(round(p.timeout_sec / DT)), 1)
    # Cycles to brake from the per-cycle velocity cap at the planning accel cap, plus margin.
    decel_cycles_cap = int(round(float(np.max(p.plan_vel / np.maximum(p.plan_acc, 1e-9))) / DT)) + 50
    # ... plus the velocity smoother settling to ~1e-6 (two first-order stages).
    decel_cycles_cap += int(round(40.0 * p.vel_smoothing_sec / DT))

    stop_reason = "timeout"
    e_pos = e_rot = float("inf")
    sigma = float("nan")
    best_err, best_err_cycle = float("inf"), 0
    last_log_cycle = -10**9

    q_plan = stream.measured_q()
    stream.hold(q_plan, p.hold_first_steps)
    v_cmd = np.zeros(7)

    # Each cycle: write (answering the state just read) -> plan -> read. The robot
    # only accepts a command that arrives within the tick of the state it answers,
    # so nothing slow may sit between read and write: the command written in cycle k
    # was planned in cycle k-1, from the state read then. That's 1 ms of extra
    # latency for the planner, harmless at these gains; planning between read and
    # write instead got every DLS command dropped on the real FR3.
    v_target = np.zeros(7)
    need_read = False

    smoother = VelocitySmoother(p.vel_smoothing_sec)

    def write_velocity(v_goal: np.ndarray) -> np.ndarray:
        # Smooth, then slew-limit, the commanded velocity so the reference is jerk- and
        # acceleration-smooth before the rate limiter (see module docstring).
        # q_plan/v_cmd come from write()'s own return value, never a later readOnce().
        nonlocal q_plan, v_cmd, need_read
        v_prev = v_cmd
        v_cmd = slew_limit_velocity(v_cmd, smoother.step(v_goal), p.plan_acc, DT)
        sent = stream.write(q_plan + v_cmd * DT)
        v_cmd = (sent - q_plan) / DT
        q_plan = sent
        need_read = True
        return v_prev

    def read_next() -> None:
        nonlocal need_read
        stream.read()
        need_read = False

    def notify(cycle: int | None, v_prev: np.ndarray) -> None:
        # Right after a write, never right after a read: that gap must stay empty.
        if on_send is not None:
            on_send(cycle, v_prev, v_cmd)

    for cycle in range(max_cycles):
        v_prev = write_velocity(v_target)  # critical path: first thing after the read
        notify(cycle, v_prev)

        R_meas, t_meas, J = read_pose_and_jacobian(stream.state)
        J = np.asarray(J, dtype=float).reshape(6, 7)
        # Steer the *commanded* pose, not the measured one: under joint impedance the
        # arm sits ~1e-4 rad behind the command (gravity, friction), and closing the
        # loop through that lag made the planner hunt at near-zero speed (stick-slip),
        # flipping acceleration every cycle: an audible buzz on the real FR3. The
        # robot's impedance controller already does the tracking.
        R_cur, t_cur = commanded_pose(R_meas, t_meas, J, q_plan - stream.measured_q())
        err = pose_error_6d(R_cur, t_cur, R_des, t_des)
        e_pos = float(np.linalg.norm(err[:3]))
        e_rot = float(np.linalg.norm(err[3:]))
        err_norm = float(np.linalg.norm(err))
        if e_pos <= p.pos_tol and e_rot <= p.rot_tol:
            stop_reason = "converged"
            break

        J = np.asarray(J, dtype=float).reshape(6, 7)
        sigma = min_singular_value(J)
        if sigma < p.sigma_stop:
            stop_reason = f"near singularity (sigma_min {sigma:.4f} < {p.sigma_stop})"
            log.append(f"t={cycle * DT:.2f}s: stopping, dominant_joints={format_singularity_report(J)}")
            break

        # Stall: less than min_improve progress over a whole stall_window_sec (per-cycle
        # progress at 1 kHz is always tiny). Includes the arm resting against the barrier.
        if cycle - best_err_cycle >= stall_cycles:
            if best_err - err_norm < p.min_improve:
                stop_reason = "stalled (closest reachable)"
                break
            best_err, best_err_cycle = err_norm, cycle

        dq_step, sigma, lam = continuous_dls_step(q_plan, J, err, p)
        if float(np.max(np.abs(dq_step))) < 1e-9:
            stop_reason = "no feasible step (joint limits / singularity barrier)"
            break

        if cycle - last_log_cycle >= log_every:
            last_log_cycle = cycle
            report = (f" dominant_joints={format_singularity_report(J)}"
                      if sigma < p.sigma_avoid else "")
            at_margin = np.flatnonzero((q_plan >= p.upper) | (q_plan <= p.lower)) + 1
            if at_margin.size:
                report += f" joints_within_limit_margin={at_margin.tolist()}"
            log.append(
                f"t={cycle * DT:.2f}s: pos_err={e_pos:.4f}m rot_err={e_rot:.4f}rad "
                f"sigma_min={sigma:.4f} lambda={lam:.3f}{report}")

        v_target = dq_step / DT
        read_next()

    if need_read:  # broke out after this cycle's write
        read_next()

    # Ramp velocity to zero on the same slew limit before finish()'s hold.
    for _ in range(decel_cycles_cap):
        if max(float(np.max(np.abs(v_cmd))), smoother.peak()) < 1e-6:
            break
        notify(None, write_velocity(np.zeros(7)))
        read_next()

    stream.finish(q_plan, p.finish_max_steps)
    # Report where the arm actually is (the loop steered the commanded pose).
    R_meas, t_meas, _ = read_pose_and_jacobian(stream.state)
    err = pose_error_6d(R_meas, t_meas, R_des, t_des)
    return DlsMoveResult(stop_reason, float(np.linalg.norm(err[:3])),
                         float(np.linalg.norm(err[3:])), sigma)
