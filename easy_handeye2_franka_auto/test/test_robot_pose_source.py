import threading
import time
import numpy as np
from types import SimpleNamespace
from easy_handeye2_franka_auto.robot_pose_source import RobotPoseSource


class FakeRobot:
    def __init__(self, pos=(0.1, 0.2, 0.3), quat=(1.0, 0.0, 0.0, 0.0), delay=0.0):
        self.pos, self.quat, self.delay = np.array(pos), np.array(quat), delay
        self.calls = []
        self._active = 0
        self.max_concurrent = 0
        self._g = threading.Lock()

    def _enter(self, name):
        with self._g:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.calls.append(name)
        time.sleep(self.delay)
        with self._g:
            self._active -= 1

    def refresh_state_snapshot(self):
        self._enter("refresh")

    def get_state_snapshot(self):
        return SimpleNamespace(ee_pos=self.pos, ee_quat=self.quat)

    def reset_to_start_pose(self, T):
        self._enter("reset")
        self.pos = np.asarray(T)[:3, 3].copy()

    def move_to_pose_dls(self, T):
        self._enter("dls")
        self.pos = np.asarray(T)[:3, 3].copy()
        return 0.0

    def error_recovery(self):
        self._enter("recover")


def test_latest_none_before_refresh():
    assert RobotPoseSource(FakeRobot()).latest() is None


def test_refresh_populates_cache():
    src = RobotPoseSource(FakeRobot())
    src.refresh()
    pos, quat = src.latest()
    np.testing.assert_allclose(pos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(quat, [1, 0, 0, 0])


def test_move_invalidates_cache_until_refresh():
    robot = FakeRobot()
    src = RobotPoseSource(robot)
    src.refresh()
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.4]
    src.move_to(T)
    assert src.latest() is None
    src.refresh()
    np.testing.assert_allclose(src.latest()[0], [0.5, 0.0, 0.4])


def test_failed_refresh_invalidates_and_raises():
    robot = FakeRobot()
    src = RobotPoseSource(robot)
    src.refresh()

    def boom():
        raise RuntimeError("boom")

    robot.refresh_state_snapshot = boom
    try:
        src.refresh()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert src.latest() is None


def test_robot_calls_never_overlap_across_threads():
    robot = FakeRobot(delay=0.02)
    src = RobotPoseSource(robot)
    T = np.eye(4)
    threads = [threading.Thread(target=src.refresh) for _ in range(4)]
    threads += [threading.Thread(target=src.move_to, args=(T,)) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert robot.max_concurrent == 1


def test_latest_does_not_block_behind_robot_call():
    robot = FakeRobot(delay=0.3)
    src = RobotPoseSource(robot)
    src.refresh()
    t = threading.Thread(target=src.refresh)
    t.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    src.latest()
    assert time.monotonic() - t0 < 0.1
    t.join()


def test_polling_refreshes_and_stops():
    robot = FakeRobot()
    src = RobotPoseSource(robot)
    src.start_polling(50.0)
    time.sleep(0.15)
    src.stop_polling()
    n = robot.calls.count("refresh")
    assert n >= 2
    time.sleep(0.1)
    assert robot.calls.count("refresh") == n


def test_go_to_moves_then_refreshes_and_leaves_cache_populated():
    robot = FakeRobot()
    src = RobotPoseSource(robot)
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.4]
    src.go_to(T, settle_sec=0.0, dwell_sec=0.0)
    assert robot.calls == ["reset", "refresh"]
    np.testing.assert_allclose(src.latest()[0], [0.5, 0.0, 0.4])


def test_move_recovers_then_reraises_on_reset_fault():
    robot = FakeRobot()
    src = RobotPoseSource(robot)

    def boom(_T):
        robot._enter("reset")
        raise RuntimeError("reflex")

    robot.reset_to_start_pose = boom
    T = np.eye(4)
    try:
        src.move_to(T)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert "recover" in robot.calls


def test_move_to_dls_invalidates_cache_and_returns_residual():
    robot = FakeRobot()
    src = RobotPoseSource(robot)
    src.refresh()
    T = np.eye(4)
    T[:3, 3] = [0.6, 0.1, 0.2]
    residual = src.move_to_dls(T)
    assert residual == 0.0
    assert src.latest() is None  # cache cleared; caller must refresh
    assert robot.calls == ["refresh", "dls"]


def test_go_to_dls_mode_dispatches_and_returns_residual():
    robot = FakeRobot()
    src = RobotPoseSource(robot)
    T = np.eye(4)
    T[:3, 3] = [0.6, 0.1, 0.2]
    residual = src.go_to(T, settle_sec=0.0, dwell_sec=0.0, motion="dls")
    assert residual == 0.0
    assert robot.calls == ["dls", "refresh"]
    np.testing.assert_allclose(src.latest()[0], [0.6, 0.1, 0.2])


def test_go_to_rejects_unknown_motion_mode():
    src = RobotPoseSource(FakeRobot())
    try:
        src.go_to(np.eye(4), 0.0, 0.0, motion="warp")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dls_recovers_then_reraises_on_fault():
    robot = FakeRobot()
    src = RobotPoseSource(robot)

    def boom(_T):
        robot._enter("dls")
        raise RuntimeError("dls failed")

    robot.move_to_pose_dls = boom
    try:
        src.move_to_dls(np.eye(4))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert "recover" in robot.calls
