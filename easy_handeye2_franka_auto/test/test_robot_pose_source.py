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
