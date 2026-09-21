"""Single owner of FrankaInterface calls, plus a cached EE pose for TF."""
from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).copy()


class RobotPoseSource:
    def __init__(self, robot):
        self._robot = robot
        self._robot_lock = threading.RLock()
        self._cache_lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()

    def _set_cache(self, value):
        with self._cache_lock:
            self._latest = value

    def latest(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self._cache_lock:
            if self._latest is None:
                return None
            pos, quat = self._latest
            return pos.copy(), quat.copy()

    def refresh(self) -> None:
        with self._robot_lock:
            try:
                self._robot.refresh_state_snapshot()
                snap = self._robot.get_state_snapshot()
                pos = _to_numpy(snap.ee_pos).reshape(3)
                quat = _to_numpy(snap.ee_quat).reshape(4)  # wxyz
            except Exception:
                self._set_cache(None)
                raise
            self._set_cache((pos, quat))

    def move_to(self, target_pose_4x4: np.ndarray) -> None:
        with self._robot_lock:
            self._set_cache(None)
            self._robot.reset_to_start_pose(np.asarray(target_pose_4x4, dtype=float))

    def start_polling(self, hz: float) -> None:
        if self._poll_thread is not None or hz <= 0:
            return
        self._poll_stop.clear()

        def _loop():
            while not self._poll_stop.wait(1.0 / hz):
                try:
                    self.refresh()
                except Exception as exc:  # noqa: BLE001
                    _log.warning("pose poll failed: %s", exc)

        self._poll_thread = threading.Thread(target=_loop, daemon=True)
        self._poll_thread.start()

    def stop_polling(self) -> None:
        if self._poll_thread is None:
            return
        self._poll_stop.set()
        self._poll_thread.join()
        self._poll_thread = None
