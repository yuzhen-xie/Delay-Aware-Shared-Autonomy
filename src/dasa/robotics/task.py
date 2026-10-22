from __future__ import annotations

import threading
import time

import numpy as np


class GripperWorker:
    def __init__(self, device, interval=.05):
        self.device, self.interval = device, interval
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._requested = self._applied = None
        self._status, self._error = None, None
        self.events = []
        self._thread = threading.Thread(target=self._run, name="measured-gripper", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def request(self, closed):
        if not isinstance(closed, bool):
            raise ValueError("A gripper request must be boolean")
        with self._lock:
            self._requested = closed

    def _run(self):
        try:
            while not self._stop.is_set():
                with self._lock:
                    requested = self._requested
                if requested is not None and requested != self._applied:
                    result = self.device.command(requested)
                    with self._lock:
                        self.events.append({"host_monotonic": time.monotonic(), "closed": requested,
                                            "command": result})
                    self._applied = requested
                status = {**self.device.status(), "host_monotonic": time.monotonic(),
                          "requested_closed": self._applied}
                with self._lock:
                    self._status = status
                self._stop.wait(self.interval)
        except BaseException as exc:
            with self._lock:
                self._error = exc

    def status(self, maximum_age=.3):
        with self._lock:
            error, status = self._error, self._status
        if error is not None:
            raise RuntimeError("Physical gripper communication failed") from error
        if status is None:
            return None
        if time.monotonic() - status["host_monotonic"] > maximum_age or status["fault"]:
            raise RuntimeError("Gripper measurements are stale or report a device fault")
        return dict(status)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=self.device.timeout + 1.)
        if self._thread.is_alive():
            raise RuntimeError("Gripper worker did not close within its communication deadline")


class GraspPlaceTask:
    def __init__(self, config, scene, autonomous_goal=None):
        self.config, self.scene = config["task"], scene
        if autonomous_goal is not None and (type(autonomous_goal) is not int or autonomous_goal not in range(3)):
            raise ValueError("An autonomous trial needs one assigned candidate goal")
        self.autonomous_goal = autonomous_goal
        self.phase, self.grasped_goal = 0, None
        self.requested_closed = False
        self.grip_started = self.release_started = None
        self.dwell_started = None
        self.events, self.stop_reason = [], None

    def _dwell(self, ready, timestamp):
        if not ready:
            self.dwell_started = None
            return False
        if self.dwell_started is None:
            self.dwell_started = timestamp
        return timestamp - self.dwell_started >= self.config["dwell_seconds"]

    def update(self, timestamp, position, velocity, human_grip, status, enabled):
        if not enabled or status is None:
            self.dwell_started = None
            return self.requested_closed
        speed = np.linalg.norm(velocity)
        distances = np.linalg.norm(self.scene.goals - position, axis=1)
        candidate = int(np.argmin(distances)) if self.autonomous_goal is None else self.autonomous_goal
        autonomous = self.autonomous_goal is not None
        if self.phase == 0:
            if not autonomous and not human_grip and self.requested_closed:
                self.requested_closed, self.grip_started = False, None
                self.events.append({"time": timestamp, "event": "grip_cancelled_by_operator"})
            ready = distances[candidate] <= self.config["grasp_tolerance"] and speed <= self.config["max_grasp_speed"]
            close = self._dwell(ready, timestamp) if autonomous else bool(human_grip)
            if close and not self.requested_closed:
                self.requested_closed, self.grip_started = True, timestamp
                self.events.append({"time": timestamp, "event": "grip_requested", "candidate_goal": candidate})
            if self.requested_closed and status.get("requested_closed") is True:
                if status["object_status"] in {1, 2} and ready:
                    self.phase, self.grasped_goal = 1, candidate
                    self.events.append({"time": timestamp, "event": "grasp_contact_detected",
                                        "candidate_goal": candidate,
                                        "identity_source": "nearest_calibrated_target_requires_independent_annotation"})
                elif timestamp - self.grip_started > 5.:
                    self.stop_reason = "grasp_not_confirmed"
        elif self.phase == 1:
            if not autonomous and not human_grip:
                self.requested_closed, self.release_started = False, timestamp
                self.phase = 2
                self.events.append({"time": timestamp, "event": "release_requested_before_lift_completion",
                                    "inside_place_tolerance": bool(np.linalg.norm(position - self.scene.place) <= self.config["place_tolerance"])})
            if status["object_status"] == 3:
                self.stop_reason = "object_contact_lost"
            if self.phase == 1 and position[2] >= self.config["lift_height"] - .02:
                self.phase = 2
                self.events.append({"time": timestamp, "event": "lift_height_reached"})
        elif self.phase == 2:
            ready = np.linalg.norm(position - self.scene.place) <= self.config["place_tolerance"]
            ready = ready and speed <= self.config["max_grasp_speed"]
            release = self._dwell(ready, timestamp) if autonomous else not bool(human_grip)
            if release and self.requested_closed:
                self.requested_closed, self.release_started = False, timestamp
                self.events.append({"time": timestamp, "event": "release_requested", "inside_place_tolerance": bool(ready)})
            if (self.release_started is not None and status.get("requested_closed") is False
                    and status["object_status"] == 3 and status["position"] <= 10):
                self.phase = 3
                self.stop_reason = "release_completed_requires_independent_annotation"
                self.events.append({"time": timestamp, "event": "gripper_open_measured",
                                    "inside_place_tolerance": bool(ready)})
            elif self.release_started is not None and timestamp - self.release_started > 5.:
                self.stop_reason = "release_not_confirmed"
        return self.requested_closed
