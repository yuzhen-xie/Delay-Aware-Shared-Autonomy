"""Measured-state RTDE access and bounded, explicitly enabled motion commands."""

from __future__ import annotations

import json
import socket
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dasa.config import write_json
from dasa.geometry import clamp_norm


@dataclass
class RobotState:
    timestamp: float
    position: np.ndarray
    rotation_vector: np.ndarray
    velocity: np.ndarray
    wrench: np.ndarray
    joints: np.ndarray
    safety_mode: int


class URRTDE:
    def __init__(self, host: str, execute: bool = False, maximum_speed: float = 0.08):
        try:
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError("Install the workstation dependency ur-rtde before connecting") from exc
        self.receive = rtde_receive.RTDEReceiveInterface(host)
        self.control = None
        self.execute, self.maximum_speed = execute, maximum_speed
        if execute:
            import rtde_control
            self.control = rtde_control.RTDEControlInterface(host)
        self.commands = []

    def state(self) -> RobotState:
        pose = np.asarray(self.receive.getActualTCPPose())
        return RobotState(float(self.receive.getTimestamp()), pose[:3], pose[3:],
                          np.asarray(self.receive.getActualTCPSpeed())[:3],
                          np.asarray(self.receive.getActualTCPForce()),
                          np.asarray(self.receive.getActualQ()), int(self.receive.getSafetyMode()))

    def speed(self, velocity, duration: float, enabled: bool):
        velocity = np.asarray(velocity, float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("A finite 3D velocity command is required")
        velocity = clamp_norm(velocity, self.maximum_speed)
        state = self.state()
        if state.safety_mode not in {1, 2} or not enabled:
            self.stop()
            return False
        self.commands.append({"timestamp": state.timestamp, "velocity": velocity.tolist(),
                              "duration": duration, "executed": self.execute})
        if self.execute:
            # Rotational velocity remains zero; tool orientation is held by the robot controller.
            self.control.speedL(np.r_[velocity, np.zeros(3)].tolist(), 0.3, duration)
        return self.execute

    def stop(self):
        if self.control is not None:
            self.control.speedStop(0.5)

    def close(self):
        self.stop()
        if self.control is not None:
            self.control.stopScript()
            self.control.disconnect()
        self.receive.disconnect()


class MasterUDP:
    """SDK bridge packets include an optional displayed observation acknowledgement.

    Host-local monotonic receive time governs the watchdog. Remote device clock
    values are recorded, never subtracted from the host clock without synchronization.
    """
    def __init__(self, port=55001, timeout=0.1):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", port))
        self.socket.settimeout(timeout)
        self.last_sequence = -1

    def read(self):
        raw, address = self.socket.recvfrom(8192)
        data = json.loads(raw.decode("utf-8"))
        sequence = data["sequence"]
        if (data["frame"] != "base" or not isinstance(sequence, int)
                or isinstance(sequence, bool) or sequence <= self.last_sequence):
            raise ValueError("Master packets must be base-frame and strictly ordered")
        acknowledgement = data.get("observation_sequence")
        if acknowledgement is not None and (not isinstance(acknowledgement, int)
                or isinstance(acknowledgement, bool) or acknowledgement < 0):
            raise ValueError("An observation acknowledgement must be a nonnegative integer")
        for key in ("force_N", "velocity_m_s"):
            value = np.asarray(data[key], float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid master signal {key}")
        if not isinstance(data["enable"], bool) or not isinstance(data["grip"], bool):
            raise ValueError("Master enable/grip must be boolean")
        self.last_sequence = data["sequence"]
        data["host_receive_monotonic"] = time.monotonic()
        return data, address

    def reflect_force(self, force, address, feedback=None):
        # Device-side actuator limits and interlocks remain part of the master SDK bridge.
        force = np.asarray(force, float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("Reflected force must be a finite 3D vector")
        packet = {"sequence": self.last_sequence, "reflected_force_N": force.tolist(),
                  "feedback": feedback}
        self.socket.sendto(json.dumps(packet).encode(), address)

    def close(self):
        self.socket.close()


@dataclass(frozen=True)
class FeedbackFrame:
    observation_sequence: int
    source_time: float
    host_source_time: float
    host_send_time: float
    state: np.ndarray
    wrench: np.ndarray

    def payload(self) -> dict:
        return {"observation_sequence": self.observation_sequence,
                "source_time_s": self.source_time, "host_source_time_s": self.host_source_time,
                "host_send_time_s": self.host_send_time, "frame": "base",
                "position_m": self.state[:3].tolist(), "velocity_m_s": self.state[3:].tolist(),
                "measured_wrench_N_Nm": self.wrench.tolist()}


class FeedbackLedger:
    """Bounded sent-frame lookup; acknowledgements refer to host-owned timestamps."""

    def __init__(self, maximum_frames=4096):
        if not isinstance(maximum_frames, int) or maximum_frames < 1:
            raise ValueError("Feedback history must have a positive finite frame count")
        self.maximum_frames = maximum_frames
        self.frames = OrderedDict()
        self.next_sequence = 0
        self.last_acknowledged = -1

    def issue(self, state, wrench, source_time: float, host_source_time: float,
              host_send_time: float) -> FeedbackFrame:
        state, wrench = np.asarray(state, float), np.asarray(wrench, float)
        if (state.shape != (6,) or wrench.shape != (6,) or not np.isfinite(state).all()
                or not np.isfinite(wrench).all() or not np.isfinite(
                    [source_time, host_source_time, host_send_time]).all()
                or source_time < 0 or host_source_time > host_send_time):
            raise ValueError("Feedback needs finite state/wrench and causal host timestamps")
        frame = FeedbackFrame(self.next_sequence, source_time, host_source_time,
                              host_send_time, state.copy(), wrench.copy())
        self.frames[frame.observation_sequence] = frame
        self.next_sequence += 1
        while len(self.frames) > self.maximum_frames:
            self.frames.popitem(last=False)
        return frame

    def acknowledge(self, observation_sequence, host_receive_time: float,
                    maximum_age_s: float) -> FeedbackFrame | None:
        if not np.isfinite([host_receive_time, maximum_age_s]).all() or maximum_age_s < 0:
            raise ValueError("Acknowledgement freshness needs a finite nonnegative age limit")
        if (not isinstance(observation_sequence, int) or isinstance(observation_sequence, bool)
                or observation_sequence < self.last_acknowledged):
            return None
        frame = self.frames.get(observation_sequence)
        if frame is None:
            return None
        age = host_receive_time - frame.host_send_time
        if age < 0 or age > maximum_age_s:
            return None
        self.last_acknowledged = observation_sequence
        return frame


class RobotiqSocket:
    """ASCII bridge used by the Robotiq URCap; explicitly selected, never assumed."""
    def __init__(self, host, port=63352, execute=False):
        self.host, self.port, self.execute = host, port, execute

    def command(self, closed: bool):
        command = f"SET POS {255 if closed else 0} SPE 64 FOR 64 GTO 1\n"
        if not self.execute:
            return {"executed": False, "command": command.strip()}
        with socket.create_connection((self.host, self.port), timeout=1) as client:
            client.sendall(command.encode("ascii"))
            reply = client.recv(1024).decode("ascii").strip()
        if reply.lower() != "ack":
            raise RuntimeError(f"Gripper rejected command: {reply}")
        return {"executed": True, "reply": reply}


def record_robot(host: str, samples: int, output: Path):
    if samples <= 0:
        raise ValueError("samples must be positive")
    robot = URRTDE(host, execute=False)
    result = []
    try:
        for _ in range(samples):
            state = robot.state()
            result.append({k: getattr(state, k) for k in state.__dataclass_fields__})
            time.sleep(0.02)
    finally:
        robot.close()
    write_json(output, {"provenance": "UR10_measured_state", "motion_commands_sent": 0,
                        "samples": result})
