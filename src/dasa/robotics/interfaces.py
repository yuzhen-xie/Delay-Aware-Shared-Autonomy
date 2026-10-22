from __future__ import annotations

import json
import socket
import time
import re
import select
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
    def __init__(self, host: str, execute: bool = False, maximum_speed: float = 0.08,
                 model: str = "UR10", force_limit: float = 18.0):
        if not isinstance(host, str) or not host.strip() or not np.isfinite(maximum_speed) or maximum_speed <= 0:
            raise ValueError("Robot host and a positive finite maximum speed are required")
        try:
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError("Install the workstation dependency ur-rtde before connecting") from exc
        self.receive, self.control, self.closed = None, None, False
        if model not in {"UR10", "UR10e"} or not np.isfinite(force_limit) or force_limit <= 0:
            raise ValueError("An explicit supported robot model and positive wrench limit are required")
        self.execute, self.maximum_speed = execute, maximum_speed
        self.model, self.force_limit = model, force_limit
        self.frequency = 125.0 if model == "UR10" else 500.0
        self.force_active = False
        self.speed_active = False
        self.commands = []
        try:
            self.receive = rtde_receive.RTDEReceiveInterface(host, self.frequency)
            if hasattr(self.receive, "isConnected") and not self.receive.isConnected():
                raise ConnectionError("The RTDE receive connection was not established")
            if execute:
                import rtde_control
                self.control = rtde_control.RTDEControlInterface(host, self.frequency)
                if hasattr(self.control, "isConnected") and not self.control.isConnected():
                    raise ConnectionError("The RTDE control connection was not established")
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def state(self) -> RobotState:
        if self.closed:
            raise RuntimeError("The RTDE connection is closed")
        for _ in range(4):
            before = float(self.receive.getTimestamp())
            pose = np.asarray(self.receive.getActualTCPPose(), float)
            velocity = np.asarray(self.receive.getActualTCPSpeed(), float)
            wrench = np.asarray(self.receive.getActualTCPForce(), float)
            joints = np.asarray(self.receive.getActualQ(), float)
            safety = int(self.receive.getSafetyMode())
            stamp = float(self.receive.getTimestamp())
            if before == stamp:
                break
        else:
            raise RuntimeError("Could not obtain a coherent measured RTDE sample")
        if (any(value.shape != (6,) or not np.isfinite(value).all()
                for value in (pose, velocity, wrench, joints)) or not np.isfinite(stamp) or stamp < 0):
            raise ValueError("RTDE returned invalid measured robot state")
        return RobotState(stamp, pose[:3], pose[3:], velocity[:3], wrench, joints, safety)

    def speed(self, velocity, duration: float, enabled: bool):
        velocity = np.asarray(velocity, float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("A finite 3D velocity command is required")
        if not np.isfinite(duration) or duration <= 0 or not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("Velocity commands need a positive duration and boolean enable")
        velocity = clamp_norm(velocity, self.maximum_speed)
        state = self.state()
        if state.safety_mode not in {1, 2} or not enabled:
            self.stop()
            return False
        executed = False
        if self.execute:
            self.speed_active = True
            executed = bool(self.control.speedL(np.r_[velocity, np.zeros(3)].tolist(), 0.3, duration))
            if not executed:
                self.stop()
        self.commands.append({"timestamp": state.timestamp, "velocity": velocity.tolist(),
                              "duration": duration, "executed": executed})
        return executed

    def stop(self):
        if self.control is not None:
            active = self.force_active or self.speed_active
            if self.force_active:
                if not self.control.forceModeStop():
                    raise RuntimeError("The controller did not acknowledge force-mode stop")
                self.force_active = False
            if active:
                if not self.control.speedStop(0.5):
                    raise RuntimeError("The controller did not acknowledge Cartesian stop")
                self.speed_active = False

    def configure_tool(self, tcp_pose, payload_mass, payload_cog, damping=.1, gain=.5):
        pose, cog = np.asarray(tcp_pose, float), np.asarray(payload_cog, float)
        if (pose.shape != (6,) or cog.shape != (3,)
                or not np.isfinite(np.r_[pose, cog, payload_mass, damping, gain]).all()
                or payload_mass < 0 or not 0 <= damping <= 1 or not 0 < gain <= 1):
            raise ValueError("Measured tool TCP, payload and bounded force-mode gains are required")
        if self.execute:
            operations = [self.control.setTcp(pose.tolist()),
                          self.control.setPayload(float(payload_mass), cog.tolist()),
                          self.control.forceModeSetDamping(float(damping)),
                          self.control.forceModeSetGainScaling(float(gain))]
            if not all(operations):
                raise RuntimeError("The controller did not acknowledge the measured tool/force configuration")

    def configure_watchdog(self, maximum_hold_seconds):
        if not np.isfinite(maximum_hold_seconds) or maximum_hold_seconds <= 0:
            raise ValueError("The controller watchdog interval must be positive")
        if self.execute:
            native_deadline = maximum_hold_seconds - 1.0 / self.frequency
            if native_deadline <= 0:
                raise ValueError("The reserved hold must include at least one native controller tick")
            if not self.control.setWatchdog(1.0 / native_deadline):
                raise RuntimeError("The controller did not acknowledge its command watchdog")

    def force(self, wrench, duration: float, enabled: bool):
        force = np.asarray(wrench, float)
        if (force.shape != (3,) or not np.isfinite(force).all()
                or not np.isfinite(duration) or duration < 1 / self.frequency - 1e-9):
            raise ValueError("Force commands require finite translational wrench and native-rate period")
        state = self.state()
        if self.execute:
            if not self.control.kickWatchdog():
                self.stop()
                raise RuntimeError("The native controller heartbeat was not acknowledged")
        if not enabled or state.safety_mode != 1:
            self.stop()
            return False
        force = clamp_norm(force, self.force_limit)
        executed = False
        sent_at = time.monotonic()
        if self.execute:
            self.force_active = True
            executed = bool(self.control.forceMode(
                [0.] * 6, [1, 1, 1, 0, 0, 0], np.r_[force, np.zeros(3)].tolist(), 2,
                [self.maximum_speed / np.sqrt(3)] * 3 + [.03] * 3))
            if not executed:
                self.stop()
        self.commands.append({"timestamp": state.timestamp, "force_N": force.tolist(),
                              "duration": duration, "executed": executed,
                              "host_send_monotonic": sent_at, "host_return_monotonic": time.monotonic(),
                              "mode": "base_frame_translational_force"})
        return executed

    def close(self):
        if self.closed:
            return
        errors = []
        actions = []
        if self.control is not None:
            actions += [self.stop, self.control.stopScript, self.control.disconnect]
        if self.receive is not None:
            actions.append(self.receive.disconnect)
        for action in actions:
            try:
                action()
            except Exception as exc:
                errors.append(exc)
        self.closed = True
        if errors:
            raise RuntimeError("RTDE cleanup reported an error") from errors[0]


class MasterUDP:





    def __init__(self, port=55001, timeout=0.1):
        if (not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535
                or not np.isfinite(timeout) or timeout <= 0):
            raise ValueError("UDP port and watchdog timeout are invalid")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(("127.0.0.1", port))
            self.socket.settimeout(timeout)
        except BaseException:
            self.socket.close()
            raise
        self.last_sequence = -1
        self.peer_address = None

    def _decode(self, raw):
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Master packet must be a JSON object")
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
        return data

    def read(self):
        raw, address = self.socket.recvfrom(8192)
        selected, selected_address = None, None
        peer = self.peer_address
        processed, invalid, foreign = 0, 0, 0
        while True:
            processed += 1
            if peer is not None and address != peer:
                foreign += 1
            else:
                try:
                    packet = self._decode(raw)
                except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                    invalid += 1
                else:
                    if peer is None:
                        peer = address
                    if selected is None or packet["sequence"] > selected["sequence"]:
                        packet["host_receive_monotonic"] = time.monotonic()
                        selected, selected_address = packet, address
            if processed >= 4096 or not select.select([self.socket], [], [], 0)[0]:
                break
            raw, address = self.socket.recvfrom(8192)
        if selected is None:
            raise ValueError("No valid fresh master command was available")
        self.peer_address = selected_address
        self.last_sequence = selected["sequence"]
        selected["skipped_packets"] = processed - 1
        selected["invalid_packets"] = invalid
        selected["foreign_peer_packets"] = foreign
        return selected, selected_address

    def reflect_force(self, force, address, feedback=None, energy_packet=None):

        force = np.asarray(force, float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("Reflected force must be a finite 3D vector")
        packet = {"sequence": self.last_sequence, "reflected_force_N": force.tolist(),
                  "feedback": feedback, "energy_packet": energy_packet}
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

    def __init__(self, host, port=63352, execute=False, speed=64, force=64, timeout=1.0):
        if (not isinstance(host, str) or not host or not isinstance(port, int)
                or not 1 <= port <= 65535 or not np.isfinite(timeout) or timeout <= 0
                or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255
                       for value in (speed, force))):
            raise ValueError("Invalid Robotiq endpoint or actuator parameters")
        self.host, self.port, self.execute = host, port, execute
        self.speed, self.force, self.timeout = speed, force, timeout

    def _exchange(self, command: str) -> str:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as client:
            client.sendall((command + "\n").encode("ascii"))
            payload = b""
            while len(payload) < 4096:
                chunk = client.recv(1024)
                if not chunk:
                    break
                payload += chunk
                if b"\n" in payload or payload.strip().lower() == b"ack":
                    break
        reply = payload.decode("ascii").strip()
        if not reply:
            raise ConnectionError("The Robotiq endpoint returned no reply")
        return reply

    def _get(self, variable: str) -> int:
        reply = self._exchange(f"GET {variable}")
        match = re.fullmatch(rf"{variable}\s+(\d+)", reply)
        if not match:
            raise RuntimeError(f"Invalid Robotiq {variable} reply")
        return int(match.group(1))

    def _set(self, **variables):
        command = "SET " + " ".join(f"{key} {value}" for key, value in variables.items())
        reply = self._exchange(command)
        if reply.lower() != "ack":
            raise RuntimeError("The Robotiq endpoint rejected a command")
        return command

    def status(self) -> dict:
        return {"active": self._get("ACT") == 1, "activation_status": self._get("STA"),
                "fault": self._get("FLT"), "position": self._get("POS"),
                "object_status": self._get("OBJ")}

    def activate(self, timeout=10.0):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Activation timeout must be positive")
        if not self.execute:
            return {"executed": False, "operation": "activate"}
        status = self.status()
        if status["active"] and status["activation_status"] == 3 and status["fault"] == 0:
            return {"executed": False, "already_active": True, "status": status}
        self._set(ACT=0)
        deadline = time.monotonic() + timeout
        while self._get("ACT") != 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("Robotiq reset timed out")
            time.sleep(.02)
        self._set(ACT=1)
        while time.monotonic() < deadline:
            status = self.status()
            if status["fault"] >= 8:
                raise RuntimeError(f"Robotiq activation fault {status['fault']}")
            if status["active"] and status["activation_status"] == 3 and status["fault"] == 0:
                return {"executed": True, "status": status}
            time.sleep(.02)
        raise TimeoutError("Robotiq activation timed out")

    def command(self, closed: bool):
        if not isinstance(closed, bool):
            raise ValueError("Gripper closed state must be boolean")
        command = f"SET POS {255 if closed else 0} SPE {self.speed} FOR {self.force} GTO 1"
        if not self.execute:
            return {"executed": False, "command": command}
        self._set(POS=255 if closed else 0, SPE=self.speed, FOR=self.force, GTO=1)
        return {"executed": True, "command": command, "completion": "not_yet_measured"}

    def wait_until_stopped(self, timeout=5.0) -> dict:
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Gripper motion timeout must be positive")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status()
            if status["fault"]:
                raise RuntimeError(f"Robotiq motion fault {status['fault']}")
            if status["object_status"] in {1, 2, 3}:
                return status
            time.sleep(.02)
        raise TimeoutError("Robotiq motion did not complete before the timeout")


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
