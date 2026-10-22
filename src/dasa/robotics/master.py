from __future__ import annotations

import ctypes
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from dasa.geometry import clamp_norm


class DHDDevice:
    def __init__(self, library_path, device_index=0):
        if isinstance(device_index, bool) or not isinstance(device_index, int) or not 0 <= device_index <= 127:
            raise ValueError("DHD device index must be an integer in [0,127]")
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Install the vendor DHD SDK and provide its shared library: {path}")
        self.device_index, self._id, self._dll_directory = device_index, None, None
        if os.name == "nt":
            self._dll_directory = os.add_dll_directory(str(path.parent))
        try:
            self._library = ctypes.CDLL(str(path))
            scalar = ctypes.POINTER(ctypes.c_double)
            identifier = ctypes.c_byte
            signatures = {
                "dhdOpenID": ([identifier], ctypes.c_int),
                "dhdGetPosition": ([scalar, scalar, scalar, identifier], ctypes.c_int),
                "dhdGetLinearVelocity": ([scalar, scalar, scalar, identifier], ctypes.c_int),
                "dhdGetButton": ([ctypes.c_int, identifier], ctypes.c_int),
                "dhdSetForce": ([ctypes.c_double] * 3 + [identifier], ctypes.c_int),
                "dhdEnableForce": ([ctypes.c_ubyte, identifier], ctypes.c_int),
                "dhdClose": ([identifier], ctypes.c_int),
            }
            for name, (arguments, result) in signatures.items():
                function = getattr(self._library, name)
                function.argtypes, function.restype = arguments, result
            if hasattr(self._library, "dhdErrorGetLastStr"):
                self._library.dhdErrorGetLastStr.argtypes = []
                self._library.dhdErrorGetLastStr.restype = ctypes.c_char_p
        except (OSError, AttributeError) as exc:
            if self._dll_directory is not None:
                self._dll_directory.close()
                self._dll_directory = None
            raise RuntimeError("Unable to load the required DHD SDK C functions") from exc

    def _error_text(self):
        try:
            value = self._library.dhdErrorGetLastStr()
            return value.decode("utf-8", errors="replace") if value else "no SDK error text"
        except (AttributeError, OSError):
            return "no SDK error text"

    def _shutdown(self):
        errors = []
        identifier, self._id = self._id, None
        if identifier is not None:
            for name, arguments in (("dhdSetForce", (0.0, 0.0, 0.0, identifier)),
                                    ("dhdEnableForce", (0, identifier)),
                                    ("dhdClose", (identifier,))):
                try:
                    result = getattr(self._library, name)(*arguments)
                    if result != 0:
                        errors.append(f"{name} returned {result}")
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None
        return errors

    def _failure(self, operation, result):
        detail = self._error_text()
        self._shutdown()
        raise RuntimeError(f"DHD {operation} returned {result}: {detail}")

    def _check(self, result, operation):
        if result != 0:
            self._failure(operation, result)

    def _identifier(self):
        if self._id is None:
            raise RuntimeError("DHD device is not open")
        return self._id

    def _invalid(self, message):
        self._shutdown()
        raise ValueError(message)

    def open(self):
        if self._id is not None:
            return self
        identifier = self._library.dhdOpenID(self.device_index)
        if identifier < 0 or identifier > 127:
            self._failure("dhdOpenID", identifier)
        self._id = identifier
        self.enableForce(False)
        self.setForce(np.zeros(3))
        return self

    def _vector(self, name):
        values = [ctypes.c_double() for _ in range(3)]
        result = getattr(self._library, name)(*[ctypes.byref(value) for value in values],
                                              self._identifier())
        self._check(result, name)
        vector = np.array([value.value for value in values])
        if not np.isfinite(vector).all():
            self._invalid(f"DHD {name} returned nonfinite coordinates")
        return vector

    def getPosition(self):
        return self._vector("dhdGetPosition")

    def getLinearVelocity(self):
        return self._vector("dhdGetLinearVelocity")

    def getButton(self, button):
        if isinstance(button, bool) or not isinstance(button, int) or not 0 <= button < 32:
            self._invalid("DHD button index must be an integer in [0,31]")
        result = self._library.dhdGetButton(button, self._identifier())
        if result not in (0, 1):
            self._failure("dhdGetButton", result)
        return bool(result)

    def setForce(self, force, fy=None, fz=None):
        force = np.asarray(force if fy is None and fz is None else [force, fy, fz], dtype=float)
        if force.shape != (3,) or not np.isfinite(force).all():
            self._invalid("DHD force command must be a finite three-coordinate vector in N")
        result = self._library.dhdSetForce(*force.tolist(), self._identifier())
        self._check(result, "dhdSetForce")

    def enableForce(self, enabled):
        if not isinstance(enabled, bool):
            self._invalid("DHD force-enable flag must be boolean")
        result = self._library.dhdEnableForce(int(enabled), self._identifier())
        self._check(result, "dhdEnableForce")

    def close(self):
        errors = self._shutdown()
        if errors:
            raise RuntimeError("DHD shutdown: " + "; ".join(errors))

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except RuntimeError:
            if exc_type is None:
                raise
        return False


def _load_rotation(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    rotation = np.asarray(record["rotation"], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("Master calibration must provide a proper device-to-base rotation")
    if record.get("source_frame", "device") != "device" or record.get("target_frame", "base") != "base":
        raise ValueError("Master calibration must map source_frame device to target_frame base")
    return rotation


def _feedback(packet, last_sequence, sent_sequence):
    if not isinstance(packet, dict):
        raise ValueError("Invalid feedback packet")
    reply_to = packet["sequence"]
    frame = packet["feedback"]
    if (isinstance(reply_to, bool) or not isinstance(reply_to, int)
            or not 0 <= reply_to < sent_sequence or not isinstance(frame, dict)):
        raise ValueError("Feedback does not acknowledge an issued command")
    sequence = frame["observation_sequence"]
    if (frame["frame"] != "base" or isinstance(sequence, bool) or not isinstance(sequence, int)
            or sequence < 0 or (last_sequence is not None and sequence <= last_sequence)):
        raise ValueError("Feedback observation sequence is invalid or stale")
    for name, dimension in (("position_m", 3), ("velocity_m_s", 3), ("measured_wrench_N_Nm", 6)):
        vector = np.asarray(frame[name], dtype=float)
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid feedback vector: {name}")
    for name in ("source_time_s", "host_source_time_s", "host_send_time_s"):
        if not isinstance(frame[name], (int, float)) or not np.isfinite(frame[name]) or frame[name] < 0:
            raise ValueError(f"Invalid feedback timestamp: {name}")
    if frame["host_source_time_s"] > frame["host_send_time_s"]:
        raise ValueError("Feedback host timestamps are not causal")
    force = np.asarray(packet["reflected_force_N"], dtype=float)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError("Invalid reflected-force vector")
    return frame, force


def _console_display(frame):
    position = frame["position_m"]
    print(f"\rFeedback {frame['observation_sequence']:06d} | delayed TCP [m]: "
          f"{position[0]:+.3f} {position[1]:+.3f} {position[2]:+.3f}", end="", flush=True)


def run_master(library_path, rotation_path, host="127.0.0.1", port=55001, device_index=0,
               frequency=500, stiffness=100.0, maxcommand=12.0, maxfeedback=2.0,
               deadman_button=0, grip_button=1, feedback_timeout=0.1, display_callback=None,
               stop_event=None, maximum_samples=None):
    numeric = [frequency, stiffness, maxcommand, maxfeedback, feedback_timeout]
    if (not np.isfinite(numeric).all() or not 0 < frequency <= 1000 or stiffness <= 0
            or maxcommand <= 0 or maxfeedback < 0 or feedback_timeout <= 0):
        raise ValueError("Invalid master frequency, stiffness, force limits or feedback timeout")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("UDP port must be an integer in [1,65535]")
    if any(isinstance(button, bool) or not isinstance(button, int) or not 0 <= button < 32
           for button in (deadman_button, grip_button)) or deadman_button == grip_button:
        raise ValueError("Select distinct valid deadman and grip button indices")
    if maximum_samples is not None and (isinstance(maximum_samples, bool)
            or not isinstance(maximum_samples, int) or maximum_samples < 1):
        raise ValueError("maximum_samples must be a positive integer")
    rotation = _load_rotation(rotation_path)
    display = _console_display if display_callback is None else display_callback
    if not callable(display):
        raise ValueError("display_callback must render feedback before returning")
    device, client = None, None
    sequence, observation_sequence, displayed, rejected = 0, None, 0, 0
    sent_times = {}
    feedback_force, last_feedback, previous_enabled = np.zeros(3), None, False
    anchor, period = None, 1.0 / frequency
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.connect((host, port))
        client.setblocking(False)
        device = DHDDevice(library_path, device_index).open()
        device.enableForce(True)
        deadline = time.monotonic()
        while ((maximum_samples is None or sequence < maximum_samples)
               and (stop_event is None or not stop_event.is_set())):
            while True:
                try:
                    raw = client.recv(8192)
                except BlockingIOError:
                    break
                try:
                    reply = json.loads(raw.decode("utf-8"))
                    frame, force = _feedback(reply, observation_sequence, sequence)
                    received_at = time.monotonic()
                    command_time = sent_times.get(reply["sequence"])
                    if command_time is None or not 0 <= received_at - command_time <= feedback_timeout:
                        raise ValueError("Feedback reply is older than the local freshness limit")
                except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                    rejected += 1
                    continue
                if display(frame) is False:
                    rejected += 1
                    continue
                observation_sequence = frame["observation_sequence"]
                feedback_force = clamp_norm(force, maxfeedback)
                last_feedback = received_at
                displayed += 1
            position, velocity = device.getPosition(), device.getLinearVelocity()
            deadman, grip = device.getButton(deadman_button), device.getButton(grip_button)
            now = time.monotonic()
            fresh = last_feedback is not None and 0 <= now - last_feedback <= feedback_timeout
            enabled = bool(deadman and fresh)
            if anchor is None or not enabled or not previous_enabled:
                anchor = position.copy()
            command = clamp_norm(stiffness * (rotation @ (position - anchor)), maxcommand) if enabled else np.zeros(3)
            reflected = rotation.T @ feedback_force if enabled else np.zeros(3)
            device.setForce(reflected)
            packet = {"sequence": sequence, "device_timestamp": now, "frame": "base",
                      "force_N": command.tolist(), "velocity_m_s": (rotation @ velocity).tolist(),
                      "enable": enabled, "grip": bool(grip and enabled),
                      "observation_sequence": observation_sequence if fresh else None,
                      "command_source": "virtual_spring_from_measured_master_displacement",
                      "command_stiffness_N_m": stiffness}
            client.send(json.dumps(packet, allow_nan=False).encode("utf-8"))
            sent_times[sequence] = now
            if len(sent_times) > max(64, int(frequency * feedback_timeout * 4)):
                del sent_times[next(iter(sent_times))]
            sequence += 1
            previous_enabled = enabled
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                deadline = time.monotonic()
    finally:
        try:
            if device is not None:
                device.close()
        finally:
            if client is not None:
                try:
                    client.send(json.dumps({"sequence": sequence, "device_timestamp": time.monotonic(),
                                            "frame": "base", "force_N": [0.0] * 3,
                                            "velocity_m_s": [0.0] * 3, "enable": False,
                                            "grip": False, "observation_sequence": None}).encode("utf-8"))
                except OSError:
                    pass
                finally:
                    client.close()
    return {"samples_sent": sequence, "feedback_frames_displayed": displayed,
            "feedback_packets_rejected": rejected, "requested_frequency_hz": frequency,
            "clock": "local_monotonic; no hard-real-time scheduling guarantee",
            "command_source": "virtual_spring_from_measured_master_displacement",
            "master_velocity_source": "DHD measured linear velocity",
            "robot_motion_commands_sent_by_bridge": 0}
