from __future__ import annotations

import ctypes
import json
import os
import hashlib
import queue
import socket
import sys
import threading
import time
import uuid
import errno
from pathlib import Path

import numpy as np

from dasa.geometry import clamp_norm
from dasa.config import write_json
from dasa.control.physical_passivity import MasterPassivity
from dasa.robotics.camera import CameraUnavailable, VideoDisplay


class DHDDevice:
    def __init__(self, library_path, device_index=0):
        if isinstance(device_index, bool) or not isinstance(device_index, int) or not 0 <= device_index <= 127:
            raise ValueError("DHD device index must be an integer in [0,127]")
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Install the vendor DHD SDK and provide its shared library: {path}")
        self.device_index, self._id, self._dll_directory = device_index, None, None
        self._watchdog, self._watchdog_attempted = None, False
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
                "dhdGetForce": ([scalar, scalar, scalar, identifier], ctypes.c_int),
                "dhdGetButton": ([ctypes.c_int, identifier], ctypes.c_int),
                "dhdSetForce": ([ctypes.c_double] * 3 + [identifier], ctypes.c_int),
                "dhdEnableForce": ([ctypes.c_ubyte, identifier], ctypes.c_int),
                "dhdClose": ([identifier], ctypes.c_int),
            }
            for name, (arguments, result) in signatures.items():
                function = getattr(self._library, name)
                function.argtypes, function.restype = arguments, result
            self._optional_signatures = {
                "dhdGetOrientationRad": ([scalar, scalar, scalar, identifier], 3),
                "dhdGetAngularVelocityRad": ([scalar, scalar, scalar, identifier], 3),
                "dhdGetGripperAngleRad": ([scalar, identifier], 1)}
            for name, (arguments, _) in self._optional_signatures.items():
                if hasattr(self._library, name):
                    function = getattr(self._library, name)
                    function.argtypes, function.restype = arguments, ctypes.c_int
            for name in ("dhdHasWrist", "dhdHasGripper"):
                if hasattr(self._library, name):
                    function = getattr(self._library, name)
                    function.argtypes, function.restype = [identifier], ctypes.c_bool
            watchdog_signatures = {
                "dhdSetWatchdog": [ctypes.c_ubyte, identifier],
                "dhdGetWatchdog": [ctypes.POINTER(ctypes.c_ubyte), identifier],
                "dhdSetTimeGuard": [ctypes.c_int, identifier],
                "dhdEnableExpertMode": [], "dhdDisableExpertMode": [],
                "dhdGetStatus": [ctypes.POINTER(ctypes.c_int), identifier]}
            for name, arguments in watchdog_signatures.items():
                if hasattr(self._library, name):
                    function = getattr(self._library, name)
                    function.argtypes, function.restype = arguments, ctypes.c_int
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
                                    ("dhdEnableForce", (0, identifier))):
                try:
                    result = getattr(self._library, name)(*arguments)
                    if result != 0:
                        errors.append(f"{name} returned {result}")
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
            if self._watchdog_attempted:
                for name, arguments in (("dhdEnableExpertMode", ()),
                                        ("dhdSetWatchdog", (0, identifier)),
                                        ("dhdSetTimeGuard", (-1, identifier)),
                                        ("dhdDisableExpertMode", ())):
                    try:
                        result = getattr(self._library, name)(*arguments)
                        if result != 0:
                            errors.append(f"{name} returned {result}")
                    except Exception as exc:
                        errors.append(f"{name}: {exc}")
            try:
                result = self._library.dhdClose(identifier)
                if result != 0:
                    errors.append(f"dhdClose returned {result}")
            except Exception as exc:
                errors.append(f"dhdClose: {exc}")
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

    def getForce(self):
        return self._vector("dhdGetForce")

    def configureWatchdog(self, maximum_hold_seconds, unit_seconds):
        if (unit_seconds is None or not np.isfinite([maximum_hold_seconds, unit_seconds]).all()
                or maximum_hold_seconds <= 0 or unit_seconds <= 0):
            raise ValueError("Provide the actual manufacturer SDK watchdog unit and a positive hold bound")
        required = ("dhdSetWatchdog", "dhdGetWatchdog", "dhdEnableExpertMode",
                    "dhdDisableExpertMode", "dhdGetStatus", "dhdSetTimeGuard")
        if any(not hasattr(self._library, name) for name in required):
            self._invalid("This SDK cannot configure and verify the native force watchdog; force actuation is prohibited")
        ticks = int(np.floor(maximum_hold_seconds / unit_seconds))
        if not 1 <= ticks <= 255:
            self._invalid("The commissioned watchdog units cannot represent the requested hold within 1..255 ticks")
        self.enableForce(False)
        self.setForce(np.zeros(3))
        self._check(self._library.dhdEnableExpertMode(), "dhdEnableExpertMode")
        self._watchdog_attempted = True
        try:
            self._check(self._library.dhdSetTimeGuard(0, self._identifier()), "dhdSetTimeGuard")
            self._check(self._library.dhdSetWatchdog(ticks, self._identifier()), "dhdSetWatchdog")
            actual = ctypes.c_ubyte()
            self._check(self._library.dhdGetWatchdog(ctypes.byref(actual), self._identifier()), "dhdGetWatchdog")
            if actual.value != ticks or actual.value == 0 or actual.value * unit_seconds > maximum_hold_seconds:
                self._invalid("The native force-watchdog readback does not satisfy the declared maximum hold")
            self._watchdog = {
                "requested_ticks": ticks, "readback_ticks": actual.value,
                "declared_unit_seconds": unit_seconds,
                "readback_timeout_seconds": actual.value * unit_seconds,
                "maximum_hold_seconds": maximum_hold_seconds,
                "unit_provenance": "runtime_commissioned_manufacturer_SDK_unit",
                "communication_timeguard_microseconds": 0,
                "communication_timeguard_scope": "SDK communication throttling only; native force cutoff remains enabled",
                "binding_reference": "https://github.com/EmDash00/forcedimension_core-python/blob/main/forcedimension_core/dhd/expert/__init__.py",
                "native_watchdog_configured": True}
        finally:
            if self._id is not None:
                self._check(self._library.dhdDisableExpertMode(), "dhdDisableExpertMode")
        return dict(self._watchdog)

    def getStatus(self):
        if not hasattr(self._library, "dhdGetStatus"):
            self._invalid("Native force-status monitoring is unavailable")
        status = (ctypes.c_int * 64)()
        self._check(self._library.dhdGetStatus(status, self._identifier()), "dhdGetStatus")
        result = {"force_enabled": bool(status[5]), "force_off_cause": int(status[14]),
                  "native_status": list(status[:17]),
                  "status_reference": "DHD_STATUS_FORCE_5_DHD_STATUS_FORCEOFFCAUSE_14"}
        if not result["force_enabled"] and result["force_off_cause"] == 3:
            self._invalid("The device firmware force watchdog expired; restart after commissioning the host cadence")
        return result

    def getOptionalState(self):
        result = {}
        capabilities = {name: bool(getattr(self._library, function)(self._identifier()))
                        if hasattr(self._library, function) else None
                        for name, function in (("wrist", "dhdHasWrist"), ("gripper", "dhdHasGripper"))}
        labels = {"dhdGetOrientationRad": "orientation_rad",
                  "dhdGetAngularVelocityRad": "angular_velocity_rad_s",
                  "dhdGetGripperAngleRad": "gripper_angle_rad"}
        for name, (_, dimensions) in self._optional_signatures.items():
            label = labels[name]
            capability = "gripper" if name == "dhdGetGripperAngleRad" else "wrist"
            if capabilities[capability] is False:
                result[label] = {"available": False, "source": name,
                                 "reason": f"physical_device_has_no_{capability}"}
                continue
            if not hasattr(self._library, name):
                result[label] = {"available": False, "source": name, "reason": "SDK_symbol_unavailable"}
                continue
            values = [ctypes.c_double() for _ in range(dimensions)]
            status = getattr(self._library, name)(*[ctypes.byref(value) for value in values],
                                                 self._identifier())
            coordinates = np.array([value.value for value in values])
            if status != 0 or not np.isfinite(coordinates).all():
                result[label] = {"available": False, "source": name, "sdk_result": int(status)}
            else:
                result[label] = {"available": True, "source": name,
                                 "value": coordinates.tolist() if dimensions > 1 else float(coordinates[0])}
        result["native_dofs_observed"] = (3 + 3 * int(result["orientation_rad"]["available"])
                                           + int(result["gripper_angle_rad"]["available"]))
        result["policy_coordinates"] = "three_base_frame_translations_with_robot_orientation_held"
        result["native_frame"] = "device"
        result["native_orientation_convention"] = "DHD_SDK_native_angles"
        result["SDK_physical_capabilities"] = capabilities
        return result

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
        if enabled and self._watchdog is None:
            self._invalid("Configure and verify a bounded native firmware watchdog before enabling forces")
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


class MasterRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(maxsize=100000)
        self.error = None
        self.thread = threading.Thread(target=self._write, name="master-measured-recorder", daemon=True)
        self.thread.start()

    @staticmethod
    def _encode(value):
        def convert(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, np.generic):
                return item.item()
            raise TypeError(type(item).__name__)
        return json.dumps(value, default=convert, allow_nan=False, separators=(",", ":")) + "\n"

    def append(self, value):
        if self.error is not None:
            raise RuntimeError("Measured master recording failed") from self.error
        try:
            self.queue.put_nowait(self._encode(value))
        except queue.Full as exc:
            raise RuntimeError("Measured master logger cannot retain every physical sample") from exc

    def _write(self):
        try:
            with (self.directory / "samples.jsonl").open("x", encoding="utf-8") as stream:
                while True:
                    value = self.queue.get()
                    if value is None:
                        break
                    stream.write(value)
                    if self.queue.empty():
                        stream.flush()
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as exc:
            self.error = exc

    def close(self):
        if self.error is not None:
            raise RuntimeError("Measured master recording failed") from self.error
        self.queue.put(None, timeout=1.)
        self.thread.join(timeout=5.)
        if self.thread.is_alive() or self.error is not None:
            raise RuntimeError("Measured master records were not fully flushed") from self.error


def _transient_udp_error(error):
    return isinstance(error, OSError) and error.errno in {
        errno.ECONNREFUSED, errno.ECONNRESET, errno.ENETUNREACH, errno.EHOSTUNREACH, 10054, 10061}


def run_master(library_path, rotation_path, host="127.0.0.1", port=55001, device_index=0,
               frequency=1000, stiffness=100.0, maxcommand=12.0, maxfeedback=2.0,
               deadman_button=0, grip_button=1, feedback_timeout=0.1, display_callback=None,
               stop_event=None, maximum_samples=None, config=None, video_host=None,
               video_port=55002, video_timeout=.2, record_directory=None,
               watchdog_unit_seconds=None):
    numeric = [frequency, stiffness, maxcommand, maxfeedback, feedback_timeout, video_timeout]
    if (not np.isfinite(numeric).all() or not 0 < frequency <= 1000 or stiffness <= 0
            or maxcommand <= 0 or maxfeedback < 0 or feedback_timeout <= 0 or video_timeout <= 0):
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
    if display_callback is not None and not callable(display_callback):
        raise ValueError("display_callback must render an actual decoded image and paired state")
    session_id = uuid.uuid4().hex
    observer = MasterPassivity(config or {}, session_id)
    maximum_hold_seconds = observer.diagnostics()["maximum_hold_seconds"]
    if (watchdog_unit_seconds is None or not np.isfinite(watchdog_unit_seconds)
            or watchdog_unit_seconds <= 0 or 1.0 / frequency >= maximum_hold_seconds):
        raise ValueError("Supply the manufacturer watchdog unit and a requested servo period shorter than its hold bound")
    directory = Path(record_directory) if record_directory is not None else Path("artifacts") / f"master_{session_id}"
    recorder = MasterRecorder(directory)
    device, client, video = None, None, None
    sequence, observation_sequence, rejected, displayed = 0, None, 0, 0
    last_udp_sequence, last_displayed_sequence = None, None
    sent_times = {}
    pending_feedback = {}
    previous_enabled, previous_sample = False, None
    network_available, native_dofs_observed = False, set()
    peer_epoch, pending_epoch, deadman_released = None, None, False
    epochs = []
    precharge_count = 1
    intervals = []
    stop_reason, cleanup_errors = "requested_stop", []
    watchdog = None
    anchor, period = None, 1.0 / frequency
    try:
        video = VideoDisplay(video_host or host, video_port, video_timeout,
                              renderer=display_callback).start()
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.connect((host, port))
        client.setblocking(False)
        if os.name == "nt" and hasattr(socket, "SIO_UDP_CONNRESET"):
            client.ioctl(socket.SIO_UDP_CONNRESET, False)
        device = DHDDevice(library_path, device_index).open()
        watchdog = device.configureWatchdog(maximum_hold_seconds, watchdog_unit_seconds)
        device.enableForce(True)
        deadline = time.monotonic()
        while ((maximum_samples is None or sequence < maximum_samples)
               and (stop_event is None or not stop_event.is_set())):
            while True:
                try:
                    raw = client.recv(8192)
                except BlockingIOError:
                    break
                except OSError as exc:
                    if not _transient_udp_error(exc):
                        raise
                    network_available = False
                    pending_feedback.clear()
                    break
                try:
                    reply = json.loads(raw.decode("utf-8"))
                    received_at = time.monotonic()
                    if not isinstance(reply, dict) or not isinstance(reply.get("sequence"), int):
                        raise ValueError("Invalid robot feedback reply")
                    command_time = sent_times.get(reply["sequence"])
                    if command_time is None or not 0 <= received_at - command_time <= feedback_timeout:
                        raise ValueError("Feedback reply is older than the local freshness limit")
                    network_available = True
                    energy = reply.get("energy_packet")
                    if energy is not None:
                        if not isinstance(energy, dict) or not isinstance(energy.get("sender_session"), str):
                            raise ValueError("Invalid robot acquisition epoch")
                        sender = energy["sender_session"]
                        if peer_epoch is None:
                            observer.accept_packet(energy)
                            peer_epoch = sender
                            epochs.append({"peer_session": sender, "master_session": session_id,
                                           "started_monotonic": received_at})
                        elif sender != peer_epoch:
                            if pending_epoch is not None and sender != pending_epoch["sender_session"]:
                                raise RuntimeError("Multiple robot acquisition epochs arrived before disarming")
                            pending_epoch = energy
                        elif pending_epoch is None:
                            observer.accept_packet(energy)
                    if reply.get("feedback") is None:
                        continue
                    frame, force = _feedback(reply, None if pending_epoch is not None else last_udp_sequence,
                                               sequence)
                except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                    rejected += 1
                    continue
                if pending_epoch is not None:
                    continue
                last_udp_sequence = frame["observation_sequence"]
                pending_feedback[last_udp_sequence] = (frame, force, received_at)
                if len(pending_feedback) > 512:
                    del pending_feedback[next(iter(pending_feedback))]
            position, velocity, reported_force = device.getPosition(), device.getLinearVelocity(), device.getForce()
            native_state = device.getOptionalState()
            device_status = device.getStatus()
            native_dofs_observed.add(native_state["native_dofs_observed"])
            deadman, grip = device.getButton(deadman_button), device.getButton(grip_button)
            now = time.monotonic()
            actual_dt = period if previous_sample is None else now - previous_sample
            if previous_sample is not None:
                intervals.append(actual_dt)
            previous_sample = now
            world_position, world_velocity, world_reported_force = rotation @ position, rotation @ velocity, rotation @ reported_force
            observer.begin_sample(now, world_reported_force, world_position, world_velocity)
            if not deadman:
                deadman_released = True
            if pending_epoch is not None:
                previous_enabled = False
                observation_sequence = None
                if not deadman:
                    observer.limit_force(np.zeros(3), world_velocity, actual_dt)
                    device.setForce(np.zeros(3))
                    recorder.append({"event": "acquisition_epoch_closed", "host_monotonic": now,
                                     "peer_session": peer_epoch, "master_session": session_id,
                                     "passivity": observer.diagnostics()})
                    session_id = uuid.uuid4().hex
                    observer = MasterPassivity(config or {}, session_id)
                    precharge_count += 1
                    observer.accept_packet(pending_epoch)
                    peer_epoch = pending_epoch["sender_session"]
                    pending_epoch = None
                    observer.begin_sample(now, world_reported_force, world_position, world_velocity)
                    pending_feedback.clear()
                    last_udp_sequence, last_displayed_sequence = None, None
                    anchor = position.copy()
                    epochs.append({"peer_session": peer_epoch, "master_session": session_id,
                                   "started_monotonic": now})
            presented, feedback_force, fresh = None, np.zeros(3), False
            try:
                presented = video.latest()
                metadata = presented.metadata
                observation_sequence = metadata["observation_sequence"]
                feedback = pending_feedback.get(observation_sequence)
                if feedback is not None:
                    frame, force, received_at = feedback
                    matched = (frame.get("video_sequence") == metadata["video_sequence"]
                               and frame.get("video_session_id") == metadata["video_session_id"]
                               and np.allclose(frame["position_m"], metadata["position_m"], atol=1e-9, rtol=0)
                               and np.allclose(frame["velocity_m_s"], metadata["velocity_m_s"], atol=1e-9, rtol=0)
                               and frame["host_source_time_s"] == metadata["host_source_time_s"])
                    fresh = bool(matched and 0 <= now - received_at <= feedback_timeout)
                    if fresh:
                        feedback_force = clamp_norm(force, maxfeedback)
                if observation_sequence != last_displayed_sequence:
                    displayed += 1
                    last_displayed_sequence = observation_sequence
            except CameraUnavailable:
                observation_sequence = None
            enabled = bool(deadman and fresh and deadman_released and pending_epoch is None
                           and network_available and device_status["force_enabled"])
            if anchor is None or not enabled or not previous_enabled:
                anchor = position.copy()
            command = clamp_norm(stiffness * (rotation @ (position - anchor)), maxcommand) if enabled else np.zeros(3)
            requested_reflection = feedback_force if enabled else np.zeros(3)
            reflected_world = observer.limit_force(requested_reflection, world_velocity, actual_dt)
            device.setForce(rotation.T @ reflected_world)
            passivity_diagnostics = observer.diagnostics()
            packet = {"sequence": sequence, "device_timestamp": now, "frame": "base",
                      "force_N": command.tolist(), "velocity_m_s": world_velocity.tolist(),
                      "enable": enabled, "grip": bool(grip and enabled),
                      "observation_sequence": observation_sequence if fresh else None,
                      "video_sequence": presented.metadata["video_sequence"] if fresh else None,
                      "video_session_id": presented.metadata["video_session_id"] if fresh else None,
                      "video_received_monotonic": presented.received_monotonic if fresh else None,
                      "video_presented_monotonic": presented.presented_monotonic if fresh else None,
                      "video_camera_host_source_monotonic": presented.metadata["camera_host_receive_monotonic"] if fresh else None,
                      "device_reported_force_N": world_reported_force.tolist(),
                      "native_state": native_state,
                      "sample_interval_seconds": actual_dt, "energy_packet": observer.packet(),
                      "robot_session_ack": peer_epoch if pending_epoch is None else None,
                      "rho": observer.rho,
                      "rho_window_removed_work_j": passivity_diagnostics["rho_window_removed_work_j"],
                      "rho_window_nominal_positive_work_j": passivity_diagnostics["rho_window_nominal_positive_work_j"],
                      "command_source": "virtual_spring_from_measured_master_displacement",
                      "command_stiffness_N_m": stiffness}
            sent = False
            try:
                client.send(json.dumps(packet, allow_nan=False).encode("utf-8"))
                sent = True
            except OSError as exc:
                if not _transient_udp_error(exc):
                    raise
                network_available, enabled = False, False
                device.setForce(np.zeros(3))
                reflected_world = np.zeros(3)
                packet.update(enable=False, force_N=[0., 0., 0.], grip=False,
                              observation_sequence=None, video_sequence=None, video_session_id=None)
            recorder.append({"event": "sample", **packet, "position_m": world_position,
                             "command_packet_sent": sent,
                             "requested_reflected_force_N": requested_reflection,
                             "applied_reflected_force_N": reflected_world,
                             "passivity": passivity_diagnostics, "device_status": device_status,
                             "force_measurement_source": "DHD dhdGetForce applied-force telemetry"})
            if sent:
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
    except BaseException as exc:
        stop_reason = f"exception_{type(exc).__name__}"
        raise
    finally:
        unwinding = sys.exc_info()[0] is not None
        if client is not None:
            try:
                client.send(json.dumps({"sequence": sequence, "device_timestamp": time.monotonic(),
                                        "frame": "base", "force_N": [0.0] * 3,
                                        "velocity_m_s": [0.0] * 3, "enable": False,
                                        "grip": False, "observation_sequence": None,
                                        "video_sequence": None}).encode("utf-8"))
            except OSError:
                pass
        for name, resource in (("DHD", device), ("video", video), ("UDP", client), ("recording", recorder)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    cleanup_errors.append({"resource": name, "error_type": type(exc).__name__})
        summary = {"samples_sent": sequence, "feedback_frames_displayed": displayed,
            "feedback_packets_rejected": rejected, "requested_frequency_hz": frequency,
            "clock": "local_monotonic; no hard-real-time scheduling guarantee",
            "stop_reason": stop_reason, "cleanup_errors": cleanup_errors,
            "actual_sample_interval_min_s": min(intervals) if intervals else None,
            "actual_sample_interval_max_s": max(intervals) if intervals else None,
            "actual_sample_interval_mean_s": float(np.mean(intervals)) if intervals else None,
            "command_source": "virtual_spring_from_measured_master_displacement",
            "master_velocity_source": "DHD measured linear velocity",
            "force_measurement_source": "DHD dhdGetForce applied-force telemetry, not an external human-force sensor",
            "native_dofs_observed": sorted(native_dofs_observed),
            "native_watchdog": watchdog,
            "policy_coordinates": "three_base_frame_translations_with_robot_orientation_held",
            "rotation_sha256": hashlib.sha256(Path(rotation_path).read_bytes()).hexdigest(),
            "session_id": session_id, "passivity": observer.diagnostics(),
            "acquisition_epochs": epochs, "declared_precharge_count": precharge_count,
            "record_directory": str(directory.resolve()), "robot_motion_commands_sent_by_bridge": 0}
        write_json(directory / "metadata.json", summary)
        if cleanup_errors and not unwinding:
            raise RuntimeError("Physical master shutdown or recording flush failed")
    return summary
