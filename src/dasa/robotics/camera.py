from __future__ import annotations

import copy
import hashlib
import json
import socket
import struct
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dasa.config import write_json
from dasa.robotics.calibration import load_calibration
from dasa.robotics.perception import CameraIntrinsics, ColorObject, detect_scene, snapshot_to_scene


class CameraUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    host_receive_monotonic: float
    sensor_timestamp_ms: float
    sensor_clock_domain: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    depth_scale_m: float
    jpeg: bytes
    calibration_id: str
    camera_serial: str
    record_path: str


class RealSenseCamera:
    def __init__(self, calibration_path, record_directory, width=640, height=480,
                 fps=30, serial=None, jpeg_quality=90):
        if (any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (width, height, fps)) or not 1 <= jpeg_quality <= 100):
            raise ValueError("Camera dimensions, rate and JPEG quality must be valid positive integers")
        self.calibration_path = Path(calibration_path).resolve(strict=True)
        self.calibration = load_calibration(self.calibration_path)
        self.record_directory = Path(record_directory)
        self.width, self.height, self.fps = width, height, fps
        self.serial, self.jpeg_quality = serial, jpeg_quality
        self.pipeline, self.align, self.profile = None, None, None
        self.sequence = 0
        self._maps = None
        self._intrinsics = None
        self._lock = threading.Lock()

    def start(self):
        import cv2
        import pyrealsense2 as rs
        with self._lock:
            if self.pipeline is not None:
                raise RuntimeError("The camera is already running")
            if self.record_directory.exists() and any(self.record_directory.iterdir()):
                raise FileExistsError("Select a new directory for immutable physical camera recordings")
            devices = list(rs.context().query_devices())
            if not devices:
                raise CameraUnavailable("No physical RealSense camera is available")
            if self.serial is None:
                if len(devices) != 1:
                    raise ValueError("Select the actual camera serial when multiple cameras are present")
                self.serial = devices[0].get_info(rs.camera_info.serial_number)
            if self.serial not in [device.get_info(rs.camera_info.serial_number) for device in devices]:
                raise CameraUnavailable("The selected physical camera serial is not available")
            configuration = rs.config()
            configuration.enable_device(self.serial)
            configuration.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            configuration.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            pipeline = rs.pipeline()
            try:
                profile = pipeline.start(configuration)
                scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
                if not np.isfinite(scale) or scale <= 0:
                    raise CameraUnavailable("The camera returned an invalid metric depth scale")
                intrinsic = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                if intrinsic.model not in {rs.distortion.none, rs.distortion.brown_conrady}:
                    raise CameraUnavailable("This camera distortion model needs an explicit validated rectification")
                k = np.array([[intrinsic.fx, 0., intrinsic.ppx],
                              [0., intrinsic.fy, intrinsic.ppy], [0., 0., 1.]])
                self._intrinsics = CameraIntrinsics(intrinsic.width, intrinsic.height,
                                                    intrinsic.fx, intrinsic.fy,
                                                    intrinsic.ppx, intrinsic.ppy)
                self._intrinsics.validate()
                if intrinsic.model == rs.distortion.brown_conrady and np.any(intrinsic.coeffs):
                    self._maps = cv2.initUndistortRectifyMap(
                        k, np.asarray(intrinsic.coeffs, float), None, k,
                        (intrinsic.width, intrinsic.height), cv2.CV_32FC1)
                self.depth_scale_m, self.profile = scale, profile
                self.align = rs.align(rs.stream.color)
                self.pipeline = pipeline
                self.record_directory.mkdir(parents=True, exist_ok=True)
                write_json(self.record_directory / "camera.json", {
                    "camera_serial": self.serial, "requested_fps": self.fps,
                    "width": intrinsic.width, "height": intrinsic.height,
                    "depth_scale_m": scale, "distortion_model": str(intrinsic.model),
                    "distortion_coefficients": list(intrinsic.coeffs),
                    "rectification_applied": self._maps is not None,
                    "calibration_id": self.calibration["calibration_id"],
                    "calibration_sha256": hashlib.sha256(self.calibration_path.read_bytes()).hexdigest(),
                    "host_time_reference": "monotonic_at_registered_frame_receive",
                    "sensor_time_reference": "SDK timestamp and reported SDK clock domain"})
            except BaseException:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
                self.pipeline = None
                raise
        return self

    def capture(self, timeout_ms=1000):
        import cv2
        if self.pipeline is None:
            raise CameraUnavailable("The physical camera is not running")
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        received = time.monotonic()
        if not color or not depth:
            raise CameraUnavailable("A complete registered RGB-D frame was not received")
        rgb = np.asanyarray(color.get_data()).copy()
        depth_array = np.asanyarray(depth.get_data()).copy()
        if rgb.shape != (self._intrinsics.height, self._intrinsics.width, 3):
            raise CameraUnavailable("The delivered color dimensions differ from the calibrated profile")
        if depth_array.shape != rgb.shape[:2] or not np.isfinite(rgb).all():
            raise CameraUnavailable("The registered RGB-D frame has invalid dimensions")
        if self._maps is not None:
            rgb = cv2.remap(rgb, *self._maps, interpolation=cv2.INTER_LINEAR)
            depth_array = cv2.remap(depth_array, *self._maps, interpolation=cv2.INTER_NEAREST)
        encoded, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                     [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not encoded:
            raise CameraUnavailable("The actual camera image could not be encoded")
        timestamp = float(color.get_timestamp())
        if not np.isfinite(timestamp):
            raise CameraUnavailable("The camera sensor timestamp is not finite")
        record = self.record_directory / f"frame_{self.sequence:09d}.npz"
        if record.exists():
            raise FileExistsError("Camera recordings are immutable; select a new capture directory")
        metadata = {"sequence": self.sequence, "host_receive_monotonic": received,
                    "sensor_timestamp_ms": timestamp,
                    "sensor_clock_domain": str(color.get_frame_timestamp_domain()),
                    "depth_sensor_timestamp_ms": float(depth.get_timestamp()),
                    "camera_serial": self.serial, "calibration_id": self.calibration["calibration_id"],
                    "depth_scale_m": self.depth_scale_m,
                    "intrinsics": vars(self._intrinsics),
                    "host_time_reference": "monotonic_at_registered_frame_receive"}
        temporary = record.with_name(record.stem + ".tmp.npz")
        np.savez_compressed(temporary, rgb=rgb, depth=depth_array,
                            metadata=np.array(json.dumps(metadata, allow_nan=False)))
        temporary.replace(record)
        result = CameraFrame(self.sequence, received, timestamp, metadata["sensor_clock_domain"],
                             rgb, depth_array, self._intrinsics, self.depth_scale_m, jpeg.tobytes(),
                             self.calibration["calibration_id"], self.serial, str(record.resolve()))
        self.sequence += 1
        return result

    def close(self):
        with self._lock:
            pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            pipeline.stop()


def detect_frame(frame: CameraFrame, calibration: dict, specification: dict, update_mode):
    options = copy.deepcopy(specification)
    objects = [ColorObject(**item) for item in options.pop("objects")]
    task = options.pop("scene")
    for name in ("intrinsics", "camera_to_base", "calibration_record", "calibration_id",
                 "timestamp_s", "sequence", "camera_frame", "update_mode", "depth_scale_m"):
        options.pop(name, None)
    snapshot = detect_scene(frame.rgb, frame.depth, frame.intrinsics,
                             np.asarray(calibration["transform"]), objects,
                             timestamp_s=frame.host_receive_monotonic, sequence=frame.sequence,
                             calibration_id=calibration["calibration_id"],
                             camera_frame=calibration["source_frame"], update_mode=update_mode,
                             depth_scale_m=frame.depth_scale_m, **options)
    snapshot["scene"] = snapshot_to_scene(snapshot, **task).to_dict()
    snapshot["camera_record"] = frame.record_path
    return snapshot


class RGBDStream:
    def __init__(self, camera: RealSenseCamera, perception_path, scene_mode="static_once",
                 history_seconds=5.0):
        if scene_mode not in {"static_once", "online"} or history_seconds <= 0:
            raise ValueError("Select static_once or online scene updates and a positive history duration")
        self.camera = camera
        self.specification = json.loads(Path(perception_path).read_text(encoding="utf-8"))
        self.scene_mode, self.history_seconds = scene_mode, history_seconds
        self._frames, self._snapshot, self._error = deque(), None, None
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("The RGB-D acquisition worker has already started")
        self.camera.start()
        self._thread = threading.Thread(target=self._acquire, name="rgbd-acquisition", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self.camera.close()
            raise
        return self

    def _acquire(self):
        try:
            while not self._stop.is_set():
                frame = self.camera.capture()
                snapshot = None
                if self._snapshot is None or self.scene_mode == "online":
                    snapshot = detect_frame(frame, self.camera.calibration, self.specification,
                                              "online" if self.scene_mode == "online" else "snapshot")
                    write_json(self.camera.record_directory / f"scene_{frame.sequence:09d}.json", snapshot)
                with self._lock:
                    self._frames.append(frame)
                    if snapshot is not None:
                        self._snapshot = snapshot
                    while len(self._frames) > 1 and self._frames[0].host_receive_monotonic < (
                            frame.host_receive_monotonic - self.history_seconds):
                        self._frames.popleft()
        except BaseException as exc:
            with self._lock:
                self._error = exc
        finally:
            try:
                self.camera.close()
            except Exception as exc:
                with self._lock:
                    self._error = self._error or exc

    def latest(self, max_age=.2):
        if not np.isfinite(max_age) or max_age <= 0:
            raise ValueError("Camera freshness must be a finite positive duration")
        with self._lock:
            if self._error is not None:
                raise CameraUnavailable("Physical RGB-D acquisition failed") from self._error
            if not self._frames:
                raise CameraUnavailable("No complete physical RGB-D frame is available")
            frame = self._frames[-1]
        if not 0 <= time.monotonic() - frame.host_receive_monotonic <= max_age:
            raise CameraUnavailable("The physical RGB-D stream is stale")
        return frame

    def at(self, host_source_time, maximum_skew_s=.1, max_age=.2):
        if not np.isfinite([host_source_time, maximum_skew_s]).all() or maximum_skew_s < 0:
            raise ValueError("Camera matching needs a finite source time and nonnegative skew limit")
        self.latest(max_age)
        with self._lock:
            candidates = [frame for frame in self._frames
                          if frame.host_receive_monotonic <= host_source_time]
        if not candidates or host_source_time - candidates[-1].host_receive_monotonic > maximum_skew_s:
            raise CameraUnavailable("No actual camera frame matches the requested robot observation time")
        return candidates[-1]

    def snapshot(self):
        self.latest()
        with self._lock:
            if self._snapshot is None:
                raise CameraUnavailable("No complete measured scene is available")
            return copy.deepcopy(self._snapshot)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.)
            if self._thread.is_alive():
                raise CameraUnavailable("The camera acquisition worker did not stop")


class VideoServer:
    def __init__(self, host="127.0.0.1", port=55002, maximum_pairing_skew_s=.1):
        if not isinstance(port, int) or not 1 <= port <= 65535 or maximum_pairing_skew_s <= 0:
            raise ValueError("Video endpoint and pairing tolerance are invalid")
        self.host, self.port, self.maximum_pairing_skew_s = host, port, maximum_pairing_skew_s
        self.session_id = uuid.uuid4().hex
        self._latest, self._revision, self._error, self._connected = None, 0, None, False
        self._condition, self._stop = threading.Condition(), threading.Event()
        self._listener, self._thread = None, None

    @property
    def connected(self):
        with self._condition:
            return self._connected and not self._stop.is_set()

    def start(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(1)
            listener.settimeout(.2)
        except BaseException:
            listener.close()
            raise
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="rgbd-video-publisher", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            listener.close()
            raise
        return self

    def publish(self, frame, observation_sequence, source_time, host_source_time,
                state6, wrench3, reflected_force3):
        state, wrench, reflected = map(lambda value: np.asarray(value, float),
                                       (state6, wrench3, reflected_force3))
        if (not isinstance(frame, CameraFrame) or state.shape != (6,) or wrench.shape != (3,)
                or reflected.shape != (3,) or not np.isfinite(np.r_[state, wrench, reflected,
                  source_time, host_source_time]).all() or not isinstance(observation_sequence, int)
                or observation_sequence < 0 or source_time < 0):
            raise ValueError("A real camera frame and finite matched robot feedback are required")
        if abs(host_source_time - frame.host_receive_monotonic) > self.maximum_pairing_skew_s:
            raise CameraUnavailable("Camera and robot observation source times do not match")
        metadata = {"schema_version": 1, "video_session_id": self.session_id,
                    "video_sequence": frame.sequence,
                    "observation_sequence": observation_sequence, "frame": "base",
                    "source_time_s": source_time, "host_source_time_s": host_source_time,
                    "camera_host_receive_monotonic": frame.host_receive_monotonic,
                    "camera_sensor_timestamp_ms": frame.sensor_timestamp_ms,
                    "camera_sensor_clock_domain": frame.sensor_clock_domain,
                    "camera_time_reference": "host_registered_frame_receive_not_sensor_exposure_time",
                    "camera_serial": frame.camera_serial, "calibration_id": frame.calibration_id,
                    "camera_record": frame.record_path,
                    "position_m": state[:3].tolist(), "velocity_m_s": state[3:].tolist(),
                    "measured_wrench_N": wrench.tolist(), "reflected_force_N": reflected.tolist(),
                    "width": frame.intrinsics.width, "height": frame.intrinsics.height}
        with self._condition:
            self._latest = (metadata, frame.jpeg)
            self._revision += 1
            self._condition.notify_all()
        return {"video_session_id": self.session_id, "video_sequence": frame.sequence,
                "observation_sequence": observation_sequence,
                "camera_host_receive_monotonic": frame.host_receive_monotonic}

    def _serve(self):
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with client:
                client.settimeout(.2)
                last_revision = -1
                with self._condition:
                    self._connected = True
                try:
                    while not self._stop.is_set():
                        with self._condition:
                            self._condition.wait_for(lambda: self._stop.is_set() or (
                                self._latest is not None and self._revision != last_revision), timeout=.2)
                            if self._stop.is_set():
                                break
                            if self._latest is None or self._revision == last_revision:
                                continue
                            metadata, jpeg = self._latest
                            metadata = dict(metadata)
                            last_revision = self._revision
                        metadata["host_send_time_s"] = time.monotonic()
                        header = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
                        client.sendall(struct.pack("!II", len(header), len(jpeg)) + header + jpeg)
                except (OSError, ValueError) as exc:
                    self._error = exc
                finally:
                    with self._condition:
                        self._connected = False

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=1.)
            if self._thread.is_alive():
                raise CameraUnavailable("The video publisher did not stop")


@dataclass(frozen=True)
class VideoPacket:
    metadata: dict
    jpeg: bytes
    received_monotonic: float


class VideoClient:
    def __init__(self, host="127.0.0.1", port=55002, timeout=.2):
        if not np.isfinite(timeout) or timeout <= 0 or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("The live video connection requires a valid port and timeout")
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.socket.settimeout(timeout)

    def _exact(self, count):
        result = bytearray()
        while len(result) < count:
            chunk = self.socket.recv(min(count - len(result), 65536))
            if not chunk:
                raise ConnectionError("The live camera video stream disconnected")
            result.extend(chunk)
        return bytes(result)

    def receive(self):
        header_size, image_size = struct.unpack("!II", self._exact(8))
        if not 1 <= header_size <= 65536 or not 1 <= image_size <= 20000000:
            raise ValueError("Invalid TCP video frame length")
        metadata = json.loads(self._exact(header_size).decode("utf-8"))
        jpeg = self._exact(image_size)
        for name in ("observation_sequence", "video_sequence"):
            if not isinstance(metadata[name], int) or isinstance(metadata[name], bool) or metadata[name] < 0:
                raise ValueError("Video frame identifiers must be nonnegative integers")
        for name, size in (("position_m", 3), ("velocity_m_s", 3), ("measured_wrench_N", 3),
                           ("reflected_force_N", 3)):
            vector = np.asarray(metadata[name], float)
            if vector.shape != (size,) or not np.isfinite(vector).all():
                raise ValueError("Invalid matched visual feedback signal")
        times = [metadata[name] for name in ("host_source_time_s", "host_send_time_s",
                                             "camera_host_receive_monotonic")]
        if (metadata.get("schema_version") != 1 or metadata.get("frame") != "base"
                or not np.isfinite(times).all() or min(times) < 0
                or times[0] > times[1] or times[2] > times[1]):
            raise ValueError("Video feedback timing and frame provenance are invalid")
        return VideoPacket(metadata, jpeg, time.monotonic())

    def close(self):
        self.socket.close()


@dataclass(frozen=True)
class PresentedVideo:
    metadata: dict
    received_monotonic: float
    presented_monotonic: float


class VideoDisplay:
    def __init__(self, host="127.0.0.1", port=55002, maximum_age_s=.2, renderer=None):
        if not np.isfinite(maximum_age_s) or maximum_age_s <= 0:
            raise ValueError("A positive video freshness deadline is required")
        self.host, self.port, self.maximum_age_s, self.renderer = host, port, maximum_age_s, renderer
        self._latest, self._error, self._thread, self._client = None, None, None, None
        self._lock, self._stop = threading.Lock(), threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._receive_render, name="rgbd-video-display", daemon=True)
        self._thread.start()
        return self

    def _receive_render(self):
        try:
            import cv2
        except ImportError as exc:
            with self._lock:
                self._error = exc
            return
        while not self._stop.is_set():
            try:
                self._client = VideoClient(self.host, self.port, self.maximum_age_s)
                last_observation = -1
                while not self._stop.is_set():
                    packet = self._client.receive()
                    metadata = packet.metadata
                    if metadata["observation_sequence"] <= last_observation:
                        raise ValueError("Presented observation sequence must strictly increase")
                    image = cv2.imdecode(np.frombuffer(packet.jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2] != (metadata["height"], metadata["width"]):
                        raise ValueError("The received live image cannot be decoded at its stated dimensions")
                    if self.renderer is None:
                        cv2.imshow("UR10 live RGB-D feedback", image)
                        if cv2.waitKey(1) & 0xff == 27:
                            self._stop.set()
                            raise CameraUnavailable("The operator closed live camera feedback")
                        if cv2.getWindowProperty("UR10 live RGB-D feedback", cv2.WND_PROP_VISIBLE) < 1:
                            self._stop.set()
                            raise CameraUnavailable("The camera feedback window is not visible")
                    elif self.renderer(image, metadata) is False:
                        raise CameraUnavailable("The actual video renderer did not confirm presentation")
                    presented = time.monotonic()
                    if presented - packet.received_monotonic > self.maximum_age_s:
                        raise CameraUnavailable("The actual camera frame exceeded the rendering deadline")
                    with self._lock:
                        self._latest = PresentedVideo(copy.deepcopy(metadata), packet.received_monotonic, presented)
                        self._error = None
                    last_observation = metadata["observation_sequence"]
            except Exception as exc:
                with self._lock:
                    self._latest, self._error = None, exc
            finally:
                if self._client is not None:
                    self._client.close()
                    self._client = None
            self._stop.wait(.1)
        if self.renderer is None:
            cv2.destroyAllWindows()

    def latest(self):
        with self._lock:
            frame = self._latest
            error = self._error
        if frame is None or not 0 <= time.monotonic() - frame.received_monotonic <= self.maximum_age_s:
            raise CameraUnavailable("No fresh actually presented RGB-D image is available") from error
        return frame

    def close(self):
        self._stop.set()
        client = self._client
        if client is not None:
            try:
                client.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.)
            if self._thread.is_alive():
                raise CameraUnavailable("The video rendering worker did not stop")
