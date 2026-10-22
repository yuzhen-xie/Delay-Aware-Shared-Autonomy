from __future__ import annotations

import json
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.optimize import least_squares

from dasa.config import write_json
from dasa.geometry import Scene
from dasa.robotics.calibration import load_calibration, transform_points


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        if (self.width < 1 or self.height < 1 or self.fx <= 0 or self.fy <= 0
                or not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all()):
            raise ValueError("Invalid pinhole camera intrinsics")


@dataclass(frozen=True)
class ColorObject:
    object_id: str
    role: str
    rgb_lower: tuple[int, int, int]
    rgb_upper: tuple[int, int, int]
    geometry: str = "marker"
    radius_m: float = 0.0
    center_offset_camera_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        low, high = np.asarray(self.rgb_lower), np.asarray(self.rgb_upper)
        if (not self.object_id or self.role not in {"goal", "obstacle"}
                or low.shape != (3,) or high.shape != (3,) or np.any(low < 0)
                or np.any(high > 255) or np.any(low > high)):
            raise ValueError("Invalid color-object specification")
        if self.geometry not in {"marker", "sphere"}:
            raise ValueError("Object geometry must be marker or sphere")
        if (not np.isfinite(self.radius_m) or self.radius_m < 0
                or (self.role == "obstacle" and self.radius_m <= 0)
                or (self.geometry == "sphere" and self.radius_m <= 0)):
            raise ValueError("A sphere or obstacle requires a positive known radius")
        offset = np.asarray(self.center_offset_camera_m, float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Center offset must be a finite 3D vector")
        if self.geometry == "sphere" and np.any(offset):
            raise ValueError("A fitted sphere center cannot also use a marker offset")


def validate_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("A finite 4x4 camera-to-base transform is required")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError("Camera-to-base transform must be a proper rigid transform")
    return transform


def deproject(pixels_uv: np.ndarray, depths_m: np.ndarray,
              intrinsics: CameraIntrinsics) -> np.ndarray:

    intrinsics.validate()
    pixels, depths = np.asarray(pixels_uv, float), np.asarray(depths_m, float)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or depths.shape != (len(pixels),)
            or not np.isfinite(pixels).all() or not np.isfinite(depths).all()
            or np.any(depths <= 0)):
        raise ValueError("Deprojection needs finite pixel pairs and positive metric depths")
    if (np.any(pixels < 0) or np.any(pixels[:, 0] > intrinsics.width - 1)
            or np.any(pixels[:, 1] > intrinsics.height - 1)):
        raise ValueError("Deprojection pixels must lie inside the calibrated image")
    return np.c_[(pixels[:, 0] - intrinsics.cx) * depths / intrinsics.fx,
                 (pixels[:, 1] - intrinsics.cy) * depths / intrinsics.fy, depths]


def _sphere_center(points: np.ndarray, radius: float,
                   maximum_residual_m: float) -> tuple[np.ndarray, float]:
    initial = np.median(points, axis=0) + np.array([0, 0, radius])
    fit = least_squares(lambda center: np.linalg.norm(points - center, axis=1) - radius,
                        initial, loss="soft_l1", f_scale=maximum_residual_m,
                        max_nfev=200)
    residual = np.linalg.norm(points - fit.x, axis=1) - radius
    rms = float(np.sqrt(np.mean(residual**2)))
    if (not fit.success or np.linalg.matrix_rank(fit.jac, tol=1e-5) < 3
            or rms > maximum_residual_m or fit.x[2] <= np.median(points[:, 2])):
        raise ValueError("Visible sphere depth does not support a reliable center fit")
    return fit.x, rms


def detect_scene(rgb: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics,
                 camera_to_base: np.ndarray, objects: list[ColorObject], *,
                 timestamp_s: float, sequence: int, calibration_id: str,
                 camera_frame: str = "camera_optical", update_mode: str = "snapshot",
                 depth_scale_m: float = 0.001, minimum_pixels: int = 30,
                 minimum_depth_m: float = 0.1, maximum_depth_m: float = 3.0,
                 maximum_sphere_residual_m: float = 0.005) -> dict:





    intrinsics.validate()
    transform = validate_transform(camera_to_base)
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    shape = (intrinsics.height, intrinsics.width)
    if rgb.shape != (*shape, 3) or depth.shape != shape:
        raise ValueError("RGB and registered depth sizes must match the camera intrinsics")
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 255):
        raise ValueError("RGB must contain finite values on the 0..255 scale")
    if (not np.isfinite(timestamp_s) or timestamp_s < 0 or not isinstance(sequence, int)
            or sequence < 0 or not calibration_id or not camera_frame
            or update_mode not in {"snapshot", "online"}):
        raise ValueError("Perception requires timestamps, calibration and update provenance")
    if (minimum_pixels < 3 or not np.isfinite(
            [depth_scale_m, minimum_depth_m, maximum_depth_m,
             maximum_sphere_residual_m]).all() or depth_scale_m <= 0
            or minimum_depth_m <= 0 or maximum_depth_m <= minimum_depth_m
            or maximum_sphere_residual_m <= 0):
        raise ValueError("Invalid depth or segmentation thresholds")
    if not objects or len({obj.object_id for obj in objects}) != len(objects):
        raise ValueError("A nonempty list of uniquely identified objects is required")
    depth_m = depth.astype(float) * depth_scale_m
    detections = []
    used_mask = np.zeros(shape, dtype=bool)
    for obj in objects:
        obj.validate()
        mask = np.all((rgb >= np.asarray(obj.rgb_lower))
                      & (rgb <= np.asarray(obj.rgb_upper)), axis=-1)
        components, component_count = label(mask)
        if component_count == 0:
            raise ValueError(f"No color component detected for {obj.object_id}")
        counts = np.bincount(components.ravel())
        counts[0] = 0
        largest = int(np.argmax(counts))
        mask = components == largest
        if np.any(used_mask & mask):
            raise ValueError("Object color intervals produce overlapping detections")
        used_mask |= mask
        mask &= (np.isfinite(depth_m) & (depth_m >= minimum_depth_m)
                 & (depth_m <= maximum_depth_m))
        v, u = np.nonzero(mask)
        if len(u) < minimum_pixels:
            raise ValueError(f"Insufficient valid depth for {obj.object_id}")
        values = depth_m[v, u]
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        keep = np.abs(values - median) <= max(0.01, 3 * 1.4826 * mad)
        u, v, values = u[keep], v[keep], values[keep]
        if len(u) < minimum_pixels:
            raise ValueError(f"Insufficient robust depth for {obj.object_id}")
        points = deproject(np.c_[u, v], values, intrinsics)
        surface = deproject(np.array([[np.median(u), np.median(v)]]),
                            np.array([np.median(values)]), intrinsics)[0]
        if obj.geometry == "sphere":
            center, residual = _sphere_center(points, obj.radius_m, maximum_sphere_residual_m)
        else:
            center = surface + np.asarray(obj.center_offset_camera_m)
            residual = None
        base_center = transform_points(center[None], transform)[0]
        detections.append({"object_id": obj.object_id, "role": obj.role,
                           "geometry": obj.geometry, "center_camera_m": center.tolist(),
                           "center_base_m": base_center.tolist(), "radius_m": obj.radius_m,
                           "surface_point_camera_m": surface.tolist(),
                           "center_offset_camera_m": list(obj.center_offset_camera_m),
                           "valid_depth_pixels": len(points), "sphere_fit_rms_m": residual,
                           "rgb_lower": list(obj.rgb_lower), "rgb_upper": list(obj.rgb_upper)})
    return {"schema_version": 1, "method": "registered_rgbd_color_geometry_v1",
            "provenance": "calibrated_rgbd_geometry", "timestamp_s": float(timestamp_s),
            "sequence": sequence, "update_mode": update_mode, "camera_frame": camera_frame,
            "target_frame": "base", "calibration_id": calibration_id,
            "camera_to_base": transform.tolist(), "intrinsics": asdict(intrinsics),
            "depth_scale_m": depth_scale_m, "minimum_pixels": minimum_pixels,
            "depth_range_m": [minimum_depth_m, maximum_depth_m],
            "maximum_sphere_residual_m": maximum_sphere_residual_m,
            "obstacle_representation": "center_and_known_bounding_sphere_radius",
            "detections": detections}


def snapshot_to_scene(snapshot: dict, goal_ids: list[str], place: list[float],
                      start: list[float], *, maximum_age_s: float | None = None,
                      current_time_s: float | None = None) -> Scene:

    if (snapshot.get("schema_version") != 1 or snapshot.get("target_frame") != "base"
            or snapshot.get("provenance") != "calibrated_rgbd_geometry"):
        raise ValueError("A calibrated base-frame perception snapshot is required")
    if len(goal_ids) != 3 or len(set(goal_ids)) != 3:
        raise ValueError("Exactly three ordered goal identifiers are required")
    if maximum_age_s is not None:
        if (current_time_s is None or maximum_age_s < 0
                or not np.isfinite([current_time_s, maximum_age_s]).all()):
            raise ValueError("A finite current time and nonnegative maximum age are required")
        age = current_time_s - snapshot["timestamp_s"]
        if age < 0 or age > maximum_age_s:
            raise ValueError("Perception snapshot is stale or from a future clock")
    objects = {item["object_id"]: item for item in snapshot["detections"]}
    if len(objects) != len(snapshot["detections"]):
        raise ValueError("Duplicate object identifiers in perception snapshot")
    if any(key not in objects or objects[key]["role"] != "goal" for key in goal_ids):
        raise ValueError("The ordered candidate goals must all be detected")
    obstacles = [item for item in snapshot["detections"] if item["role"] == "obstacle"]
    if not obstacles:
        raise ValueError("This controller requires at least one declared obstacle")
    scene = Scene(np.array([objects[key]["center_base_m"] for key in goal_ids], float),
                  np.asarray(place, float), np.asarray(start, float),
                  np.array([item["center_base_m"] for item in obstacles], float),
                  np.array([item["radius_m"] for item in obstacles], float),
                  [item["object_id"] for item in obstacles], "calibrated_rgbd_geometry", "base")
    if (scene.goals.shape != (3, 3) or scene.start.shape != (3,) or scene.place.shape != (3,)
            or scene.centers.shape != (len(obstacles), 3)
            or not all(np.isfinite(x).all() for x in
                       [scene.goals, scene.start, scene.place, scene.centers, scene.radii])
            or np.any(scene.radii <= 0)):
        raise ValueError("Invalid metric geometry in perception snapshot")
    return scene


class PerceptionStream:


    def __init__(self, goal_ids: list[str], place: list[float], start: list[float], *,
                 maximum_age_s: float = 0.2):
        if len(goal_ids) != 3 or len(set(goal_ids)) != 3:
            raise ValueError("Exactly three ordered goal identifiers are required")
        if not np.isfinite(maximum_age_s) or maximum_age_s < 0:
            raise ValueError("Maximum perception age must be finite and nonnegative")
        self.goal_ids, self.place, self.start = list(goal_ids), list(place), list(start)
        self.maximum_age_s = maximum_age_s
        self._snapshot = None

    def update(self, snapshot: dict, current_time_s: float) -> None:
        if snapshot.get("update_mode") != "online":
            raise ValueError("An online stream requires online perception snapshots")
        snapshot_to_scene(snapshot, self.goal_ids, self.place, self.start,
                          maximum_age_s=self.maximum_age_s, current_time_s=current_time_s)
        if self._snapshot is not None:
            if (snapshot["sequence"] <= self._snapshot["sequence"]
                    or snapshot["timestamp_s"] <= self._snapshot["timestamp_s"]):
                raise ValueError("Perception sequence and acquisition time must both increase")
            if snapshot["calibration_id"] != self._snapshot["calibration_id"]:
                raise ValueError("Changing camera calibration requires a new stream")
        self._snapshot = copy.deepcopy(snapshot)

    def scene(self, current_time_s: float) -> Scene:
        if self._snapshot is None:
            raise ValueError("No complete perception snapshot is available")
        return snapshot_to_scene(self._snapshot, self.goal_ids, self.place, self.start,
                                 maximum_age_s=self.maximum_age_s,
                                 current_time_s=current_time_s)


def perceive_file(rgbd_path: str | Path, specification_path: str | Path,
                  output: str | Path) -> dict:

    specification = json.loads(Path(specification_path).read_text(encoding="utf-8"))
    calibration_record = specification.pop("calibration_record", None)
    calibration_digest = None
    if calibration_record is not None:
        calibration_path = Path(calibration_record)
        if not calibration_path.is_absolute():
            calibration_path = Path(specification_path).resolve().parent / calibration_path
        calibration = load_calibration(calibration_path)
        for name, value in (("calibration_id", calibration["calibration_id"]),
                            ("camera_frame", calibration["source_frame"])):
            if name in specification and specification[name] != value:
                raise ValueError("Perception specification conflicts with its calibration record")
            specification[name] = value
        if "camera_to_base" in specification and not np.allclose(
                specification["camera_to_base"], calibration["transform"], atol=1e-8, rtol=0):
            raise ValueError("Perception transform conflicts with its calibration record")
        specification["camera_to_base"] = calibration["transform"]
        calibration_digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    intrinsics = CameraIntrinsics(**specification.pop("intrinsics"))
    objects = [ColorObject(**item) for item in specification.pop("objects")]
    transform = np.asarray(specification.pop("camera_to_base"), float)
    scene_specification = specification.pop("scene", None)
    with np.load(rgbd_path, allow_pickle=False) as arrays:
        snapshot = detect_scene(arrays["rgb"], arrays["depth"], intrinsics,
                                transform, objects, **specification)
    if scene_specification is not None:
        snapshot["scene"] = snapshot_to_scene(snapshot, **scene_specification).to_dict()
    if calibration_digest is not None:
        snapshot["calibration_record_sha256"] = calibration_digest
    write_json(output, snapshot)
    return snapshot
