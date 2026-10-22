import json
import hashlib
from pathlib import Path

import numpy as np

from dasa.config import write_json


def rigid_registration(source: np.ndarray, target: np.ndarray):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("At least three matching 3D landmarks are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Calibration landmarks must be finite")
    centered_source = source - source.mean(0)
    centered_target = target - target.mean(0)
    if min(np.linalg.matrix_rank(centered_source), np.linalg.matrix_rank(centered_target)) < 2:
        raise ValueError("Collinear landmarks do not determine a rigid transform")
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    translation = target.mean(0) - rotation @ source.mean(0)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return transform, residual


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def transform_wrench(wrench, transform):

    force = transform[:3, :3] @ np.asarray(wrench)[:3]
    torque = transform[:3, :3] @ np.asarray(wrench)[3:] + np.cross(transform[:3, 3], force)
    return np.r_[force, torque]


def calibration_identifier(record: dict) -> str:
    payload = {key: record[key] for key in ("source_frame", "target_frame", "transform")}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    return record.get("calibration_id", f"rigid_{digest[:20]}")


def load_calibration(path: str | Path, maximum_rms_m: float | None = None) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    transform = np.asarray(record.get("transform"), float)
    if (not isinstance(record.get("source_frame"), str) or not record["source_frame"]
            or record.get("target_frame") != "base" or transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Calibration must contain a proper camera-to-base rigid transform")
    rms = record.get("rms_m")
    if not isinstance(rms, (int, float)) or not np.isfinite(rms) or rms < 0:
        raise ValueError("Calibration RMS must be a finite nonnegative metric residual")
    if maximum_rms_m is not None and (not np.isfinite(maximum_rms_m)
            or maximum_rms_m <= 0 or rms > maximum_rms_m):
        raise ValueError("Calibration RMS exceeds the configured tolerance")
    residual = np.asarray(record.get("residuals_m", []), float)
    if residual.ndim != 1 or not np.isfinite(residual).all() or np.any(residual < 0):
        raise ValueError("Calibration residuals must be finite nonnegative lengths")
    record["calibration_id"] = calibration_identifier(record)
    return record


def calibrate_file(landmarks: Path, output: Path):
    data = json.loads(landmarks.read_text())
    transform, residual = rigid_registration(data["camera_points_m"], data["base_points_m"])
    tolerance = data.get("maximum_residual_m", 0.005)
    if not np.isfinite(tolerance) or tolerance <= 0 or residual.max() > tolerance:
        raise ValueError("Landmark residual exceeds the declared calibration tolerance")
    if not isinstance(data["camera_frame"], str) or not data["camera_frame"]:
        raise ValueError("A camera source frame is required")
    record = {"source_frame": data["camera_frame"], "target_frame": "base",
              "transform": transform.tolist(), "residuals_m": residual.tolist(),
              "rms_m": float(np.sqrt(np.mean(residual**2))), "method": "SVD_rigid_registration",
              "landmarks_sha256": hashlib.sha256(landmarks.read_bytes()).hexdigest(),
              "maximum_residual_m": tolerance}
    record["calibration_id"] = calibration_identifier(record)
    write_json(output, record)
    return record
