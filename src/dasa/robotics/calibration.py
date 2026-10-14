"""Rigid registration of camera/task landmarks into the robot base frame."""

import json
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
    if np.linalg.matrix_rank(centered_source) < 2:
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
    """Spatial force transform including the moment introduced by frame translation."""
    force = transform[:3, :3] @ np.asarray(wrench)[:3]
    torque = transform[:3, :3] @ np.asarray(wrench)[3:] + np.cross(transform[:3, 3], force)
    return np.r_[force, torque]


def calibrate_file(landmarks: Path, output: Path):
    data = json.loads(landmarks.read_text())
    transform, residual = rigid_registration(data["camera_points_m"], data["base_points_m"])
    if residual.max() > data.get("maximum_residual_m", 0.005):
        raise ValueError("Landmark residual exceeds the declared calibration tolerance")
    write_json(output, {"source_frame": data["camera_frame"], "target_frame": "base",
                        "transform": transform, "residuals_m": residual,
                        "rms_m": np.sqrt(np.mean(residual**2)), "method": "SVD_rigid_registration"})
