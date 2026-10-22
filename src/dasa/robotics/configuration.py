from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import re
from pathlib import Path

import numpy as np
import yaml

from dasa.config import merge
from dasa.geometry import Scene
from dasa.robotics.calibration import load_calibration
from dasa.robotics.perception import snapshot_to_scene


def _host(host) -> str:
    if not isinstance(host, str) or not host or host != host.strip():
        raise ValueError("An actual robot hostname or IP address is required")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if (len(host) > 253 or not all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?",
                                                 part) for part in host.rstrip(".").split("."))):
            raise ValueError("Invalid robot hostname or IP address") from None
    return host


def default_profile(config: dict) -> dict:
    return {"schema_version": 3, "robot": {"model": "UR10", "mode": "force"},
            "master": {"udp_port": 55001, "servo_seconds": .001},
            "gripper": {"type": "robotiq_socket", "port": 63352, "speed": 64,
                        "force": 64, "timeout_seconds": 1.0},
            "loop_seconds": .008, "maximum_seconds": float(config.get("study", {}).get("max_seconds", 60.)),
            "feedback_delay_seconds": float(config["task"]["feedback_delay"]),
            "camera": {"source": "realsense", "width": 640, "height": 480, "fps": 30,
                       "maximum_age_seconds": .2, "scene_mode": "static_once",
                       "history_seconds": 5., "maximum_pairing_skew_seconds": .1},
            "video": {"host": "127.0.0.1", "port": 55002},
            "force_mode": {"damping": .1, "gain": .5},
            "limits": {"maximum_speed_m_s": .08, "maximum_master_force_N": 12.,
                       "maximum_reflection_force_N": 2., "watchdog_seconds": .10,
                       "feedback_acknowledgement_seconds": .10, "calibration_rms_m": .005,
                       "minimum_jacobian_singular_value": .02, "obstacle_margin_m": .04,
                       "start_tolerance_m": .05,
                       "workspace_min_m": copy.deepcopy(config["task"]["workspace_min"]),
                       "workspace_max_m": copy.deepcopy(config["task"]["workspace_max"])}}


def _scene_geometry(scene: Scene) -> None:
    if (scene.frame != "base" or scene.provenance not in
            {"calibrated_scene", "calibrated_rgbd_geometry"} or scene.goals.shape != (3, 3)
            or scene.start.shape != (3,) or scene.place.shape != (3,)
            or scene.centers.ndim != 2 or scene.centers.shape[1:] != (3,)
            or len(scene.centers) < 1 or scene.radii.shape != (len(scene.centers),)
            or len(scene.obstacle_ids) != len(scene.centers)
            or len(set(scene.obstacle_ids)) != len(scene.obstacle_ids)
            or not all(np.isfinite(x).all() for x in
                       [scene.goals, scene.place, scene.start, scene.centers, scene.radii])
            or np.any(scene.radii <= 0)):
        raise ValueError("A finite measured base-frame scene with three goals and obstacles is required")


def _snapshot_scene(snapshot: dict, calibration: dict) -> Scene:
    if (snapshot.get("calibration_id") != calibration["calibration_id"]
            or snapshot.get("camera_frame") != calibration["source_frame"]
            or not np.allclose(np.asarray(snapshot.get("camera_to_base"), float),
                               calibration["transform"], rtol=0, atol=1e-8)):
        raise ValueError("Perception snapshot and calibration record must identify the same transform")
    if not isinstance(snapshot.get("scene"), dict):
        raise ValueError("The perception snapshot must include ordered task scene geometry")
    scene = Scene.from_dict(snapshot["scene"])
    _scene_geometry(scene)
    goals = [item for item in snapshot.get("detections", []) if item.get("role") == "goal"]
    if len(goals) != 3:
        raise ValueError("A measured snapshot requires three detected candidate goals")
    remaining = goals.copy()
    ordered = []
    for goal in scene.goals:
        matches = [item for item in remaining if np.allclose(item["center_base_m"], goal,
                                                            atol=1e-8, rtol=0)]
        if len(matches) != 1:
            raise ValueError("Scene goal order must identify the detected candidate positions")
        ordered.append(matches[0]["object_id"])
        remaining.remove(matches[0])
    observed = snapshot_to_scene(snapshot, ordered, scene.place.tolist(), scene.start.tolist())
    if (observed.obstacle_ids != scene.obstacle_ids
            or not np.allclose(observed.centers, scene.centers, atol=1e-8, rtol=0)
            or not np.allclose(observed.radii, scene.radii, atol=1e-8, rtol=0)
            or observed.provenance != scene.provenance):
        raise ValueError("Saved task geometry must match its actual perception detections")
    return observed


def validate_hardware(hardware: dict, base: Path | None = None) -> dict:
    data = copy.deepcopy(hardware)
    if not isinstance(data, dict):
        raise ValueError("Hardware configuration must be a mapping")
    base = Path.cwd() if base is None else Path(base)
    try:
        _host(data["robot"]["host"])
        if data["robot"]["model"] not in {"UR10", "UR10e"}:
            raise ValueError("Robot model must be UR10 or UR10e")
        native_period = .008 if data["robot"]["model"] == "UR10" else .002
        if data["robot"].get("mode") != "force" or data["loop_seconds"] < native_period - 1e-9:
            raise ValueError("Force-mode cadence must respect the selected robot's native rate")
        for port in [data["master"]["udp_port"], data["gripper"]["port"]]:
            if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
                raise ValueError("Device ports must be integers in 1..65535")
        if data["gripper"]["type"] not in {"robotiq_socket", "external"}:
            raise ValueError("Unsupported gripper transport")
        for key in ("speed", "force"):
            value = data["gripper"].get(key, 64)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255:
                raise ValueError("Robotiq speed and force must be integers in 0..255")
        limits = data["limits"]
        positive = [data["loop_seconds"], data["maximum_seconds"],
                    data["gripper"].get("timeout_seconds", 1.),
                    *[limits[key] for key in ("maximum_speed_m_s", "maximum_master_force_N",
                       "maximum_reflection_force_N", "watchdog_seconds", "calibration_rms_m",
                       "minimum_jacobian_singular_value")],
                    limits.get("feedback_acknowledgement_seconds", limits["watchdog_seconds"])]
        nonnegative = [data["feedback_delay_seconds"], limits["obstacle_margin_m"],
                       limits.get("start_tolerance_m", .05)]
        if (not np.isfinite(positive + nonnegative).all() or np.any(np.asarray(positive) <= 0)
                or np.any(np.asarray(nonnegative) < 0)
                or data["loop_seconds"] > limits["watchdog_seconds"]):
            raise ValueError("Hardware cadence, delay and physical limits must be finite and ordered")
        lower, upper = (np.asarray(limits[key], float)
                        for key in ("workspace_min_m", "workspace_max_m"))
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.isfinite(np.r_[lower, upper]).all() or np.any(lower >= upper)):
            raise ValueError("Hardware workspace limits must be finite ordered 3D vectors")
        scene = Scene.from_dict(data["scene"])
        _scene_geometry(scene)
        points = np.vstack([scene.goals, scene.start, scene.place])
        if np.any(points < lower) or np.any(points > upper):
            raise ValueError("Measured task coordinates lie outside the configured workspace")
        calibration_path = Path(data["calibration_record"])
        if not calibration_path.is_absolute():
            calibration_path = base / calibration_path
        calibration_path = calibration_path.resolve(strict=True)
        calibration = load_calibration(calibration_path, limits["calibration_rms_m"])
        digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        if data.get("calibration_sha256", digest) != digest:
            raise ValueError("Calibration record changed after hardware configuration")
        data["calibration_record"] = str(calibration_path)
        data["calibration_sha256"] = digest
        if "snapshot_record" in data:
            snapshot_path = Path(data["snapshot_record"])
            if not snapshot_path.is_absolute():
                snapshot_path = base / snapshot_path
            snapshot_path = snapshot_path.resolve(strict=True)
            snapshot_digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            if data.get("snapshot_sha256", snapshot_digest) != snapshot_digest:
                raise ValueError("Scene snapshot changed after hardware configuration")
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            observed = _snapshot_scene(snapshot, calibration)
            if observed.to_dict() != scene.to_dict():
                raise ValueError("Hardware scene differs from its linked perception snapshot")
            data["snapshot_record"] = str(snapshot_path)
            data["snapshot_sha256"] = snapshot_digest
        elif scene.provenance == "calibrated_rgbd_geometry":
            raise ValueError("RGB-D scenes require their linked perception snapshot")
        data["scene"] = scene.to_dict()
        tool = data["tool"]
        pose, cog = np.asarray(tool["tcp_pose"], float), np.asarray(tool["payload_cog_m"], float)
        if (pose.shape != (6,) or cog.shape != (3,) or tool["payload_mass_kg"] < 0
                or not np.isfinite(np.r_[pose, cog, tool["payload_mass_kg"]]).all()):
            raise ValueError("Measured tool TCP and payload must be finite and dimensionally valid")
        camera = data["camera"]
        if (camera["source"] != "realsense" or camera["scene_mode"] not in {"static_once", "online"}
                or not 1 <= data["video"]["port"] <= 65535 or camera["maximum_age_seconds"] <= 0):
            raise ValueError("A live calibrated RGB-D camera and valid video endpoint are required")
        for key in ("perception_record", "tool_record"):
            source = Path(data[key])
            if not source.is_absolute():
                source = base / source
            source = source.resolve(strict=True)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if data.get(key + "_sha256", digest) != digest:
                raise ValueError(f"Changed commissioned input: {key}")
            data[key], data[key + "_sha256"] = str(source), digest
        if json.loads(Path(data["tool_record"]).read_text(encoding="utf-8")) != tool:
            raise ValueError("Tool configuration differs from its measured input record")
        data.setdefault("schema_version", 3)
    except KeyError as exc:
        raise ValueError(f"Missing required hardware input: {exc.args[0]}") from exc
    return data


def load_hardware(path: str | Path) -> dict:
    path = Path(path)
    return validate_hardware(yaml.safe_load(path.read_text(encoding="utf-8")), path.resolve().parent)


def configure(config: dict, host: str, calibration_path: str | Path,
              scene_snapshot_path: str | Path, output: str | Path, model: str = "UR10",
              gripper: str = "robotiq_socket", profile_path: str | Path | None = None,
              perception_path: str | Path | None = None,
              tool_path: str | Path | None = None) -> dict:
    host = _host(host)
    hardware = default_profile(config)
    if profile_path is not None:
        hardware = merge(hardware, yaml.safe_load(Path(profile_path).read_text(encoding="utf-8")))
    calibration_path, snapshot_path = (Path(path).resolve(strict=True)
                                       for path in (calibration_path, scene_snapshot_path))
    calibration = load_calibration(calibration_path, hardware["limits"]["calibration_rms_m"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    scene = _snapshot_scene(snapshot, calibration)
    if perception_path is None or tool_path is None:
        raise ValueError("Supply measured tool and calibrated live-perception specification files")
    perception_path, tool_path = (Path(path).resolve(strict=True)
                                  for path in (perception_path, tool_path))
    hardware["robot"].update(host=_host(host), model=model)
    hardware["loop_seconds"] = max(hardware["loop_seconds"], .008 if model == "UR10" else .002)
    if profile_path is None and model == "UR10e":
        hardware["loop_seconds"] = .002
    hardware["gripper"]["type"] = gripper
    hardware.update(calibration_record=str(calibration_path), snapshot_record=str(snapshot_path),
                    scene=scene.to_dict(), perception_record=str(perception_path),
                    tool_record=str(tool_path), tool=json.loads(tool_path.read_text(encoding="utf-8")))
    hardware["camera"].update(calibration_record=str(calibration_path),
                               perception_record=str(perception_path))
    hardware = validate_hardware(hardware)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(yaml.safe_dump(hardware, sort_keys=False), encoding="utf-8")
    temporary.replace(output)
    return hardware
