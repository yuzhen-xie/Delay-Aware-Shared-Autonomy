from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


from dasa.geometry import Scene


@dataclass
class Demonstration:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    raw_force: np.ndarray
    observed_state: np.ndarray
    phase: np.ndarray
    scene: Scene
    goal: int
    metadata: dict

    def validate(self) -> None:
        n = len(self.time)
        if n < 3 or not np.isfinite(self.time).all() or np.any(np.diff(self.time) <= 0):
            raise ValueError("A demonstration needs at least three increasing timestamps")
        for key, dims in (("position", 3), ("velocity", 3), ("raw_force", 3),
                          ("observed_state", 6)):
            x = getattr(self, key)
            if x.shape != (n, dims) or not np.isfinite(x).all():
                raise ValueError(f"Invalid {key} shape or non-finite data")
        if (self.phase.shape != (n,) or not np.isin(self.phase, [0, 1, 2, 3]).all()
                or not 0 <= self.goal < len(self.scene.goals)):
            raise ValueError("Invalid phase labels or intended goal")
        if self.scene.frame != "base":
            raise ValueError("Transform signals to the robot base frame before import")
        scene = self.scene
        if (scene.goals.shape != (3, 3) or scene.start.shape != (3,) or scene.place.shape != (3,)
                or scene.centers.ndim != 2 or scene.centers.shape[1:] != (3,)
                or scene.radii.shape != (len(scene.centers),) or len(scene.centers) < 1
                or len(scene.obstacle_ids) != len(scene.centers)
                or len(set(scene.obstacle_ids)) != len(scene.centers)
                or np.any(scene.radii <= 0)
                or not all(np.isfinite(x).all() for x in
                           (scene.goals, scene.start, scene.place, scene.centers, scene.radii))):
            raise ValueError("Invalid finite three-goal scene geometry or obstacle radii")
        for key in ("episode_id", "operator_id", "provenance", "success"):
            if key not in self.metadata:
                raise ValueError(f"Missing metadata field: {key}")
        if any(not isinstance(self.metadata[key], str) or not self.metadata[key].strip()
               for key in ("episode_id", "operator_id", "provenance")):
            raise ValueError("Episode, operator and provenance identifiers must be nonempty strings")
        if not isinstance(self.metadata["success"], bool):
            raise ValueError("success must be a JSON boolean")
        if self.metadata["provenance"] != "UR10_measured_teleoperation":
            raise ValueError("Demonstrations must originate from executed, annotated hardware recordings")
        for key in ("source_recording_sha256", "hardware_timing", "execution_contract"):
            if key not in self.metadata:
                raise ValueError(f"Measured demonstration is missing its acquisition link: {key}")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "schema_version": 1,
                    "scene": self.scene.to_dict(), "goal": self.goal}
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, time=self.time, position=self.position,
                            velocity=self.velocity, raw_force=self.raw_force,
                            observed_state=self.observed_state, phase=self.phase,
                            metadata=np.array(json.dumps(metadata)))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Demonstration":
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
            if metadata.pop("schema_version") != 1:
                raise ValueError("Unsupported demonstration schema")
            scene, goal = Scene.from_dict(metadata.pop("scene")), metadata.pop("goal")
            result = cls(**{key: arrays[key].copy() for key in (
                "time", "position", "velocity", "raw_force", "observed_state", "phase")},
                scene=scene, goal=goal, metadata=metadata)
        result.validate()
        return result


def validate_partitions(root: Path, splits: dict) -> None:
    seen_episodes, seen_operators, seen_signals = set(), set(), set()
    for name, paths in splits.items():
        episodes, operators, signals = set(), set(), set()
        for path in paths:
            resolved = (root / path).resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise ValueError("Dataset entry escapes its root")
            demo = Demonstration.load(resolved)
            digest = hashlib.sha256()
            for signal in (demo.time, demo.position, demo.velocity, demo.raw_force, demo.observed_state):
                digest.update(np.asarray(signal, dtype='<f8').tobytes())
            signal_hash = digest.hexdigest()
            if signal_hash in signals or signal_hash in seen_signals:
                raise ValueError("Duplicate signal data or leakage within or across partitions")
            signals.add(signal_hash)
            episode = demo.metadata["episode_id"]
            if episode in episodes:
                raise ValueError(f"Duplicate episode in {name}: {episode}")
            episodes.add(episode)
            operators.add(demo.metadata["operator_id"])
        if episodes & seen_episodes or operators & seen_operators:
            raise ValueError("Episode or operator leakage across partitions")
        seen_episodes |= episodes
        seen_operators |= operators
        seen_signals |= signals


def assemble_dataset(config: dict, training: Path, validation: Path, test: Path, output: Path):
    from dasa.config import write_json

    sources = {"train": Path(training), "validation": Path(validation), "test": Path(test)}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty destination to preserve prior datasets")
    splits, inventory = {}, []
    for name, source in sources.items():
        paths = sorted(source.glob("*.npz"))
        if not paths or (name == "train" and len(paths) != config["experiments"]["bootstrap_demonstrations"]):
            raise ValueError("Supply 50 training recordings and nonempty independent validation/test sets")
        splits[name] = []
        for path in paths:
            demo = Demonstration.load(path)
            if name == "train" and float(demo.metadata["delay_seconds"]) != 0:
                raise ValueError("Initial training demonstrations must have zero configured feedback delay")
            if name == "train":
                accepted, reasons = successful(demo, config)
                if not accepted:
                    raise ValueError(f"Rejected initial demonstration {path.name}: {reasons}")
            relative = Path(name) / path.name
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            splits[name].append(relative.as_posix())
            inventory.append({"file": relative.as_posix(), "split": name,
                              "episode_id": demo.metadata["episode_id"],
                              "operator_id": demo.metadata["operator_id"],
                              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                              "source_recording_sha256": demo.metadata["source_recording_sha256"]})
    validate_partitions(output, splits)
    write_json(output / "splits.json", splits)
    write_json(output / "inventory.json", {"provenance": "UR10_measured_teleoperation",
                                           "partition_unit": "operator", "records": inventory})
    return splits


def successful(demo: Demonstration, config: dict) -> tuple[bool, list[str]]:
    demo.validate()
    reasons = []
    m = demo.metadata
    limits = {"maximum_seconds": config["task"]["max_seconds"],
              "robot_radius_m": config["task"]["robot_radius"],
              "place_tolerance_m": config["task"]["place_tolerance"],
              "workspace_min_m": config["task"]["workspace_min"],
              "workspace_max_m": config["task"]["workspace_max"]}
    if m["provenance"] == "UR10_measured_teleoperation" and "execution_contract" in m:
        recorded = m["execution_contract"]
        if not isinstance(recorded, dict) or any(key not in recorded for key in limits):
            return False, ["invalid_execution_contract"]
        limits = {key: recorded[key] for key in limits}
        try:
            lengths = np.asarray([limits[key] for key in
                                  ("maximum_seconds", "robot_radius_m", "place_tolerance_m")], float)
            lower, upper = np.asarray(limits["workspace_min_m"], float), np.asarray(limits["workspace_max_m"], float)
            valid = (lengths.shape == (3,) and lower.shape == upper.shape == (3,)
                     and np.isfinite(np.r_[lengths, lower, upper]).all()
                     and np.all(lengths > 0) and np.all(lower < upper))
        except (ValueError, TypeError):
            valid = False
        if not valid:
            return False, ["invalid_execution_contract"]
    for field in ("success", "placed", "collision", "bounds_violation"):
        if field not in m:
            reasons.append(f"missing_annotation:{field}")
        elif not isinstance(m[field], bool):
            reasons.append(f"invalid_annotation:{field}")
    if "grasped_goal" not in m:
        reasons.append("missing_annotation:grasped_goal")
    elif m["grasped_goal"] is not None and (
            isinstance(m["grasped_goal"], bool) or not isinstance(m["grasped_goal"], int)
            or not 0 <= m["grasped_goal"] < len(demo.scene.goals)):
        reasons.append("invalid_annotation:grasped_goal")
    if m.get("success") is not True or m.get("placed") is not True:
        reasons.append("task_incomplete")
    if m.get("grasped_goal") != demo.goal:
        reasons.append("wrong_object")
    if m.get("collision") is not False or m.get("bounds_violation") is not False:
        reasons.append("contact_or_bounds")
    if np.any(demo.position < np.asarray(limits["workspace_min_m"])) or np.any(
            demo.position > np.asarray(limits["workspace_max_m"])):
        reasons.append("recorded_position_outside_workspace")
    radii = limits["robot_radius_m"]
    if m.get("hardware_timing", {}).get("scene_mode") == "online":
        minimum = m.get("observed_minimum_surface_clearance_m")
        if minimum is None or not np.isfinite(minimum):
            reasons.append("missing_online_geometry_measurements")
        elif minimum <= 0:
            reasons.append("observed_online_clearance_violation")
    elif any(demo.scene.swept_clearance(a, b, radii) <= 0
             for a, b in zip(demo.position[:-1], demo.position[1:])):
        reasons.append("swept_collision")
    if np.linalg.norm(demo.position[-1] - demo.scene.place) > limits["place_tolerance_m"]:
        reasons.append("placement_error")
    if demo.time[-1] - demo.time[0] > limits["maximum_seconds"]:
        reasons.append("timeout")
    return not reasons, reasons


def bootstrap_admission(demos: list[Demonstration], config: dict) -> list[dict]:
    if not demos:
        raise ValueError("Bootstrap training requires successful complete demonstrations")
    audit = []
    for demo in demos:
        accepted, reasons = successful(demo, config)
        audit.append({"episode_id": demo.metadata["episode_id"],
                      "accepted": accepted, "reasons": reasons})
    rejected = [row for row in audit if not row["accepted"]]
    if rejected:
        raise ValueError("Bootstrap demonstrations failed admission: " + json.dumps(rejected))
    return audit


def resample_positions(demo: Demonstration, count: int) -> np.ndarray:
    phase_time = (demo.time - demo.time[0]) / (demo.time[-1] - demo.time[0])
    target = np.linspace(0, 1, count)
    return np.stack([np.interp(target, phase_time, demo.position[:, d]) for d in range(3)], -1)
