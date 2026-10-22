from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dasa.config import write_json
from dasa.geometry import Scene, make_scene, waypoints


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


def synthetic_demonstration(config: dict, seed: int, operator: str, episode: str,
                            goal: int, noise: float | None = None) -> Demonstration:

    rng = np.random.default_rng(seed)
    bootstrap = config.get("bootstrap", {})
    noise = bootstrap.get("force_noise_std_N", 0.15) if noise is None else noise
    scene = make_scene(config, rng)
    points = waypoints(scene, goal, config["simulation"]["lift_height"])
    dt = config["simulation"]["decision_dt"]
    chunks, phases = [], []
    for j, (start, end) in enumerate(zip(points[:-1], points[1:])):
        speed = bootstrap.get("cruise_speed_m_s", [0.16, 0.24])
        duration = max(bootstrap.get("segment_min_seconds", 0.45),
                       np.linalg.norm(end - start) / rng.uniform(*speed))
        count = max(bootstrap.get("minimum_segment_samples", 12), int(duration / dt))
        u = np.arange(count) / count
        smooth = 10 * u**3 - 15 * u**4 + 6 * u**5
        path = start + smooth[:, None] * (end - start)

        side = rng.normal(0, bootstrap.get("lateral_std_m", 0.008), 3)
        path += np.sin(np.pi * u)[:, None] ** 2 * side
        chunks.append(path)
        phases.extend([0 if j <= 2 else (1 if j == 3 else 2)] * count)
    position = np.vstack([*chunks, points[-1][None]])
    phase = np.r_[phases, 3]
    time = np.arange(len(position)) * dt
    velocity = np.gradient(position, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    force = config["simulation"]["mass"] * acceleration + config["simulation"]["damping"] * velocity
    force += rng.normal(0, noise, force.shape)
    metadata = {"episode_id": episode, "operator_id": operator,
                "provenance": "synthetic_quintic_demonstration", "seed": seed,
                "success": True, "collision": False, "bounds_violation": False,
                "grasped_goal": goal, "placed": True, "delay_seconds": 0.0,
                "source_condition": "bootstrap", "noise_std_newton": noise}
    result = Demonstration(time, position, velocity, force,
                           np.c_[position, velocity], phase, scene, goal, metadata)
    result.validate()
    return result


def generate_dataset(config: dict, root: Path) -> dict:
    splits = {}
    inventory = []
    e = config["experiments"]
    count_fields = [("train", "bootstrap_demonstrations"),
                    ("validation", "validation_demonstrations"), ("test", "test_demonstrations")]
    for split_index, (split, field) in enumerate(count_fields):
        paths = []
        for j in range(e[field]):
            seed = config["seed"] + 10000 * split_index + j
            operator = f"synthetic_{split}_{j % 5:02d}"
            episode = f"{split}_{j:04d}"
            path = root / split / f"{episode}.npz"
            demo = synthetic_demonstration(config, seed, operator, episode, j % 3)
            demo.save(path)
            paths.append(str(path.relative_to(root)))
            inventory.append({"path": str(path.relative_to(root)), "split": split,
                              "generation_order": len(inventory), "episode_id": episode,
                              "operator_id": operator, "seed": seed, "goal": j % 3,
                              "samples": len(demo.time), "duration_seconds": float(demo.time[-1]),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "provenance": demo.metadata["provenance"], "scene": demo.scene.to_dict()})
        splits[split] = paths
    validate_partitions(root, splits)
    write_json(root / "splits.json", splits)
    write_json(root / "inventory.json", {"schema_version": 1,
        "generator": "quintic_bootstrap_v1", "generation_order": "train, validation, test; index ascending",
        "split_seed_rule": "seed + 10000*split_index + episode_index",
        "counts": {s: len(p) for s, p in splits.items()},
        "fractions": {s: len(p) / len(inventory) for s, p in splits.items()}, "records": inventory})
    return splits


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


def successful(demo: Demonstration, config: dict) -> tuple[bool, list[str]]:
    demo.validate()
    reasons = []
    m = demo.metadata
    limits = {"maximum_seconds": config["simulation"]["max_seconds"],
              "robot_radius_m": config["simulation"]["robot_radius"],
              "place_tolerance_m": config["simulation"]["place_tolerance"],
              "workspace_min_m": config["simulation"]["workspace_min"],
              "workspace_max_m": config["simulation"]["workspace_max"]}
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
    if any(demo.scene.swept_clearance(a, b, radii) <= 0
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
