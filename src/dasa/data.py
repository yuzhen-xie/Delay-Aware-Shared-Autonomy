"""Versioned demonstrations, atomic recording and leakage-resistant partitions."""

from __future__ import annotations

import json
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
        if self.phase.shape != (n,) or not 0 <= self.goal < len(self.scene.goals):
            raise ValueError("Invalid phase labels or intended goal")
        if self.scene.frame != "base":
            raise ValueError("Transform signals to the robot base frame before import")
        for key in ("episode_id", "operator_id", "provenance", "success"):
            if key not in self.metadata:
                raise ValueError(f"Missing metadata field: {key}")

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
                            goal: int, noise: float = 0.15) -> Demonstration:
    """Quintic task-space demonstrations; these are never classified as participant data."""
    rng = np.random.default_rng(seed)
    scene = make_scene(config, rng)
    points = waypoints(scene, goal, config["simulation"]["lift_height"])
    dt = config["simulation"]["decision_dt"]
    chunks, phases = [], []
    for j, (start, end) in enumerate(zip(points[:-1], points[1:])):
        duration = max(0.45, np.linalg.norm(end - start) / rng.uniform(0.16, 0.24))
        count = max(12, int(duration / dt))
        u = np.arange(count) / count
        smooth = 10 * u**3 - 15 * u**4 + 6 * u**5
        path = start + smooth[:, None] * (end - start)
        # Variability vanishes at task anchors and remains bounded between them.
        side = rng.normal(0, 0.008, 3)
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
        splits[split] = paths
    validate_partitions(root, splits)
    write_json(root / "splits.json", splits)
    return splits


def validate_partitions(root: Path, splits: dict) -> None:
    seen_episodes, seen_operators = set(), set()
    for name, paths in splits.items():
        episodes, operators = set(), set()
        for path in paths:
            demo = Demonstration.load(root / path)
            episode = demo.metadata["episode_id"]
            if episode in episodes:
                raise ValueError(f"Duplicate episode in {name}: {episode}")
            episodes.add(episode)
            operators.add(demo.metadata["operator_id"])
        if episodes & seen_episodes or operators & seen_operators:
            raise ValueError("Episode or operator leakage across partitions")
        seen_episodes |= episodes
        seen_operators |= operators


def successful(demo: Demonstration, config: dict) -> tuple[bool, list[str]]:
    reasons = []
    m = demo.metadata
    if not m["success"] or not m.get("placed", False):
        reasons.append("task_incomplete")
    if m.get("grasped_goal") != demo.goal:
        reasons.append("wrong_object")
    if m.get("collision", False) or m.get("bounds_violation", False):
        reasons.append("contact_or_bounds")
    radii = config["simulation"]["robot_radius"]
    if any(demo.scene.swept_clearance(a, b, radii) <= 0
           for a, b in zip(demo.position[:-1], demo.position[1:])):
        reasons.append("swept_collision")
    if np.linalg.norm(demo.position[-1] - demo.scene.place) > config["simulation"]["place_tolerance"]:
        reasons.append("placement_error")
    if demo.time[-1] > config["simulation"]["max_seconds"]:
        reasons.append("timeout")
    return not reasons, reasons


def resample_positions(demo: Demonstration, count: int) -> np.ndarray:
    phase_time = (demo.time - demo.time[0]) / (demo.time[-1] - demo.time[0])
    target = np.linspace(0, 1, count)
    return np.stack([np.interp(target, phase_time, demo.position[:, d]) for d in range(3)], -1)
