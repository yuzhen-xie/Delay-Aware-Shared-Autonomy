from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dasa.config import write_json
from dasa.geometry import Scene, make_scene, waypoints


SCENARIOS = ("demonstrated", "novel", "wider", "extra_obstacles")
SCHEMA_VERSION = 1
GENERATOR_VERSION = "constrained_translation_v1"


@dataclass(frozen=True)
class SceneConstraints:


    object_size_m: tuple[float, float, float] = (0.04, 0.04, 0.06)
    separation_margin_m: float = 0.005
    maximum_base_radius_m: float = 1.1
    check_nominal_paths: bool = True
    maximum_attempts: int = 1000

    def validate(self) -> None:
        size = np.asarray(self.object_size_m, float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Object dimensions must contain three positive finite lengths")
        if not np.isfinite([self.separation_margin_m, self.maximum_base_radius_m]).all():
            raise ValueError("Scene constraint lengths must be finite")
        if self.separation_margin_m < 0 or self.maximum_base_radius_m <= 0:
            raise ValueError("Invalid scene separation or base-radius constraint")
        if not isinstance(self.maximum_attempts, int) or self.maximum_attempts < 1:
            raise ValueError("At least one scene sampling attempt is required")


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _default_constraints(config: dict) -> SceneConstraints:
    return SceneConstraints(object_size_m=tuple(config["scene"].get(
        "object_size_m", (0.04, 0.04, 0.06))))


def validate_scene(scene: Scene, config: dict,
                   constraints: SceneConstraints | None = None) -> list[str]:

    constraints = constraints or _default_constraints(config)
    constraints.validate()
    failures = []
    g, o, r = scene.goals, scene.centers, scene.radii
    if (g.shape != (3, 3) or scene.start.shape != (3,) or scene.place.shape != (3,)
            or o.ndim != 2 or o.shape[1:] != (3,) or r.shape != (len(o),)
            or len(scene.obstacle_ids) != len(o)):
        return ["geometry_shape"]
    if not all(np.isfinite(x).all() for x in (g, o, r, scene.start, scene.place)):
        return ["nonfinite_geometry"]
    if scene.frame != "base":
        failures.append("frame_not_base")
    if len(set(scene.obstacle_ids)) != len(scene.obstacle_ids) or np.any(r <= 0):
        failures.append("invalid_obstacles")
    sim = config["simulation"]
    lower, upper = np.asarray(sim["workspace_min"]), np.asarray(sim["workspace_max"])
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("Workspace bounds must be ordered 3D vectors")
    object_radius = float(np.linalg.norm(constraints.object_size_m) / 2)
    margin = constraints.separation_margin_m
    tcp_radius = float(sim["robot_radius"])
    task_points = np.vstack([g, scene.start, scene.place])
    task_radii = np.array([max(object_radius, tcp_radius)] * 3
                         + [tcp_radius, max(object_radius, tcp_radius)])
    if np.any(task_points - task_radii[:, None] < lower) or np.any(
            task_points + task_radii[:, None] > upper):
        failures.append("task_envelope_outside_workspace")
    if len(o) and (np.any(o - r[:, None] < lower) or np.any(o + r[:, None] > upper)):
        failures.append("obstacle_outside_workspace")
    if np.any(np.linalg.norm(task_points, axis=1) > constraints.maximum_base_radius_m):
        failures.append("outside_declared_base_radius")
    separation = np.linalg.norm(g[:, None] - g[None], axis=-1)
    if np.any(separation[np.triu_indices(3, k=1)] < 2 * object_radius + margin):
        failures.append("overlapping_targets")
    if len(o):
        distances = np.linalg.norm(task_points[:, None] - o[None], axis=-1)
        if np.any(distances <= task_radii[:, None] + r[None] + margin):
            failures.append("task_obstacle_intersection")
        obstacle_distances = np.linalg.norm(o[:, None] - o[None], axis=-1)
        pairs = np.triu_indices(len(o), k=1)
        if np.any(obstacle_distances[pairs] <= (r[:, None] + r[None] + margin)[pairs]):
            failures.append("overlapping_obstacles")
    if constraints.check_nominal_paths and len(o):
        for goal in range(3):
            path = waypoints(scene, goal, sim["lift_height"])
            if np.any(path - tcp_radius < lower) or np.any(path + tcp_radius > upper):
                failures.append(f"nominal_path_{goal}_outside_workspace")
            if any(scene.swept_clearance(a, b, tcp_radius) <= margin
                   for a, b in zip(path[:-1], path[1:])):
                failures.append(f"nominal_path_{goal}_blocked")
    return failures


def generate_manifest(config: dict, count: int, scenario: str = "demonstrated",
                      seed: int | None = None,
                      constraints: SceneConstraints | None = None) -> dict:

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scene scenario: {scenario}")
    if not isinstance(count, int) or count < 1:
        raise ValueError("A manifest needs at least one scene")
    constraints = constraints or _default_constraints(config)
    constraints.validate()
    seed = int(config["seed"] if seed is None else seed)
    if seed < 0:
        raise ValueError("Scene seed must be nonnegative")
    entries = []
    for index in range(count):
        attempts = []
        for attempt in range(constraints.maximum_attempts):
            child_seed = int(np.random.SeedSequence([seed, index, attempt]).generate_state(
                1, dtype=np.uint64)[0])
            scene = make_scene(config, np.random.default_rng(child_seed), scenario)
            failures = validate_scene(scene, config, constraints)
            attempts.append({"attempt": attempt, "seed": child_seed, "failures": failures})
            if not failures:
                entries.append({"scene_id": f"{scenario}_{index:05d}", "index": index,
                                "accepted_attempt": attempt, "sampling_seed": child_seed,
                                "attempts": attempts, "scene": scene.to_dict()})
                break
        else:
            raise ValueError(f"No feasible scene at index {index} after "
                             f"{constraints.maximum_attempts} attempts: {attempts[-1]['failures']}")
    generation_config = {"scene": config["scene"], "simulation": {
        key: config["simulation"][key] for key in (
            "workspace_min", "workspace_max", "robot_radius", "lift_height")}}
    constraint_values = asdict(constraints)
    constraint_values["object_size_m"] = list(constraints.object_size_m)
    result = {"schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
              "numpy_version": np.__version__, "master_seed": seed, "scenario": scenario,
              "coordinate_frame": "base", "length_unit": "metre",
              "geometry_status": "declared_benchmark_geometry_not_hardware_measurements",
              "object_envelope": "axis_aligned_box_centered_on_candidate_grasp_position",
              "constraints": constraint_values, "generation_config": generation_config,
              "generation_config_sha256": _digest(generation_config), "scenes": entries}
    result["content_sha256"] = _digest(result)
    return result


def write_scene_manifest(config: dict, output: str | Path, count: int,
                         scenario: str = "demonstrated", seed: int | None = None,
                         constraints: SceneConstraints | None = None) -> dict:
    result = generate_manifest(config, count, scenario, seed, constraints)
    write_json(output, result)
    return result


def load_scene_manifest(path: str | Path) -> dict:

    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if (result.get("schema_version") != SCHEMA_VERSION
            or result.get("generator_version") != GENERATOR_VERSION):
        raise ValueError("Unsupported scene manifest version")
    digest = result.get("content_sha256")
    payload = {key: value for key, value in result.items() if key != "content_sha256"}
    if digest != _digest(payload):
        raise ValueError("Scene manifest integrity check failed")
    config = result["generation_config"]
    if result["generation_config_sha256"] != _digest(config):
        raise ValueError("Scene generation config integrity check failed")
    constraints = SceneConstraints(**result["constraints"])
    ids = []
    for entry in result["scenes"]:
        ids.append(entry["scene_id"])
        failures = validate_scene(Scene.from_dict(entry["scene"]), config, constraints)
        if failures:
            raise ValueError(f"Invalid scene {entry['scene_id']}: {failures}")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Scene manifest identifiers must be nonempty and unique")
    return result
