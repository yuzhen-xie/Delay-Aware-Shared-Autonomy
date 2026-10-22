from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, override: str | Path | None = None) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if override:
        with open(override, encoding="utf-8") as stream:
            config = merge(config, yaml.safe_load(stream))
    validate(config)
    return config


def validate(c: dict) -> None:
    def finite_tree(value):
        if isinstance(value, dict):
            return all(finite_tree(v) for v in value.values())
        if isinstance(value, list):
            return all(finite_tree(v) for v in value)
        return bool(np.isfinite(value)) if isinstance(value, (float, int)) else True
    if not finite_tree(c):
        raise ValueError("Configuration contains a non-finite numeric value")
    task = c["task"]
    dt, decision = task["control_dt"], task["decision_dt"]
    if dt <= 0 or decision <= 0:
        raise ValueError("Device and decision sampling periods must be positive")
    for field in ("feedback_delay", "command_delay"):
        if task[field] < 0:
            raise ValueError(f"{field} must be nonnegative")
    if c["motion"]["distance_max"] <= c["motion"]["distance_min"]:
        raise ValueError("Collision-distance thresholds must be ordered")
    for section, fields in {
        "task": ("max_seconds", "robot_radius", "force_limit", "grasp_tolerance",
                       "place_tolerance", "dwell_seconds", "max_grasp_speed"),
        "motion": ("repulsive_sigma", "repulsive_limit", "lambda_gain", "kalman_measurement",
                   "kalman_initial_variance"),
        "arbitration": ("epsilon", "roughness_window", "rho_window_seconds", "low_confidence_threshold"),
        "skill": ("states", "trajectory_samples", "covariance_floor", "duration_floor", "em_iterations",
                  "lqt_control_cost", "original_weight", "new_weight"),
        "intent": ("hidden_dim", "latent_dim", "sequence_window", "batch_size", "learning_rate"),
    }.items():
        for field in fields:
            if field in c[section] and c[section][field] <= 0:
                raise ValueError(f"{section}.{field} must be positive")
    if c["arbitration"]["covariance_upper"] <= c["arbitration"]["covariance_lower"]:
        raise ValueError("Confidence covariance bounds must be ordered")
    if c["motion"]["kalman_process"] < 0:
        raise ValueError("Kalman process variance must be nonnegative")
    for field in ("attractive_weight", "repulsive_weight"):
        if c["motion"].get(field, 1.0) < 0:
            raise ValueError(f"{field} must be nonnegative")
    for field in ("fixed_weight", "guarded_corrective_max"):
        if not 0 <= c["arbitration"].get(field, .5) <= 1:
            raise ValueError(f"{field} must lie in [0, 1]")
    lower, upper = np.asarray(task["workspace_min"]), np.asarray(task["workspace_max"])
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("Workspace bounds must be ordered 3-vectors")
    p = c["passivity"]["physical"]
    for port in ("master", "robot", "autonomy"):
        if not 0 <= p[port + "_initial_energy_j"] <= p[port + "_maximum_energy_j"]:
            raise ValueError("Physical port storage bounds must be finite and ordered")
    for port in ("master", "robot"):
        if p[port + "_maximum_hold_seconds"] <= 0 or p[port + "_speed_bound_m_s"] <= 0:
            raise ValueError("Physical port reservations need positive hold and velocity bounds")
        if not 0 <= p[port + "_energy_export_fraction"] <= 1:
            raise ValueError("Exported energy fractions must lie in [0,1]")
    goals = np.asarray(c["scene"]["goals"], dtype=float)
    if goals.shape != (3, 3) or not np.isfinite(goals).all():
        raise ValueError("The task requires three finite 3D candidate goals")
    if c["intent"]["hidden_dim"] % c["intent"]["transformer_heads"]:
        raise ValueError("Transformer dimension must be divisible by its head count")
    if c["intent"]["deviation_objective"] not in {"auxiliary", "joint"}:
        raise ValueError("Unknown deviation loss objective")
    objective = c["intent"].get("objective", "joint_elbo")
    if objective not in {"joint_elbo", "action_elbo", "legacy_weighted_auxiliary"}:
        raise ValueError("Unknown cVAE training objective")
    if objective != "legacy_weighted_auxiliary" and c["intent"].get("kl_weight", 1.0) != 1.0:
        raise ValueError("Standard ELBO requires kl_weight=1; choose an explicit legacy objective otherwise")
    if objective == "joint_elbo" and c["intent"].get("deviation_weight", 1.0) != 1.0:
        raise ValueError("Joint ELBO requires deviation_weight=1")
    if objective == "action_elbo" and c["intent"].get("deviation_weight", 0.0) != 0.0:
        raise ValueError("Action-only ELBO requires deviation_weight=0")
    if not 0 < c["intent"]["confidence_threshold"] < 1:
        raise ValueError("Intent confidence threshold must be in (0, 1)")
    if c["motion"]["risk_distance"] not in {"center", "surface"}:
        raise ValueError("Risk distance must be center or surface")
    if c["skill"]["trajectory_samples"] < c["skill"]["states"]:
        raise ValueError("Each HSMM state needs at least one time sample")
    if c["intent"]["epochs"] < 1:
        raise ValueError("At least one training epoch is required")
    budgets = c["experiments"].get("refinement_budgets", [0, 5, 10, 20, 30])
    if any(not isinstance(b, int) or b < 0 for b in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Refinement budgets must be distinct nonnegative integers")
    study = c.get("study", {})
    if study.get("participants", 20) < 1:
        raise ValueError("Invalid planned study cohort counts")
    if c["experiments"]["bootstrap_demonstrations"] != 50:
        raise ValueError("Initial intent training requires exactly 50 measured demonstrations")
    if c["skill"]["refinement_count"] != 30:
        raise ValueError("The full refinement comparison requires 30 accepted demonstrations")
    if c["intent"].get("latent_inference", "sample") not in {"sample", "mean"}:
        raise ValueError("Unknown latent inference rule")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, default=default, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def manifest(config: dict, command: str, inputs: list[Path] | None = None) -> dict:
    encoded = json.dumps(config, sort_keys=True).encode()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    from importlib.metadata import version
    source_root = Path(__file__).resolve().parents[2]
    source_paths = sorted((source_root / "src/dasa").rglob("*.py"))
    source_hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths}
    return {
        "command": command, "config": config,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()), "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__,
        "dependencies": {package: version(package) for package in (
            "numpy", "scipy", "torch", "matplotlib", "PyYAML")},
        "source_sha256": source_hashes,
        "data_provenance": "unspecified_until_inputs_are_inspected",
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs or []},
    }
