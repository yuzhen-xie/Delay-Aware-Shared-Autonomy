from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from dasa.config import manifest, write_json



_UNITS = {'simulation.control_dt': 's',
 'simulation.decision_dt': 's',
 'simulation.mass': 'kg',
 'simulation.damping': 'N s/m',
 'simulation.force_limit': 'N',
 'simulation.robot_radius': 'm',
 'simulation.workspace_min': 'm',
 'simulation.workspace_max': 'm',
 'simulation.max_seconds': 's',
 'simulation.grasp_tolerance': 'm',
 'simulation.place_tolerance': 'm',
 'simulation.dwell_seconds': 's',
 'simulation.max_grasp_speed': 'm/s',
 'simulation.lift_height': 'm',
 'simulation.feedback_delay': 's, one-way',
 'simulation.command_delay': 's, one-way',
 'simulation.jitter_seconds': 's',
 'scene.goals': 'm',
 'scene.start': 'm',
 'scene.place': 'm',
 'scene.object_size_m': 'm',
 'scene.training_translation': 'm',
 'scene.novel_translation': 'm per xy axis',
 'scene.added_obstacle_radius': 'm',
 'motion.attractive_gain': 'N/m',
 'motion.repulsive_gain': 'N m',
 'motion.repulsive_sigma': 'm',
 'motion.repulsive_limit': 'N per obstacle',
 'motion.attractive_weight': '1',
 'motion.repulsive_weight': '1',
 'motion.distance_min': 'm',
 'motion.distance_max': 'm',
 'motion.lambda_gain': '1',
 'motion.kalman_process': 'N^2/sample',
 'motion.kalman_measurement': 'N^2',
 'motion.kalman_initial_variance': 'N^2',
 'arbitration.stiffness': 'N/m',
 'arbitration.damping': 'N s/m',
 'arbitration.covariance_lower': 'm^2',
 'arbitration.covariance_upper': 'm^2',
 'arbitration.roughness_gamma': 's/N',
 'arbitration.roughness_window': 'samples',
 'arbitration.epsilon': '1',
 'arbitration.rho_window_seconds': 's',
 'arbitration.low_confidence_threshold': '1',
 'arbitration.correction_distance': 'm',
 'arbitration.fixed_weight': '1',
 'arbitration.guarded_corrective_max': '1',
 'passivity.initial_energy': 'J',
 'passivity.maximum_energy': 'J',
 'passivity.input_efficiency': '1',
 'passivity.tolerance': 'J',
 'intent.beta': 'rad^-1',
 'intent.confidence_threshold': '1',
 'intent.sustained_seconds': 's',
 'intent.memory_decay': '1',
 'intent.learning_rate': '1/update',
 'intent.kl_weight': '1',
 'intent.deviation_weight': '1',
 'skill.covariance_floor': 'm^2',
 'skill.duration_floor': 'resampled samples',
 'skill.lqt_control_cost': 'relative cost',
 'skill.original_weight': '1 per episode',
 'skill.new_weight': '1 per episode',
 'skill.proximity_threshold_m': 'm',
 'skill.place_snap_distance_m': 'm',
 'bootstrap.segment_min_seconds': 's',
 'bootstrap.cruise_speed_m_s': 'm/s',
 'bootstrap.lateral_std_m': 'm',
 'bootstrap.force_noise_std_N': 'N',
 'operator.kp': 'N/m',
 'operator.kd': 'N s/m',
 'operator.expert_noise': 'N',
 'operator.noisy_noise': 'N',
 'operator.corrective_gain': '1',
 'experiments.intent_delays': 's, one-way feedback',
 'experiments.sharing_delays': 's, one-way feedback',
 'study.max_seconds': 's'}


def flatten(value: dict, prefix=""):
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            yield from flatten(item, name)
        else:
            yield name, item


def parameter_rows(config: dict):
    rows = []
    for path, value in flatten(config):
        unit = _UNITS.get(path, 'count' if type(value) is int else '1')
        rows.append({'parameter': path, 'value': value, 'unit': unit})
    return rows


def export_protocol(config: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rows = parameter_rows(config)
    write_json(output / "parameters.json", rows)
    (output / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (output / "parameters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, "value": json.dumps(r["value"])} for r in rows)
    write_json(output / "provenance.json", manifest(config, "protocol"))


def audit_run(config: dict, root: Path):

    records, malformed = [], []
    for path in sorted(root.rglob("*.npz")):
        if "audit" in path.relative_to(root).parts:
            continue
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if "metadata" not in arrays:
                    continue
                metadata = json.loads(str(arrays["metadata"]))
                if "scene" not in metadata:
                    continue
                records.append({"path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "trial_id": metadata.get("trial_id"), "episode_id": metadata.get("episode_id"),
                    "seed": metadata.get("seed"), "unit": metadata.get("operator_id"),
                    "goal": metadata.get("intended_goal", metadata.get("goal")),
                    "condition": metadata.get("condition", metadata.get("source_condition")),
                    "provenance": metadata.get("provenance"), "scene": metadata["scene"]})
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            malformed.append({"path": str(path.relative_to(root)), "error": str(error)})
    write_json(root / "audit/actual_scene_records.json", {"schema_version": 1,
        "records": records, "malformed_records": malformed,
        "interpretation": "record inventory, not a count of independent trials or operators"})
    export_protocol(config, root / "audit")
    write_json(root / "audit/run_inventory.json", manifest(config, "audit-run", [
        root / r["path"] for r in records]))
    return {"record_count": len(records), "malformed_count": len(malformed)}
