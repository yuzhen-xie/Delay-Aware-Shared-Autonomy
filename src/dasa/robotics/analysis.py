from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from dasa.control.arbitration import CONDITIONS
from dasa.data import Demonstration
from dasa.metrics import measured_motion_metrics


FIELDS = ["trial_id", "participant_id", "condition", "delay_seconds", "success", "completion_seconds",
          "stability_index", "safety_index", "minimum_clearance_m", "jerk_rms", "practice", "termination",
          "elapsed_seconds", "record_status", "provenance",
          "robot_radius_m", "timeout_seconds", "candidate_id", "config_sha256", "stage",
          "layout_group", "scene_id", "source_file", "source_sha256"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _radius(config, metadata, assignment, contract):
    radius = config["task"]["robot_radius"]
    if isinstance(radius, bool) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Configured robot_radius must be a positive metric radius")
    for record in (metadata, assignment, contract):
        if "robot_radius_m" in record:
            value = record["robot_radius_m"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not np.isclose(value, radius, atol=1e-12, rtol=0)):
                raise ValueError("Recorded robot radius differs from the supplied measurement configuration")
    return float(radius)


def _annotations(demo):
    metadata = demo.metadata
    if metadata["provenance"] != "UR10_measured_teleoperation":
        raise ValueError("Measured metrics require imported UR10 measurements")
    if demo.scene.provenance not in {"calibrated_scene", "calibrated_rgbd_geometry"}:
        raise ValueError("Measured metrics require calibrated scene geometry")
    for field in ("success", "placed", "collision", "bounds_violation"):
        if not isinstance(metadata.get(field), bool):
            raise ValueError(f"Measured outcome annotation must be boolean: {field}")
    if "grasped_goal" not in metadata:
        raise ValueError("Measured outcome annotation requires grasped_goal")
    goal = metadata["grasped_goal"]
    if goal is not None and (isinstance(goal, bool) or not isinstance(goal, int)
                             or not 0 <= goal < len(demo.scene.goals)):
        raise ValueError("Invalid annotated grasped_goal")
    if metadata["success"] and (not metadata["placed"] or metadata["collision"]
                                or metadata["bounds_violation"] or goal != demo.goal):
        raise ValueError("Successful outcome conflicts with recorded task annotations")
    condition = metadata.get("source_condition")
    if condition not in CONDITIONS:
        raise ValueError("A recorded source_condition is required")
    delay = metadata.get("delay_seconds")
    if (isinstance(delay, bool) or not isinstance(delay, (int, float))
            or not np.isfinite(delay) or delay < 0):
        raise ValueError("Recorded delay_seconds must be a finite nonnegative value")
    assignment = metadata.get("experiment_assignment", {})
    if not isinstance(assignment, dict):
        raise ValueError("Invalid experiment assignment")
    for source, destination in (("trial_id", "episode_id"), ("participant_id", "operator_id")):
        if source in assignment and assignment[source] != metadata[destination]:
            raise ValueError("Trial annotations conflict with their acquisition assignment")
    if "intended_goal" in assignment and assignment["intended_goal"] != demo.goal:
        raise ValueError("Goal annotation conflicts with its acquisition assignment")
    practices = [record["practice"] for record in (metadata, assignment) if "practice" in record]
    if any(not isinstance(value, bool) for value in practices) or len(set(practices)) > 1:
        raise ValueError("Practice annotations must be consistent JSON booleans")
    return condition, float(delay), assignment, practices[0] if practices else False


def export_metrics(config, recordings: Path, output: Path):
    recordings, output = Path(recordings).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("Measured metric output must be a CSV file")
    audit_path = output.with_suffix(".audit.json")
    signal_directory = output.with_name(output.stem + ".signals")
    if output.exists() or audit_path.exists() or signal_directory.exists():
        raise FileExistsError("Measured metric exports preserve existing outputs")
    paths = sorted(recordings.rglob("*.npz")) if recordings.is_dir() else [recordings]
    if not paths or any(not path.is_file() or path.suffix.lower() != ".npz" for path in paths):
        raise ValueError("At least one processed measured demonstration NPZ is required")
    rows, sources, episodes, derived_series = [], [], set(), []
    for path in paths:
        digest = _sha(path)
        demo = Demonstration.load(path)
        if _sha(path) != digest:
            raise ValueError("Input demonstration changed during metric calculation")
        metadata = demo.metadata
        episode = metadata["episode_id"]
        if episode in episodes:
            raise ValueError("Duplicate trial identifiers across measured demonstrations")
        episodes.add(episode)
        condition, delay, assignment, practice = _annotations(demo)
        contract = metadata.get("execution_contract", {})
        if not isinstance(contract, dict):
            raise ValueError("Invalid recorded execution contract")
        radius = _radius(config, metadata, assignment, contract)
        timeouts = [value for value in (assignment.get("timeout_seconds"), contract.get("maximum_seconds"))
                    if value is not None]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not np.isfinite(value) or value <= 0 for value in timeouts):
            raise ValueError("Recorded trial timeout must be finite and positive")
        if len(timeouts) == 2 and not np.isclose(timeouts[0], timeouts[1], atol=1e-12, rtol=0):
            raise ValueError("Trial assignment and execution contract have different timeouts")
        timeout = float(timeouts[0]) if timeouts else None
        candidate_id = assignment.get("candidate_id")
        candidate_hash = assignment.get("candidate_config_sha256")
        if candidate_id is not None or candidate_hash is not None:
            if (not isinstance(candidate_id, str) or not candidate_id.strip()
                    or not isinstance(candidate_hash, str) or len(candidate_hash) != 64
                    or any(character not in "0123456789abcdef" for character in candidate_hash)):
                raise ValueError("Physical validation requires a bound candidate ID and configuration-file SHA256")
        scene_mode = metadata.get("hardware_timing", {}).get("scene_mode", "static_once")
        measured_clearance = None
        if scene_mode == "online":
            with np.load(path, allow_pickle=False) as archive:
                if "surface_clearance_m" not in archive:
                    raise ValueError("Online scene analysis requires actual per-sample surface_clearance_m")
                measured_clearance = np.asarray(archive["surface_clearance_m"], float)
        elif scene_mode not in {"static_once", "snapshot", "static"}:
            raise ValueError("Unsupported recorded scene-update mode")
        elapsed = float(demo.time[-1] - demo.time[0])
        motion, motion_audit, series = measured_motion_metrics(
            config, demo.time, demo.position, demo.velocity, demo.scene, radius, measured_clearance)
        motion_audit["scene_mode"] = scene_mode
        termination = metadata.get("termination") or metadata.get("stop_reason") \
            or ("annotated_success" if metadata["success"] else "annotated_failure")
        if not isinstance(termination, str):
            raise ValueError("Recorded termination must be a string")
        relative = str(path.relative_to(recordings)) if recordings.is_dir() else path.name
        row = {"trial_id": episode, "participant_id": metadata["operator_id"], "condition": condition,
               "delay_seconds": delay, "success": metadata["success"],
               "completion_seconds": elapsed if metadata["success"] else None,
               **motion,
               "practice": practice, "termination": termination, "elapsed_seconds": elapsed,
               "record_status": "collected", "provenance": metadata["provenance"],
               "robot_radius_m": radius, "timeout_seconds": timeout,
               "candidate_id": candidate_id, "config_sha256": candidate_hash,
               "stage": metadata.get("stage") or assignment.get("stage"),
               "layout_group": assignment.get("layout_group"), "scene_id": assignment.get("scene_id"),
               "source_file": relative, "source_sha256": digest}
        rows.append(row)
        signal_file = hashlib.sha256(episode.encode()).hexdigest()[:24] + ".npz"
        if series:
            derived_series.append((signal_file, series))
        sources.append({"source_file": relative, "source_sha256": digest, "trial_id": episode,
                        "provenance": metadata["provenance"], "samples": len(demo.time),
                        "time_window_seconds": [float(demo.time[0]), float(demo.time[-1])],
                        "robot_radius_m": radius, "intended_goal": demo.goal,
                        "practice_source": "recorded" if "practice" in metadata or "practice" in assignment
                                           else "legacy_default_false",
                        "source_recording_sha256": metadata.get("source_recording_sha256"),
                        "planned_order": assignment.get("planned_order"),
                        "timeout_seconds": timeout,
                        "candidate_id": candidate_id, "candidate_config_sha256": candidate_hash,
                        "stage": row["stage"], "layout_group": assignment.get("layout_group"),
                        "scene_id": assignment.get("scene_id"), "motion_metrics": motion_audit,
                        "derived_signal_file": str(signal_directory.name + "/" + signal_file) if series else None})
    audit = {
        "schema_version": 1, "provenance": "derived_imported_UR10_measurements",
        "n_trials": len(rows), "n_participants": len({row["participant_id"] for row in rows}),
        "n_success": sum(row["success"] for row in rows), "n_failures": sum(not row["success"] for row in rows),
        "n_practice": sum(row["practice"] for row in rows),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "calculation": {"elapsed": "last_minus_first_retained_timestamp",
                        "completion": "retained_duration_if_annotated_success_else_null",
                        "clearance": "per_record_static_swept_or_sampled_online_geometry_scope",
                        "interpolation": "per_record_static_position_or_online_clearance_as_audited",
                        "motion_indices": "configured_uniform_resampling_Savitzky_Golay_time_averaging",
                        "omega": "norm_filtered_translational_TCP_velocity_m_per_s",
                        "stability": "exp_negative_absolute_gamma_st_omega_ddot",
                        "safety": "exp_negative_gamma_sa_omega_squared_over_positive_surface_clearance_else_zero",
                        "outcome": "independent_annotations_no_success_reclassification",
                        "units": {"elapsed": "s", "completion": "s", "clearance": "m", "jerk_rms": "m/s^3",
                                  "stability_index": "1", "safety_index": "1"}},
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    created_csv, created_audit, created_signals = False, False, False
    try:
        signal_directory.mkdir(exist_ok=False)
        created_signals = True
        for filename, series in derived_series:
            np.savez_compressed(signal_directory / filename, **series)
        audit["derived_signal_sha256"] = {filename: _sha(signal_directory / filename)
                                          for filename, _ in derived_series}
        with output.open("x", newline="", encoding="utf-8") as stream:
            created_csv = True
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        audit["csv_sha256"] = _sha(output)
        with audit_path.open("x", encoding="utf-8") as stream:
            created_audit = True
            json.dump(audit, stream, indent=2, allow_nan=False)
    except BaseException:
        if created_audit:
            audit_path.unlink(missing_ok=True)
        if created_csv:
            output.unlink(missing_ok=True)
        if created_signals:
            for filename, _ in derived_series:
                (signal_directory / filename).unlink(missing_ok=True)
            signal_directory.rmdir()
        raise
    return audit
