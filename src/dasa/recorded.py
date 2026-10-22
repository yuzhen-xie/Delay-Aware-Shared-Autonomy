from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dasa.config import manifest, seed_everything, write_json
from dasa.data import Demonstration, validate_partitions
from dasa.experiments import require_measured
from dasa.metrics import intent_metrics
from dasa.models.dmp import DMP, DMPIntent
from dasa.models.hsmm import TPHSMM
from dasa.models.intent import (IntentEstimator, ReferenceBank, condition, inference_spec,
                              load_models, model_spec)
from dasa.statistics import comparisons


METHODS = ("full", "no_latent", "no_delay", "likelihood_only", "br", "memory", "dmp", "lstm", "transformer")
METRICS = ("ipa", "accuracy", "nll", "brier", "early_prediction_seconds", "prediction_stability")


def _recorded_age(path: Path, demo: Demonstration, nominal: float,
                  allow_nominal: bool = False) -> tuple[np.ndarray, str]:

    with np.load(path, allow_pickle=False) as arrays:
        for key in ("observed_source_time", "observation_source_time", "source_time"):
            if key not in arrays:
                continue
            source = np.asarray(arrays[key], dtype=float)
            if source.shape != demo.time.shape or not np.isfinite(source).all():
                raise ValueError(f"Invalid recorded {key}: {path.name}")
            command_time = demo.time
            if "command_receive_time" in arrays:
                command_time = np.asarray(arrays["command_receive_time"], dtype=float)
                if command_time.shape != demo.time.shape or not np.isfinite(command_time).all():
                    raise ValueError("Command receive timestamps must be finite and sample-aligned")
            age = command_time - source
            if np.any(age < -1e-9):
                raise ValueError("Observation source timestamps cannot come from the future")
            clock = "command_receive_time" if "command_receive_time" in arrays else "time"
            return np.maximum(age, 0), f"recorded_timestamps:{clock}-{key}; requires_same_synchronized_clock"
        if "feedback_age" in arrays:
            age = np.asarray(arrays["feedback_age"], dtype=float)
            if age.shape != demo.time.shape or not np.isfinite(age).all() or np.any(age < 0):
                raise ValueError("Recorded feedback_age must be finite, nonnegative and sample-aligned")
            return age, "recorded_feedback_age_array"
    if not allow_nominal:
        raise ValueError("Physical recording requires measured source timestamps or a recorded feedback_age array")
    return np.full(len(demo.time), nominal), "nominal_metadata_only; effective_age_unmeasured"


def _signal_hash(demo):
    digest = hashlib.sha256()
    for name in ("time", "position", "velocity", "raw_force", "observed_state", "phase"):
        value = np.ascontiguousarray(getattr(demo, name), dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _heldout_files(dataset, splits, recordings_path):
    if recordings_path is None:
        if not splits.get("test"):
            raise ValueError("Recorded intent evaluation requires a nonempty held-out test split")
        files = [dataset / name for name in splits["test"]]
    else:
        recordings_path = Path(recordings_path)
        if not recordings_path.is_dir():
            raise ValueError("The external held-out recordings directory does not exist")
        files = sorted(recordings_path.rglob("*.npz"))
        if not files:
            raise ValueError("The external held-out directory contains no measured NPZ recordings")
    protected = [Demonstration.load(dataset / name) for split in ("train", "validation")
                 for name in splits[split]]
    operators = {demo.metadata["operator_id"] for demo in protected}
    ids = {demo.metadata["episode_id"] for demo in protected}
    hashes = {_signal_hash(demo) for demo in protected}
    source_hashes = {demo.metadata["source_recording_sha256"] for demo in protected}
    seen_ids, seen_hashes, seen_sources = set(), set(), set()
    for path in files:
        demo = Demonstration.load(path)
        require_measured(demo)
        identifier, digest = demo.metadata["episode_id"], _signal_hash(demo)
        source_digest = demo.metadata["source_recording_sha256"]
        if (demo.metadata["operator_id"] in operators or identifier in ids or digest in hashes
                or source_digest in source_hashes):
            raise ValueError("Held-out recordings overlap frozen training/validation operators, episodes or signals")
        if identifier in seen_ids or digest in seen_hashes or source_digest in seen_sources:
            raise ValueError("Held-out recordings contain duplicate episodes, raw signals or source acquisitions")
        seen_ids.add(identifier)
        seen_hashes.add(digest)
        seen_sources.add(source_digest)
    return files


def intent_recordings(config: dict, prepared_root: Path, output: Path,
                      participants_path: Path | None = None,
                      recordings_path: Path | None = None) -> list[dict]:
    prepared_root, output = Path(prepared_root), Path(output)
    stage_output = output / "intent"
    if stage_output.exists() and any(stage_output.iterdir()):
        raise ValueError("The intent stage already contains an analysis; choose a new analysis root")
    dataset = prepared_root / "dataset"
    splits_path = dataset / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    validate_partitions(dataset, splits)
    recording_files = _heldout_files(dataset, splits, recordings_path)
    episodes = []
    for path in recording_files:
        demo = Demonstration.load(path)
        require_measured(demo)
        delay = demo.metadata.get("delay_seconds")
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not np.isfinite(delay) or delay < 0:
            raise ValueError("Each recording requires an explicit nonnegative delay_seconds")
        if not isinstance(demo.metadata["success"], bool):
            raise ValueError("Recorded success must be a JSON boolean")
        age, age_source = _recorded_age(path, demo, float(delay),
                                       config["intent"].get("allow_nominal_recorded_age", False))
        episodes.append((path, demo, float(delay), age, age_source))
    seed_everything(config["seed"])
    skill = TPHSMM.load(config, prepared_root / "models/bootstrap_hsmm.npz")
    cvae, networks = load_models(config, prepared_root / "models")
    if config["intent"].get("require_verified_training", True) and (
            cvae.training_metadata.get("schema_version", 0) < 2 or not cvae.context_prior_trained):
        raise ValueError("Physical analysis requires audited classifier training and the trained context-only ablation")
    dmp = DMP.load(prepared_root / "models/dmp.npz")
    predictions = stage_output / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    rows, inventory = [], []
    inference = inference_spec(config)
    for episode_index, (path, demo, delay, age, age_source) in enumerate(episodes):
        indices = np.flatnonzero(demo.phase == 0)
        elapsed = demo.time - demo.time[0]
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        source = {"episode_id": demo.metadata["episode_id"], "unit": demo.metadata["operator_id"],
                  "source_file": str(path.resolve()), "source_sha256": source_hash,
                  "provenance": demo.metadata["provenance"], "delay_seconds": delay,
                  "age_source": age_source, "pickup_samples": len(indices),
                  "total_samples": len(demo.time), "duration_seconds": float(elapsed[-1]),
                  "success": demo.metadata["success"], "true_goal": demo.goal,
                  "selection_status": "included" if len(indices) else "no_pickup_samples"}
        inventory.append(source)
        if not len(indices):
            continue
        times = elapsed[indices]
        current = np.c_[demo.position, demo.velocity]
        completion = float(elapsed[-1]) if demo.metadata["success"] else None
        episode_seed = (inference["seed"] + 100003 * episode_index) % 2**32
        for method in METHODS:
            estimator = DMPIntent(dmp, demo.scene, config["intent"]["beta"]) if method == "dmp" else \
                IntentEstimator(config, ReferenceBank(skill, demo.scene), cvae, method,
                                networks.get(method), seed=episode_seed)
            posterior, native, contexts, latent, means, logvars = [], [], [], [], [], []
            native_kind = None
            for index in indices:
                probability, embedding = estimator.step(demo.raw_force[index].copy(),
                                                        demo.observed_state[index].copy(),
                                                        current[index].copy(), demo.scene,
                                                        float(age[index]))
                posterior.append(probability)
                descriptor = getattr(estimator, "last_representation", None)
                if descriptor is None:
                    current_kind, feature = "categorical_goal_posterior_probability", probability
                else:
                    current_kind, feature = descriptor["native_kind"], descriptor["native"]
                if native_kind is not None and current_kind != native_kind:
                    raise ValueError("Representation definition changed within one episode")
                native_kind = current_kind
                native.append(feature)
                state = current[index] if method in {"br", "memory", "no_delay"} else demo.observed_state[index]
                contexts.append(condition(state, demo.scene, config["task"]["robot_radius"]))
                if method in {"full", "no_delay"}:
                    if descriptor is None or not descriptor["latent_available"]:
                        raise ValueError("Latent method did not expose its actual encoder representation")
                    latent.append(embedding)
                    means.append(descriptor["latent_mean"])
                    logvars.append(descriptor["latent_logvar"])
            posterior = np.asarray(posterior, dtype=float)
            native = np.asarray(native, dtype=float)
            if (posterior.shape != (len(indices), len(demo.scene.goals))
                    or not np.isfinite(posterior).all() or np.any(posterior < 0)
                    or not np.allclose(posterior.sum(1), 1, atol=1e-6)):
                raise ValueError(f"Invalid categorical posterior from {method}")
            if native.ndim != 2 or native.shape[0] != len(indices) or not np.isfinite(native).all():
                raise ValueError(f"Invalid declared native representation from {method}")
            metrics = intent_metrics(times, posterior, demo.goal, config, completion)
            rows.append({"condition": method, **source, **metrics,
                         "inference_seed": episode_seed, "latent_mode": inference["latent_mode"],
                         "native_representation_kind": native_kind})
            metadata = {**source, "method": method,
                        "evaluation": "offline_recorded_signals_no_generated_interaction",
                        "time_axis": "seconds_relative_to_recording_start",
                        "observation_alignment": "provided_observed_state; no additional delay applied",
                        "representation_kind": "categorical_goal_posterior_probability",
                        "native_representation_kind": native_kind,
                        "latent_available": method in {"full", "no_delay"},
                        "inference_seed": episode_seed, "inference": inference,
                        "phase_scope": "pickup_phase_zero_only",
                        "confidence_threshold": config["intent"]["confidence_threshold"],
                        "sustained_prediction_seconds": config["intent"]["sustained_seconds"]}
            arrays = {"time": times, "recorded_time": demo.time[indices], "source_indices": indices,
                      "posterior": posterior, "representation": posterior,
                      "native_representation": native, "model_context": np.asarray(contexts),
                      "true_goal": demo.goal, "feedback_age": age[indices],
                      "metadata": np.array(json.dumps(metadata))}
            if method in {"full", "no_delay"}:
                arrays.update(latent=np.asarray(latent), latent_mean=np.asarray(means),
                              latent_logvar=np.asarray(logvars))
            np.savez_compressed(predictions / f"{method}_episode{episode_index:05d}.npz",
                                **arrays)
    observed_delays = sorted({entry[2] for entry in episodes})
    configured_delays = config["experiments"]["intent_delays"]
    planned_delays = sorted(set(configured_delays) | set(observed_delays))
    statistics = comparisons(rows, "full", [method for method in METHODS if method != "full"],
                             list(METRICS), config,
                             family="recorded_intent_all_methods_metrics_declared_and_observed_delays",
                             planned_delays=planned_delays)
    for row in statistics:
        row["delay_scope"] = "configured_delay" if row["delay_seconds"] in configured_delays else \
            "observed_unplanned_delay_exploratory"
    write_json(output / "intent/metrics.json", rows)
    metrics_sha256 = hashlib.sha256((stage_output / "metrics.json").read_bytes()).hexdigest()
    for row in statistics:
        row.update(metric_source_sha256=metrics_sha256, metric_source_file="intent/metrics.json")
    write_json(output / "intent/statistics.json", statistics)
    write_json(output / "intent/input_inventory.json", inventory)
    write_json(output / "intent/evaluation_protocol.json", {
        "evaluation": "actual_heldout_recorded_signals_at_independently_acquired_delays",
        "methods": list(METHODS), "configured_delay_seconds": configured_delays,
        "observed_delay_seconds": observed_delays, "statistical_delay_seconds": planned_delays,
        "missing_delay_policy": "retain_unavailable_planned_contrasts; do_not_synthesize_recordings",
        "additional_delay_policy": "include_in_expanded_Holm_family_and_label_exploratory",
        "statistical_unit": "operator; repeated episodes averaged within operator/method/delay",
        "recording_alignment": "observed_state supplied in the processed NPZ contract",
        "age_fallback": "nominal delay metadata only when timestamps/feedback_age absent; not measured age",
        "timing_assumption": "source timestamp arrays must share recording time clock and units",
        "completion": "duration of successfully annotated recording; otherwise unavailable",
        "inference": inference, "model_inputs": model_spec(config),
        "no_latent_control": "trained context-only prior; same input context and likelihood as full estimator",
        "likelihood_only_control": "uniform goal prior with delayed-state angular likelihood",
        "representation_scope": "common posterior-probability space for cross-method comparisons; native features separately identified",
        "prediction_stability": "one minus goal switches divided by pickup sample count; stable wrong predictions remain incorrect",
        "probability_calibration": "NLL, Brier score and ECE reported; no unvalidated claim of absolute reliability",
        "label_scope": "goal and success used for outcome scoring only",
        "recording_source": "external_physical_holdout_directory" if recordings_path is not None
        else "frozen_dataset_test_partition",
        "holdout_policy": "episode, operator, source acquisition and raw signal disjoint from frozen training and validation"})
    inputs = [splits_path, *[dataset / name for paths in splits.values() for name in paths]]
    inputs.extend(recording_files)
    inputs.extend(path for path in sorted((prepared_root / "models").iterdir()) if path.is_file())
    record = manifest(config, "intent-recordings", inputs)
    record["data_provenance"] = sorted({entry[1].metadata["provenance"] for entry in episodes})
    record["evaluation_mode"] = "offline_replay_of_recorded_signals"
    record["prepared_source"] = str(prepared_root.resolve())
    record["observed_delay_seconds"] = observed_delays
    record["external_recordings_source"] = str(Path(recordings_path).resolve()) \
        if recordings_path is not None else None
    write_json(stage_output / "manifest.json", record)
    if participants_path is not None:
        from dasa.representation import analyze
        analyze(config, output, participants_path, output / "representation")
    return rows
