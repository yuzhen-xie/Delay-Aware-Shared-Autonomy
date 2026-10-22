from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dasa.config import manifest, seed_everything, write_json
from dasa.data import Demonstration, validate_partitions
from dasa.metrics import intent_metrics
from dasa.models.dmp import DMP, DMPIntent
from dasa.models.hsmm import TPHSMM
from dasa.models.intent import IntentEstimator, ReferenceBank, load_models
from dasa.statistics import comparisons


METHODS = ("full", "no_latent", "no_delay", "br", "memory", "dmp", "lstm", "transformer")
METRICS = ("ipa", "accuracy", "nll", "brier", "early_prediction_seconds", "prediction_stability")


def _recorded_age(path: Path, demo: Demonstration, nominal: float) -> tuple[np.ndarray, str]:

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
    return np.full(len(demo.time), nominal), "nominal_metadata_only; effective_age_unmeasured"


def intent_recordings(config: dict, prepared_root: Path, output: Path) -> list[dict]:







    prepared_root, output = Path(prepared_root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Recorded evaluation output must be empty to preserve existing analyses")
    dataset = prepared_root / "dataset"
    splits_path = dataset / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    validate_partitions(dataset, splits)
    if not splits.get("test"):
        raise ValueError("Recorded intent evaluation requires a nonempty held-out test split")
    episodes = []
    for relative in splits["test"]:
        path = dataset / relative
        demo = Demonstration.load(path)
        delay = demo.metadata.get("delay_seconds")
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not np.isfinite(delay) or delay < 0:
            raise ValueError("Each recording requires an explicit nonnegative delay_seconds")
        if not isinstance(demo.metadata["success"], bool):
            raise ValueError("Recorded success must be a JSON boolean")
        age, age_source = _recorded_age(path, demo, float(delay))
        episodes.append((path, demo, float(delay), age, age_source))
    seed_everything(config["seed"])
    skill = TPHSMM.load(config, prepared_root / "models/bootstrap_hsmm.npz")
    cvae, networks = load_models(config, prepared_root / "models")
    dmp = DMP.load(prepared_root / "models/dmp.npz")
    predictions = output / "intent/predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    rows, inventory = [], []
    for episode_index, (path, demo, delay, age, age_source) in enumerate(episodes):
        indices = np.flatnonzero(demo.phase == 0)
        elapsed = demo.time - demo.time[0]
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        source = {"episode_id": demo.metadata["episode_id"], "unit": demo.metadata["operator_id"],
                  "source_file": str(path.relative_to(prepared_root)), "source_sha256": source_hash,
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
        for method in METHODS:
            estimator = DMPIntent(dmp, demo.scene, config["intent"]["beta"]) if method == "dmp" else \
                IntentEstimator(config, ReferenceBank(skill, demo.scene), cvae, method, networks.get(method))
            posterior, latent = [], []
            for index in indices:
                probability, embedding = estimator.step(demo.raw_force[index].copy(),
                                                        demo.observed_state[index].copy(),
                                                        current[index].copy(), demo.scene,
                                                        float(age[index]))
                posterior.append(probability)
                latent.append(embedding)
            posterior = np.asarray(posterior, dtype=float)
            latent = np.asarray(latent, dtype=float)
            if (posterior.shape != (len(indices), len(demo.scene.goals))
                    or not np.isfinite(posterior).all() or np.any(posterior < 0)
                    or not np.allclose(posterior.sum(1), 1, atol=1e-6)):
                raise ValueError(f"Invalid categorical posterior from {method}")
            if latent.ndim != 2 or latent.shape[0] != len(indices) or not np.isfinite(latent).all():
                raise ValueError(f"Invalid recorded-evaluation embedding from {method}")
            metrics = intent_metrics(times, posterior, demo.goal, config, completion)
            rows.append({"condition": method, **source, **metrics})
            metadata = {**source, "method": method,
                        "evaluation": "offline_recorded_signals_no_generated_interaction",
                        "time_axis": "seconds_relative_to_recording_start",
                        "observation_alignment": "provided_observed_state; no additional delay applied"}
            np.savez_compressed(predictions / f"{method}_episode{episode_index:05d}.npz",
                                time=times, recorded_time=demo.time[indices],
                                source_indices=indices, posterior=posterior,
                                latent=latent, true_goal=demo.goal,
                                feedback_age=age[indices], metadata=np.array(json.dumps(metadata)))
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
    write_json(output / "intent/statistics.json", statistics)
    write_json(output / "intent/input_inventory.json", inventory)
    write_json(output / "intent/evaluation_protocol.json", {
        "evaluation": "recorded_holdout_only; no interaction simulation or synthetic delay injection",
        "methods": list(METHODS), "configured_delay_seconds": configured_delays,
        "observed_delay_seconds": observed_delays, "statistical_delay_seconds": planned_delays,
        "missing_delay_policy": "retain_unavailable_planned_contrasts; do_not_synthesize_recordings",
        "additional_delay_policy": "include_in_expanded_Holm_family_and_label_exploratory",
        "statistical_unit": "operator; repeated episodes averaged within operator/method/delay",
        "recording_alignment": "observed_state supplied in the processed NPZ contract",
        "age_fallback": "nominal delay metadata only when timestamps/feedback_age absent; not measured age",
        "timing_assumption": "source timestamp arrays must share recording time clock and units",
        "completion": "duration of successfully annotated recording; otherwise unavailable",
        "label_scope": "goal and success used for outcome scoring only",
        "holdout_policy": "episode, operator and signal disjoint from training and validation"})
    inputs = [splits_path, *[dataset / name for paths in splits.values() for name in paths]]
    inputs.extend(path for path in sorted((prepared_root / "models").iterdir()) if path.is_file())
    record = manifest(config, "intent-recordings", inputs)
    record["data_provenance"] = sorted({entry[1].metadata["provenance"] for entry in episodes})
    record["evaluation_mode"] = "offline_replay_of_recorded_signals"
    record["prepared_source"] = str(prepared_root.resolve())
    record["observed_delay_seconds"] = observed_delays
    write_json(output / "manifest.json", record)
    return rows
