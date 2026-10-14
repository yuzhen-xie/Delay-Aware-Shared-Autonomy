"""Dataset preparation, shared-control comparisons and independent policy refinement."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from dasa.config import manifest, seed_everything, write_json
from dasa.data import Demonstration, generate_dataset, validate_partitions, synthetic_demonstration
from dasa.geometry import make_scene
from dasa.metrics import intent_metrics, silhouette, trial_metrics
from dasa.models.dmp import DMP, DMPIntent
from dasa.models.hsmm import TPHSMM, refine
from dasa.models.intent import IntentEstimator, ReferenceBank, load_models, train_intent, training_arrays
from dasa.simulation import run_trial
from dasa.statistics import comparisons


def load_split(root: Path, split: str):
    splits = json.loads((root / "splits.json").read_text())
    validate_partitions(root, splits)
    return [Demonstration.load(root / p) for p in splits[split]]


def prepare(config: dict, root: Path):
    seed_everything(config["seed"])
    data = root / "dataset"
    generate_dataset(config, data)
    train_dataset(config, data, root)


def train_dataset(config: dict, data: Path, root: Path):
    seed_everything(config["seed"])
    train, validation = load_split(data, "train"), load_split(data, "validation")
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    skill = TPHSMM(config).fit(train)
    skill.save(models / "bootstrap_hsmm.npz")
    write_json(models / "hsmm_fit.json", skill.history)
    DMP().fit(train).save(models / "dmp.npz")
    history = train_intent(config, training_arrays(train, skill, config),
                           training_arrays(validation, skill, config), models)
    write_json(models / "training.json", history)
    files = sorted(data.glob("*/*.npz"))
    run_manifest = manifest(config, "train_dataset", files)
    run_manifest["data_provenance"] = sorted({d.metadata["provenance"] for d in train + validation})
    write_json(root / "manifest.json", run_manifest)
    print(f"Prepared {len(train)} bootstrap and {len(validation)} validation demonstrations", flush=True)


def refine_dataset(config: dict, data: Path, candidates: Path, output: Path):
    original = load_split(data, "train")
    new = [Demonstration.load(p) for p in sorted(candidates.glob("*.npz"))]
    held_out = load_split(data, "validation") + load_split(data, "test")
    held_out_operators = {d.metadata["operator_id"] for d in held_out}
    if any(d.metadata["operator_id"] in held_out_operators for d in new):
        raise ValueError("Refinement operators overlap held-out partitions")
    model, audit, accepted = refine(config, original, new)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "refined_hsmm.npz")
    write_json(output / "selection_audit.json", audit)
    write_json(output / "refinement_metadata.json", {
        "original_count": len(original), "candidate_count": len(new), "accepted_count": len(accepted),
        "required_count": config["skill"]["refinement_count"],
        "quota_met": len(accepted) == config["skill"]["refinement_count"],
        "provenance": sorted({d.metadata["provenance"] for d in original + new}),
        "weights": {"original": config["skill"]["original_weight"], "new": config["skill"]["new_weight"]}})


def sharing(config: dict, root: Path):
    skill = TPHSMM.load(config, root / "models/bootstrap_hsmm.npz")
    cvae, _ = load_models(config, root / "models")
    rows = []
    e = config["experiments"]
    trials = root / "sharing/trials"
    for unit in range(e["simulated_operators"]):
        for repeat in range(e["trials_per_condition"]):
            seed = config["seed"] + 100000 + unit * 100 + repeat
            scene = make_scene(config, np.random.default_rng(seed))
            goal = (unit + repeat) % 3
            # All conditions receive the same scene, operator seed and delay process.
            for delay in e["sharing_delays"]:
                c = copy.deepcopy(config)
                c["simulation"]["feedback_delay"] = delay
                order = np.random.default_rng(seed + int(delay * 1000)).permutation(e["conditions"])
                for name in order:
                    trial_id = f"shared_u{unit:02d}_r{repeat:02d}_d{int(delay * 1000)}_{name}"
                    trial = run_trial(c, scene, skill, cvae, seed, goal, str(name),
                                      f"synthetic_u{unit:02d}", noisy=unit % 2 == 1, trial_id=trial_id)
                    trial.save(trials / f"{trial_id}.npz")
                    rows.append(trial_metrics(trial, c))
                print(f"Shared control: simulated operator {unit + 1}, delay {delay:.1f}s", flush=True)
    write_json(root / "sharing/metrics.json", rows)
    methods = [name for name in e["conditions"] if name != "C3"]
    stats = comparisons(rows, "C3", methods,
                        ["success_rate", "completion_seconds", "minimum_clearance_m", "jerk_rms"], config)
    write_json(root / "sharing/statistics.json", stats)
    return rows


def intent_experiment(config: dict, root: Path):
    skill = TPHSMM.load(config, root / "models/bootstrap_hsmm.npz")
    cvae, networks = load_models(config, root / "models")
    dmp = DMP.load(root / "models/dmp.npz")
    demos = load_split(root / "dataset", "test")
    methods = ["full", "no_latent", "no_delay", "br", "memory", "dmp", "lstm", "transformer"]
    rows, latent_values, labels = [], [], []
    for delay in config["experiments"]["intent_delays"]:
        for j, demo in enumerate(demos):
            c = copy.deepcopy(config)
            c["simulation"]["feedback_delay"] = delay
            trace = run_trial(c, demo.scene, skill, cvae, config["seed"] + 200000 + j,
                              demo.goal, "C1", demo.metadata["operator_id"], noisy=j % 2 == 1)
            trace.save(root / "intent/traces" / f"test_{j:03d}_d{int(delay * 1000)}.npz")
            a, m = trace.arrays, trace.metadata
            pick = a["phase"] == 0
            times = a["decision_time"][pick]
            if len(times) < 1:
                continue
            for method in methods:
                bank = ReferenceBank(skill, demo.scene)
                estimator = DMPIntent(dmp, demo.scene, c["intent"]["beta"]) if method == "dmp" else IntentEstimator(
                    c, bank, cvae, method, networks.get(method))
                probabilities, embeddings = [], []
                for k in np.flatnonzero(pick):
                    probability, embedding = estimator.step(
                        a["raw_force"][k], a["observed_state"][k],
                        a["current_state"][k], demo.scene, a["feedback_age"][k])
                    probabilities.append(probability)
                    embeddings.append(embedding)
                probability = np.asarray(probabilities)
                metrics = intent_metrics(times, probability, demo.goal, c,
                                         float(a["time"][-1]) if m["success"] else None)
                rows.append({"unit": demo.metadata["operator_id"], "episode_id": demo.metadata["episode_id"],
                             "condition": method, "delay_seconds": delay,
                             "provenance": m["provenance"], **metrics})
                prediction_path = root / "intent/predictions" / f"{method}_t{j:03d}_d{int(delay * 1000)}.npz"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(prediction_path, time=times, posterior=probability,
                                    latent=np.asarray(embeddings), true_goal=demo.goal)
                if method == "full" and delay == 0.4:
                    latent_values.extend(embeddings)
                    labels.extend([demo.goal] * len(embeddings))
        print(f"Intent inference: delay {delay:.1f}s, {len(demos)} held-out episodes", flush=True)
    write_json(root / "intent/metrics.json", rows)
    write_json(root / "intent/statistics.json", comparisons(
        rows, "full", [m for m in methods if m != "full"], ["ipa", "accuracy", "nll", "brier"], config))
    write_json(root / "intent/latent_diagnostics.json", {
        "silhouette": silhouette(np.asarray(latent_values), np.asarray(labels), config["seed"])
        if latent_values else None, "interpretation": "goal separability only; cognitive state and skill are unvalidated"})
    return rows


def refinement_experiment(config: dict, root: Path):
    cvae, _ = load_models(config, root / "models")
    original = load_split(root / "dataset", "train")
    rows, audits, budgets = [], [], []
    e = config["experiments"]
    for repeat in range(e["independent_repeats"]):
        # Independent bootstrap perturbation and training fit per replicate; shared across C1-C3.
        seed_everything(config["seed"] + 300000 + repeat)
        bootstrap = []
        for j, demo in enumerate(original):
            d = synthetic_demonstration(config, config["seed"] + 300000 + repeat * 1000 + j,
                                        f"synthetic_boot_r{repeat}_{j % 5}",
                                        f"bootstrap_r{repeat}_{j}", demo.goal)
            bootstrap.append(d)
        base = TPHSMM(config).fit(bootstrap)
        for name in ("C1", "C2", "C3"):
            candidates, accepted_count = [], 0
            max_attempts = config["skill"]["refinement_count"] * 5
            c = copy.deepcopy(config)
            c["simulation"]["feedback_delay"] = 0.8
            from dasa.data import successful
            for attempt in range(max_attempts):
                seed = config["seed"] + 400000 + repeat * 1000 + attempt
                scene = make_scene(c, np.random.default_rng(seed))
                trial = run_trial(c, scene, base, cvae, seed, attempt % 3, name,
                                  f"synthetic_refinement_r{repeat}_{attempt % 5}", noisy=attempt % 2 == 1,
                                  trial_id=f"refine_{name}_r{repeat}_a{attempt}")
                trial.save(root / f"refinement/collection/r{repeat}/{name}/{attempt:03d}.npz")
                if len(trial.arrays["time"]) < 3:
                    continue
                demo = trial.demonstration()
                candidates.append(demo)
                ok, _ = successful(demo, c)
                accepted_count += int(ok)
                if accepted_count >= config["skill"]["refinement_count"]:
                    break
            model, audit, accepted = refine(c, bootstrap, candidates)
            model.save(root / f"refinement/models/{name}_r{repeat}.npz")
            for a in audit:
                audits.append({"repeat": repeat, "condition": name, **a})
            budgets.append({"repeat": repeat, "condition": name, "attempted": len(candidates),
                            "accepted": len(accepted), "requested": config["skill"]["refinement_count"],
                            "quota_met": len(accepted) == config["skill"]["refinement_count"]})
            for scenario in e["scenarios"]:
                for j in range(e["autonomy_trials"]):
                    seed = config["seed"] + 500000 + repeat * 1000 + j
                    scene = make_scene(c, np.random.default_rng(seed), scenario)
                    trial = run_trial(c, scene, model, cvae, seed, j % 3, "autonomous",
                                      f"training_repeat_{repeat}",
                                      trial_id=f"autonomy_{name}_r{repeat}_{scenario}_{j}")
                    trial.save(root / f"refinement/evaluation/r{repeat}/{name}/{scenario}/{j:03d}.npz")
                    metrics = trial_metrics(trial, c)
                    metrics["condition"] = name
                    metrics["scenario"] = scenario
                    metrics["repeat"] = repeat
                    metrics["refinement_quota_met"] = budgets[-1]["quota_met"]
                    rows.append(metrics)
            print(f"Refinement {name}, repeat {repeat + 1}: {len(accepted)}/{c['skill']['refinement_count']} accepted", flush=True)
    write_json(root / "refinement/metrics.json", rows)
    write_json(root / "refinement/selection_audit.json", audits)
    write_json(root / "refinement/budgets.json", budgets)
    stats = []
    for scenario in e["scenarios"]:
        subset = [r for r in rows if r["scenario"] == scenario]
        results = comparisons(subset, "C3", ["C1", "C2"],
                              ["success_rate", "reference_error_m"], config)
        stats.extend([{**r, "scenario": scenario} for r in results])
    # Correct across all scenarios, rather than separately within each scenario.
    from dasa.statistics import holm
    for row, corrected in zip(stats, holm([r["p_raw"] for r in stats])):
        row["p_holm"] = corrected
    write_json(root / "refinement/statistics.json", stats)
    return rows


def stress_experiment(config: dict, root: Path):
    skill = TPHSMM.load(config, root / "models/bootstrap_hsmm.npz")
    cvae, _ = load_models(config, root / "models")
    rows = []
    cases = [("jitter", {"jitter_seconds": 0.20}),
             ("bidirectional", {"command_delay": 0.4}),
             ("low_confidence", {}), ("sudden_correction", {})]
    for case, changes in cases:
        for unit in range(config["experiments"]["simulated_operators"]):
            seed = config["seed"] + 600000 + unit
            scene = make_scene(config, np.random.default_rng(seed),
                               "extra_obstacles" if case == "sudden_correction" else "demonstrated")
            c = copy.deepcopy(config)
            c["simulation"].update(changes)
            if case == "low_confidence":
                c["arbitration"]["covariance_upper"] = 0.00004
                c["operator"]["noisy_noise"] = 2.5
            for name in ("C3", "C3_guarded"):
                trial = run_trial(c, scene, skill, cvae, seed, unit % 3, name,
                                  f"synthetic_stress_{unit}", noisy=True,
                                  trial_id=f"stress_{case}_{name}_{unit}")
                trial.save(root / f"stress/{case}/{name}_{unit}.npz")
                rows.append({"case": case, **trial_metrics(trial, c)})
    write_json(root / "stress/metrics.json", rows)
    return rows


def run_all(config: dict, root: Path):
    started = time.monotonic()
    prepare(config, root)
    intent_experiment(config, root)
    sharing(config, root)
    refinement_experiment(config, root)
    stress_experiment(config, root)
    from dasa.portbench import run as run_ports
    run_ports(config, root / "ports")
    from dasa.reporting import report
    report(config, root)
    write_json(root / "run_status.json", {"completed": True, "elapsed_seconds": time.monotonic() - started,
                                          "config": config, "provenance": "synthetic_task_space_simulation"})
