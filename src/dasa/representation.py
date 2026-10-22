from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from dasa.config import manifest, write_json
from dasa.statistics import comparisons, holm


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def _participants(path, group_column):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"participant_id", group_column}.issubset(reader.fieldnames or []):
            raise ValueError(f"Participant CSV requires participant_id and explicit {group_column}")
        rows = list(reader)
    result = {}
    for row in rows:
        identifier = (row.get("participant_id") or "").strip()
        group = (row.get(group_column) or "").strip()
        if not identifier or identifier in result:
            raise ValueError("Participant identifiers must be nonempty and unique")
        status = row.get("inclusion_status", "included").strip() or "included"
        if status not in {"included", "excluded", "withdrawn"}:
            raise ValueError("Unknown participant inclusion status")
        if status == "included" and not group:
            raise ValueError("Included participants require an explicitly recorded experience group")
        if row.get("record_status", "collected") != "collected":
            raise ValueError("Planned participant rows cannot supply observed experience labels")
        result[identifier] = {"group": group, "included": status == "included"}
    if not any(row["included"] for row in result.values()):
        raise ValueError("No included participants with recorded group labels")
    return result


def _settings(config):
    supplied = config.get("representation", {})
    settings = {
        "delay_seconds": 0.4, "delay_tolerance_seconds": 1e-9,
        "methods": ["full", "no_latent", "no_delay"],
        "samples_per_episode": 64, "samples_per_goal_per_operator": 64,
        "required_goals": [0, 1, 2], "maximum_tsne_points": 600,
        "bootstrap_samples": config["experiments"]["bootstrap_samples"],
        "seed": config["seed"],
        "tsne": {"perplexity": 30.0, "iterations": 1000, "learning_rate": 200.0,
                 "early_exaggeration": 12.0, "early_exaggeration_iterations": 250,
                 "initialization": "pca", "initial_scale": 0.0001,
                 "initial_momentum": 0.5, "final_momentum": 0.8,
                 "minimum_gain": 0.01, "entropy_tolerance": 0.00001,
                 "precision_search_iterations": 80}}
    for key, value in supplied.items():
        if key not in settings:
            raise ValueError(f"Unknown representation setting: {key}")
        if key == "tsne":
            if not isinstance(value, dict) or any(name not in settings["tsne"] for name in value):
                raise ValueError("Unknown t-SNE setting")
            settings["tsne"].update(value)
        else:
            settings[key] = value
    for key in ("samples_per_episode", "samples_per_goal_per_operator", "maximum_tsne_points",
                "bootstrap_samples", "seed"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid representation integer: {key}")
    if (not np.isfinite([settings["delay_seconds"], settings["delay_tolerance_seconds"]]).all()
            or settings["delay_seconds"] < 0 or settings["delay_tolerance_seconds"] < 0):
        raise ValueError("Representation delay settings must be finite and nonnegative")
    if not settings["methods"] or len(set(settings["methods"])) != len(settings["methods"]):
        raise ValueError("Representation methods must be nonempty and unique")
    if settings["required_goals"] != [0, 1, 2]:
        raise ValueError("This three-goal analysis requires the declared goal set [0,1,2]")
    t = settings["tsne"]
    numeric = [value for value in t.values() if isinstance(value, (int, float))]
    if not np.isfinite(numeric).all() or t["perplexity"] < 1 or t["learning_rate"] <= 0:
        raise ValueError("Invalid t-SNE numeric settings")
    for key in ("iterations", "early_exaggeration_iterations", "precision_search_iterations"):
        if isinstance(t[key], bool) or not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"Invalid t-SNE iteration count: {key}")
    if (t["iterations"] <= t["early_exaggeration_iterations"] or t["early_exaggeration"] < 1
            or t["initialization"] not in {"pca", "random"} or t["initial_scale"] <= 0
            or t["minimum_gain"] <= 0 or t["entropy_tolerance"] <= 0
            or not 0 <= t["initial_momentum"] < 1 or not 0 <= t["final_momentum"] < 1):
        raise ValueError("Invalid t-SNE initialization, momentum or exaggeration settings")
    return settings


def _episodes(root, participants, settings):
    records, excluded, seen = [], [], set()
    files = sorted((root / "intent/predictions").glob("*.npz"))
    if not files:
        raise ValueError("Recorded prediction files are required")
    for path in files:
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
            method = metadata["method"]
            if method not in settings["methods"]:
                continue
            if not np.isclose(metadata["delay_seconds"], settings["delay_seconds"],
                              atol=settings["delay_tolerance_seconds"], rtol=0):
                continue
            identifier = metadata["unit"]
            if identifier not in participants:
                raise ValueError("A recorded operator has no explicit participant registration")
            if not participants[identifier]["included"]:
                excluded.append({"file": path.name, "reason": "participant_not_included"})
                continue
            if metadata["provenance"] != "UR10_measured_teleoperation":
                raise ValueError("Representation analysis requires physical participant recordings")
            if metadata.get("evaluation") != "offline_recorded_signals_no_generated_interaction":
                raise ValueError("Expected the recorded-signal analysis contract")
            posterior = np.asarray(arrays["posterior"], dtype=float)
            native = np.asarray(arrays["native_representation"], dtype=float)
            context = np.asarray(arrays["model_context"], dtype=float)
            time = np.asarray(arrays["time"], dtype=float)
            goal = np.asarray(arrays["true_goal"]).item()
            if (isinstance(goal, bool) or not isinstance(goal, int) or goal not in settings["required_goals"]
                    or posterior.shape != (len(time), 3) or native.ndim != 2
                    or native.shape[0] != len(time) or native.shape[1] < 1 or context.shape != (len(time), 19)
                    or not len(time) or not all(np.isfinite(value).all() for value in
                                               (posterior, native, context, time))
                    or np.any(posterior < 0) or not np.allclose(posterior.sum(1), 1, atol=1e-6)
                    or np.any(np.diff(time) <= 0)):
                raise ValueError("Invalid recorded representation arrays")
            native_kind = metadata["native_representation_kind"]
            if "latent" in native_kind and not metadata.get("latent_available", False):
                raise ValueError("A method without an encoder cannot be labelled as a latent representation")
            key = (method, metadata["episode_id"])
            if key in seen:
                raise ValueError("Duplicate method/episode prediction artifacts")
            seen.add(key)
            selection = np.unique(np.linspace(0, len(time) - 1,
                                  min(len(time), settings["samples_per_episode"])).astype(int))
            prediction = posterior.argmax(1)
            records.append({"path": path, "method": method, "unit": identifier,
                            "group": participants[identifier]["group"], "episode": metadata["episode_id"],
                            "goal": goal, "posterior": posterior[selection], "native": native[selection],
                            "context": context[selection], "time": time[selection],
                            "source_indices": np.asarray(arrays["source_indices"])[selection],
                            "native_kind": native_kind,
                            "accuracy": float(np.mean(prediction == goal)),
                            "prediction_stability": float(1 - np.sum(prediction[1:] != prediction[:-1]) / len(prediction))})
    return records, excluded


def _silhouette(features, labels):
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or len(classes) >= len(labels):
        return None
    distances = cdist(features, features, metric="euclidean")
    within = np.zeros(len(labels))
    between = np.full(len(labels), np.inf)
    for label, count in zip(classes, counts):
        mask = labels == label
        mean = distances[:, mask].sum(1) / count
        within[mask] = distances[mask][:, mask].sum(1) / max(count - 1, 1)
        between[~mask] = np.minimum(between[~mask], mean[~mask])
    score = (between - within) / np.maximum(np.maximum(within, between), 1e-12)
    for label, count in zip(classes, counts):
        if count == 1:
            score[labels == label] = 0
    return float(score.mean())


def _balanced_silhouette(records, space, settings):
    features, labels = [], []
    by_goal = {}
    for record in records:
        by_goal.setdefault(record["goal"], []).append(record[space])
    if sorted(by_goal) != settings["required_goals"]:
        return None, "incomplete_goal_coverage", 0
    joined = {goal: np.vstack(values) for goal, values in by_goal.items()}
    count = min(settings["samples_per_goal_per_operator"], min(len(values) for values in joined.values()))
    if count < 2:
        return None, "fewer_than_two_samples_per_goal", 0
    for goal in settings["required_goals"]:
        values = joined[goal]
        indices = np.linspace(0, len(values) - 1, count).astype(int)
        features.append(values[indices])
        labels.extend([goal] * count)
    return _silhouette(np.vstack(features), np.asarray(labels)), "available", len(labels)


def _bootstrap(values, seed, count):
    values = np.asarray(values, float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(count, len(values)))].mean(1)
    return np.quantile(means, [.025, .975]).tolist()


def _operator_rows(records, settings):
    result = []
    keys = sorted({(row["method"], row["unit"]) for row in records})
    for method, unit in keys:
        selected = [row for row in records if row["method"] == method and row["unit"] == unit]
        for space in ("posterior", "native"):
            kinds = {row["native_kind"] for row in selected} if space == "native" else {
                "categorical_goal_posterior_probability"}
            if len(kinds) != 1:
                raise ValueError("Native feature kind differs between episodes of one model")
            value, status, samples = _balanced_silhouette(selected, space, settings)
            result.append({"condition": method, "unit": unit, "group": selected[0]["group"],
                           "delay_seconds": settings["delay_seconds"], "feature_space": space,
                           "feature_kind": next(iter(kinds)), "silhouette": value,
                           "silhouette_status": status, "silhouette_samples": samples,
                           "n_episodes": len(selected), "observed_goals": sorted({row["goal"] for row in selected}),
                           "prediction_stability": float(np.mean([row["prediction_stability"] for row in selected])),
                           "accuracy": float(np.mean([row["accuracy"] for row in selected]))})
    return result


def _group_rows(rows, participants, settings):
    result = []
    groups = sorted({row["group"] for row in participants.values() if row["included"]})
    for group in groups:
        for method in settings["methods"]:
            for space in ("posterior", "native"):
                selected = [row for row in rows if row["group"] == group and row["condition"] == method
                            and row["feature_space"] == space]
                for metric in ("silhouette", "prediction_stability", "accuracy"):
                    usable = [row for row in selected if row[metric] is not None]
                    values = [row[metric] for row in usable]
                    tag = f"{group}|{method}|{space}|{metric}"
                    offset = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")
                    result.append({"group": group, "condition": method, "feature_space": space,
                                   "metric": metric, "n_units": len(usable),
                                   "n_units_missing": len(selected) - len(usable),
                                   "units": [row["unit"] for row in usable],
                                   "mean": float(np.mean(values)) if values else None,
                                   "sd_between_operators": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                   "ci95": _bootstrap(values, (settings["seed"] + offset) % 2**32,
                                                       settings["bootstrap_samples"]),
                                   "interval": "percentile bootstrap of independent operator means",
                                   "delay_seconds": settings["delay_seconds"]})
    return result


def exact_tsne(features, settings, seed):
    features = np.asarray(features, float)
    n = len(features)
    if (features.ndim != 2 or not np.isfinite(features).all() or n < 3
            or not 1 <= settings["perplexity"] < n):
        raise ValueError("t-SNE requires finite features and perplexity smaller than the sample count")
    distances = cdist(features, features, metric="sqeuclidean")
    conditional = np.zeros((n, n))
    target_entropy = np.log(settings["perplexity"])
    entropy_errors = []
    for index in range(n):
        mask = np.arange(n) != index
        row = distances[index, mask]
        precision, lower, upper = 1.0, 0.0, np.inf
        for _ in range(settings["precision_search_iterations"]):
            weights = np.exp(-(row - row.min()) * precision)
            probabilities = weights / weights.sum()
            entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
            error = entropy - target_entropy
            if abs(error) <= settings["entropy_tolerance"]:
                break
            if error > 0:
                lower = precision
                precision = 2 * precision if np.isinf(upper) else .5 * (precision + upper)
            else:
                upper = precision
                precision = .5 * (precision + lower)
        conditional[index, mask] = probabilities
        entropy_errors.append(float(abs(error)))
    p = (conditional + conditional.T) / (2 * n)
    p /= p.sum()
    rng = np.random.default_rng(seed)
    if settings["initialization"] == "pca" and features.shape[1] >= 2:
        centered = features - features.mean(0)
        left, singular, _ = np.linalg.svd(centered, full_matrices=False)
        y = left[:, :2] * singular[:2]
        y *= settings["initial_scale"] / max(float(y[:, 0].std()), 1e-12)
    else:
        y = rng.normal(0, settings["initial_scale"], (n, 2))
    y += rng.normal(0, settings["initial_scale"] * 1e-4, y.shape)
    velocity, gains = np.zeros_like(y), np.ones_like(y)
    history = []
    for iteration in range(settings["iterations"]):
        numerator = 1 / (1 + cdist(y, y, metric="sqeuclidean"))
        np.fill_diagonal(numerator, 0)
        q = numerator / numerator.sum()
        early = iteration < settings["early_exaggeration_iterations"]
        factor = settings["early_exaggeration"] if early else 1.0
        coefficients = (factor * p - q) * numerator
        gradient = 4 * (coefficients.sum(1)[:, None] * y - coefficients @ y)
        gains = np.where(np.sign(gradient) != np.sign(velocity), gains + .2, gains * .8)
        gains = np.maximum(gains, settings["minimum_gain"])
        momentum = settings["initial_momentum"] if early else settings["final_momentum"]
        velocity = momentum * velocity - settings["learning_rate"] * gains * gradient
        y += velocity
        y -= y.mean(0)
        if not np.isfinite(y).all():
            raise ValueError("t-SNE optimization diverged; preserve settings and inspect the input features")
        if iteration % 50 == 0 or iteration == settings["iterations"] - 1:
            divergence = float(np.sum(p * (np.log(np.maximum(p, 1e-300)) -
                                            np.log(np.maximum(q, 1e-300)))))
            history.append({"iteration": iteration, "kl_p_q": divergence,
                            "exaggeration_active": early})
    final_numerator = 1 / (1 + cdist(y, y, metric="sqeuclidean"))
    np.fill_diagonal(final_numerator, 0)
    final_q = final_numerator / final_numerator.sum()
    final_kl = float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(final_q, 1e-300)))))
    return y, {"implementation": "exact symmetric Student-t SNE; Euclidean feature distances",
               "entropy_search_maximum_error": max(entropy_errors),
               "entropy_search_converged": max(entropy_errors) <= settings["entropy_tolerance"],
               "final_kl_p_q": final_kl, "optimization_history": history,
               "seed": seed, "settings": settings, "n_points": n}


def _tsne_inputs(records, space, kind, maximum):
    points = []
    for record in records:
        if space == "native" and record["native_kind"] != kind:
            continue
        for index, feature in enumerate(record[space]):
            points.append({"feature": feature, "method": record["method"], "group": record["group"],
                           "participant_id": record["unit"], "episode_id": record["episode"],
                           "goal": record["goal"], "source_index": int(record["source_indices"][index]),
                           "time_s": float(record["time"][index])})
    buckets = {}
    for point in points:
        buckets.setdefault((point["method"], point["group"], point["goal"], point["participant_id"]), []).append(point)
    if len(buckets) > maximum:
        return []
    quota = max(1, maximum // max(len(buckets), 1))
    for key, values in buckets.items():
        indices = np.unique(np.linspace(0, len(values) - 1, min(len(values), quota)).astype(int))
        buckets[key] = [values[index] for index in indices]
    selected, depth = [], 0
    while len(selected) < maximum:
        added = 0
        for key in sorted(buckets):
            values = buckets[key]
            if depth < len(values) and len(selected) < maximum:
                selected.append(values[depth])
                added += 1
        if not added:
            break
        depth += 1
    return selected


def _plot_tsne(points, coordinates, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods, groups = sorted({row["method"] for row in points}), sorted({row["group"] for row in points})
    fig, axes = plt.subplots(len(groups), len(methods), figsize=(4 * len(methods), 3.5 * len(groups)),
                             squeeze=False, sharex=True, sharey=True)
    colors = ["#2479b2", "#ce852b", "#32916c"]
    for row_index, group in enumerate(groups):
        for column, method in enumerate(methods):
            axis = axes[row_index, column]
            for goal in (0, 1, 2):
                mask = np.array([point["group"] == group and point["method"] == method
                                 and point["goal"] == goal for point in points])
                axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=10, alpha=.7,
                             c=colors[goal], label=f"Goal {goal + 1}", linewidths=0)
            axis.set_title(f"{method} | {group}", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, .94))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _operator_diagnostic(records, settings):
    results = []
    groups = sorted({row["group"] for row in records})
    for method in ("full", "no_delay"):
        for group in groups:
            for goal in settings["required_goals"]:
                selected = [row for row in records if row["method"] == method and row["group"] == group
                            and row["goal"] == goal]
                counts = {unit: sum(row["unit"] == unit for row in selected) for unit in
                          {row["unit"] for row in selected}}
                eligible = {unit for unit, count in counts.items() if count >= 2}
                selected = [row for row in selected if row["unit"] in eligible]
                status = "available" if len(eligible) >= 2 else "requires_two_operators_with_two_episodes_each_for_same_goal"
                for space in ("native", "context"):
                    result = {"method": method, "group": group, "goal": goal, "feature_space": space,
                              "status": status, "n_operators": len(eligible), "n_episodes": len(selected),
                              "excluded_operator_episode_counts": {unit: count for unit, count in counts.items()
                                                                     if unit not in eligible},
                              "design": "leave-one-episode-out nearest-operator-centroid within one goal",
                              "standardization": "training-fold mean and SD only",
                              "interpretation": "repeated-operator identifiability; scene, session and time remain possible confounds; not causal habit isolation"}
                    if status != "available":
                        results.append(result)
                        continue
                    features = np.vstack([row[space].mean(0) for row in selected])
                    labels = np.array([row["unit"] for row in selected])
                    predictions = []
                    for held_out in range(len(selected)):
                        training = np.arange(len(selected)) != held_out
                        mean, sd = features[training].mean(0), features[training].std(0)
                        normalized = (features - mean) / np.maximum(sd, 1e-8)
                        classes = sorted(set(labels[training]))
                        centers = np.vstack([normalized[training & (labels == label)].mean(0) for label in classes])
                        predictions.append(classes[np.argmin(np.linalg.norm(centers - normalized[held_out], axis=1))])
                    predictions = np.asarray(predictions)
                    recalls = [{"unit": unit, "recall": float(np.mean(predictions[labels == unit] == unit))}
                               for unit in sorted(eligible)]
                    result.update(balanced_accuracy=float(np.mean([row["recall"] for row in recalls])),
                                  chance=1 / len(eligible), operator_recalls=recalls,
                                  ci95=_bootstrap([row["recall"] for row in recalls], settings["seed"],
                                                  settings["bootstrap_samples"]),
                                  interval_scope="operator bootstrap conditional on these fixed held-out-episode predictions",
                                  episode_predictions=[{"episode_id": row["episode"], "true_operator": row["unit"],
                                                        "predicted_operator": str(prediction)}
                                                       for row, prediction in zip(selected, predictions)])
                    results.append(result)
    return results


def analyze(config: dict, recorded_root: Path, participants_path: Path, output: Path,
            group_column: str = "experience_group") -> dict:
    recorded_root, participants_path, output = Path(recorded_root), Path(participants_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Representation output must be empty")
    settings = _settings(config)
    participants = _participants(participants_path, group_column)
    episodes, excluded = _episodes(recorded_root, participants, settings)
    output.mkdir(parents=True, exist_ok=True)
    rows = _operator_rows(episodes, settings)
    groups = _group_rows(rows, participants, settings)
    write_json(output / "operator_metrics.json", rows)
    write_json(output / "group_summary.json", groups)
    _csv(output / "operator_metrics.csv", rows)
    _csv(output / "group_summary.csv", groups)
    statistics = []
    for group in sorted({row["group"] for row in participants.values() if row["included"]}):
        selected = [row for row in rows if row["group"] == group and row["feature_space"] == "posterior"]
        statistics.extend({"experience_group": group, "feature_space": "categorical_goal_posterior_probability", **row}
                          for row in comparisons(selected, "full", [method for method in settings["methods"] if method != "full"],
                                                 ["silhouette", "prediction_stability"], config,
                                                 family="representation_all_groups_methods_metrics",
                                                 planned_delays=[settings["delay_seconds"]]))
    for row, corrected in zip(statistics, holm([row["p_raw"] for row in statistics], family_size=len(statistics))):
        row.update(p_holm=corrected, family_size_planned=len(statistics),
                   family_size_estimable=sum(value["p_raw"] is not None for value in statistics),
                   significant_holm=corrected < .05 if corrected is not None else None)
    write_json(output / "statistics.json", statistics)
    _csv(output / "statistics.csv", statistics)
    write_json(output / "within_goal_operator_information.json", _operator_diagnostic(episodes, settings))
    tsne = []
    spaces = [("posterior", "categorical_goal_posterior_probability")]
    spaces.extend(("native", kind) for kind in sorted({row["native_kind"] for row in episodes}))
    for space, kind in spaces:
        points = _tsne_inputs(episodes, space, kind, settings["maximum_tsne_points"])
        name = "".join(character if character.isalnum() else "_" for character in f"{space}_{kind}")
        item = {"feature_space": space, "feature_kind": kind, "n_points": len(points),
                "silhouette_space": "original declared features, never t-SNE coordinates"}
        if len(points) <= settings["tsne"]["perplexity"]:
            item.update(status="insufficient_points_for_declared_perplexity")
        elif np.max(np.std(np.vstack([point["feature"] for point in points]), axis=0)) < 1e-12:
            item.update(status="constant_features_no_informative_embedding")
        else:
            features = np.vstack([point["feature"] for point in points])
            coordinates, audit = exact_tsne(features, settings["tsne"], settings["seed"])
            point_rows = [{key: value for key, value in point.items() if key != "feature"}
                          for point in points]
            for record, coordinate in zip(point_rows, coordinates):
                record.update(tsne_1=float(coordinate[0]), tsne_2=float(coordinate[1]))
            np.savez_compressed(output / f"{name}.npz", features=features, coordinates=coordinates,
                                metadata=np.array(json.dumps(point_rows)))
            _csv(output / f"{name}.csv", point_rows)
            _plot_tsne(points, coordinates, output / name,
                       f"{kind} | {settings['delay_seconds'] * 1000:g} ms | shared fit within this feature space")
            item.update(status="available" if audit["entropy_search_converged"] else
                        "available_with_affinity_entropy_tolerance_warning", output_stem=name, **audit)
        tsne.append(item)
    write_json(output / "tsne_audit.json", tsne)
    protocol = {"settings": settings, "group_source": f"explicit CSV column {group_column}",
                "included_registrations": sum(row["included"] for row in participants.values()),
                "observed_operators": sorted({row["unit"] for row in episodes}),
                "included_prediction_episodes": len(episodes), "excluded": excluded,
                "status": "analyzed" if episodes else "no_physical_recordings_at_declared_delay",
                "silhouette": "per operator, three observed goals, balanced deterministic samples, Euclidean original features",
                "native_space_caution": "different native feature kinds are distinct analyses; context-only features are not a latent embedding",
                "cross_method_primary_space": "three-dimensional categorical goal posterior; supervised readout separation is not independent latent validation",
                "sample_selection": "evenly spaced within episodes, then balanced by goal within operator; no outcome-based selection",
                "bootstrap_unit": "operator; repeated episodes averaged first",
                "missingness": "insufficient per-operator goal coverage remains unavailable, never filled from pooled participants",
                "operator_information": "within-goal repeated-episode identity diagnostic with context-feature control; no causal operator-habit claim"}
    write_json(output / "protocol.json", protocol)
    record = manifest(config, "representation-analyze", [participants_path, *[row["path"] for row in episodes]])
    record["data_provenance"] = "physical_recorded_participant_signals_only"
    record["participant_labels"] = "explicit experience groups; no performance-based or inferred labels"
    write_json(output / "manifest.json", record)
    return {"operator_rows": len(rows), "group_rows": len(groups), "tsne": tsne,
            "status": protocol["status"]}
