import numpy as np

from dasa.simulation import Trial


def intent_metrics(time, posterior, true_goal: int, config: dict, completion: float | None = None):
    prediction = np.argmax(posterior, axis=1)
    confidence = np.max(posterior, axis=1)
    correct = prediction == true_goal
    confident = confidence >= config["intent"]["confidence_threshold"]
    target = np.eye(posterior.shape[1])[np.repeat(true_goal, len(posterior))]
    ece = 0.0
    bins = np.minimum((confidence * 10).astype(int), 9)
    for index in range(10):
        selected = bins == index
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    earliest = None
    start = None
    for t, usable in zip(time, correct & confident):
        if usable:
            start = float(t) if start is None else start
            if t - start >= config["intent"]["sustained_seconds"]:
                earliest = start
                break
        else:
            start = None
    return {"accuracy": float(correct.mean()), "ipa": float((correct & confident).mean()),
            "prediction_stability": float(1 - np.sum(prediction[1:] != prediction[:-1]) / len(prediction)),
            "nll": float(-np.log(np.clip(posterior[:, true_goal], 1e-12, 1)).mean()),
            "brier": float(np.sum((posterior - target) ** 2, axis=1).mean()), "ece": float(ece),
            "early_prediction_seconds": max(0.0, completion - earliest)
            if completion is not None and earliest is not None else None}


def trial_metrics(trial: Trial, config: dict) -> dict:
    a, m = trial.arrays, trial.metadata
    time, velocity = a["time"], a["velocity"]
    if len(time) >= 3:
        acceleration = np.gradient(velocity, time, axis=0)
        jerk = np.gradient(acceleration, time, axis=0)
        jerk_rms = float(np.sqrt(np.mean(np.sum(jerk**2, axis=1))))
    else:
        jerk_rms = None
    step_time = np.diff(np.r_[0, time])
    completion = float(time[-1]) if m["success"] else None
    pick = a["phase"] == 0
    inference = intent_metrics(a["decision_time"][pick], a["posterior"][pick], m["intended_goal"], config, completion) if pick.any() else {}
    return {"trial_id": m["trial_id"], "unit": m["operator_id"], "condition": m["condition"],
            "delay_seconds": m["delay_seconds"], "seed": m["seed"], "provenance": m["provenance"],
            "success_rate": float(m["success"]), "completion_seconds": completion,
            "elapsed_seconds": float(time[-1]), "collision_rate": float(m["collision"]),
            "minimum_clearance_m": float(a["clearances"].min()),
            "path_length_m": float(np.linalg.norm(np.diff(a["position"], axis=0), axis=1).sum()),
            "jerk_rms": jerk_rms, "reference_error_m": float(np.linalg.norm(
                a["position"] - a["reference"], axis=1).mean()),
            "integrated_force_squared": float(np.sum(step_time * np.sum(a["executed_force"]**2, axis=1))),
            "maximum_energy_residual": float(np.abs(a["energy_residual"]).max()),
            "minimum_tank_energy": float(a["tank_energy"].min()),
            "removed_work_j": float(a["removed_work"].sum()),
            "corrective_samples": int(a["corrective"].sum()),
            **{f"intent_{k}": v for k, v in inference.items()}}


def silhouette(embeddings: np.ndarray, labels: np.ndarray, seed: int, maximum=600):
    from scipy.spatial.distance import cdist
    if len(np.unique(labels)) < 2 or len(embeddings) < 3:
        return None
    rng = np.random.default_rng(seed)
    if len(embeddings) > maximum:
        ix = rng.choice(len(embeddings), maximum, replace=False)
        embeddings, labels = embeddings[ix], labels[ix]
    distances = cdist(embeddings, embeddings)
    scores = []
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = distances[i, same].mean()
        b = min(distances[i, labels == g].mean() for g in np.unique(labels) if g != labels[i])
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores))
