import numpy as np
from scipy.signal import savgol_filter


def _time_weights(time):
    time = np.asarray(time, float)
    if (time.ndim != 1 or not len(time) or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0)):
        raise ValueError("Metric timestamps must be finite and strictly increasing")
    if len(time) == 1:
        return np.ones(1)
    intervals = np.diff(time)
    weights = np.r_[intervals[0] / 2, (intervals[:-1] + intervals[1:]) / 2,
                    intervals[-1] / 2]
    return weights / weights.sum()


def intent_metrics(time, posterior, true_goal: int, config: dict, completion: float | None = None):
    weights = _time_weights(time)
    posterior = np.asarray(posterior, float)
    if (posterior.ndim != 2 or len(posterior) != len(weights)
            or not np.isfinite(posterior).all() or np.any(posterior < 0)
            or not np.allclose(posterior.sum(1), 1.) or not 0 <= true_goal < posterior.shape[1]):
        raise ValueError("Intent metrics require a finite categorical posterior and recorded goal")
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
            ece += weights[selected].sum() * abs(
                np.average(correct[selected], weights=weights[selected])
                - np.average(confidence[selected], weights=weights[selected]))
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
    return {"accuracy": float(weights @ correct),
            "ipa": float(weights @ (correct & confident)) if len(time) > 1 else None,
            "prediction_stability": float(1 - np.sum(prediction[1:] != prediction[:-1]) / len(prediction)),
            "nll": float(weights @ -np.log(np.clip(posterior[:, true_goal], 1e-12, 1))),
            "brier": float(weights @ np.sum((posterior - target) ** 2, axis=1)), "ece": float(ece),
            "early_prediction_seconds": max(0.0, completion - earliest)
            if completion is not None and earliest is not None else None,
            "prediction_time_weighting": "trapezoidal_sample_time_weights",
            "inference_samples": len(time), "observation_duration_seconds": float(time[-1] - time[0]),
            "confidence_threshold": float(config["intent"]["confidence_threshold"]),
            "sustained_prediction_seconds": float(config["intent"]["sustained_seconds"])}


def measured_motion_metrics(config, time, position, velocity, scene, robot_radius, measured_clearance=None):
    parameters = config.get("analysis", {}).get("metrics")
    if not isinstance(parameters, dict):
        raise ValueError("A frozen analysis.metrics configuration is required")
    numeric = {}
    for key in ("resample_dt_seconds", "smoothing_window_seconds", "gamma_st_s3_per_m", "gamma_sa_s2_per_m"):
        value = parameters.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"A positive configured metric parameter is required: {key}")
        numeric[key] = float(value)
    order = parameters.get("polynomial_order")
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("The configured derivative polynomial order must be an integer of at least two")
    time, position, velocity = np.asarray(time, float), np.asarray(position, float), np.asarray(velocity, float)
    _time_weights(time)
    if (len(time) < 3 or position.shape != (len(time), 3) or velocity.shape != position.shape
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()
            or not np.isfinite(robot_radius) or robot_radius <= 0 or scene.frame != "base"
            or scene.provenance not in {"calibrated_scene", "calibrated_rgbd_geometry"}):
        raise ValueError("Motion indices require measured base-frame signals and calibrated obstacle geometry")
    online_clearance = None
    if measured_clearance is not None:
        online_clearance = np.asarray(measured_clearance, float)
        if online_clearance.shape != time.shape or not np.isfinite(online_clearance).all():
            raise ValueError("Online surface clearance must be finite and aligned with every measured timestamp")
        clearance = float(online_clearance.min())
    else:
        clearance = min(scene.swept_clearance(start, end, robot_radius)
                        for start, end in zip(position[:-1], position[1:]))
    duration = float(time[-1] - time[0])
    intervals = max(1, int(np.ceil(duration / numeric["resample_dt_seconds"])))
    grid = np.linspace(time[0], time[-1], intervals + 1)
    actual_dt = duration / intervals
    window = max(order + 1, int(np.rint(numeric["smoothing_window_seconds"] / actual_dt)))
    if window % 2 == 0:
        window += 1
    audit = {"parameter_provenance": "configured_operational_values", "parameters": {**numeric,
             "polynomial_order": order}, "resampled_dt_seconds": actual_dt,
             "smoothing_window_samples": window, "resampled_samples": len(grid),
             "omega_definition": "norm_filtered_translational_TCP_velocity_m_per_s",
             "stability_definition": "exp_negative_absolute_gamma_st_times_second_derivative_omega",
             "safety_definition": "exp_negative_gamma_sa_times_omega_squared_divided_by_surface_clearance",
             "nonpositive_clearance_safety": 0., "trial_aggregation": "trapezoidal_time_mean",
             "interpolation": "linear_measured_samples_endpoint_preserving_uniform_grid",
             "filter": "Savitzky_Golay_velocity_and_omega_derivatives_interp_boundaries",
             "collision_geometry": "TCP_sphere_and_obstacle_spheres_linear_swept_segments_all_phases",
             "units": {"stability_index": "1", "safety_index": "1", "omega": "m/s",
                       "omega_ddot": "m/s^3", "gamma_st_s3_per_m": "s^3/m",
                       "gamma_sa_s2_per_m": "s^2/m", "jerk_rms": "m/s^3", "clearance": "m"}}
    if online_clearance is not None:
        audit.update(collision_geometry="sampled_online_scene_TCP_surface_clearance",
                     clearance_temporal_interpolation="linear_recorded_clearance_between_actual_timestamps",
                     clearance_scope="sampled_online_geometry_without_between_frame_swept_assertion")
    else:
        audit["clearance_scope"] = "static_scene_linear_swept_TCP_sphere"
    values = {"minimum_clearance_m": float(clearance), "jerk_rms": None,
              "stability_index": None, "safety_index": None}
    if len(grid) < window:
        audit["status"] = "insufficient_duration_for_configured_filter"
        return values, audit, {}
    sampled_position = np.column_stack([np.interp(grid, time, position[:, axis]) for axis in range(3)])
    sampled_velocity = np.column_stack([np.interp(grid, time, velocity[:, axis]) for axis in range(3)])
    filtered_velocity = savgol_filter(sampled_velocity, window, order, axis=0, mode="interp")
    omega = np.linalg.norm(filtered_velocity, axis=1)
    omega_ddot = savgol_filter(omega, window, order, deriv=2, delta=actual_dt, mode="interp")
    jerk = savgol_filter(sampled_velocity, window, order, deriv=2, delta=actual_dt, axis=0, mode="interp")
    if online_clearance is not None:
        ell = np.interp(grid, time, online_clearance)
    else:
        distances = np.linalg.norm(sampled_position[:, None] - scene.centers[None], axis=2)
        ell = (distances - scene.radii[None] - robot_radius).min(axis=1)
    stability = np.exp(-np.abs(numeric["gamma_st_s3_per_m"] * omega_ddot))
    safety = np.zeros(len(grid))
    separated = ell > 0
    safety[separated] = np.exp(-numeric["gamma_sa_s2_per_m"] * omega[separated] ** 2 / ell[separated])
    weights = _time_weights(grid)
    values.update(jerk_rms=float(np.sqrt(weights @ np.sum(jerk ** 2, axis=1))),
                  stability_index=float(weights @ stability), safety_index=float(weights @ safety))
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Measured motion indices must be finite")
    audit["status"] = "computed"
    series = {"time": grid, "position": sampled_position, "filtered_velocity": filtered_velocity,
              "omega": omega, "omega_ddot": omega_ddot, "surface_clearance": ell,
              "stability_index": stability, "safety_index": safety, "jerk": jerk}
    return values, audit, series


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
