from __future__ import annotations

from pathlib import Path
import hashlib
import json
import copy

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

from dasa.data import Demonstration, resample_positions, successful
from dasa.geometry import Scene, task_frames



def demonstration_fingerprint(demo: Demonstration) -> dict:

    demo.validate()
    digest = hashlib.sha256()
    for key in ("time", "position", "velocity", "raw_force", "observed_state", "phase"):
        array = np.ascontiguousarray(getattr(demo, key), dtype="<f8")
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    geometry_data = demo.scene.to_dict()
    geometry_data.pop("provenance", None)
    geometry_data.pop("obstacle_ids", None)
    geometry = json.dumps({"scene": geometry_data, "goal": demo.goal},
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest.update(geometry)
    signal_hash = digest.hexdigest()
    record = json.dumps(demo.metadata, sort_keys=True, separators=(",", ":"),
                        allow_nan=False).encode()
    return {"episode_id": demo.metadata["episode_id"],
            "operator_id": demo.metadata["operator_id"],
            "provenance": demo.metadata["provenance"],
            "source_condition": demo.metadata.get("source_condition"),
            "goal": int(demo.goal), "samples": len(demo.time),
            "duration_seconds": float(demo.time[-1] - demo.time[0]),
            "signal_sha256": signal_hash,
            "record_sha256": hashlib.sha256(bytes.fromhex(signal_hash) + record).hexdigest()}

def gaussian_logpdf(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    delta = x - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Emission covariance must be positive definite")
    return -0.5 * (3 * np.log(2 * np.pi) + logdet +
                   np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta))


def duration_logpmf(mean: float, std: float, maximum: int) -> np.ndarray:
    d = np.arange(1, maximum + 1)
    logp = -0.5 * ((d - mean) / max(std, 1e-6)) ** 2
    return logp - logsumexp(logp)


def segment_posterior(log_emission: np.ndarray, duration: np.ndarray):





    t_count, states = log_emission.shape
    max_duration = duration.shape[1]
    cumulative = np.vstack([np.zeros(states), np.cumsum(log_emission, axis=0)])
    forward = np.full((states + 1, t_count + 1), -np.inf)
    forward[0, 0] = 0
    for k in range(1, states + 1):
        for end in range(k, t_count + 1):
            ds = np.arange(1, min(max_duration, end - k + 1) + 1)
            start = end - ds
            score = forward[k - 1, start] + duration[k - 1, ds - 1]
            score += cumulative[end, k - 1] - cumulative[start, k - 1]
            forward[k, end] = logsumexp(score)
    log_likelihood = forward[states, t_count]
    if not np.isfinite(log_likelihood):
        raise ValueError("Duration support cannot cover the complete trajectory")
    backward = np.full_like(forward, -np.inf)
    backward[states, t_count] = 0
    for k in range(states - 1, -1, -1):
        for start in range(t_count):
            ds = np.arange(1, min(max_duration, t_count - start) + 1)
            end = start + ds
            score = duration[k, ds - 1] + cumulative[end, k] - cumulative[start, k]
            backward[k, start] = logsumexp(score + backward[k + 1, end])
    gamma = np.zeros_like(log_emission)
    duration_counts = np.zeros_like(duration)
    for k in range(states):
        differences = np.zeros(t_count + 1)
        for start in range(t_count):
            ds = np.arange(1, min(max_duration, t_count - start) + 1)
            end = start + ds
            score = forward[k, start] + duration[k, ds - 1]
            score += cumulative[end, k] - cumulative[start, k] + backward[k + 1, end]
            probability = np.exp(score - log_likelihood)
            differences[start] += probability.sum()
            np.add.at(differences, end, -probability)
            duration_counts[k, ds - 1] += probability
        gamma[:, k] = np.cumsum(differences)[:-1]
    return gamma, duration_counts, float(log_likelihood)


class TPHSMM:
    def __init__(self, config: dict):
        self.config = config
        self.k = config["skill"]["states"]
        self.samples = config["skill"]["trajectory_samples"]
        self.means = np.zeros((self.k, 3, 3))
        self.covariances = np.repeat(np.eye(3)[None, None], self.k * 3, axis=0).reshape(
            self.k, 3, 3, 3) * 0.01
        self.duration_mean = np.full(self.k, self.samples / self.k)
        self.duration_std = np.full(self.k, self.samples / self.k / 3)
        self.history = []
        self.fit_ids = []
        self.fit_manifest = {"available": False, "reason": "not_fitted"}
        self.duration_seconds = 9.0

    def world_gaussians(self, scene: Scene, goal: int):
        rotations, offsets = task_frames(scene, goal)
        means, covariances = [], []
        for state in range(self.k):
            precision, natural = np.zeros((3, 3)), np.zeros(3)
            for frame in range(3):
                r = rotations[frame]
                covariance = r @ self.covariances[state, frame] @ r.T
                mean = r @ self.means[state, frame] + offsets[frame]
                inv = np.linalg.inv(covariance)
                precision += inv
                natural += inv @ mean
            covariance = np.linalg.inv(precision)
            means.append(covariance @ natural)
            covariances.append(covariance)
        return np.array(means), np.array(covariances)

    def fit(self, demos: list[Demonstration], weights: np.ndarray | None = None) -> "TPHSMM":
        if not demos:
            raise ValueError("TP-HSMM needs complete demonstrations")
        ids = [d.metadata["episode_id"] for d in demos]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicated demonstrations would change the empirical weights")
        weights = np.ones(len(demos)) if weights is None else np.asarray(weights, dtype=float)
        if weights.shape != (len(demos),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("All demonstration weights must be positive")
        records = [demonstration_fingerprint(d) for d in demos]
        if len({record["signal_sha256"] for record in records}) != len(records):
            raise ValueError("Duplicate demonstration signals would change empirical weights")
        paths = [resample_positions(d, self.samples) for d in demos]
        local = []
        for path, demo in zip(paths, demos):
            rotations, offsets = task_frames(demo.scene, demo.goal)
            local.append(np.stack([(path - b) @ r for r, b in zip(rotations, offsets)], axis=1))
        floor = self.config["skill"]["covariance_floor"]
        labels = np.minimum(np.arange(self.samples) * self.k // self.samples, self.k - 1)
        responsibilities = [np.eye(self.k)[labels] for _ in demos]
        self.history = []
        self.duration_mean = np.full(self.k, self.samples / self.k)
        self.duration_std = np.full(self.k, self.samples / self.k / 3)
        for iteration in range(self.config["skill"]["em_iterations"] + 1):
            for k in range(self.k):
                mass = sum(w * g[:, k].sum() for w, g in zip(weights, responsibilities))
                if mass <= 1e-10:
                    continue
                for frame in range(3):
                    mean = sum(w * (g[:, k, None] * x[:, frame]).sum(0)
                               for w, g, x in zip(weights, responsibilities, local)) / mass
                    covariance = sum(w * np.einsum("t,ti,tj->ij", g[:, k],
                                                  x[:, frame] - mean, x[:, frame] - mean)
                                     for w, g, x in zip(weights, responsibilities, local)) / mass
                    self.means[k, frame] = mean
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    self.covariances[k, frame] = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
            if iteration == self.config["skill"]["em_iterations"]:
                break
            duration = np.array([duration_logpmf(m, s, self.samples)
                                 for m, s in zip(self.duration_mean, self.duration_std)])
            responsibilities, counts, likelihood = [], np.zeros_like(duration), 0.0
            for w, x in zip(weights, local):



                emission = np.stack([sum(gaussian_logpdf(x[:, frame], self.means[k, frame],
                                                        self.covariances[k, frame])
                                            for frame in range(3)) for k in range(self.k)], -1)
                gamma, count, ll = segment_posterior(emission, duration)
                responsibilities.append(gamma)
                counts += w * count
                likelihood += w * ll
            for k in range(self.k):
                def objective(parameters):
                    return -float(counts[k] @ duration_logpmf(parameters[0], parameters[1], self.samples))
                old = np.array([self.duration_mean[k], self.duration_std[k]])
                fitted = minimize(objective, old, method="L-BFGS-B", bounds=[
                    (1, self.samples), (self.config["skill"]["duration_floor"], self.samples)])
                if np.isfinite(fitted.fun) and fitted.fun <= objective(old) + 1e-9:
                    self.duration_mean[k], self.duration_std[k] = fitted.x
            self.history.append({"iteration": iteration, "weighted_log_likelihood": likelihood})
        self.fit_ids = ids
        self.fit_manifest = {
            "available": True, "schema_version": 1, "fit": "full_weighted_batch_refit",
            "initialization": "equal_normalized_time_bins",
            "states": self.k, "trajectory_samples": self.samples,
            "task_frames": ["start", "object", "place"], "frame_rotations": "identity",
            "skill_config": copy.deepcopy(self.config["skill"]),
            "records": [{**record, "weight": float(weight)}
                        for record, weight in zip(records, weights)],
            "total_weight": float(weights.sum()), "history": self.history}
        self.duration_seconds = float(np.average([d.time[-1] - d.time[0] for d in demos],
                                                weights=weights))
        return self

    def state_sequence(self, scene: Scene, goal: int) -> np.ndarray:

        duration = np.array([duration_logpmf(m, s, self.samples)
                             for m, s in zip(self.duration_mean, self.duration_std)])
        dp = np.full((self.k + 1, self.samples + 1), -np.inf)
        back = np.zeros_like(dp, dtype=int)
        dp[0, 0] = 0
        for k in range(1, self.k + 1):
            for end in range(k, self.samples + 1):
                ds = np.arange(1, end - k + 2)
                scores = dp[k - 1, end - ds] + duration[k - 1, ds - 1]
                best = int(np.argmax(scores))
                dp[k, end], back[k, end] = scores[best], ds[best]
        sequence, end = np.empty(self.samples, dtype=int), self.samples
        for k in range(self.k, 0, -1):
            d = back[k, end]
            sequence[end - d:end] = k - 1
            end -= d
        return sequence

    def reproduce(self, scene: Scene, goal: int) -> tuple[np.ndarray, np.ndarray]:
        means, covariances = self.world_gaussians(scene, goal)
        sequence = self.state_sequence(scene, goal)
        target, covariance = means[sequence], covariances[sequence]
        dt = self.duration_seconds / (self.samples - 1)
        a = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        b = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
        r = np.eye(3) * self.config["skill"]["lqt_control_cost"]
        gains, feedforward = [], []
        p, q = np.zeros((6, 6)), np.zeros(6)
        for t in reversed(range(self.samples)):
            cost = np.zeros((6, 6))
            cost[:3, :3] = np.linalg.inv(covariance[t])
            state_target = np.r_[target[t], np.zeros(3)]
            p += cost
            q -= cost @ state_target
            h = r + b.T @ p @ b
            gain, offset = np.linalg.solve(h, b.T @ p @ a), np.linalg.solve(h, b.T @ q)
            gains.append(gain)
            feedforward.append(offset)
            p, q = a.T @ p @ a - a.T @ p @ b @ gain, a.T @ q - a.T @ p @ b @ offset
        state, path = np.r_[scene.start, np.zeros(3)], []
        for gain, offset in zip(reversed(gains), reversed(feedforward)):
            path.append(state[:3].copy())
            state = a @ state + b @ (-gain @ state - offset)
        return np.array(path), covariance

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, means=self.means, covariances=self.covariances,
                            duration_mean=self.duration_mean, duration_std=self.duration_std,
                            duration_seconds=self.duration_seconds,
                            fit_ids=np.asarray(self.fit_ids, dtype=str),
                            fit_manifest=np.array(json.dumps(self.fit_manifest, allow_nan=False)))

    @classmethod
    def load(cls, config: dict, path: Path) -> "TPHSMM":
        model = cls(config)
        with np.load(path, allow_pickle=False) as d:
            for k in ("means", "covariances", "duration_mean", "duration_std"):
                setattr(model, k, d[k].copy())
            if (model.means.shape != (model.k, 3, 3) or
                    model.covariances.shape != (model.k, 3, 3, 3)):
                raise ValueError("TP-HSMM checkpoint state/frame dimensions do not match configuration")
            model.duration_seconds = float(d["duration_seconds"])
            model.fit_ids = d["fit_ids"].tolist()
            model.fit_manifest = json.loads(str(d["fit_manifest"])) if "fit_manifest" in d else {
                "available": False, "reason": "legacy_checkpoint_missing_training_records"}
            if model.fit_manifest.get("available") and model.fit_manifest["trajectory_samples"] != model.samples:
                raise ValueError("TP-HSMM checkpoint trajectory horizon does not match configuration")
        return model


def refine(config: dict, original: list[Demonstration], candidates: list[Demonstration]):






    accepted, audit = [], []
    quota = config["skill"]["refinement_count"]
    if not isinstance(quota, int) or quota < 0:
        raise ValueError("refinement_count must be a nonnegative integer")
    seen_ids = {d.metadata["episode_id"] for d in original}
    seen_hashes = {demonstration_fingerprint(d)["signal_sha256"] for d in original}
    for demo in candidates:
        record = demonstration_fingerprint(demo)
        ok, reasons = successful(demo, config)
        reasons = list(reasons)
        if record["episode_id"] in seen_ids or record["signal_sha256"] in seen_hashes:
            reasons.append("duplicate_or_original_overlap")
        seen_ids.add(record["episode_id"])
        seen_hashes.add(record["signal_sha256"])
        if len(accepted) >= quota:
            reasons.append("quota_reached")
        ok = ok and not reasons
        if ok:
            accepted.append(demo)
        audit.append({**record, "accepted": ok, "reasons": reasons,
                      "candidate_order": len(audit),
                      "accepted_index": len(accepted) - 1 if ok else None})
    weights = np.r_[np.full(len(original), config["skill"]["original_weight"]),
                    np.full(len(accepted), config["skill"]["new_weight"])]
    model = TPHSMM(config).fit(original + accepted, weights)
    model.fit_manifest["refinement"] = {
        "original_ids": [d.metadata["episode_id"] for d in original],
        "accepted_ids": [d.metadata["episode_id"] for d in accepted],
        "requested": quota, "accepted": len(accepted), "quota_met": len(accepted) == quota,
        "eligibility": "first_unique_successes_in_supplied_acquisition_order",
        "comparison_status": "eligible" if len(accepted) == quota else "diagnostic_only_incomplete_quota"}
    return model, audit, accepted
