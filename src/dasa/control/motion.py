"""Cartesian force guidance and a fully specified random-walk Kalman filter."""

import numpy as np

from dasa.geometry import Scene, clamp_norm


class ForceKalman:
    """x_(k+1)=x_k+w, y_k=x_k+v; Q=qI and R=rI in N² per sample."""
    def __init__(self, process: float, measurement: float):
        self.mean = np.zeros(3)
        self.covariance = np.eye(3)
        self.q = process * np.eye(3)
        self.r = measurement * np.eye(3)

    def step(self, measurement: np.ndarray):
        predicted = self.covariance + self.q
        gain = np.linalg.solve((predicted + self.r).T, predicted.T).T
        self.mean += gain @ (measurement - self.mean)
        residual = np.eye(3) - gain
        self.covariance = residual @ predicted @ residual.T + gain @ self.r @ gain.T
        return self.mean.copy()


def potential_force(position: np.ndarray, target: np.ndarray, scene: Scene, config: dict):
    c = config["motion"]
    force = c["attractive_gain"] * (target - position)
    # Gaussian radial repulsion: k_rep (p-o)/sigma² exp(-||p-o||²/(2 sigma²)).
    # Center distance is used here; surface clearance is separately used for risk.
    delta = position - scene.centers
    sigma = c["repulsive_sigma"]
    repulsion = c["repulsive_gain"] * delta / sigma**2
    repulsion *= np.exp(-np.sum(delta**2, axis=1) / (2 * sigma**2))[:, None]
    repulsion = np.array([clamp_norm(f, c["repulsive_limit"]) for f in repulsion])
    return force + repulsion.sum(0)


def risk_weight(clearance: float, config: dict) -> float:
    c = config["motion"]
    middle = (c["distance_min"] + c["distance_max"]) / 2
    scale = (clearance - middle) / (c["distance_max"] - c["distance_min"])
    return float(1 / (1 + np.exp(-np.clip(c["lambda_gain"] * scale, -50, 50))))
