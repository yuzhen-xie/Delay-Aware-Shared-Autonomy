import numpy as np

from dasa.geometry import Scene, clamp_norm


class ForceKalman:

    def __init__(self, process: float, measurement: float, initial_variance: float = 1.0):
        if process < 0 or measurement <= 0 or initial_variance <= 0:
            raise ValueError("Kalman Q must be nonnegative; R and initial variance must be positive")
        self.mean = np.zeros(3)
        self.covariance = initial_variance * np.eye(3)
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
    force = c.get("attractive_weight", 1.0) * c["attractive_gain"] * (target - position)


    delta = position - scene.centers
    sigma = c["repulsive_sigma"]
    repulsion = c["repulsive_gain"] * delta / sigma**2
    repulsion *= np.exp(-np.sum(delta**2, axis=1) / (2 * sigma**2))[:, None]
    repulsion = np.array([clamp_norm(f, c["repulsive_limit"]) for f in repulsion])
    return force + c.get("repulsive_weight", 1.0) * repulsion.sum(0)


def risk_weight(clearance: float, config: dict) -> float:
    c = config["motion"]
    middle = (c["distance_min"] + c["distance_max"]) / 2
    scale = (clearance - middle) / (c["distance_max"] - c["distance_min"])
    return float(1 / (1 + np.exp(-np.clip(c["lambda_gain"] * scale, -50, 50))))
