import numpy as np
from scipy.special import logsumexp

from dasa.models.hsmm import duration_logpmf, gaussian_logpdf


class HSMMFilter:
    def __init__(self, model, scene, goal):
        self.model = model
        self.means, self.covariances = model.world_gaussians(scene, goal)
        self.states, self.maximum_age = model.k, model.samples
        self.step_seconds = model.duration_seconds / (model.samples - 1)
        if not np.isfinite(self.step_seconds) or self.step_seconds <= 0:
            raise ValueError("A physical skill needs a positive measured duration")
        pmf = np.exp(np.array([duration_logpmf(m, s, model.samples)
                              for m, s in zip(model.duration_mean, model.duration_std)]))
        survival = np.flip(np.cumsum(np.flip(pmf, axis=1), axis=1), axis=1)
        self.hazard = np.clip(pmf / np.maximum(survival, 1e-300), 0, 1)
        self.belief = np.zeros_like(pmf)
        self.belief[0, 0] = 1.0
        self.first = True
        self.clock = 0.0
        self.elapsed_seconds = 0.0
        self.log_evidence = 0.0

    def _transition(self, belief):
        predicted = np.zeros_like(belief)
        predicted[:, 1:] = belief[:, :-1] * (1 - self.hazard[:, :-1])
        predicted[1:, 0] = np.sum(belief[:-1] * self.hazard[:-1], axis=1)
        predicted[-1, -1] += np.sum(belief[-1] * self.hazard[-1])
        total = predicted.sum()
        if not np.isfinite(total) or total <= 0:
            raise ArithmeticError("State filter exhausted its probability support")
        return predicted / total

    def update(self, position, dt=None):
        position = np.asarray(position, float)
        dt = self.step_seconds if dt is None else float(dt)
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(dt) or dt < 0:
            raise ValueError("State filtering requires finite measured position and nonnegative elapsed time")
        if not self.first:
            self.elapsed_seconds += dt
            self.clock += dt
            steps = int(self.clock / self.step_seconds)
            if steps == 0:
                return self.belief.sum(1)
            for _ in range(steps):
                self.belief = self._transition(self.belief)
            self.clock -= steps * self.step_seconds
        self.first = False
        emission = np.array([gaussian_logpdf(np.asarray(position), mean, covariance)
                             for mean, covariance in zip(self.means, self.covariances)])
        with np.errstate(divide="ignore"):
            log_joint = np.log(self.belief) + emission[:, None]
        evidence = logsumexp(log_joint)
        if not np.isfinite(evidence):
            raise ArithmeticError("State filter exhausted its probability support")
        self.belief = np.exp(log_joint - evidence)
        self.log_evidence += float(evidence)
        return self.belief.sum(1)

    def _moments(self, belief):
        probability = belief.sum(1)
        mean = probability @ self.means
        delta = self.means - mean
        covariance = np.einsum("k,kij->ij", probability, self.covariances)
        covariance += np.einsum("k,ki,kj->ij", probability, delta, delta)
        return mean, covariance

    def reference(self):
        return self._moments(self.belief)

    def update_scene(self, scene, goal):
        means, covariances = self.model.world_gaussians(scene, goal)
        if means.shape != self.means.shape or covariances.shape != self.covariances.shape:
            raise ValueError("Live task geometry changed the fitted state topology")
        self.means, self.covariances = means, covariances

    def plan(self, current_state, horizon=32):
        if not isinstance(horizon, int) or horizon < 2:
            raise ValueError("A causal reference plan needs at least two samples")
        belief = self.belief.copy()
        means, covariances = [], []
        for _ in range(horizon):
            mean, covariance = self._moments(belief)
            means.append(mean)
            covariances.append(covariance)
            belief = self._transition(belief)
        covariance = np.asarray(covariances)
        states = self.model.lqt(np.asarray(means), covariance, current_state, self.step_seconds)
        return states, covariance
