"""Causal explicit-duration state filtering for task-parameterized skills."""

import numpy as np
from scipy.special import logsumexp

from dasa.models.hsmm import duration_logpmf, gaussian_logpdf


class HSMMFilter:
    def __init__(self, model, scene, goal):
        self.means, self.covariances = model.world_gaussians(scene, goal)
        self.states, self.maximum_age = model.k, model.samples
        pmf = np.exp(np.array([duration_logpmf(m, s, model.samples)
                              for m, s in zip(model.duration_mean, model.duration_std)]))
        survival = np.flip(np.cumsum(np.flip(pmf, axis=1), axis=1), axis=1)
        self.hazard = np.clip(pmf / np.maximum(survival, 1e-300), 0, 1)
        self.belief = np.zeros_like(pmf)
        self.belief[0, 0] = 1.0
        self.first = True
        self.log_evidence = 0.0

    def update(self, position):
        if not self.first:
            predicted = np.zeros_like(self.belief)
            predicted[:, 1:] = self.belief[:, :-1] * (1 - self.hazard[:, :-1])
            predicted[1:, 0] = np.sum(self.belief[:-1] * self.hazard[:-1], axis=1)
            # A completed skill remains in the terminal state while the executor holds position.
            predicted[-1, -1] += np.sum(self.belief[-1] * self.hazard[-1])
            self.belief = predicted
        self.first = False
        emission = np.array([gaussian_logpdf(np.asarray(position), mean, covariance)
                             for mean, covariance in zip(self.means, self.covariances)])
        with np.errstate(divide="ignore"):
            log_joint = np.log(self.belief) + emission[:, None]
        evidence = logsumexp(log_joint)
        if not np.isfinite(evidence):
            raise ArithmeticError("State filter exhausted its probability support")
        self.belief = np.exp(log_joint - evidence)
        self.log_evidence += evidence
        return self.belief.sum(1)

    def reference(self):
        probability = self.belief.sum(1)
        mean = probability @ self.means
        delta = self.means - mean
        covariance = np.einsum("k,kij->ij", probability, self.covariances)
        covariance += np.einsum("k,ki,kj->ij", probability, delta, delta)
        return mean, covariance
