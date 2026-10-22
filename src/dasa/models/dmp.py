from pathlib import Path

import numpy as np
from scipy.special import softmax

from dasa.data import Demonstration
from dasa.geometry import Scene, unit
from dasa.models.intent import angular_cost


class DMP:
    def __init__(self, bases=25, alpha=25.0):
        self.bases, self.alpha = bases, alpha
        self.centers = np.exp(-4 * np.linspace(0, 1, bases))
        self.widths = np.full(bases, bases**2)
        self.weights = np.zeros((3, bases, 3))

    def basis(self, phase):
        psi = np.exp(-self.widths * (phase[:, None] - self.centers) ** 2)
        return psi / np.maximum(psi.sum(1, keepdims=True), 1e-12) * phase[:, None]

    def fit(self, demos: list[Demonstration]):
        s = np.linspace(0, 1, 120)
        phase = np.exp(-4 * s)
        design = self.basis(phase)
        for goal in range(3):
            targets = []
            for demo in demos:
                if demo.goal != goal:
                    continue
                x = demo.position[demo.phase == 0]
                u = np.linspace(0, 1, len(x))
                y = np.stack([np.interp(s, u, x[:, d]) for d in range(3)], -1)
                dy, ddy = np.gradient(y, s, axis=0), None
                ddy = np.gradient(dy, s, axis=0)
                scale = y[-1] - y[0]
                scale = np.where(np.abs(scale) < 0.01, 0.01, scale)
                target = (ddy - self.alpha * (self.alpha / 4 * (y[-1] - y) - dy)) / scale
                targets.append(target)
            if not targets:
                raise ValueError("DMP needs a demonstration for each candidate goal")
            target = np.mean(targets, axis=0)
            self.weights[goal] = np.linalg.solve(design.T @ design + 1e-4 * np.eye(self.bases),
                                                design.T @ target)
        return self

    def rollout(self, scene: Scene, goal: int):

        y, v = scene.start.copy(), np.zeros(3)
        target, path, dt = scene.goals[goal], [], 0.002
        scale = target - y
        scale = np.where(np.abs(scale) < 0.01, 0.01, scale)
        for t in np.arange(0, 1.3, dt):
            phase = np.exp(-4 * t)
            forcing = self.basis(np.array([phase]))[0] @ self.weights[goal] * scale
            a = self.alpha * (self.alpha / 4 * (target - y) - v) + forcing
            v += dt * a
            y += dt * v
            path.append(y.copy())
        return np.array(path)

    def save(self, path: Path):
        np.savez_compressed(path, weights=self.weights)

    @classmethod
    def load(cls, path: Path):
        model = cls()
        with np.load(path, allow_pickle=False) as arrays:
            model.weights = arrays["weights"].copy()
        return model


class DMPIntent:
    def __init__(self, model: DMP, scene: Scene, beta: float):
        self.paths = [model.rollout(scene, goal) for goal in range(3)]
        self.beta = beta

    def step(self, action, observed, current, scene, age):
        references = []
        for path in self.paths:
            near = int(np.argmin(np.linalg.norm(path - observed[:3], axis=1)))
            references.append(unit(path[min(near + 20, len(path) - 1)] - observed[:3]))
        return softmax(-self.beta * angular_cost(action, np.array(references))), np.zeros(8)
