"""Three-port energy accounting with a discrete-work output constraint.

The tank enforces a sampled controller-boundary inequality, not a proof of the
entire delayed bilateral robot/human system. Port allocations are diagnostic
shares of the removed robot work; physical PC1/PC3 deployment requires measured
master velocities and a device-specific autonomous controller realization.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyStep:
    force: np.ndarray
    energy: float
    supplied: float
    work: float
    removed_work: float
    spilled: float
    allocations: np.ndarray
    rho: float


class ThreePortTank:
    def __init__(self, config: dict):
        self.config = config
        self.energy = config["passivity"]["initial_energy"]
        self.initial = self.energy
        self.supplied = self.work = self.spilled = 0.0
        self.window = deque()
        self.window_removed = 0.0
        self.window_nominal = 0.0

    def robot_work(self, force: np.ndarray, velocity: np.ndarray) -> float:
        c = self.config["simulation"]
        dt, mass, damping = c["control_dt"], c["mass"], c["damping"]
        midpoint = (2 * mass * velocity + dt * force) / (2 * mass + dt * damping)
        return float(dt * force @ midpoint)

    def step(self, time: float, nominal: np.ndarray, velocity: np.ndarray,
             human_force: np.ndarray, master_velocity: np.ndarray,
             alpha: float, blend: float, previous_rho: float, three_port: bool = True) -> EnergyStep:
        c, p = self.config["simulation"], self.config["passivity"]
        dt, mass, damping = c["control_dt"], c["mass"], c["damping"]
        supplied = dt * max(float(human_force @ master_velocity), 0.0) * p["input_efficiency"]
        available = min(p["maximum_energy"], self.energy + supplied)
        spilled = max(0.0, self.energy + supplied - p["maximum_energy"])
        nominal_work = self.robot_work(nominal, velocity)
        force = nominal.copy()
        if nominal_work > available:
            # W(F) is a convex quadratic. Its sublevel set is a Euclidean ball.
            center = -mass * velocity / dt
            radius2 = available * (2 * mass + dt * damping) / dt**2
            radius2 += mass**2 * (velocity @ velocity) / dt**2
            delta = nominal - center
            force = center + delta * np.sqrt(max(radius2, 0)) / max(np.linalg.norm(delta), 1e-15)
        work = self.robot_work(force, velocity)
        removed = max(0.0, nominal_work - work)
        energy = available - work
        if energy < -p["tolerance"]:
            raise ArithmeticError("Output work exceeded the tank budget")
        if energy > p["maximum_energy"]:
            spilled += energy - p["maximum_energy"]
        self.energy = float(np.clip(energy, 0, p["maximum_energy"]))
        self.supplied += supplied
        self.work += work
        self.spilled += spilled
        weights = np.array([(1 - alpha) * (1 - previous_rho), alpha, 1 - blend])
        if not three_port:
            weights[2] = 0
        if weights.sum() < 1e-12:
            weights = np.array([0.0, 1.0, 0.0])
        allocation = removed * weights / weights.sum()
        self.window.append((time, removed, max(nominal_work, 0)))
        self.window_removed += removed
        self.window_nominal += max(nominal_work, 0)
        horizon = self.config["arbitration"]["rho_window_seconds"]
        while self.window and self.window[0][0] < time - horizon:
            _, removed_old, nominal_old = self.window.popleft()
            self.window_removed -= removed_old
            self.window_nominal -= nominal_old
        rho = float(np.clip(self.window_removed / max(self.window_nominal, 1e-12), 0, 1))
        return EnergyStep(force, self.energy, supplied, work, removed, spilled, allocation, rho)

    def residual(self) -> float:
        return self.initial + self.supplied - self.work - self.spilled - self.energy
