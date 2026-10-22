from dataclasses import dataclass

import numpy as np


@dataclass
class Port:
    effort: np.ndarray
    flow: np.ndarray
    actuator: str


class MultiportPassivity:
    def __init__(self, initial_energy=0.0, maximum_energy=8.0):
        if not 0 <= initial_energy <= maximum_energy:
            raise ValueError("Invalid initial storage")
        self.energy = self.initial = float(initial_energy)
        self.maximum = maximum_energy
        self.input = self.output = self.spilled = 0.0

    def regulate(self, ports: list[Port], dt: float, inbound_energy: float,
                 weights: np.ndarray):
        if len(ports) != 3 or dt <= 0 or inbound_energy < 0:
            raise ValueError("Three ports, positive dt and nonnegative inbound energy are required")
        weights = np.asarray(weights, float)
        if weights.shape != (3,) or np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("Invalid damping weights")
        for port in ports:
            if port.actuator not in {"force", "velocity"}:
                raise ValueError("Port actuator must be force or velocity")
            if np.shape(port.effort) != (3,) or np.shape(port.flow) != (3,):
                raise ValueError("Port vectors must have three components")
            if not np.isfinite(port.effort).all() or not np.isfinite(port.flow).all():
                raise ValueError("Port vectors must be finite")
        available = min(self.maximum, self.energy + inbound_energy)
        spill = max(0.0, self.energy + inbound_energy - self.maximum)
        work = dt * np.array([np.dot(p.effort, p.flow) for p in ports])
        excess = max(0.0, float(work.sum()) - available)
        removal = np.zeros(3)
        remaining = np.maximum(work, 0)
        for _ in range(3):
            eligible = remaining > 1e-12
            if excess <= 1e-12 or not eligible.any():
                break
            allocation = weights * eligible
            if allocation.sum() <= 1e-12:
                allocation = eligible.astype(float)
            requested = excess * allocation / allocation.sum()
            applied = np.minimum(requested, remaining)
            removal += applied
            remaining -= applied
            excess -= applied.sum()
        corrected = []
        for index, port in enumerate(ports):
            effort, flow = np.asarray(port.effort, float).copy(), np.asarray(port.flow, float).copy()
            if port.actuator == "force":
                effort -= removal[index] * flow / max(dt * (flow @ flow), 1e-300)
            else:
                flow -= removal[index] * effort / max(dt * (effort @ effort), 1e-300)
            corrected.append(Port(effort, flow, port.actuator))
        delivered = dt * sum(p.effort @ p.flow for p in corrected)
        storage = available - delivered
        if storage < -1e-9:
            raise ArithmeticError("Insufficient damping to enforce the port energy budget")
        spill += max(0.0, storage - self.maximum)
        self.energy = float(np.clip(storage, 0, self.maximum))
        self.input += inbound_energy
        self.output += delivered
        self.spilled += spill
        return corrected, removal

    def residual(self):
        return self.initial + self.input - self.output - self.spilled - self.energy
