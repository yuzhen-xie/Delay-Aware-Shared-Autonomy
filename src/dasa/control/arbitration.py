from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlCondition:
    name: str
    layer_two: bool
    autonomy_rule: str
    intent: str = "full"
    guarded: bool = False
    three_port: bool = True


CONDITIONS = {
    "C1": ControlCondition("C1", False, "off", three_port=False),
    "C2": ControlCondition("C2", True, "autonomy_confidence"),
    "C3": ControlCondition("C3", True, "relative_confidence"),
    "C4": ControlCondition("C4", True, "fixed"),
    "delay_br_shared": ControlCondition("delay_br_shared", True, "relative_confidence", "no_latent"),
    "C3_guarded": ControlCondition("C3_guarded", True, "relative_confidence", guarded=True),
    "autonomous": ControlCondition("autonomous", False, "full"),
}


class Confidence:
    def __init__(self, config: dict):
        self.config = config
        self.previous = np.zeros(3)
        self.derivatives = deque(maxlen=config["arbitration"]["roughness_window"])

    def values(self, human_force, covariance, rho):
        c = self.config["arbitration"]
        trace = np.trace(covariance)
        autonomous = float(np.clip((c["covariance_upper"] - trace) /
                                   (c["covariance_upper"] - c["covariance_lower"]), 0, 1))
        self.derivatives.append(np.linalg.norm(human_force - self.previous) /
                                self.config["simulation"]["decision_dt"])
        self.previous = human_force.copy()
        roughness = float(np.mean(self.derivatives))
        human = (1 - rho) * np.exp(-c["roughness_gamma"] * roughness)
        return autonomous, float(human), roughness


def authority(condition: ControlCondition, autonomous: float, human: float, config: dict,
              clearance: float, corrective: bool) -> tuple[float, float]:
    c = config["arbitration"]
    rule = condition.autonomy_rule
    if rule == "off":
        return 0.0, 1.0
    if rule == "full":
        return 1.0, 1.0
    if rule == "fixed":
        return c.get("fixed_weight", 0.5), 1.0
    if rule == "autonomy_confidence":
        return autonomous, 1.0
    alpha = autonomous / (autonomous + human + c["epsilon"])
    scale = 1.0
    if condition.guarded:

        scale = min(1.0, (autonomous + human) / c["low_confidence_threshold"])
        if corrective and clearance < c["correction_distance"]:
            alpha = min(alpha, c.get("guarded_corrective_max", 0.15))
    return float(alpha), float(scale)
