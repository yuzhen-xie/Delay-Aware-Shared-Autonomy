from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from dasa.control.arbitration import CONDITIONS
from dasa.control.hierarchy import HierarchicalController
from dasa.control.passivity import ThreePortTank
from dasa.data import Demonstration
from dasa.delay import DelayBuffer, sampled_delay
from dasa.geometry import Scene, clamp_norm


class SimulatedOperator:
    def __init__(self, config: dict, scene: Scene, intended_goal: int, seed: int, noisy: bool):
        self.config, self.scene, self.goal = config, scene, intended_goal
        self.rng = np.random.default_rng(seed)
        self.noisy = noisy
        self.previous_noise = np.zeros(3)
        self.waypoint = 0

    def command(self, observed: np.ndarray, phase: int):
        p, v = observed[:3], observed[3:]
        target = self.scene.goals[self.goal]
        lift = self.config["simulation"]["lift_height"]
        if phase == 0:
            if self.waypoint == 0 and np.linalg.norm(p[:2] - target[:2]) < 0.055:
                self.waypoint = 1
            desired = np.r_[target[:2], lift] if self.waypoint == 0 else target
        elif phase == 1:
            desired = np.r_[target[:2], lift]
        else:
            if self.waypoint < 2 and np.linalg.norm(p[:2] - self.scene.place[:2]) < 0.06:
                self.waypoint = 2
            desired = np.r_[self.scene.place[:2], lift] if self.waypoint < 2 else self.scene.place
        c = self.config["operator"]
        force = c["kp"] * (desired - p) - c["kd"] * v
        noise_std = c["noisy_noise"] if self.noisy else c["expert_noise"]
        self.previous_noise = 0.7 * self.previous_noise + self.rng.normal(0, noise_std, 3)
        force += self.previous_noise
        distances = self.scene.clearance(p, self.config["simulation"]["robot_radius"])
        near = int(np.argmin(distances))
        corrective = False
        if distances[near] < self.config["arbitration"]["correction_distance"]:
            outward = p - self.scene.centers[near]
            corrective = float(force @ outward) < 0
            if corrective:
                force += c["corrective_gain"] * outward / max(np.linalg.norm(outward), 1e-9) * 7
        return clamp_norm(force, self.config["simulation"]["force_limit"]), corrective


class ManipulationTask:
    def __init__(self, config: dict, scene: Scene, intended_goal: int):
        self.config, self.scene, self.intended_goal = config, scene, intended_goal
        self.phase = 0
        self.grasped_goal = None
        self.dwell = 0.0
        self.success = False
        self.failed = False
        self.events = []

    def update(self, time: float, position: np.ndarray, velocity: np.ndarray, dt: float):
        c = self.config["simulation"]
        if self.phase == 0:
            distance = np.linalg.norm(self.scene.goals - position, axis=1)
            nearest = int(np.argmin(distance))
            in_zone = distance[nearest] < c["grasp_tolerance"]
            self.dwell = self.dwell + dt if in_zone and np.linalg.norm(velocity) < c["max_grasp_speed"] else 0.0
            if self.dwell >= c["dwell_seconds"]:
                self.grasped_goal = nearest
                self.events.append({"time": time, "event": "grasp", "goal": nearest})
                self.failed = nearest != self.intended_goal
                self.phase, self.dwell = 1, 0.0
        elif self.phase == 1 and position[2] > c["lift_height"] - 0.05:
            self.phase = 2
            self.events.append({"time": time, "event": "lift"})
        elif self.phase == 2:
            in_zone = np.linalg.norm(position - self.scene.place) < c["place_tolerance"]
            self.dwell = self.dwell + dt if in_zone and np.linalg.norm(velocity) < c["max_grasp_speed"] else 0.0
            if self.dwell >= c["dwell_seconds"]:
                self.phase, self.success = 3, True
                self.events.append({"time": time, "event": "place"})


class Trial:
    def __init__(self, arrays: dict, metadata: dict):
        self.arrays, self.metadata = arrays, metadata

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays, metadata=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as d:
            return cls({k: d[k].copy() for k in d.files if k != "metadata"},
                       json.loads(str(d["metadata"])))

    def demonstration(self) -> Demonstration:
        a, m = self.arrays, self.metadata
        return Demonstration(a["time"], a["position"], a["velocity"], a["raw_force"],
                             a["observed_state"], a["phase"], Scene.from_dict(m["scene"]),
                             m["intended_goal"], {
                                 "episode_id": m["trial_id"], "operator_id": m["operator_id"],
                                 "provenance": m["provenance"], "success": m["success"],
                                 "collision": m["collision"], "bounds_violation": m["bounds_violation"],
                                 "grasped_goal": m["grasped_goal"], "placed": m["success"],
                                 "source_condition": m["condition"], "delay_seconds": m["delay_seconds"]})


def run_trial(config: dict, scene: Scene, skill, cvae, seed: int, intended_goal: int,
              condition_name: str = "C3", operator_id: str = "synthetic_00",
              noisy: bool = False, estimator=None, trial_id: str | None = None) -> Trial:
    config = copy.deepcopy(config)
    control = CONDITIONS[condition_name]
    autonomous = condition_name == "autonomous"
    if autonomous:
        config["passivity"]["initial_energy"] = config["passivity"]["maximum_energy"]
    c, decision = config["simulation"], config["simulation"]["decision_dt"]
    dt, substeps = c["control_dt"], round(decision / c["control_dt"])
    rng = np.random.default_rng(seed + 991)
    operator = SimulatedOperator(config, scene, intended_goal, seed, noisy)
    task = ManipulationTask(config, scene, intended_goal)
    hierarchy = HierarchicalController(config, scene, skill, cvae, condition_name, estimator)
    p, v = scene.start.copy(), np.zeros(3)
    feedback = DelayBuffer(np.r_[p, v], horizon=c["feedback_delay"] + c["jitter_seconds"] + 1)
    commands = DelayBuffer(np.r_[np.zeros(3), p, v, 0.0, 0.0],
                           horizon=c["command_delay"] + c["jitter_seconds"] + 1)
    tank = ThreePortTank(config)
    collision = bounds = False
    rho, posterior, latent = 0.0, np.ones(3) / 3, np.zeros(config["intent"]["latent_dim"])
    record = {name: [] for name in ("time", "decision_time", "current_state", "position", "velocity", "raw_force", "human_force",
              "observed_state", "source_time", "command_source_time", "feedback_age", "phase",
              "posterior", "latent", "alpha", "lambda", "autonomy_confidence", "human_confidence",
              "roughness", "corrective", "nominal_force", "executed_force", "reference",
              "clearances", "rho", "tank_energy", "supplied_energy", "robot_work", "removed_work",
              "spilled_energy", "port_allocations", "energy_residual")}
    for index in range(int(c["max_seconds"] / decision)):
        time = index * decision
        feedback.push(time, np.r_[p, v])
        observation = feedback.observe(time, sampled_delay(c["feedback_delay"], c["jitter_seconds"], rng))
        raw, corrective = operator.command(observation.state, task.phase)
        if autonomous:
            raw, corrective = np.zeros(3), False
        commands.push(time, np.r_[raw, observation.state, observation.source_time, float(corrective)])
        packet = commands.observe(time, sampled_delay(c["command_delay"], c["jitter_seconds"], rng))
        raw, perceived, stamp, corrective = packet.state[:3], packet.state[3:9], packet.state[9], bool(packet.state[10])
        age = time - stamp
        current_state = np.r_[p, v]
        command = hierarchy.decide(raw, perceived, current_state, age, task.phase,
                                   task.grasped_goal, rho, corrective,
                                   intended_goal if autonomous else None)
        human, posterior, latent, reference = [command[k] for k in
                                               ("human_force", "posterior", "latent", "reference")]
        alpha, blend, ca, ch, roughness, nominal = [command[k] for k in
            ("alpha", "lambda", "autonomy_confidence", "human_confidence", "roughness", "nominal_force")]
        total_supply = total_work = total_removed = total_spill = 0.0
        allocation = np.zeros(3)
        force = nominal.copy()
        for substep in range(substeps):
            subtime = time + substep * dt

            master_velocity = raw / 40.0 if not autonomous else np.zeros(3)
            energy = tank.step(subtime, nominal, v, human, master_velocity,
                               alpha, blend, rho, control.three_port)
            force, rho = energy.force, energy.rho
            total_supply += energy.supplied
            total_work += energy.work
            total_removed += energy.removed_work
            total_spill += energy.spilled
            allocation += energy.allocations
            v_new = ((2 * c["mass"] - dt * c["damping"]) * v + 2 * dt * force) / (2 * c["mass"] + dt * c["damping"])
            p_new = p + dt * (v + v_new) / 2
            collision |= scene.swept_clearance(p, p_new, c["robot_radius"]) <= 0
            bounds |= bool(np.any(p_new < c["workspace_min"]) or np.any(p_new > c["workspace_max"]))
            p, v = p_new, v_new
            task.update(subtime + dt, p, v, dt)
            if collision or bounds or task.failed or task.success:
                break
        row = {"time": time + (substep + 1) * dt, "decision_time": time,
               "current_state": current_state, "position": p.copy(), "velocity": v.copy(),
               "raw_force": raw.copy(), "human_force": human, "observed_state": perceived,
               "source_time": stamp, "command_source_time": packet.source_time, "feedback_age": age,
               "phase": task.phase, "posterior": posterior.copy(), "latent": latent.copy(),
               "alpha": alpha, "lambda": blend, "autonomy_confidence": ca, "human_confidence": ch,
               "roughness": roughness, "corrective": corrective, "nominal_force": nominal,
               "executed_force": force.copy(), "reference": reference,
               "clearances": scene.clearance(p, c["robot_radius"]), "rho": rho,
               "tank_energy": tank.energy, "supplied_energy": total_supply,
               "robot_work": total_work, "removed_work": total_removed, "spilled_energy": total_spill,
               "port_allocations": allocation, "energy_residual": tank.residual()}
        for key, value in row.items():
            record[key].append(value)
        if collision or bounds or task.failed or task.success:
            break
    metadata = {"schema_version": 1, "trial_id": trial_id or f"{condition_name}_{seed}_{intended_goal}",
                "operator_id": operator_id, "seed": seed, "intended_goal": intended_goal,
                "condition": condition_name, "scene": scene.to_dict(),
                "provenance": "synthetic_task_space_simulation", "noisy_operator": noisy,
                "delay_seconds": c["feedback_delay"], "command_delay_seconds": c["command_delay"],
                "success": task.success and not (collision or bounds or task.failed),
                "collision": collision, "bounds_violation": bounds,
                "grasped_goal": task.grasped_goal, "events": task.events,
                "dt": dt, "decision_dt": decision, "initial_energy": tank.initial,
                "termination": "collision" if collision else "bounds" if bounds else
                "wrong_object" if task.failed else "success" if task.success else "timeout"}
    return Trial({k: np.asarray(v) for k, v in record.items()}, metadata)
