import numpy as np

from dasa.control.arbitration import CONDITIONS, Confidence, authority
from dasa.control.motion import ForceKalman, potential_force, risk_weight
from dasa.geometry import clamp_norm
from dasa.models.intent import IntentEstimator, ReferenceBank
from dasa.models.filter import HSMMFilter


class HierarchicalController:
    def __init__(self, config, scene, skill, cvae, condition_name="C3", estimator=None):
        self.config, self.scene = config, scene
        self.condition = CONDITIONS[condition_name]
        self.estimator = estimator or IntentEstimator(config, ReferenceBank(skill, scene),
                                                       cvae, self.condition.intent)
        self.paths, self.covariances = zip(*(skill.reproduce(scene, g) for g in range(3)))
        self.kalman = ForceKalman(config["motion"]["kalman_process"],
                                  config["motion"]["kalman_measurement"],
                                  config["motion"].get("kalman_initial_variance", 1.0))
        self.confidence = Confidence(config)
        self.posterior = np.ones(3) / 3
        self.latent = np.zeros(config["intent"]["latent_dim"])
        self.progress = np.zeros((3, 3), dtype=int)
        self.filters = [HSMMFilter(skill, scene, goal) for goal in range(3)] \
            if config["skill"].get("online_filter", False) else None
        self.filter_interval = skill.duration_seconds / skill.samples
        self.filter_clock = 0.0
        self.filter_initialized = False

    def decide(self, raw, observed, current, age, phase, grasped_goal, rho,
               corrective=False, autonomous_goal=None):
        c, scene, control = self.config["simulation"], self.scene, self.condition
        tracking = self.config["skill"]
        p, v = current[:3], current[3:]
        human = self.kalman.step(raw)
        if phase == 0 and autonomous_goal is None:
            self.posterior, self.latent = self.estimator.step(raw, observed, current, scene, age)
        goal = int(np.argmax(self.posterior)) if autonomous_goal is None else autonomous_goal
        if grasped_goal is not None:
            goal = grasped_goal
        path, cov = self.paths[goal], self.covariances[goal]
        goal_index = max(tracking.get("goal_index_floor", 4), int(np.argmin(np.linalg.norm(
            path[:max(1, int(len(path) * tracking.get("goal_search_fraction", .60)))]
            - scene.goals[goal], axis=1))))
        goal_index = min(goal_index, len(path) - 1)
        if phase == 0:
            lo, hi = 0, goal_index + 1
        elif phase == 1:
            lo, hi = goal_index, min(len(path), goal_index + max(1, int(
                len(path) * tracking.get("lift_segment_fraction", .22))))
        else:
            lo, hi = goal_index, len(path)
        nearest = lo + int(np.argmin(np.linalg.norm(path[lo:hi] - p, axis=1)))
        stage = min(phase, 2)
        previous = max(lo, self.progress[goal, stage])
        reference_index = min(max(nearest + tracking.get("lookahead_samples", 5), previous), hi - 1)
        if np.linalg.norm(path[reference_index] - p) < tracking.get("proximity_threshold_m", .035):
            reference_index = min(reference_index + tracking.get("progress_advance_samples", 3), hi - 1)
        self.progress[goal, stage] = reference_index
        reference = path[reference_index].copy()
        if phase == 0 and reference_index >= goal_index - tracking.get("goal_snap_samples", 4):
            reference = scene.goals[goal].copy()
        elif phase == 1:
            reference = np.r_[scene.goals[goal][:2], c["lift_height"]]
        elif phase == 2 and (reference_index >= len(path) - tracking.get("place_snap_samples", 9)
                             or np.linalg.norm(p - path[-1]) < tracking.get("place_snap_distance_m", .05)):
            reference = scene.place.copy()
        autonomous_force = self.config["arbitration"]["stiffness"] * (reference - p)
        autonomous_force -= self.config["arbitration"]["damping"] * v
        covariance = cov[reference_index]
        if self.filters is not None:
            self.filter_clock += c["decision_dt"]
            if not self.filter_initialized or self.filter_clock >= self.filter_interval:
                for state_filter in self.filters:
                    state_filter.update(p)
                self.filter_clock %= self.filter_interval
                self.filter_initialized = True
            _, covariance = self.filters[goal].reference()
        clearance = float(scene.clearance(observed[:3], c["robot_radius"]).min())
        risk_distance = float(np.linalg.norm(observed[:3] - scene.centers, axis=1).min()) \
            if self.config["motion"]["risk_distance"] == "center" else clearance
        blend = risk_weight(risk_distance, self.config) if control.layer_two else 0.0
        target = scene.goals[goal] if phase == 0 else reference
        field = potential_force(p, target, scene, self.config)
        motion = (1 - blend) * human + blend * field
        ca, ch, roughness = self.confidence.values(human, covariance, rho)
        alpha, scale = authority(control, ca, ch, self.config, clearance, corrective)
        nominal = clamp_norm(scale * ((1 - alpha) * motion + alpha * autonomous_force), c["force_limit"])
        return {"human_force": human, "posterior": self.posterior.copy(), "latent": self.latent.copy(),
                "reference": reference, "nominal_force": nominal, "alpha": alpha, "lambda": blend,
                "autonomy_confidence": ca, "human_confidence": ch, "roughness": roughness,
                "inferred_goal": goal}
