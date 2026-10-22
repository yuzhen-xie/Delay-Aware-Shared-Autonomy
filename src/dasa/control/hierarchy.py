import numpy as np

from dasa.control.arbitration import CONDITIONS, Confidence, authority
from dasa.control.motion import ForceKalman, potential_force, risk_weight
from dasa.geometry import Scene, clamp_norm
from dasa.models.intent import IntentEstimator, ReferenceBank
from dasa.models.filter import HSMMFilter


class HierarchicalController:
    def __init__(self, config, scene, skill, cvae, condition_name="C3", estimator=None):
        if (not skill.fit_manifest.get("available") or not skill.fit_manifest.get("records")
                or any(record.get("provenance") != "UR10_measured_teleoperation"
                       for record in skill.fit_manifest["records"])):
            raise ValueError("Physical shared control requires an audited measured skill")
        self.config, self.scene = config, scene
        self.skill = skill
        self.condition = CONDITIONS[condition_name]
        self.estimator = estimator or IntentEstimator(config, ReferenceBank(skill, scene),
                                                       cvae, self.condition.intent)
        self.kalman = ForceKalman(config["motion"]["kalman_process"],
                                  config["motion"]["kalman_measurement"],
                                  config["motion"].get("kalman_initial_variance", 1.0),
                                  config["task"]["decision_dt"])
        self.confidence = Confidence(config)
        self.posterior = np.ones(3) / 3
        self.latent = np.zeros(config["intent"]["latent_dim"])
        self.filters = [HSMMFilter(skill, scene, goal) for goal in range(3)]

    def update_scene(self, scene, freeze_task_frames=False):
        if scene.frame != self.scene.frame or scene.goals.shape != self.scene.goals.shape:
            raise ValueError("Live scene updates must preserve the base frame and ordered candidate identities")
        candidate = Scene.from_dict(scene.to_dict())
        if freeze_task_frames:
            candidate.goals = self.scene.goals.copy()
            candidate.place = self.scene.place.copy()
            candidate.start = self.scene.start.copy()
        changed = candidate.obstacle_ids != self.scene.obstacle_ids
        for name in ("goals", "place", "start", "centers", "radii"):
            left, right = getattr(candidate, name), getattr(self.scene, name)
            if not np.isfinite(left).all():
                raise ValueError("Live task geometry must remain finite")
            if left.shape != right.shape:
                changed = True
            else:
                delta = np.abs(left - right) if name == "radii" else np.linalg.norm((left - right).reshape(-1, 3), axis=1)
                changed = changed or np.max(delta, initial=0.) > .001
        if not changed:
            return False
        for goal, state_filter in enumerate(self.filters):
            state_filter.update_scene(candidate, goal)
        self.scene = candidate
        self.estimator.bank = ReferenceBank(self.skill, candidate)
        return True

    def decide(self, raw, observed, current, age, phase, grasped_goal, rho,
               corrective=False, autonomous_goal=None, elapsed_seconds=None):
        task, scene, control = self.config["task"], self.scene, self.condition
        current = np.asarray(current, float)
        if current.shape != (6,) or not np.isfinite(current).all():
            raise ValueError("The controller requires measured TCP position and velocity")
        p, v = current[:3], current[3:]
        human = self.kalman.step(raw, task["decision_dt"])
        if phase == 0 and autonomous_goal is None:
            self.posterior, self.latent = self.estimator.step(raw, observed, current, scene, age)
        if autonomous_goal is not None and (type(autonomous_goal) is not int or autonomous_goal not in range(3)):
            raise ValueError("An autonomous execution requires an assigned categorical goal")
        goal = int(np.argmax(self.posterior)) if autonomous_goal is None else autonomous_goal
        if grasped_goal is not None:
            if autonomous_goal is not None and grasped_goal != autonomous_goal:
                raise ValueError("The grasped object differs from the autonomous assignment")
            goal = grasped_goal
        for state_filter in self.filters:
            state_filter.update(p, task["decision_dt"] if elapsed_seconds is None else elapsed_seconds)
        states, predicted_covariance = self.filters[goal].plan(
            current, self.config["skill"].get("prediction_horizon_samples", 32))
        reference, reference_velocity = states[1, :3], states[1, 3:]
        _, covariance = self.filters[goal].reference()
        stiffness = self.config["arbitration"]["stiffness"]
        damping = self.config["arbitration"]["damping"]
        stiffness_force = stiffness * (reference - p)
        autonomous_force = stiffness_force - damping * v
        clearance = float(scene.clearance(observed[:3], task["robot_radius"]).min())
        risk_distance = float(np.linalg.norm(observed[:3] - scene.centers, axis=1).min()) \
            if self.config["motion"]["risk_distance"] == "center" else clearance
        blend = risk_weight(risk_distance, self.config) if control.layer_two else 0.0
        target = scene.goals[goal] if phase == 0 else reference
        field = potential_force(p, target, scene, self.config)
        motion = (1 - blend) * human + blend * field
        ca, ch, roughness = self.confidence.values(human, covariance, rho)
        alpha, scale = authority(control, ca, ch, self.config, clearance, corrective)
        nominal = clamp_norm(scale * ((1 - alpha) * motion + alpha * autonomous_force), task["force_limit"])
        return {"human_force": human, "posterior": self.posterior.copy(), "latent": self.latent.copy(),
                "reference": reference.copy(), "reference_velocity": reference_velocity.copy(),
                "reference_covariance": covariance, "reference_step_seconds": self.filters[goal].step_seconds,
                "predicted_reference_covariance": predicted_covariance[1],
                "nominal_force": nominal, "autonomous_force": autonomous_force,
                "autonomous_stiffness_force": stiffness_force, "autonomous_damping_gain": damping,
                "motion_force": motion, "authority_scale": scale,
                "alpha": alpha, "lambda": blend, "autonomy_confidence": ca,
                "human_confidence": ch, "roughness": roughness, "inferred_goal": goal,
                "state_probability": self.filters[goal].belief.sum(1),
                "state_filter_age_seconds": self.filters[goal].elapsed_seconds,
                "state_filter_step_remainder_seconds": self.filters[goal].clock,
                "reference_mode": "causal_explicit_duration_filter_receding_LQT"}
