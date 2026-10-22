from dataclasses import dataclass

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return np.divide(vector, norm, out=np.zeros_like(vector), where=norm > 1e-12)


def clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-12))


@dataclass
class Scene:
    goals: np.ndarray
    place: np.ndarray
    start: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    obstacle_ids: list[str]
    provenance: str = "known_scene_simulation"
    frame: str = "base"

    def clearance(self, position: np.ndarray, robot_radius: float) -> np.ndarray:
        return np.linalg.norm(position - self.centers, axis=-1) - self.radii - robot_radius

    def swept_clearance(self, start: np.ndarray, end: np.ndarray, robot_radius: float) -> float:
        delta = end - start
        u = np.clip((self.centers - start) @ delta / max(delta @ delta, 1e-15), 0, 1)
        closest = start + u[:, None] * delta
        return float(np.min(np.linalg.norm(closest - self.centers, axis=-1)
                            - self.radii - robot_radius))

    def to_dict(self) -> dict:
        return {"goals": self.goals.tolist(), "place": self.place.tolist(),
                "start": self.start.tolist(), "centers": self.centers.tolist(),
                "radii": self.radii.tolist(), "obstacle_ids": self.obstacle_ids,
                "provenance": self.provenance, "frame": self.frame}

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        return cls(**{k: np.asarray(v) if k in {"goals", "place", "start", "centers", "radii"}
                      else v for k, v in data.items()})


def make_scene(config: dict, rng: np.random.Generator, scenario: str = "demonstrated") -> Scene:
    c = config["scene"]
    goals, place, start = map(lambda x: np.asarray(x, dtype=float),
                             (c["goals"], c["place"], c["start"]))
    centers = np.array([o["center"] for o in c["obstacles"]], dtype=float)
    radii = np.array([o["radius"] for o in c["obstacles"]])
    ids = [o["id"] for o in c["obstacles"]]
    span = c["training_translation"]
    translation = rng.uniform(-span, span, 3)
    translation[2] *= 0.25
    goals, place, start, centers = [x + translation for x in (goals, place, start, centers)]
    if scenario != "demonstrated":
        low, high = c["novel_translation"]
        scale = 1.6 if scenario == "wider" else 1.0
        shift = rng.uniform(low, high, (3, 2)) * rng.choice([-1, 1], (3, 2)) * scale
        goals[:, :2] += shift
        goals[:, :2] = np.clip(goals[:, :2], [0.30, -0.42], [0.78, 0.42])
        new_centers = [[0.49, -0.12, 0.31]]
        if scenario == "extra_obstacles":
            new_centers += [[0.68, 0.14, 0.34], [0.29, 0.12, 0.28]]
        centers = np.vstack([centers, np.asarray(new_centers) + translation])
        radii = np.r_[radii, [c["added_obstacle_radius"]] * len(new_centers)]
        ids += [f"novel_{i}" for i in range(len(new_centers))]
    return Scene(goals, place, start, centers, radii, ids, c["provenance"], c["frame"])


def waypoints(scene: Scene, goal: int, lift: float) -> np.ndarray:
    target = scene.goals[goal]
    return np.array([scene.start, [scene.start[0], scene.start[1], lift],
                     [target[0], target[1], lift], target,
                     [target[0], target[1], lift], [scene.place[0], scene.place[1], lift],
                     scene.place])


def task_frames(scene: Scene, goal: int) -> tuple[np.ndarray, np.ndarray]:

    return np.repeat(np.eye(3)[None], 3, axis=0), np.array(
        [scene.start, scene.goals[goal], scene.place])
