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
    provenance: str = "planned_physical_layout"
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
    if scenario not in {"demonstrated", "novel", "wider", "extra_obstacles"}:
        raise ValueError("Unknown physical scene condition")
    if scenario != "demonstrated":
        low, high = c["novel_translation"]
        scale = 1.6 if scenario == "wider" else 1.0
        distance = rng.uniform(low, high, 3) * scale
        angle = rng.uniform(-np.pi, np.pi, 3)
        goals[:, :2] += distance[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
        added = 3 if scenario == "extra_obstacles" else 1
        new_centers = []
        for _ in range(added):
            goal = int(rng.integers(0, 3))
            fraction = rng.uniform(.3, .7)
            new_centers.append(start + fraction * (goals[goal] - start) + rng.normal(0, .04, 3))
        centers = np.vstack([centers, np.asarray(new_centers)])
        radii = np.r_[radii, [c["added_obstacle_radius"]] * added]
        ids += [f"novel_{i}" for i in range(added)]
    return Scene(goals, place, start, centers, radii, ids, "planned_physical_layout", c["frame"])


def task_frames(scene: Scene, goal: int) -> tuple[np.ndarray, np.ndarray]:

    return np.repeat(np.eye(3)[None], 3, axis=0), np.array(
        [scene.start, scene.goals[goal], scene.place])
