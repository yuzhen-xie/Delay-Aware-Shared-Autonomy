import hashlib
import json
from pathlib import Path

import numpy as np

from dasa.config import write_json
from dasa.control.arbitration import CONDITIONS


OUTCOMES = ("success", "collision", "bounds", "wrong_object", "timeout", "aborted")


def write_annotation(recording: Path, output: Path, episode: str, operator: str,
                     intended_goal: int, outcome: str, grasped_goal: int | None = None):
    if output.exists():
        raise ValueError("Annotation output already exists")
    if not episode.strip() or not operator.strip():
        raise ValueError("Episode and operator identifiers must be nonempty")
    if type(intended_goal) is not int or intended_goal not in range(3):
        raise ValueError("Intended goal must be 0, 1 or 2")
    if grasped_goal is not None and (type(grasped_goal) is not int or grasped_goal not in range(3)):
        raise ValueError("Grasped goal must be 0, 1 or 2")
    if outcome not in OUTCOMES:
        raise ValueError("Unknown trial outcome")
    if outcome == "success" and grasped_goal != intended_goal:
        raise ValueError("A successful annotation requires the correct grasped goal")
    if outcome == "wrong_object" and (grasped_goal is None or grasped_goal == intended_goal):
        raise ValueError("A wrong-object annotation requires a different grasped goal")
    data = json.loads(recording.read_text(encoding="utf-8"))
    if (data.get("provenance") != "UR10_workstation_measurement"
            or data.get("schema_version", 0) < 2 or data.get("execute") is not True
            or not any(sample.get("executed") is True for sample in data.get("samples", []))):
        raise ValueError("Annotations require an executed workstation recording")
    condition = data.get("condition")
    delay = data.get("timing", {}).get("feedback_delay_seconds")
    if condition not in CONDITIONS or isinstance(delay, bool) or not isinstance(delay, (float, int)):
        raise ValueError("Recording condition and feedback delay are required")
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("Recorded delay must be finite and nonnegative")
    if outcome == "success":
        events = {event.get("event") for event in data.get("task_events", [])}
        required = {"grasp_contact_detected", "lift_height_reached", "gripper_open_measured"}
        if not required.issubset(events) or data.get("stop_reason") != "release_completed_requires_independent_annotation":
            raise ValueError("A successful annotation requires measured grasp, lift and release completion")
    result = {"episode_id": episode, "operator_id": operator, "intended_goal": intended_goal,
              "grasped_goal": grasped_goal, "success": outcome == "success",
              "placed": outcome == "success", "collision": outcome == "collision",
              "bounds_violation": outcome == "bounds", "termination": outcome,
              "source_condition": condition, "delay_seconds": float(delay),
              "annotation_source": "operator_entered_outcome",
              "source_recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest()}
    result["stage"] = data["stage"]
    write_json(output, result)
    return result
