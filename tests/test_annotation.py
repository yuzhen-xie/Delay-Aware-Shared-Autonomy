import json

import pytest

from dasa.robotics.annotation import write_annotation


def recording(tmp_path, execute=True):
    path = tmp_path / "recording.json"
    path.write_text(json.dumps({"provenance": "UR10_workstation_measurement", "schema_version": 2,
                               "execute": execute, "condition": "C3",
                               "samples": [{"executed": execute}],
                               "timing": {"feedback_delay_seconds": .8}}))
    return path


def test_explicit_success_annotations_preserve_source_and_reject_overwrite(tmp_path):
    path = recording(tmp_path)
    output = tmp_path / "annotation.json"
    result = write_annotation(path, output, "trial-1", "P01", 2, "success", 2)
    assert result["success"] and result["placed"]
    assert result["delay_seconds"] == .8 and result["source_condition"] == "C3"
    assert len(result["source_recording_sha256"]) == 64
    assert json.loads(output.read_text()) == result
    with pytest.raises(ValueError, match="already exists"):
        write_annotation(path, output, "trial-1", "P01", 2, "success", 2)


def test_failed_annotation_keeps_ungrasped_object_and_rejects_false_success(tmp_path):
    path = recording(tmp_path)
    result = write_annotation(path, tmp_path / "failed.json", "trial-1", "P01", 2, "timeout")
    assert result["grasped_goal"] is None and not result["success"]
    with pytest.raises(ValueError, match="correct grasped"):
        write_annotation(path, tmp_path / "false.json", "trial-1", "P01", 2, "success", 0)
    with pytest.raises(ValueError, match="executed workstation"):
        write_annotation(recording(tmp_path, False), tmp_path / "read-only.json", "trial-1", "P01", 2, "timeout")
