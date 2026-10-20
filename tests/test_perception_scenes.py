import copy
import json

import numpy as np
import pytest

from dasa.geometry import Scene
from dasa.robotics.perception import (
    CameraIntrinsics, ColorObject, PerceptionStream, deproject, detect_scene, perceive_file,
    snapshot_to_scene,
)
from dasa.scenes import (
    SceneConstraints, generate_manifest, load_scene_manifest, validate_scene, write_scene_manifest,
)


def test_deprojection_and_rigid_base_coordinates():
    camera = CameraIntrinsics(80, 60, 100., 120., 40., 30.)
    points = deproject(np.array([[40., 30.], [50., 42.]]), np.array([1., 2.]), camera)
    assert np.allclose(points, [[0, 0, 1], [.2, .2, 2]])
    with pytest.raises(ValueError, match="positive metric"):
        deproject(np.array([[40., 30.]]), np.array([0.]), camera)


def _marker_image():
    camera = CameraIntrinsics(50, 50, 100., 100., 25., 25.)
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    depth = np.full((50, 50), 1000, dtype=float)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    objects = []
    for j, (u, v) in enumerate([(5, 5), (30, 5), (5, 30), (30, 30)]):
        rgb[v:v+8, u:u+8] = colors[j]
        objects.append(ColorObject(f"object{j}", "goal" if j < 3 else "obstacle",
                                   colors[j], colors[j], radius_m=0.05 if j == 3 else 0))
    transform = np.eye(4)
    transform[:3, 3] = [.5, -.1, .2]
    return camera, rgb, depth, objects, transform


def test_snapshot_marker_geometry_and_age():
    camera, rgb, depth, objects, transform = _marker_image()
    depth[5:7, 5:7] = 0
    depth[7, 7] = 2900  # A valid-range background leak is removed by the depth MAD rule.
    snapshot = detect_scene(rgb, depth, camera, transform, objects, timestamp_s=1.2,
                            sequence=4, calibration_id="check-2026-10-20")
    first = snapshot["detections"][0]
    assert first["valid_depth_pixels"] == 59
    assert first["center_base_m"][2] == pytest.approx(1.2)
    assert np.allclose(np.asarray(first["center_camera_m"]) + transform[:3, 3],
                       first["center_base_m"])
    scene = snapshot_to_scene(snapshot, ["object0", "object1", "object2"],
                              [.6, 0, .1], [.2, 0, .5],
                              maximum_age_s=.2, current_time_s=1.3)
    assert scene.provenance == "calibrated_rgbd_geometry"
    assert scene.centers.shape == (1, 3)
    assert scene.radii[0] == .05
    with pytest.raises(ValueError, match="stale"):
        snapshot_to_scene(snapshot, ["object0", "object1", "object2"],
                           [.6, 0, .1], [.2, 0, .5],
                           maximum_age_s=.2, current_time_s=1.5)


def test_known_radius_sphere_center_uses_surface_geometry():
    camera = CameraIntrinsics(100, 100, 100., 100., 50., 50.)
    v, u = np.indices((100, 100))
    a = 1 + ((u - 50) / 100)**2 + ((v - 50) / 100)**2
    discriminant = 1 - a * (1 - .1**2)
    mask = discriminant > 0
    depth = np.zeros((100, 100))
    depth[mask] = (1 - np.sqrt(discriminant[mask])) / a[mask]
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[mask] = [255, 0, 0]
    spec = ColorObject("sphere", "obstacle", (255, 0, 0), (255, 0, 0),
                       geometry="sphere", radius_m=.1)
    snapshot = detect_scene(rgb, depth, camera, np.eye(4), [spec], timestamp_s=0,
                            sequence=0, calibration_id="cal1", depth_scale_m=1.)
    detection = snapshot["detections"][0]
    assert np.allclose(detection["center_base_m"], [0, 0, 1.], atol=1e-5)
    assert detection["surface_point_camera_m"][2] < .95
    assert detection["sphere_fit_rms_m"] < 1e-6


def test_missing_objects_and_invalid_extrinsics_fail():
    camera, rgb, depth, objects, transform = _marker_image()
    rgb[:20, :20] = 0
    with pytest.raises(ValueError, match="No color"):
        detect_scene(rgb, depth, camera, transform, objects, timestamp_s=0,
                     sequence=0, calibration_id="cal1")
    transform[0, 0] = 2
    with pytest.raises(ValueError, match="rigid transform"):
        detect_scene(rgb, depth, camera, transform, objects, timestamp_s=0,
                     sequence=0, calibration_id="cal1")


def test_perception_file_preserves_calibration_and_scene(tmp_path):
    from dataclasses import asdict
    camera, rgb, depth, objects, transform = _marker_image()
    image_path, specification, output = [tmp_path / name for name in
                                         ["rgbd.npz", "spec.json", "snapshot.json"]]
    np.savez(image_path, rgb=rgb, depth=depth)
    specification.write_text(json.dumps({
        "intrinsics": asdict(camera), "objects": [asdict(obj) for obj in objects],
        "camera_to_base": transform.tolist(), "timestamp_s": 1., "sequence": 1,
        "calibration_id": "cal1", "scene": {"goal_ids": ["object0", "object1", "object2"],
                                                  "place": [.6, 0, .1], "start": [.2, 0, .5]}}))
    snapshot = perceive_file(image_path, specification, output)
    assert json.loads(output.read_text())["scene"] == snapshot["scene"]
    assert snapshot["calibration_id"] == "cal1"


def test_online_updates_are_whole_frame_monotonic_and_expire():
    camera, rgb, depth, objects, transform = _marker_image()
    snapshot = detect_scene(rgb, depth, camera, transform, objects, timestamp_s=1.,
                            sequence=1, calibration_id="cal1", update_mode="online")
    stream = PerceptionStream(["object0", "object1", "object2"], [.6, 0, .1], [.2, 0, .5])
    stream.update(snapshot, 1.1)
    original = stream.scene(1.1).centers.copy()
    snapshot["detections"][-1]["center_base_m"][0] += 10
    assert np.array_equal(stream.scene(1.1).centers, original)
    with pytest.raises(ValueError, match="must both increase"):
        stream.update(snapshot, 1.1)
    with pytest.raises(ValueError, match="stale"):
        stream.scene(1.3)


@pytest.mark.parametrize("scenario", ["demonstrated", "novel", "wider", "extra_obstacles"])
def test_manifests_are_reproducible_feasible_and_integrity_checked(config, tmp_path, scenario):
    manifest = generate_manifest(config, 4, scenario, seed=1234)
    assert manifest == generate_manifest(config, 4, scenario, seed=1234)
    path = tmp_path / "scenes.json"
    assert write_scene_manifest(config, path, 4, scenario, seed=1234) == manifest
    assert load_scene_manifest(path) == manifest
    for entry in manifest["scenes"]:
        assert not validate_scene(Scene.from_dict(entry["scene"]), config)
        assert entry["attempts"][-1]["failures"] == []
    modified = json.loads(path.read_text())
    modified["scenes"][0]["scene"]["goals"][0][0] += .01
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="integrity"):
        load_scene_manifest(path)


def test_infeasible_scenes_are_rejected_with_recorded_reason(config):
    bad = copy.deepcopy(config)
    bad["scene"]["goals"] = [[.5, 0, .12]] * 3
    with pytest.raises(ValueError, match="overlapping_targets"):
        generate_manifest(bad, 1, constraints=SceneConstraints(maximum_attempts=2))
    with pytest.raises(ValueError, match="Unknown scene"):
        generate_manifest(config, 1, scenario="not-a-scenario")
    scene = Scene.from_dict(generate_manifest(config, 1)["scenes"][0]["scene"])
    scene.centers[0] = scene.goals[0]
    assert "task_obstacle_intersection" in validate_scene(scene, config)
