import copy

import numpy as np
import pytest

from dasa.data import generate_dataset
from dasa.geometry import Scene
from dasa.robotics.calibration import rigid_registration, transform_points, transform_wrench
from dasa.robotics.kinematics import forward, jacobian, inverse
from dasa.study import validate_protocol, write_schedule, read_csv


def test_swept_collision_detects_tunneling():
    scene = Scene(np.zeros((3, 3)), np.zeros(3), np.zeros(3),
                  np.zeros((1, 3)), np.array([.1]), ["sphere"])
    start, end = np.array([-1., 0, 0]), np.array([1., 0, 0])
    assert scene.clearance(start, .02).min() > 0
    assert scene.clearance(end, .02).min() > 0
    assert scene.swept_clearance(start, end, .02) < 0


def test_calibration_recovers_transform():
    source = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    translation = np.array([.2, .4, -.1])
    target = source @ rotation.T + translation
    transform, residual = rigid_registration(source, target)
    assert np.max(residual) < 1e-10
    assert np.allclose(transform_points(source, transform), target)
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1)


def test_collinear_calibration_is_rejected():
    points = np.c_[np.arange(3), np.zeros((3, 2))]
    with pytest.raises(ValueError, match="Collinear"):
        rigid_registration(points, points)


def test_wrench_translation_includes_moment():
    t = np.eye(4)
    t[:3, 3] = [1, 0, 0]
    wrench = transform_wrench([0, 1, 0, 0, 0, 0], t)
    assert np.allclose(wrench, [0, 1, 0, 0, 0, 1])


@pytest.mark.parametrize("model", ["UR10", "UR10e"])
def test_jacobian_matches_finite_difference(model):
    joints = np.array([.1, -.5, .7, .2, -.3, .1])
    j = jacobian(joints, model)
    eps = 1e-7
    for k in range(6):
        shifted = joints.copy()
        shifted[k] += eps
        measured = (forward(shifted, model)[:3, 3] - forward(joints, model)[:3, 3]) / eps
        assert np.allclose(measured, j[:3, k], atol=1e-6)


def test_inverse_position_from_nearby_seed():
    q = np.array([.1, -.5, .7, .2, -.3, .1])
    target = forward(q)[:3, 3]
    fitted = inverse(target, q + .02)
    assert np.linalg.norm(forward(fitted)[:3, 3] - target) < 1e-5


def test_absent_actual_approval_record_prevents_human_analysis():
    with pytest.raises(ValueError, match="approval_id"):
        validate_protocol({}, [])


def test_study_schedule_has_every_condition_for_every_participant(config, tmp_path):
    output = tmp_path / "schedule.csv"
    write_schedule(config, output)
    rows = read_csv(output)
    assert len(rows) == 12 * 4 * 4
    for participant in {r["participant_id"] for r in rows}:
        selected = [r for r in rows if r["participant_id"] == participant]
        assert len({(r["condition"], r["delay_seconds"]) for r in selected}) == 16


def test_dataset_generation_reproducible_and_has_disjoint_operators(config, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    c = copy.deepcopy(config)
    c["experiments"].update(bootstrap_demonstrations=3, validation_demonstrations=3, test_demonstrations=3)
    assert generate_dataset(c, a) == generate_dataset(c, b)
    from dasa.data import Demonstration
    first = Demonstration.load(a / "train/train_0000.npz")
    second = Demonstration.load(b / "train/train_0000.npz")
    assert np.array_equal(first.raw_force, second.raw_force)
