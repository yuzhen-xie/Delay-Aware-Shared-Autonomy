import numpy as np
import pytest

from dasa.data import Demonstration, synthetic_demonstration, validate_partitions, successful
from dasa.delay import DelayBuffer
from dasa.control.arbitration import CONDITIONS, authority
from dasa.control.motion import ForceKalman, risk_weight
from dasa.statistics import holm, paired_inference


def test_fractional_delay_uses_causal_source_time():
    buffer = DelayBuffer(np.zeros(1))
    for t in np.arange(0, 1.01, .1):
        buffer.push(t, np.array([t]))
    observation = buffer.observe(1.0, .25)
    assert np.isclose(observation.source_time, .7)
    assert observation.source_time <= .75


def test_jitter_does_not_redeliver_older_packet():
    buffer = DelayBuffer(np.zeros(1))
    for t in np.arange(0, 1.01, .1):
        buffer.push(t, np.array([t]))
    first = buffer.observe(.8, .1)
    second = buffer.observe(.9, .6)
    assert second.source_time >= first.source_time


def test_c1_disables_both_assistance_layers(config):
    assert not CONDITIONS["C1"].layer_two
    assert authority(CONDITIONS["C1"], .9, .01, config, .1, True)[0] == 0


def test_low_confidence_extension_reduces_magnitude(config):
    _, baseline_scale = authority(CONDITIONS["C3"], .01, .01, config, .3, False)
    _, guarded_scale = authority(CONDITIONS["C3_guarded"], .01, .01, config, .3, False)
    assert baseline_scale == 1
    assert guarded_scale < .2


def test_risk_weight_increases_with_clearance(config):
    assert risk_weight(.08, config) < risk_weight(.25, config)


def test_kalman_preserves_constant_force_amplitude():
    kalman = ForceKalman(.08, .8)
    signal = np.array([2., -3., 1.])
    for _ in range(100):
        filtered = kalman.step(signal)
    assert np.allclose(filtered, signal, atol=1e-6)
    assert np.linalg.eigvalsh(kalman.covariance).min() > 0


def test_demo_round_trip_and_operator_leakage_rejection(config, tmp_path):
    d = synthetic_demonstration(config, 1, "same_operator", "episode1", 0)
    d.save(tmp_path / "a.npz")
    restored = Demonstration.load(tmp_path / "a.npz")
    assert np.allclose(d.position, restored.position)
    d.metadata["episode_id"] = "episode2"
    d.save(tmp_path / "b.npz")
    with pytest.raises(ValueError, match="leakage"):
        validate_partitions(tmp_path, {"train": ["a.npz"], "test": ["b.npz"]})


def test_failed_demonstrations_are_rejected(config):
    d = synthetic_demonstration(config, 1, "operator", "episode", 0)
    d.metadata["collision"] = True
    accepted, reasons = successful(d, config)
    assert not accepted and "contact_or_bounds" in reasons


def test_recorded_geometry_and_boolean_outcomes_are_validated(config):
    d = synthetic_demonstration(config, 1, "operator", "episode", 0)
    d.metadata["success"] = "false"
    with pytest.raises(ValueError, match="JSON boolean"):
        d.validate()
    d.metadata["success"] = False
    d.scene.centers[0, 0] = np.nan
    with pytest.raises(ValueError, match="scene geometry"):
        d.validate()


def test_demonstration_timeout_uses_elapsed_duration(config):
    d = synthetic_demonstration(config, 1, "operator", "episode", 0)
    d.time += 1000.0
    _, reasons = successful(d, config)
    assert "timeout" not in reasons


def test_measured_admission_uses_recorded_device_budget(config):
    d = synthetic_demonstration(config, 1, "operator", "episode", 0)
    d.metadata["provenance"] = "UR10_measured_teleoperation"
    d.time = d.time * (30.0 / (d.time[-1] - d.time[0]))
    d.metadata["execution_contract"] = {
        "maximum_seconds": 60.0, "robot_radius_m": config["simulation"]["robot_radius"],
        "place_tolerance_m": config["simulation"]["place_tolerance"],
        "workspace_min_m": config["simulation"]["workspace_min"],
        "workspace_max_m": config["simulation"]["workspace_max"]}
    accepted, reasons = successful(d, config)
    assert accepted, reasons
    d.metadata["execution_contract"]["maximum_seconds"] = 25.0
    assert "timeout" in successful(d, config)[1]
    d.metadata["execution_contract"]["robot_radius_m"] = float("nan")
    assert "invalid_execution_contract" in successful(d, config)[1]


def test_holm_correction_and_missing_values():
    corrected = holm([.01, .04, .03, None])
    assert np.allclose(corrected[:3], [.03, .06, .06])
    assert corrected[3] is None


def test_exact_paired_test_has_correct_minimum_probability():
    result = paired_inference([1] * 5, [0] * 5, 1)
    assert result["n_units"] == 5
    assert result["p_raw"] == 2 / 2**5
    assert result["effect_dz"] is None


def test_single_training_repeat_does_not_generate_p_value():
    result = paired_inference([.9], [.7], 1)
    assert result["p_raw"] is None
