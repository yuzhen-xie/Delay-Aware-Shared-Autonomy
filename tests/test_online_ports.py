import numpy as np
import pytest

from dasa.models.hsmm import TPHSMM
from dasa.models.filter import HSMMFilter
from dasa.control.ports import MultiportPassivity, Port


def test_online_filter_remains_normalized_beyond_duration_horizon(config, scene):
    model = TPHSMM(config)
    state_filter = HSMMFilter(model, scene, 0)
    for _ in range(model.samples * 3):
        probability = state_filter.update(scene.start)
        assert np.isclose(probability.sum(), 1)
        assert np.all(probability >= 0)
    mean, covariance = state_filter.reference()
    assert np.isfinite(mean).all()
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_port_controller_modifies_both_force_and_velocity():
    observer = MultiportPassivity(0)
    ports = [Port(np.ones(3), np.ones(3), "force"),
             Port(np.ones(3), np.ones(3), "force"),
             Port(np.ones(3), np.ones(3), "velocity")]
    corrected, removal = observer.regulate(ports, .001, 0, np.ones(3))
    assert np.allclose(corrected[0].effort, 0)
    assert np.allclose(corrected[2].flow, 0)
    assert np.all(removal > 0)
    assert abs(observer.residual()) < 1e-10


def test_port_weights_redistribute_when_a_port_has_no_outbound_power():
    observer = MultiportPassivity(0)
    ports = [Port(np.ones(3), -np.ones(3), "force"),
             Port(np.ones(3) * 5, np.ones(3), "force"),
             Port(np.zeros(3), np.ones(3), "velocity")]
    corrected, _ = observer.regulate(ports, .01, 0, np.array([1, 0, 1]))
    assert .01 * sum(p.effort @ p.flow for p in corrected) <= 1e-10


def test_random_multiport_arbitration_preserves_energy():
    rng = np.random.default_rng(11)
    observer = MultiportPassivity(.01)
    for _ in range(1000):
        ports = [Port(rng.normal(size=3), rng.normal(size=3), actuator)
                 for actuator in ["force", "force", "velocity"]]
        observer.regulate(ports, .001, .00001, rng.random(3))
        assert observer.energy >= 0
        assert abs(observer.residual()) < 1e-8


def test_invalid_port_actuator_is_rejected():
    observer = MultiportPassivity(0)
    with pytest.raises(ValueError, match="actuator"):
        observer.regulate([Port(np.ones(3), np.ones(3), "invalid")] * 3, .01, 0, np.ones(3))


def test_delayed_port_bench_accounts_packet_energy_once(config, tmp_path):
    from dasa.portbench import run
    import json
    config["experiments"]["port_bench_steps"] = 1200
    run(config, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["minimum_storage_j"] >= 0
    assert metrics["maximum_budget_residual_j"] < 1e-8
    assert metrics["duplicate_energy_packets_credited"] == 0
