import numpy as np

from dasa.control.passivity import ThreePortTank


def test_work_matches_midpoint_plant_energy(config):
    tank = ThreePortTank(config)
    c = config["simulation"]
    rng = np.random.default_rng(3)
    for _ in range(100):
        v, f = rng.normal(size=(2, 3))
        dt, m, b = c["control_dt"], c["mass"], c["damping"]
        v1 = ((2 * m - dt * b) * v + 2 * dt * f) / (2 * m + dt * b)
        midpoint = (v + v1) / 2
        delta_k = 0.5 * m * (v1 @ v1 - v @ v)
        dissipation = dt * b * (midpoint @ midpoint)
        assert np.isclose(tank.robot_work(f, v), delta_k + dissipation, atol=1e-12)


def test_empty_tank_stops_energy_creation_at_rest(config):
    config["passivity"]["initial_energy"] = 0
    tank = ThreePortTank(config)
    step = tank.step(0, np.array([10., 2., 3.]), np.zeros(3), np.zeros(3), np.zeros(3), .5, .5, 0)
    assert np.linalg.norm(step.force) < 1e-10
    assert step.energy == 0


def test_random_switching_preserves_cumulative_budget(config):
    config["passivity"]["initial_energy"] = 0.005
    tank = ThreePortTank(config)
    rng = np.random.default_rng(42)
    v = np.zeros(3)
    c = config["simulation"]
    for t in range(1500):
        force = rng.normal(0, 80, 3)
        step = tank.step(t * c["control_dt"], force, v, np.zeros(3), np.zeros(3), rng.random(), rng.random(), .2)
        assert step.energy >= 0
        assert np.isclose(step.allocations.sum(), step.removed_work)
        assert abs(tank.residual()) < 1e-8
        v = ((2 * c["mass"] - c["control_dt"] * c["damping"]) * v
             + 2 * c["control_dt"] * step.force) / (2 * c["mass"] + c["control_dt"] * c["damping"])
    assert tank.work <= tank.initial + 1e-8


def test_two_port_has_no_third_allocation(config):
    config["passivity"]["initial_energy"] = 0
    tank = ThreePortTank(config)
    step = tank.step(0, np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3), 0, 0, 0, False)
    assert step.allocations[2] == 0
