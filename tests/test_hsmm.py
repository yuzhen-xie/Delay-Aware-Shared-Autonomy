import numpy as np

from dasa.models.hsmm import duration_logpmf, segment_posterior, TPHSMM
from dasa.data import synthetic_demonstration


def test_explicit_duration_posterior_normalizes_per_sample():
    emission = np.array([[0., -3], [0, -3], [-3, 0], [-3, 0]])
    duration = np.array([duration_logpmf(2, 0.5, 4)] * 2)
    gamma, counts, likelihood = segment_posterior(emission, duration)
    assert np.isfinite(likelihood)
    assert np.allclose(gamma.sum(1), 1)
    assert np.allclose(counts.sum(1), 1)
    assert gamma[0, 0] > gamma[0, 1]
    assert gamma[-1, 1] > gamma[-1, 0]


def test_task_parameterization_is_translation_equivariant(config, scene):
    model = TPHSMM(config)
    means, covariance = model.world_gaussians(scene, 0)
    shift = np.array([.12, -.08, .02])
    scene.goals += shift
    scene.place += shift
    scene.start += shift
    means2, covariance2 = model.world_gaussians(scene, 0)
    assert np.allclose(means2, means + shift)
    assert np.allclose(covariance2, covariance)


def test_fit_reproduce_and_serialization(config, scene, tmp_path):
    config["skill"].update(trajectory_samples=42, states=5, em_iterations=1)
    demos = [synthetic_demonstration(config, j, f"o{j}", f"e{j}", j % 3) for j in range(6)]
    model = TPHSMM(config).fit(demos)
    path, covariance = model.reproduce(scene, 0)
    assert path.shape == (42, 3)
    assert np.isfinite(path).all()
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert np.allclose(path[0], scene.start)
    model.save(tmp_path / "model.npz")
    restored = TPHSMM.load(config, tmp_path / "model.npz")
    assert np.allclose(restored.reproduce(scene, 0)[0], path)
