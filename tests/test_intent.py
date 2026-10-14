import numpy as np
import torch

from dasa.models.intent import angular_cost, negative_elbo, previous_latent, sequence_windows
from dasa.metrics import intent_metrics


def test_angular_likelihood_prefers_aligned_action():
    cost = angular_cost(np.array([1., 0, 0]), np.eye(3))
    assert cost[0] == 0
    assert np.isclose(cost[1], np.pi / 2)


def test_zero_action_has_uniform_evidence():
    assert np.array_equal(angular_cost(np.zeros(3), np.eye(3)), np.zeros(3))


def test_kl_is_positive_and_zero_for_standard_normal():
    a = torch.zeros(4, 3)
    nll, kl = negative_elbo(a, a, torch.zeros(4, 8), torch.zeros(4, 8))
    assert nll == 0 and kl == 0
    _, kl = negative_elbo(a, a, torch.ones(4, 8), torch.zeros(4, 8))
    assert kl > 0


def test_classifier_latent_is_lagged_and_resets_at_episode_boundary():
    mu = torch.arange(12).reshape(4, 3).float()
    shifted = previous_latent(mu, np.array(["a", "a", "b", "b"]))
    assert torch.equal(shifted[0], torch.zeros(3))
    assert torch.equal(shifted[1], mu[0])
    assert torch.equal(shifted[2], torch.zeros(3))
    assert torch.equal(shifted[3], mu[2])


def test_sequence_windows_do_not_leak_future_or_cross_episode():
    data = {"action": np.arange(4)[:, None], "context": np.zeros((4, 1)),
            "episode": np.array(["a", "a", "b", "b"])}
    windows = sequence_windows(data, 3)
    assert np.array_equal(windows[1, :, 0], [0, 0, 1])
    assert np.array_equal(windows[2, :, 0], [2, 2, 2])


def test_wrong_stable_predictions_do_not_count_as_accurate(config):
    p = np.tile([0.99, 0.005, 0.005], (20, 1))
    metrics = intent_metrics(np.arange(20) * .02, p, 1, config)
    assert metrics["prediction_stability"] == 1
    assert metrics["accuracy"] == 0
    assert metrics["early_prediction_seconds"] is None
