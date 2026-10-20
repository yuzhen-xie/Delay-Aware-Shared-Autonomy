import json

import numpy as np
import pytest
import torch

from dasa.models.intent import (angular_cost, cvae_loss, load_models, negative_elbo,
                               objective_spec, previous_latent, sequence_windows, train_intent)
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


def test_joint_elbo_is_unweighted_reconstruction_plus_kl_and_regularizes_posterior(config):
    config["intent"]["objective"] = "joint_elbo"
    action = torch.zeros(4, 3)
    predicted = torch.ones(4, 3)
    mu = torch.ones(4, 2, requires_grad=True)
    logvar = torch.full((4, 2), np.log(2.0), requires_grad=True)
    loss, parts = cvae_loss(action, predicted, action, 2 * predicted, mu, logvar, config)
    assert torch.equal(parts["reconstruction_nll"], parts["action_nll"] + parts["deviation_nll"])
    assert torch.equal(loss, parts["reconstruction_nll"] + parts["kl_divergence"])
    loss.backward()
    assert torch.allclose(mu.grad, mu.detach() / len(action))
    assert torch.all(logvar.grad > 0)


def test_legacy_weights_require_named_legacy_objective(config):
    config["intent"].pop("objective", None)
    config["intent"]["kl_weight"] = 0.05
    config["intent"]["deviation_weight"] = 0.2
    config["intent"]["deviation_objective"] = "auxiliary"
    assert objective_spec(config)["name"] == "joint_elbo"
    assert objective_spec(config)["kl_weight"] == 1.0
    config["intent"]["objective"] = "legacy_weighted_auxiliary"
    assert objective_spec(config)["kl_weight"] == 0.05
    assert objective_spec(config)["deviation_weight"] == 0.2
    assert not objective_spec(config)["standard_negative_elbo"]
    config["intent"]["objective"] = "unknown"
    with pytest.raises(ValueError, match="Unknown intent objective"):
        objective_spec(config)


def test_training_records_actual_objective_and_checkpoint_integrity(config, tmp_path):
    config["intent"].update(objective="joint_elbo", epochs=1, batch_size=4, hidden_dim=8,
                             latent_dim=2, sequence_window=3, transformer_heads=2,
                             transformer_layers=1)
    rng = np.random.default_rng(14)

    def arrays(episode):
        return {"action": rng.normal(size=(8, 3)).astype(np.float32),
                "deviation": rng.normal(size=(8, 3)).astype(np.float32),
                "context": rng.normal(size=(8, 19)).astype(np.float32),
                "label": np.arange(8, dtype=np.int64) % 3,
                "episode": np.array([episode] * 8)}

    history = train_intent(config, arrays("train"), arrays("validation"), tmp_path)
    audit = json.loads((tmp_path / "training-audit.json").read_text())
    row = next(item for item in history if item["model"] == "cvae")
    assert row["objective"] == "joint_elbo"
    assert np.isclose(row["training_loss"], row["reconstruction_nll"] + row["kl_divergence"])
    assert abs(row["objective_identity_error"]) < 1e-5
    assert audit["objective"]["standard_negative_elbo"]
    assert audit["train_episode_ids"] == ["train"]
    assert audit["architecture"]["encoder"] == [25, 8, 8]
    model, _ = load_models(config, tmp_path)
    assert model.training_metadata["objective"]["name"] == "joint_elbo"
    audit_path = tmp_path / "training-audit.json"
    audit_text = audit_path.read_text()
    audit_path.unlink()
    historical, _ = load_models(config, tmp_path)
    assert historical.training_metadata["objective"]["name"] == "unrecorded"
    audit_path.write_text(audit_text)
    config["intent"]["objective"] = "action_elbo"
    with pytest.raises(ValueError, match="Objective configuration"):
        load_models(config, tmp_path)
    config["intent"]["objective"] = "joint_elbo"
    checkpoint = tmp_path / "cvae.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="does not match training audit"):
        load_models(config, tmp_path)
