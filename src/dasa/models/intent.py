from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from torch import nn

from dasa.data import Demonstration
from dasa.geometry import Scene, unit
from dasa.models.hsmm import TPHSMM


CONDITION_DIM = 19
ACTION_DIM = 3
GOALS = 3


def condition(state: np.ndarray, scene: Scene, robot_radius: float) -> np.ndarray:
    p, v = state[:3], state[3:]
    distances = scene.clearance(p, robot_radius)
    nearest = int(np.argmin(distances))
    return np.r_[p, v, (scene.goals - p).ravel(), scene.centers[nearest] - p,
                 distances[nearest]].astype(np.float32)


def angular_cost(action: np.ndarray, references: np.ndarray) -> np.ndarray:
    if np.linalg.norm(action) < 1e-9:
        return np.zeros(len(references))
    cosine = unit(references) @ unit(action)
    angles = np.arccos(np.clip(cosine, -1, 1))

    angles[np.linalg.norm(references, axis=1) < 1e-9] = np.pi / 2
    return angles


class ReferenceBank:
    def __init__(self, skill: TPHSMM, scene: Scene):
        self.paths = []
        self.covariances = []
        for goal in range(GOALS):
            path, cov = skill.reproduce(scene, goal)
            first_half = max(2, int(len(path) * 0.60))
            stop = int(np.argmin(np.linalg.norm(path[:first_half] - scene.goals[goal], axis=1)))
            self.paths.append(path[:max(stop + 1, 5)])
            self.covariances.append(cov)

    def actions(self, state: np.ndarray) -> np.ndarray:
        p = state[:3]
        references = []
        for path in self.paths:
            nearest = int(np.argmin(np.linalg.norm(path - p, axis=1)))
            target = path[min(nearest + 6, len(path) - 1)]
            references.append(unit(target - p))
        return np.asarray(references)


class ConditionalVAE(nn.Module):
    def __init__(self, hidden: int, latent: int):
        super().__init__()
        self.latent = latent
        self.encoder = nn.Sequential(nn.Linear(CONDITION_DIM + ACTION_DIM + GOALS, hidden),
                                     nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(CONDITION_DIM + latent, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.action_head = nn.Linear(hidden, ACTION_DIM)
        self.deviation_head = nn.Linear(hidden, GOALS)
        self.classifier = nn.Sequential(nn.Linear(CONDITION_DIM + latent, hidden), nn.SiLU(),
                                        nn.Linear(hidden, GOALS))

    def encode(self, action, deviation, context):
        h = self.encoder(torch.cat([action, deviation, context], -1))
        return self.mu(h), self.logvar(h).clamp(-8, 6)

    def forward(self, action, deviation, context):
        mu, logvar = self.encode(action, deviation, context)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        h = self.decoder(torch.cat([context, z], -1))
        return self.action_head(h), self.deviation_head(h), mu, logvar


def negative_elbo(action, predicted, mu, logvar):

    nll = 0.5 * (action - predicted).square().sum(-1).mean()
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    return nll, kl


def objective_spec(config: dict) -> dict:

    c = config["intent"]
    name = c.get("objective", "joint_elbo")
    if name == "joint_elbo":
        kl_weight, deviation_weight = 1.0, 1.0
    elif name == "action_elbo":
        kl_weight, deviation_weight = 1.0, 0.0
    elif name == "legacy_weighted_auxiliary":
        kl_weight = float(c["kl_weight"])
        deviation_weight = (1.0 if c.get("deviation_objective") == "joint"
                            else float(c["deviation_weight"]))
        if not np.isfinite(kl_weight) or not np.isfinite(deviation_weight):
            raise ValueError("Objective weights must be finite")
        if kl_weight < 0 or deviation_weight < 0:
            raise ValueError("Objective weights must be nonnegative")
    else:
        raise ValueError(f"Unknown intent objective: {name}")
    return {"name": name, "kl_weight": kl_weight, "deviation_weight": deviation_weight,
            "prior": "N(0,I)", "likelihood_variance": 1.0,
            "gaussian_additive_constant_omitted": True,
            "reduction": "sum feature coordinates; mean batch observations",
            "standard_negative_elbo": name in {"joint_elbo", "action_elbo"}}


def cvae_loss(action, predicted, deviation, predicted_deviation, mu, logvar, config: dict):

    action_nll, kl = negative_elbo(action, predicted, mu, logvar)
    deviation_nll = 0.5 * (predicted_deviation - deviation).square().sum(-1).mean()
    spec = objective_spec(config)
    reconstruction = action_nll + spec["deviation_weight"] * deviation_nll
    loss = reconstruction + spec["kl_weight"] * kl
    return loss, {"action_nll": action_nll, "deviation_nll": deviation_nll,
                  "reconstruction_nll": reconstruction, "kl_divergence": kl,
                  "weighted_kl_divergence": spec["kl_weight"] * kl}


def model_spec(config: dict) -> dict:

    c = config["intent"]
    hidden, latent = c["hidden_dim"], c["latent_dim"]
    return {"action_dimension": ACTION_DIM, "deviation_dimension": GOALS,
            "context_dimension": CONDITION_DIM, "goals": GOALS,
            "encoder": [CONDITION_DIM + ACTION_DIM + GOALS, hidden, hidden],
            "encoder_heads": {"mu": latent, "log_variance": latent},
            "log_variance_clip": [-8.0, 6.0],
            "decoder": [CONDITION_DIM + latent, hidden, hidden],
            "decoder_heads": {"action": ACTION_DIM, "deviation": GOALS},
            "goal_classifier": [CONDITION_DIM + latent, hidden, GOALS],
            "hidden_activation": "SiLU", "classifier_output": "categorical logits",
            "prior": "independent standard normal coordinates",
            "initial_inference_latent": "zero prior mean; deterministic inference uses encoder mean",
            "action_preprocessing": "unit raw force direction; zero force remains zero",
            "context_preprocessing": "SI units; no dataset normalization or age feature",
            "deviation_preprocessing": "three Euclidean distances to candidate unit references",
            "classification_latent": "previous observation encoder mean; reset to zero per episode",
            "lstm": {"input": CONDITION_DIM + ACTION_DIM, "hidden": hidden, "layers": 1},
            "transformer": {"input": CONDITION_DIM + ACTION_DIM, "hidden": hidden,
                            "heads": c["transformer_heads"], "layers": c["transformer_layers"],
                            "feedforward": 2 * hidden, "activation": "ReLU", "dropout": 0.0,
                            "causal_attention": True, "position_embedding_std": 0.02},
            "sequence_window": c["sequence_window"]}


def _arrays_digest(arrays: dict) -> dict:

    result = {}
    for key in ("action", "deviation", "context", "label", "episode"):
        value = np.ascontiguousarray(arrays[key])
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
        result[key] = {"sha256": digest.hexdigest(), "shape": list(value.shape),
                       "dtype": str(value.dtype)}
    return result


class SequenceClassifier(nn.Module):
    def __init__(self, kind: str, config: dict):
        super().__init__()
        c = config["intent"]
        self.kind = kind
        hidden = c["hidden_dim"]
        dimension = CONDITION_DIM + ACTION_DIM
        if kind == "lstm":
            self.network = nn.LSTM(dimension, hidden, batch_first=True)
        elif kind == "transformer":
            self.projection = nn.Linear(dimension, hidden)
            self.position = nn.Parameter(torch.zeros(1, c["sequence_window"], hidden))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(hidden, c["transformer_heads"], 2 * hidden,
                                               dropout=0.0, batch_first=True)
            self.network = nn.TransformerEncoder(layer, c["transformer_layers"],
                                                  enable_nested_tensor=False)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(hidden, GOALS)

    def forward(self, x):
        if self.kind == "lstm":
            h, _ = self.network(x)
        else:
            n = x.shape[1]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
            h = self.network(self.projection(x) + self.position[:, :n], mask=mask)
        return self.head(h[:, -1])


def training_arrays(demos: list[Demonstration], skill: TPHSMM, config: dict) -> dict:
    actions, contexts, deviations, labels, episodes = [], [], [], [], []
    for demo in demos:
        bank = ReferenceBank(skill, demo.scene)
        indices = np.flatnonzero(demo.phase == 0)
        for t in indices:
            action = unit(demo.raw_force[t]).astype(np.float32)
            state = demo.observed_state[t]
            refs = bank.actions(state)
            actions.append(action)
            contexts.append(condition(state, demo.scene, config["simulation"]["robot_radius"]))
            deviations.append(np.linalg.norm(action - refs, axis=1))
            labels.append(demo.goal)
            episodes.append(demo.metadata["episode_id"])
    return {"action": np.asarray(actions, dtype=np.float32),
            "context": np.asarray(contexts, dtype=np.float32),
            "deviation": np.asarray(deviations, dtype=np.float32),
            "label": np.asarray(labels), "episode": np.asarray(episodes)}


def sequence_windows(arrays: dict, window: int) -> np.ndarray:
    features = np.c_[arrays["action"], arrays["context"]]
    result = np.empty((len(features), window, features.shape[1]), dtype=np.float32)
    start = 0
    for j in range(len(features)):
        if j == 0 or arrays["episode"][j] != arrays["episode"][j - 1]:
            start = j
        indices = np.maximum(np.arange(j - window + 1, j + 1), start)
        result[j] = features[indices]
    return result


def previous_latent(mu: torch.Tensor, episode: np.ndarray) -> torch.Tensor:
    previous = torch.zeros_like(mu)
    same = torch.tensor(episode[1:] == episode[:-1])
    previous[1:] = torch.where(same[:, None], mu[:-1], 0)
    return previous


def train_intent(config: dict, train: dict, validation: dict, output: Path) -> list[dict]:
    c = config["intent"]
    spec = objective_spec(config)
    if c["epochs"] < 1 or c["batch_size"] < 1:
        raise ValueError("Training requires positive epochs and batch size")
    for split in (train, validation):
        if not len(split["label"]):
            raise ValueError("Training and validation arrays must both be nonempty")
    output.mkdir(parents=True, exist_ok=True)
    rng_state = torch.get_rng_state().numpy().tobytes()
    rng_record = {"declared_run_seed": config["seed"], "torch_initial_seed": torch.initial_seed(),
                  "torch_state_before_training_sha256": hashlib.sha256(rng_state).hexdigest()}
    model = ConditionalVAE(c["hidden_dim"], c["latent_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
    tensors = {k: torch.from_numpy(train[k]) for k in ("action", "deviation", "context", "label")}
    history = []
    for epoch in range(c["epochs"]):
        model.train()
        totals, count = {}, 0
        for ix in torch.randperm(len(train["label"])).split(c["batch_size"]):
            action, deviation, context = [tensors[k][ix] for k in ("action", "deviation", "context")]
            predicted, devhat, mu, logvar = model(action, deviation, context)
            loss, components = cvae_loss(action, predicted, deviation, devhat, mu, logvar,
                                         config)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite cVAE objective; check training observations")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in {"training_loss": loss, **components}.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(ix)
            count += len(ix)
        means = {key: total / count for key, total in totals.items()}
        identity_error = (means["training_loss"] - means["reconstruction_nll"]
                          - means["weighted_kl_divergence"])
        history.append({"model": "cvae", "epoch": epoch, "objective": spec["name"],
                        "objective_identity_error": identity_error, **means})
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(tensors["action"], tensors["deviation"], tensors["context"])
    previous = previous_latent(mu, train["episode"])
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=c["learning_rate"])
    best, best_state = np.inf, None
    validation_t = {k: torch.from_numpy(validation[k]) for k in ("action", "deviation", "context", "label")}
    with torch.no_grad():
        validation_mu, _ = model.encode(validation_t["action"], validation_t["deviation"], validation_t["context"])
        validation_previous = previous_latent(validation_mu, validation["episode"])
    for epoch in range(c["epochs"]):
        training_total, training_count = 0.0, 0
        for ix in torch.randperm(len(train["label"])).split(c["batch_size"]):
            logits = model.classifier(torch.cat([previous[ix], tensors["context"][ix]], -1))
            loss = nn.functional.cross_entropy(logits, tensors["label"][ix])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_total += float(loss.detach()) * len(ix)
            training_count += len(ix)
        with torch.no_grad():
            logits = model.classifier(torch.cat([validation_previous, validation_t["context"]], -1))
            val = float(nn.functional.cross_entropy(logits, validation_t["label"]))
        history.append({"model": "goal_prior", "epoch": epoch,
                        "training_nll": training_total / training_count, "validation_nll": val})
        if val < best:
            best = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output / "cvae.pt")
    for kind in ("lstm", "transformer"):
        network = SequenceClassifier(kind, config)
        optimizer = torch.optim.Adam(network.parameters(), lr=c["learning_rate"])
        x = torch.from_numpy(sequence_windows(train, c["sequence_window"]))
        xv = torch.from_numpy(sequence_windows(validation, c["sequence_window"]))
        best, best_state = np.inf, None
        for epoch in range(c["epochs"]):
            network.train()
            training_total, training_count = 0.0, 0
            for ix in torch.randperm(len(x)).split(c["batch_size"]):
                loss = nn.functional.cross_entropy(network(x[ix]), tensors["label"][ix])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                optimizer.step()
                training_total += float(loss.detach()) * len(ix)
                training_count += len(ix)
            network.eval()
            with torch.no_grad():
                val = float(nn.functional.cross_entropy(network(xv), validation_t["label"]))
            history.append({"model": kind, "epoch": epoch,
                            "training_nll": training_total / training_count,
                            "validation_nll": val})
            if val < best:
                best, best_state = val, {k: v.clone() for k, v in network.state_dict().items()}
        network.load_state_dict(best_state)
        torch.save(network.state_dict(), output / f"{kind}.pt")
    metadata = {
        "schema_version": 1, "objective": spec, "architecture": model_spec(config),
        "torch_version": str(torch.__version__),
        "randomness": rng_record, "intent_config": c,
        "optimizer": {"name": "Adam", "learning_rate": c["learning_rate"],
                      "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.0,
                      "cvae_gradient_norm_clip": 5.0, "sequence_gradient_norm_clip": 5.0,
                      "classifier_gradient_norm_clip": None},
        "epochs": c["epochs"], "batch_size": c["batch_size"],
        "selection": {"cvae": "final training epoch", "goal_classifier": "validation NLL",
                      "lstm": "validation NLL", "transformer": "validation NLL"},
        "train_arrays": _arrays_digest(train), "validation_arrays": _arrays_digest(validation),
        "train_episode_ids": sorted(set(train["episode"].tolist())),
        "validation_episode_ids": sorted(set(validation["episode"].tolist())),
        "checkpoint_sha256": {name: hashlib.sha256((output / f"{name}.pt").read_bytes()).hexdigest()
                              for name in ("cvae", "lstm", "transformer")},
        "history": history}
    (output / "training-audit.json").write_text(json.dumps(metadata, indent=2, allow_nan=False),
                                               encoding="utf-8")
    return history


class IntentEstimator:
    def __init__(self, config: dict, bank: ReferenceBank, model: ConditionalVAE | None,
                 kind: str = "full", network: SequenceClassifier | None = None):
        self.config, self.bank, self.model = config, bank, model
        self.kind, self.network = kind, network
        self.previous = np.zeros(config["intent"]["latent_dim"], dtype=np.float32)
        self.log_memory = np.zeros(GOALS)
        self.history = []

    def step(self, action: np.ndarray, observed: np.ndarray, current: np.ndarray,
             scene: Scene, age: float) -> tuple[np.ndarray, np.ndarray]:
        state = current if self.kind in {"br", "memory", "no_delay"} else observed
        age = 0.0 if self.kind in {"br", "memory", "no_delay"} else age
        refs = self.bank.actions(state)
        context = condition(state, scene, self.config["simulation"]["robot_radius"])
        a = unit(action).astype(np.float32)
        logits = -self.config["intent"]["beta"] * angular_cost(a, refs)
        prior = np.zeros(GOALS)
        if self.kind == "memory":
            self.log_memory = self.config["intent"]["memory_decay"] * self.log_memory + logits
            self.log_memory -= self.log_memory.max()
            return softmax(self.log_memory), self.previous.copy()
        if self.kind in {"lstm", "transformer"}:
            self.history.append(np.r_[a, context])
            window = self.config["intent"]["sequence_window"]
            self.history = self.history[-window:]
            x = [self.history[0]] * (window - len(self.history)) + self.history
            with torch.no_grad():
                prediction = self.network(torch.tensor(np.array(x)[None], dtype=torch.float32))
            return softmax(prediction.numpy()[0]), self.previous.copy()
        if self.kind in {"full", "no_delay"}:
            if self.model is None:
                raise ValueError("Latent inference requires a trained cVAE")
            with torch.no_grad():
                prior = self.model.classifier(torch.from_numpy(
                    np.r_[self.previous, context].astype(np.float32))[None]).numpy()[0]
                deviation = np.linalg.norm(a - refs, axis=1).astype(np.float32)
                mu, _ = self.model.encode(torch.from_numpy(a)[None],
                                          torch.from_numpy(deviation)[None],
                                          torch.from_numpy(context)[None])

                self.previous = mu.numpy()[0]
        return softmax(prior + logits), self.previous.copy()


def load_models(config: dict, directory: Path):
    c = config["intent"]
    audit_path = directory / "training-audit.json"
    metadata = {"schema_version": 0, "objective": {"name": "unrecorded"}}
    if audit_path.exists():
        metadata = json.loads(audit_path.read_text(encoding="utf-8"))
        for name in ("cvae", "lstm", "transformer"):
            actual = hashlib.sha256((directory / f"{name}.pt").read_bytes()).hexdigest()
            if actual != metadata["checkpoint_sha256"][name]:
                raise ValueError(f"Checkpoint does not match training audit: {name}")
        if metadata["architecture"] != model_spec(config):
            raise ValueError("Network configuration does not match the training audit")
        if metadata["objective"] != objective_spec(config):
            raise ValueError("Objective configuration does not match the training audit")
    cvae = ConditionalVAE(c["hidden_dim"], c["latent_dim"])
    cvae.load_state_dict(torch.load(directory / "cvae.pt", weights_only=True, map_location="cpu"))
    cvae.eval()
    cvae.training_metadata = metadata
    networks = {}
    for kind in ("lstm", "transformer"):
        model = SequenceClassifier(kind, config)
        model.load_state_dict(torch.load(directory / f"{kind}.pt", weights_only=True, map_location="cpu"))
        model.eval()
        networks[kind] = model
    return cvae, networks
