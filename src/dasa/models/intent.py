"""Causal intent inference with a learned behavioral prior and angular likelihood."""

from __future__ import annotations

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
    # A vanishing reference direction carries no directional evidence.
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
    """Unit-variance Gaussian NLL (constant omitted) plus KL to N(0,I)."""
    nll = 0.5 * (action - predicted).square().sum(-1).mean()
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    return nll, kl


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
    output.mkdir(parents=True, exist_ok=True)
    model = ConditionalVAE(c["hidden_dim"], c["latent_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
    tensors = {k: torch.from_numpy(train[k]) for k in ("action", "deviation", "context", "label")}
    history = []
    for epoch in range(c["epochs"]):
        model.train()
        total, count = 0.0, 0
        for ix in torch.randperm(len(train["label"])).split(c["batch_size"]):
            action, deviation, context = [tensors[k][ix] for k in ("action", "deviation", "context")]
            predicted, devhat, mu, logvar = model(action, deviation, context)
            nll, kl = negative_elbo(action, predicted, mu, logvar)
            auxiliary = 0.5 * (devhat - deviation).square().sum(-1).mean()
            devweight = 1.0 if c["deviation_objective"] == "joint" else c["deviation_weight"]
            loss = nll + c["kl_weight"] * kl + devweight * auxiliary
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
            count += len(ix)
        history.append({"model": "cvae", "epoch": epoch, "training_loss": total / count})
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
        for ix in torch.randperm(len(train["label"])).split(c["batch_size"]):
            logits = model.classifier(torch.cat([previous[ix], tensors["context"][ix]], -1))
            loss = nn.functional.cross_entropy(logits, tensors["label"][ix])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model.classifier(torch.cat([validation_previous, validation_t["context"]], -1))
            val = float(nn.functional.cross_entropy(logits, validation_t["label"]))
        history.append({"model": "goal_prior", "epoch": epoch, "validation_nll": val})
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
            for ix in torch.randperm(len(x)).split(c["batch_size"]):
                loss = nn.functional.cross_entropy(network(x[ix]), tensors["label"][ix])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                optimizer.step()
            network.eval()
            with torch.no_grad():
                val = float(nn.functional.cross_entropy(network(xv), validation_t["label"]))
            history.append({"model": kind, "epoch": epoch, "validation_nll": val})
            if val < best:
                best, best_state = val, {k: v.clone() for k, v in network.state_dict().items()}
        network.load_state_dict(best_state)
        torch.save(network.state_dict(), output / f"{kind}.pt")
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
                # The prior uses z_(t-1); the current action only updates the next prior.
                self.previous = mu.numpy()[0]
        return softmax(prior + logits), self.previous.copy()


def load_models(config: dict, directory: Path):
    c = config["intent"]
    cvae = ConditionalVAE(c["hidden_dim"], c["latent_dim"])
    cvae.load_state_dict(torch.load(directory / "cvae.pt", weights_only=True, map_location="cpu"))
    cvae.eval()
    networks = {}
    for kind in ("lstm", "transformer"):
        model = SequenceClassifier(kind, config)
        model.load_state_dict(torch.load(directory / f"{kind}.pt", weights_only=True, map_location="cpu"))
        model.eval()
        networks[kind] = model
    return cvae, networks
