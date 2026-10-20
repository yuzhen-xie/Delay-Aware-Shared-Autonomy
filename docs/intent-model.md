# Intent models and objective audit

This document describes the implemented models. The parameter values below are
the repository reference configuration; they are not recovered measurements of
an earlier training session. A checkpoint without its training audit has an
unrecorded objective. Retraining creates a new artifact and cannot establish the
objective used to obtain an older artifact.

## Inputs, preprocessing and causality

The candidate-goal set has exactly three elements. The robot observation is
`[p_x,p_y,p_z,v_x,v_y,v_z]` in the robot base frame (metres and metres/second).
The context vector has 19 real coordinates:

| Feature | Coordinates | Units |
| --- | ---: | --- |
| End-effector position | 3 | m |
| End-effector velocity | 3 | m/s |
| Goal displacement, in configured goal order | 3 × 3 | m |
| Displacement to the obstacle with smallest surface clearance | 3 | m |
| That obstacle's surface clearance, including end-effector radius | 1 | m |

The action is the three-coordinate unit direction of the raw master force.
Force norms at or below `1e-12` map to zero. Force amplitude is retained by the
motion controller, but is not an intent-model input. For each of the three
candidate goals, a TP-HSMM/LQT pickup reference supplies a unit action. The
encoder receives all three Euclidean distances between the human direction and
these reference directions. Selecting a deviation using the true goal at
inference is not permitted. There is no dataset standardization, whitening or
separate observation-age feature in this implementation.

Only demonstration samples in pickup phase `0` are used for these models.
Episode identifiers are preserved when making causal windows and lagged latent
features. Data partitioning occurs at the episode/operator level before feature
extraction. The held-out test partition is not used for checkpoint selection.

The full estimator computes both the likelihood and context from the state
actually displayed to the operator, not the current robot state. It computes
the goal prior from the preceding encoder mean, then encodes the current action
to supply the next prior. This order prevents the current action from entering
both the prior and likelihood through the same latent update. The first latent
is the zero prior mean. Inference uses posterior means rather than random
latent samples. Observation age is logged and used to select the delayed state;
it does not directly enter the network.

## Architecture

`intent.hidden_dim=64` and `intent.latent_dim=8` give the following dimensions.
All outputs are real-valued except the final goal probability vector.

| Component | Layers and output |
| --- | --- |
| Encoder input | action `3` + candidate deviations `3` + context `19` = `25` |
| Encoder trunk | Linear `25→64`, SiLU, Linear `64→64`, SiLU |
| Posterior heads | Separate Linear `64→8` for mean and log variance |
| Log variance | Clamped to `[-8,6]` before reparameterization and KL evaluation |
| Training latent | `z=mu+exp(0.5*logvar)*eta`, `eta~N(0,I_8)` |
| Decoder trunk | Linear `(19+8)→64`, SiLU, Linear `64→64`, SiLU |
| Decoder heads | Linear `64→3` action and Linear `64→3` candidate deviations |
| Goal classifier `f_cls` | Linear `(8+19)→64`, SiLU, Linear `64→3` |
| Goal probabilities | Softmax of classifier logits |

The latent prior is exactly `p(z)=N(0,I_8)`. The approximate posterior is diagonal
Gaussian. It has no learned temporal transition or goal-conditioned prior. The
lagged classifier supplies temporal information; interpreting latent coordinates
as measured operator expertise would require a separate validation study.

## Actual optimized objective

The reference mode is `intent.objective: joint_elbo`. For each minibatch:

```text
action_nll       = mean_batch(0.5 * sum_coordinates((a - a_hat)^2))
deviation_nll    = mean_batch(0.5 * sum_coordinates((e - e_hat)^2))
reconstruction_nll = action_nll + deviation_nll
kl_divergence    = mean_batch(0.5 * sum_latent(mu^2 + exp(logvar) - 1 - logvar))
loss             = reconstruction_nll + kl_divergence
```

Both Gaussian decoder variances are fixed to one in the preprocessed feature
coordinates. Additive Gaussian normalization constants are omitted consistently.
The same minibatch mean is used for reconstruction and KL; there is no hidden
division by latent width, KL annealing, free-bits term or additional KL multiplier.
Because deviations are deterministic features of actions and references, this
is a Gaussian feature-reconstruction model, not six independent physical
measurements. Reconstructing deviations does not provide additional independent
demonstrations.

`action_elbo` is a separately named ablation with reconstruction of the action
alone and unit KL weight. `legacy_weighted_auxiliary` is an explicitly named
compatibility mode: its optimized value is action NLL plus configured deviation
weight times deviation NLL plus configured KL weight times KL. Historical
weights `0.2` and `0.05` therefore do **not** optimize the unweighted joint ELBO.
The `joint` legacy deviation switch sets its deviation weight to one. Do not
relabel a legacy checkpoint after changing configuration files. The likelihood
rationality coefficient `intent.beta=3` belongs to the angular action likelihood;
it is unrelated to the KL multiplier.

The reconstruction model trains for 35 epochs with Adam, learning rate `0.001`,
betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0`, minibatches of 256 and
global gradient norm clipping at `5`. The final reconstruction epoch is used.
The encoder is then fixed and the goal classifier trains separately for 35
epochs with cross entropy and Adam at the same learning rate, without gradient
clipping. Its lowest validation cross-entropy checkpoint is selected. These are
reference engineering choices, not a completed hyperparameter-search result.
Any tuning study must use training/validation data only, freeze its selection
rule before test evaluation, and archive every candidate and selection score.

## Baselines and ablations

The LSTM baseline receives a left-padded window of 16 action-plus-context
vectors (`22` coordinates). Padding repeats the first available sample of the
same episode. It has one 64-unit recurrent layer and a Linear `64→3` output.
The Transformer uses Linear `22→64`, learned position embeddings initialized
with standard deviation `0.02`, two causal encoder layers, four heads,
128-unit feedforward layers, ReLU, dropout `0`, and a Linear `64→3` output from
the last token. Adam, minibatch size, epochs and gradient clipping match cVAE
training; each baseline is selected by validation cross entropy. Neither sees
future samples or crosses episode boundaries.

The full categorical posterior is proportional to the lagged learned prior
times `exp(-beta*angle(action,reference_goal))`. Zero human force contributes
uniform directional evidence. The `no_latent` ablation uses a uniform goal
prior. `no_delay` uses the current robot state in place of the delayed state.
The `br` and exponentially decayed `memory` baselines also use current state;
their delays and information sets must be reported explicitly when comparing
them. A zero-length reference direction carries angle `pi/2` for nonzero action.
The separate DMP baseline fits 25 Gaussian basis functions on 120 resampled
pickup points per demonstration. Its canonical decay rate is `4`, spring gain
`25`, damping parameter `25/4=6.25`, ridge coefficient `1e-4`, basis centres
`exp(-4*linspace(0,1,25))`, and basis widths `25^2`. Per-goal forcing targets
are averaged across bootstrap demonstrations before ridge regression. Movement
scale coordinates with magnitude below `0.01 m` are replaced by `0.01 m`.
Rollouts use integration step `0.002 s` for `1.3 s` and a reference lookahead of
20 rollout samples. DMP intent uses the delayed observation and the same angular
likelihood, with no learned latent prior. These constants are implementation
defaults; saved DMP files contain weights only, so the source version must
accompany them.

## Training artifacts and verification

Each new training run writes `models/training-audit.json` with the resolved
objective, architecture, actual optimizer settings, preprocessing description,
ordered training/validation array hashes, episode identifiers, random-state
fingerprint, checkpoint hashes and per-epoch histories. Every cVAE epoch records
action NLL, deviation NLL, reconstruction NLL, KL, weighted KL, optimized loss
and the numerical loss-identity residual. Classifier/baseline histories contain
training and validation cross entropy. `models/training.json` remains the
pipeline's compact history output.

The loader checks checkpoint SHA-256, architecture and effective objective
against the audit before evaluation. Existing plain state dictionaries remain
loadable when an audit is absent, but their attached metadata states
`objective.name="unrecorded"`. Absence of an audit must be disclosed when using
these artifacts; current defaults cannot determine their historical objective.

Run `pytest tests/test_intent.py`. The tests check the exact unweighted objective,
the nonzero KL gradients, named legacy behavior, a real short optimizer run,
component histories, configuration mismatch detection and checksum rejection.
The main experiment manifest supplies the code commit and complete configuration.
