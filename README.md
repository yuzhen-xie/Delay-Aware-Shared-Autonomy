# Delay-Aware Shared Autonomy

A modular Python engineering platform for delayed teleoperation, intent estimation,
task-parameterized skill learning and shared-control experiments.

## Installation

Python 3.12 x64 is required. CPU execution is supported; no GPU or background
service is necessary. Scientific and RTDE dependencies are pinned with artifact hashes.

Linux:

```bash
bash scripts/setup.sh
source /workspace/.venvs/dasa/bin/activate
dasa --profile configs/smoke.yaml run --output artifacts/example
```

For installations outside `/workspace`, choose a writable environment path:

```bash
DASA_VENV="$PWD/.venv" DASA_CACHE="$PWD/.cache/pip" bash scripts/setup.sh
source .venv/bin/activate
```

Windows 10/11:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Hardware
powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1 -Profile smoke
```

Open the generated `report/index.html`. Each run uses a new output directory.
The complete experiment configuration is `configs/default.yaml`; `smoke.yaml`
reduces the training and evaluation budget for an installation check.

## Experiment specification

Read [实验实施与数据记录细则](docs/experiment-details.md) for the complete operational
definitions and [figure recipes](docs/figure-recipes.md) for Fig. 5, Fig. 6,
Fig. 8 and learning commands. Supporting specifications cover the
[actual training objective and architectures](docs/intent-model.md),
[demonstration selection and weighted refitting](docs/skill-refinement.md),
[RGB-D geometry and frozen scene manifests](docs/perception-scenes.md),
[anonymous study procedure and statistical analysis](docs/human-study.md),
[validation-only sensitivity experiments](docs/tuning.md) and
[default values with units](docs/parameters.md).

[Implementation validation](docs/validation.md) records the executed checks and
observed simulation failures, including incomplete demonstration budgets.

The reference cVAE optimizes `reconstruction_nll + kl_divergence` with unit
weights and records both terms in a checkpoint-bound training audit. The explicitly
named `legacy_objective.yaml` profile preserves the earlier weighted objective.
Default values and prospective protocols describe this implementation; measured
outcomes come from recorded data. Generated examples are labeled synthetic.

## Components

- Timestamped feedback and command channels, measured packet age, delay jitter and stale-packet handling.
- Conditional variational encoding, causal latent goal priors and angular action likelihoods.
- Boltzmann, memory, DMP, LSTM and causal Transformer estimation baselines.
- Task-parameterized explicit-duration models, segment EM, online duration filtering and LQT.
- Kalman force estimation, potential-field guidance, uncertainty arbitration and corrective priority.
- Discrete robot-work storage constraints and coordinated force/velocity port regulation.
- Demonstration admission, weighted skill updates, scenario generalization and paired statistical analysis.
- UR10/UR10e kinematics, RTDE state/velocity interfaces, rigid calibration, UDP master bridge and recording import.

## Commands

```bash
dasa --profile configs/smoke.yaml prepare --output artifacts/staged
dasa --profile configs/smoke.yaml intent --output artifacts/staged
dasa --profile configs/smoke.yaml sharing --output artifacts/staged
dasa --profile configs/smoke.yaml refinement --output artifacts/staged
dasa --profile configs/smoke.yaml stress --output artifacts/staged
dasa --profile configs/smoke.yaml port-bench --output artifacts/staged/ports
dasa --profile configs/smoke.yaml report --output artifacts/staged
dasa --profile configs/smoke.yaml figures --output artifacts/staged
dasa --profile configs/smoke.yaml audit-run --output artifacts/staged
dasa validate-data artifacts/staged/dataset
dasa protocol --output artifacts/specification
dasa scene-manifest --scenario wider --count 100 --output artifacts/scenes/wider.json
```

Run `dasa run --output artifacts/full` for the complete configured budget. All
failures, admission rejections and unmet demonstration quotas remain in the outputs.
Use `--profile configs/online.yaml` to enable causal duration-filter uncertainty.

Measured datasets use operator-disjoint `train`, `validation` and `test` entries
in a `splits.json` file. Train and update them with:

```bash
dasa train-data data/processed --output artifacts/measured
dasa intent-recordings artifacts/measured --output artifacts/recorded-intent
dasa refine-data data/processed data/raw/new_demos --output artifacts/updated
```

Training snapshots the declared dataset into the new run directory. Budget and
generalization experiments consume ordered collections, preserve failed attempts,
and evaluate bootstrap-only and first-success-prefix policies on shared frozen scenes:

```bash
dasa --profile configs/experiments/generalization.yaml generalization artifacts/measured data/collections --output artifacts/measured/generalization
dasa --profile configs/experiments/sensitivity.yaml validation-sweep artifacts/measured --output artifacts/sensitivity
dasa --profile configs/experiments/generalization.yaml figures --output artifacts/measured
dasa study-schedule --output artifacts/study/schedule.csv
```

Unmet successful-demonstration budgets are excluded from fixed-budget comparisons.
Statistical exports include paired-unit effects and confidence intervals, raw p
values, Holm adjustment, multiplicity families and missingness. Anonymous study
and NASA-TLX imports consume recorded CSV data; no questionnaire data are generated.

## Device integration

Copy `configs/hardware.example.yaml` to an untracked local configuration. Supply
the controller IP, explicit UR10/UR10e model, measured base-frame scene,
calibration record, device limits and gripper interface. Read-only acquisition:

```bash
dasa robot-read --host <controller-IP> --samples 100 --output artifacts/robot-state.json
```

The master SDK bridge sends ordered JSON packets to localhost UDP port 55001:

```json
{"sequence":1,"device_timestamp":0.02,"frame":"base","force_N":[0,0,0],"velocity_m_s":[0,0,0],"enable":false,"grip":false}
```

The device bridge performs axis calibration and applies its actuator limits.
`scripts/master_keyboard.py` provides Windows command-input checks without haptic output.
`robot-session` defaults to acquisition without motion; `--execute` explicitly
enables RTDE motion and the selected gripper interface. `robot-replay` evaluates
the nominal controller offline. `import-recording` requires independent task annotations.

The simulator uses a translational point-mass plant, spherical obstacles and an
abstract grasp/lift/place state machine. Its energy inequality applies to that
discrete plant. Windows velocity control is a different execution interface;
hardware cadence, collisions, force limits and port-energy behavior require device
validation. Potential-field guidance alone does not guarantee collision avoidance.
The multiport controller assumes held measured effort/flow values over each sample.

## Layout and checks

`src/dasa/models` contains learned and movement models; `control` contains
arbitration and energy regulation; `robotics` contains device interfaces. Data,
experiments, statistics and reporting have independent modules. `configs`,
`scripts`, `study` and `tests` contain execution settings, entry points, empty
data-contract templates and automated checks. Generated data and local hardware
settings are excluded from version control.

```bash
python -m pytest -q
ruff check src tests scripts
python scripts/package.py --output artifacts/project.zip
```

The source bundle includes code, configuration, tests and operational templates.
Experimental records, trained models and private notes are not included by default.
