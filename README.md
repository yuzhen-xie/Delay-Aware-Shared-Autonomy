# Delay-Aware Shared Autonomy

Python 3.12 software for physical UR10/UR10e human–robot experiments with a Force Dimension master, Robotiq gripper and calibrated RealSense RGB-D camera. Experimental outputs are generated from recorded measurements and independently assessed task outcomes.

## Installation

On Windows 10/11, run `powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1`, then `.\.venv\Scripts\Activate.ps1`. On Linux x86-64, run `bash scripts/setup.sh`, then `source .venv/bin/activate`. Setup installs dependencies without acquiring data or moving devices. Install the Force Dimension vendor SDK and camera drivers on the workstation.

## Calibrated platform

Supply the actual robot address through `--host`. Required measured input files are:

- `landmarks.json`: `camera_frame`, paired `camera_points_m[N,3]` and `base_points_m[N,3]`.
- `tool.json`: `tcp_pose[6]`, `payload_mass_kg` and `payload_cog_m[3]`, matching the mounted tool.
- `perception.json`: ordered object IDs, HSV segmentation ranges, physical geometry, task start/place coordinates and the calibrated camera transform. Live acquisition reads intrinsics and depth scale from the camera.
- `rotation.json`: orthogonal `rotation[3,3]` mapping master coordinates to the robot base frame.

```bash
dasa calibrate data/raw/landmarks.json --output data/raw/calibration.json
dasa camera-capture --calibration data/raw/calibration.json --perception data/raw/perception.json --output data/raw/capture --samples 1
dasa hardware-config --host "$ROBOT_IP" --model UR10 --calibration data/raw/calibration.json --scene data/raw/capture/scene.json --perception data/raw/perception.json --tool data/raw/tool.json --feedback-delay 0 --output configs/ur10.local.yaml
dasa gripper --host "$ROBOT_IP" --activate --execute
dasa gripper --host "$ROBOT_IP" --open --execute
```

Hardware configuration binds the calibration, perception, tool and scene records by hashes. Prescribed layouts are checked against the measured scene before acquisition. All positions use the robot base frame, lengths are metres, forces are newtons and energy is joules. Obstacles are conservative spherical envelopes around measured centers. The default risk-blending input is the nearest TCP-to-obstacle **center distance**. Collision checks and safety metrics use **surface clearance**, subtracting obstacle and TCP-envelope radii. `configs/surface_risk.yaml` explicitly selects the alternative surface-distance blending variant. TCP-envelope checks operate alongside the UR controller's native protective functions; they do not certify full-arm collision clearance.

## Acquisition

Start the master bridge in a separate terminal:

```bash
dasa master-device --library "$DHD_LIBRARY" --rotation data/raw/rotation.json --watchdog-unit-us "$DHD_WATCHDOG_UNIT_US" --frequency 1000 --video-port 55002 --output artifacts/master/session-001
```

Set `DHD_WATCHDOG_UNIT_US` to the watchdog tick duration, in microseconds, specified by the installed device SDK; the bridge verifies the configured tick count by SDK readback. Button 0 enables motion and button 1 requests grasp/release. Master displacement generates a virtual-spring command; its construction is distinct from a measured human force. Camera frames are displayed with paired robot-state sequence IDs. Acknowledgement occurs after presentation. 

Collect initial zero-delay demonstrations with manual control:

```bash
dasa robot-session --hardware configs/ur10.local.yaml --condition C1 --output data/raw/bootstrap-001.json --execute
dasa annotate-recording data/raw/bootstrap-001.json --episode bootstrap-001 --operator B01 --goal 0 --grasped-goal 0 --outcome success --output data/raw/bootstrap-001.annotation.json
dasa import-recording data/raw/bootstrap-001.json data/raw/bootstrap-001.annotation.json --output data/train/bootstrap-001.npz
```

Each execution produces an immutable recording. Independently annotate success, wrong-object grasp, collision, bounds violation, timeout or abortion; unsuccessful trials remain task outcomes. Collect 50 accepted zero-delay training demonstrations, plus separate validation and test recordings from disjoint operators. Episodes, operators and signal hashes cannot overlap partitions. Initial-training IDs such as `B01` must differ from held-out study IDs such as `P01`. Intent evaluation uses separately acquired trials at every feedback delay being evaluated. Supply additional annotated, imported held-out episodes through `--recordings`; their operators and signals must remain disjoint from frozen training and validation data. Without this option, evaluation uses the prepared dataset's test partition. Offline analysis reuses logged observations and never creates additional delay conditions. For representation analysis, `representation-schedule` assigns two trials per candidate goal and participant at 400 ms; explicit experience labels come from the participant registry.

```bash
dasa dataset-create --train data/train --validation data/validation --test data/test --output data/processed
dasa train-data data/processed --output artifacts/measured
dasa protocol --output artifacts/protocol
dasa study-schedule --output artifacts/study/schedule.csv
dasa representation-schedule study/local/participants.csv --output artifacts/study/representation.csv
dasa hardware-trial artifacts/study/schedule.csv --participant P01 --trial-id P01_r00_c00 --hardware configs/ur10.local.yaml --models artifacts/measured/models --output artifacts/hardware --execute
```

A scheduled invocation executes one selected trial. C1 is direct teleoperation, C2 uses autonomy confidence, C3 uses relative human/autonomy confidence, and C4 uses fixed authority 0.5. The default prospective plan contains 20 participants, eight zero-delay practice trials and one trial per participant × condition × delay. Increase and freeze `study.trial_repeats` before collection when repeated-trial success rates are required. A single trial contributes a binary outcome.

The master requests 1 kHz; UR10 uses 125 Hz and UR10e uses 500 Hz. Actual periods and held-command intervals are measured. PC1 regulates master feedback force, PC2 regulates commanded robot wrench, and PC3 modifies the velocity supplied to autonomous damping. Energy packets use cumulative counters and deduplication; measured timing/budget violations stop execution. Robot-port energy accounting concerns commanded Cartesian wrench and measured TCP displacement. External TCP wrench is recorded separately. 

## Models and refinement

The cVAE has 25 encoder inputs: action direction 3, three candidate-goal deviations, and context 19 (TCP position/velocity, three goal displacements, nearest-obstacle displacement and surface clearance). Two 64-unit SiLU layers produce an 8-dimensional Gaussian latent. The decoder reconstructs action and deviation coordinates; the prior is `N(0,I)`. Training uses `reconstruction_nll + kl_divergence`. Inference samples the preceding latent with a recorded seed; the context-only ablation has its own trained classifier. Training components, input definitions and checkpoint hashes are exported.

TP-HSMM uses seven ordered states, explicit durations and start/goal/place task frames. Causal state filtering drives the reference and covariance used for confidence. Refinement retains the original corpus and refits it together with the first 30 unique successful demonstrations per condition in acquisition order, with equal per-episode weights. The intent model remains frozen. Collection audits retain rejected and failed attempts.

```bash
dasa refinement-schedule artifacts/measured study/local/refinement-operators.csv --output artifacts/refinement-collection
dasa refine-data data/processed data/collections/r0/C3 --output artifacts/refined/r0/C3
dasa --profile configs/experiments/generalization.yaml generalization artifacts/measured data/collections --output artifacts/learning
dasa generalization-analyze artifacts/learning data/autonomous-recordings --output artifacts/learning-analysis
dasa figures --output artifacts/learning-analysis
dasa learning-trajectories data/autonomous-recordings --scene-id "$SCENE_ID" --output artifacts/learning-analysis/trajectories
```

`refinement-operators.csv` lists actual anonymous `operator_id` values outside the held-out partitions. Execute one assigned row from the resulting `schedule.csv`; use assigned extension attempts only until each condition reaches 30 accepted successes. Learning schedules require ten independently collected refinement sets and 100 physical autonomous trials per fitted policy and scenario. Baseline and novel layouts are saved as explicit scene manifests. Novel goals move 5–15 cm in Euclidean horizontal distance; layouts are constrained by physical workspace and geometric non-overlap. Human demonstration collection and supervised autonomous-policy evaluation have separate assignment stages.

## Analysis and outputs

Freeze the operational metric definitions, questionnaire scale and analysis plan before acquisition with `dasa protocol`. The exported `study_protocol.yaml` records the actual freeze time and configuration hash. `configs/default.yaml` exposes all values; `study/protocol.template.yaml` describes the prospective structure. 

```bash
dasa measured-metrics data/study/recordings --output artifacts/study/trials.csv
dasa study-analyze artifacts/study/trials.csv study/local/participants.csv artifacts/protocol/study_protocol.yaml --tlx study/local/tlx.csv --output artifacts/measured
dasa intent-recordings artifacts/measured --recordings data/intent-heldout --output artifacts/measured
dasa representation-analyze artifacts/measured study/local/participants.csv --output artifacts/measured/representation
dasa recording-figures artifacts/hardware/acquisition/P01_r00_c00/execute/recording.json --output artifacts/measured/figures/trace
dasa figures --output artifacts/measured
dasa report --output artifacts/measured
dasa protocol --output artifacts/protocol
```

Safety is the time-weighted mean of `exp(-gamma_sa * speed² / clearance)`, with zero at nonpositive clearance. Stability is the time-weighted mean of `exp(-abs(gamma_st * d²speed/dt²))`. Speed is measured translational TCP speed; resampling, Savitzky–Golay filtering and both coefficients are explicit configured values. Clearance and jerk remain secondary outcomes. IPA uses confidence 0.6; EPT requires a correct confident prediction sustained for 0.20 s during pickup. Completion-time comparisons condition on success and are accompanied by failure rates and timeout-penalized duration.

Questionnaire inputs retain the declared 1–7 scale and anchors, with explicit burden direction and affine 0–100 exports. Participant-level paired sign-flip tests report raw/adjusted p values, Holm families, effect sizes, pointwise bootstrap intervals and missing pairs. Representation analysis records its feature space, t-SNE settings, goal-controlled operator diagnostic and participant-level aggregation. Significance markers are drawn only from matching numerical contrasts.

Raw RGB-D frames and master logs remain beside acquisitions; imported signals are NPZ files. Each analysis writes CSV/JSON summaries, provenance and input hashes. Figures are PNG/PDF/SVG under the selected output directory. `report/index.html` links the numerical tables. Create a source-only archive with `python scripts/package.py --output artifacts/project.zip`.
