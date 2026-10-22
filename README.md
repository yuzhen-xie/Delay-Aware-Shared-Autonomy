# Delay-Aware Shared Autonomy

Python 3.12 software for delayed intent inference, TP-HSMM/LQT, shared control, skill refinement, hardware recording and statistical analysis. Supports Windows 10/11, Linux, UR10/UR10e RTDE, Robotiq URCap grippers, Force Dimension DHD devices and UDP master interfaces.

## Installation and simulation

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Hardware
.\.venv\Scripts\Activate.ps1
dasa --profile configs/smoke.yaml run --output artifacts/check
```

On Linux, run `bash scripts/setup.sh`, then `source /workspace/.venvs/dasa/bin/activate`. Use `configs/default.yaml` for the full experimental budget or profiles under `configs/experiments/` for individual experiments. Trajectories, models, statistics and figures are saved in the selected output directory.

## Hardware experiments

Supply the actual controller address through `--host`. Calibration, camera intrinsics and scene geometry must come from measured inputs. Generate the hardware configuration first:

```bash
dasa calibrate data/raw/landmarks.json --output data/raw/calibration.json
dasa perceive data/raw/rgbd.npz data/raw/perception.json --output data/raw/scene.json
dasa hardware-config --host "$ROBOT_IP" --model UR10 --calibration data/raw/calibration.json --scene data/raw/scene.json --feedback-delay 0 --output configs/ur10.local.yaml
dasa robot-read --host "$ROBOT_IP" --samples 100 --output artifacts/robot-state.json
dasa gripper --host "$ROBOT_IP"
dasa gripper --host "$ROBOT_IP" --activate --execute
```

`landmarks.json` contains `camera_frame`, `camera_points_m[N,3]` and `base_points_m[N,3]`. RGB-D NPZ files contain `rgb[H,W,3]` and `depth[H,W]`. The perception configuration supplies pinhole intrinsics, depth scale, calibration transform, color ranges and object geometry. Goal and obstacle centers use the robot base frame; lengths are in meters and forces in newtons.

Run the master interface in a separate terminal. DHD requires the vendor SDK shared library. `rotation.json` supplies an orthogonal `rotation[3,3]` from the device frame to the base frame. Button 0 enables control; button 1 controls grasping. Displacement from the neutral position generates a virtual spring command, while feedback uses bounded robot force:

```bash
dasa master-device --library "$DHD_LIBRARY" --rotation data/raw/rotation.json
```

The Windows keyboard bridge is `python scripts/master_keyboard.py`: WASD/QE move, Space enables control, G toggles grasping and Esc exits. UDP master packets contain `sequence`, `frame=base`, `force_N[3]`, `velocity_m_s[3]`, `enable`, `grip` and `observation_sequence`. Acknowledge a feedback frame only after receiving and displaying it.

Select one scheduled trial:

```bash
dasa study-schedule --output artifacts/study/schedule.csv
dasa hardware-trial artifacts/study/schedule.csv --participant P01 --trial-id P01_r00_c00 --hardware configs/ur10.local.yaml --models artifacts/measured/models --output artifacts/hardware --execute
```

Without `--execute`, the workstation only records measurements. Use `robot-session` for standalone acquisition and `--skill` to select a refined model. Hardware control uses translational velocity commands with held tool orientation, speed limits, feedback acknowledgement, clearance checks and watchdogs. The measured scene must match the selected trial. Software tests do not establish physical hardware validation.

Collect initial zero-delay demonstrations with `dasa robot-session --hardware configs/ur10.local.yaml --condition C1 --output data/raw/bootstrap-001.json --execute`; no pretrained model is required. Include successful bootstrap demonstrations for every candidate goal. Scheduled trials use their assigned delays.

## Data, training and analysis

Enter independently assessed outcomes after acquisition, then import the recordings:

```bash
dasa annotate-recording recording.json --episode P01_r00_c00 --operator P01 --goal 0 --grasped-goal 0 --outcome success --output annotation.json
dasa import-recording recording.json annotation.json --output data/study/recordings/P01_r00_c00.npz
dasa measured-metrics data/study/recordings --output artifacts/study/trials.csv
dasa validate-data data/processed
dasa train-data data/processed --output artifacts/measured
dasa intent-recordings artifacts/measured --output artifacts/intent
dasa refine-data data/processed data/collections/r0/C3 --output artifacts/refined
dasa --profile configs/experiments/generalization.yaml generalization artifacts/measured data/collections --output artifacts/measured/generalization
dasa figures --output artifacts/measured
```

Store bootstrap data in `data/processed`, study recordings in `data/study/recordings`, and new demonstrations by repeat and condition under paths such as `data/collections/r0/C3`. Demonstration NPZ files contain `time`, `position`, `velocity`, `raw_force`, `observed_state`, `phase` and JSON `metadata`. `splits.json` defines operator-disjoint train, validation and test partitions. Refinement refits weighted original and accepted new data; incomplete quotas are excluded from fixed-budget comparisons. The cVAE objective is `reconstruction_nll + kl_divergence`, with component histories and checkpoint hashes saved together.

CSV field templates for anonymous trials and NASA-TLX are under `study/`. `study-analyze` processes recorded observations and exports paired permutation tests, Holm corrections, effect sizes and confidence intervals. `validation-sweep` uses validation data only. Simulation and measured records retain separate provenance labels.

## Checks and packaging

```bash
python -m pytest -q
ruff check src tests scripts
python scripts/package.py --output artifacts/project.zip
```
