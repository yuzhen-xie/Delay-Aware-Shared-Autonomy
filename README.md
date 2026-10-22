# Delay-Aware Shared Autonomy

Python 3.12 工程：延迟意图推断、TP-HSMM/LQT、共享控制、示范精炼、实机采集与统计。支持 Windows 10/11、Linux、UR10/UR10e RTDE、Robotiq URCap 夹爪及 Force Dimension DHD 主端，也可接入 UDP 主端。

## 安装与仿真

Windows：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Hardware
.\.venv\Scripts\Activate.ps1
dasa --profile configs/smoke.yaml run --output artifacts/check
```

Linux：`bash scripts/setup.sh`，随后激活 `/workspace/.venvs/dasa/bin/activate`。完整实验使用 `configs/default.yaml`；分项配置位于 `configs/experiments/`。轨迹、模型、统计及图像保存在指定输出目录。

## 真实实验

设备地址通过 `--host` 输入；标定、相机内参和场景均使用现场测量文件。先生成实机配置：

```bash
dasa calibrate data/raw/landmarks.json --output data/raw/calibration.json
dasa perceive data/raw/rgbd.npz data/raw/perception.json --output data/raw/scene.json
dasa hardware-config --host "$ROBOT_IP" --model UR10 --calibration data/raw/calibration.json --scene data/raw/scene.json --feedback-delay 0 --output configs/ur10.local.yaml
dasa robot-read --host "$ROBOT_IP" --samples 100 --output artifacts/robot-state.json
dasa gripper --host "$ROBOT_IP"
dasa gripper --host "$ROBOT_IP" --activate --execute
```

`landmarks.json` 包含 `camera_frame`、`camera_points_m[N,3]`、`base_points_m[N,3]`。RGB-D NPZ 包含 `rgb[H,W,3]`、`depth[H,W]`；感知配置提供针孔内参、深度尺度、标定变换及目标/障碍的颜色区间与几何尺寸。目标和障碍中心统一转换到机器人基坐标系，长度单位 m、力单位 N。

主端在独立终端运行。DHD 使用厂家 SDK 动态库，`rotation.json` 提供设备到基坐标系的正交 `rotation[3,3]`。按钮 0 为使能，按钮 1 控制抓取；输入为相对中性位置的虚拟弹簧指令，反馈为限幅后的机器人力：

```bash
dasa master-device --library "$DHD_LIBRARY" --rotation data/raw/rotation.json
```

也可运行 Windows 键盘桥 `python scripts/master_keyboard.py`：WASD/QE 移动，空格使能，G 抓取，Esc 退出。UDP 主端报文包含 `sequence`、`frame=base`、`force_N[3]`、`velocity_m_s[3]`、`enable`、`grip`、`observation_sequence`；收到并显示反馈后回传帧编号。

选择一个试次执行：

```bash
dasa study-schedule --output artifacts/study/schedule.csv
dasa hardware-trial artifacts/study/schedule.csv --participant P01 --trial-id P01_r00_c00 --hardware configs/ur10.local.yaml --models artifacts/measured/models --output artifacts/hardware --execute
```

不加 `--execute` 时仅采集。单独采集可用 `robot-session`；`--skill` 可选择精炼模型。实机采用保持姿态的平移速度控制，包含限速、反馈确认、净空检查和看门狗。场景必须与所选试次的实测布局一致；代码测试不等同于设备现场验收。

首次零时延启动示范可用 `dasa robot-session --hardware configs/ur10.local.yaml --condition C1 --output data/raw/bootstrap-001.json --execute`，无需预训练模型。每个候选目标均应有成功启动示范；正式试次按计划表设置延迟。

## 数据、训练与分析

完成试次后输入独立判定结果，再导入训练数据：

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

启动示范放入 `data/processed`，正式试次放入 `data/study/recordings`，新增示范按重复和条件放入 `data/collections/r0/C3` 等目录。示范 NPZ 保存 `time`、`position`、`velocity`、`raw_force`、`observed_state`、`phase` 和 JSON `metadata`；`splits.json` 按操作者隔离 `train/validation/test`。新增成功示范与原始数据加权重训，不足配额的结果不进入固定预算比较。cVAE 目标为 `reconstruction_nll + kl_divergence`，训练分量与模型校验值一并保存。

匿名试次及 NASA-TLX CSV 格式见 `study/` 的字段文件。使用 `study-analyze` 分析真实记录，输出配对置换检验、Holm 校正、效应量和置信区间；`validation-sweep` 仅使用验证集。仿真与实测数据保留独立来源标记。

## 检查与打包

```bash
python -m pytest -q
ruff check src tests scripts
python scripts/package.py --output artifacts/project.zip
```
