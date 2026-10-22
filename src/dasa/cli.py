import argparse
import json
from pathlib import Path

from dasa.config import load_config, seed_everything


def parser():
    p = argparse.ArgumentParser(prog="dasa", description="Measured human–robot teleoperation experiments")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--profile")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("figures", "report", "protocol", "audit-run"):
        s = sub.add_parser(command)
        s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("dataset-create")
    for name in ("train", "validation", "test"):
        s.add_argument("--" + name, type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("validate-data")
    s.add_argument("dataset", type=Path)
    for command in ("train-data", "intent-recordings"):
        s = sub.add_parser(command)
        s.add_argument("dataset", type=Path)
        s.add_argument("--output", type=Path, required=True)
        if command == "intent-recordings":
            s.add_argument("--recordings", type=Path)
            s.add_argument("--participants", type=Path)
    s = sub.add_parser("refine-data")
    s.add_argument("dataset", type=Path)
    s.add_argument("candidates", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("generalization")
    s.add_argument("prepared", type=Path)
    s.add_argument("candidates", type=Path)
    s.add_argument("--scenes", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("generalization-analyze")
    s.add_argument("experiment", type=Path)
    s.add_argument("recordings", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("refinement-schedule")
    s.add_argument("prepared", type=Path)
    s.add_argument("operators", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("representation-analyze")
    s.add_argument("recorded", type=Path)
    s.add_argument("participants", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("representation-schedule")
    s.add_argument("participants", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("study-schedule")
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("study-analyze")
    for name in ("trials", "participants", "protocol"):
        s.add_argument(name, type=Path)
    s.add_argument("--tlx", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("scene-manifest")
    s.add_argument("--count", type=int, default=100)
    s.add_argument("--scenario", choices=["demonstrated", "novel", "wider", "extra_obstacles"], default="demonstrated")
    s.add_argument("--seed", type=int)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("calibrate")
    s.add_argument("landmarks", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("perceive")
    s.add_argument("rgbd", type=Path)
    s.add_argument("specification", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("camera-capture")
    s.add_argument("--calibration", type=Path, required=True)
    s.add_argument("--perception", type=Path, required=True)
    s.add_argument("--serial")
    s.add_argument("--samples", type=int, default=1)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("hardware-config")
    s.add_argument("--host", required=True)
    for name in ("calibration", "scene", "perception", "tool"):
        s.add_argument("--" + name, type=Path, required=True)
    s.add_argument("--model", choices=["UR10", "UR10e"], default="UR10")
    s.add_argument("--hardware-profile", type=Path)
    s.add_argument("--feedback-delay", type=float)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("robot-read")
    s.add_argument("--host", required=True)
    s.add_argument("--samples", type=int, default=100)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("robot-session")
    s.add_argument("--hardware", type=Path, required=True)
    s.add_argument("--models", type=Path)
    s.add_argument("--skill", type=Path)
    s.add_argument("--condition", choices=["C1", "C2", "C3", "C4", "autonomous"], default="C3")
    s.add_argument("--goal", type=int, choices=range(3))
    s.add_argument("--stage", choices=["human_bootstrap", "human_validation", "human_intent_evaluation", "human_shared_control", "intent_representation"])
    s.add_argument("--execute", action="store_true")
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("hardware-trial")
    s.add_argument("schedule", type=Path)
    s.add_argument("--hardware", type=Path, required=True)
    s.add_argument("--models", type=Path)
    s.add_argument("--skill", type=Path)
    s.add_argument("--participant")
    s.add_argument("--trial-id", required=True)
    s.add_argument("--execute", action="store_true")
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("master-device")
    s.add_argument("--library", type=Path, required=True)
    s.add_argument("--rotation", type=Path, required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=55001)
    s.add_argument("--video-port", type=int, default=55002)
    s.add_argument("--device-index", type=int, default=0)
    s.add_argument("--frequency", type=float, default=1000.)
    s.add_argument("--stiffness", type=float, default=100.)
    s.add_argument("--max-command", type=float, default=12.)
    s.add_argument("--max-feedback", type=float, default=2.)
    s.add_argument("--deadman-button", type=int, default=0)
    s.add_argument("--grip-button", type=int, default=1)
    s.add_argument("--feedback-timeout", type=float, default=.1)
    s.add_argument("--watchdog-unit-us", type=float, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("gripper")
    s.add_argument("--host", required=True)
    s.add_argument("--port", type=int, default=63352)
    actions = s.add_mutually_exclusive_group()
    actions.add_argument("--activate", action="store_true")
    actions.add_argument("--open", action="store_true")
    actions.add_argument("--close", action="store_true")
    s.add_argument("--execute", action="store_true")
    s = sub.add_parser("annotate-recording")
    s.add_argument("recording", type=Path)
    s.add_argument("--episode", required=True)
    s.add_argument("--operator", required=True)
    s.add_argument("--goal", type=int, choices=range(3), required=True)
    s.add_argument("--grasped-goal", type=int, choices=range(3))
    s.add_argument("--outcome", choices=["success", "collision", "bounds", "wrong_object", "timeout", "aborted"], required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("import-recording")
    s.add_argument("recording", type=Path)
    s.add_argument("annotations", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("measured-metrics")
    s.add_argument("recordings", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("recording-figures")
    s.add_argument("recording", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("learning-trajectories")
    s.add_argument("recordings", type=Path)
    s.add_argument("--scene-id", required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("validation-plan")
    s.add_argument("prepared", type=Path)
    s.add_argument("--parameters", nargs="+")
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("validation-select")
    s.add_argument("plan", type=Path)
    s.add_argument("trials", type=Path)
    s.add_argument("--output", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    c = load_config(args.config, args.profile)
    seed_everything(c["seed"])
    name = args.command
    if name == "dataset-create":
        from dasa.data import assemble_dataset
        assemble_dataset(c, args.train, args.validation, args.test, args.output)
    elif name == "validate-data":
        from dasa.data import validate_partitions
        validate_partitions(args.dataset, json.loads((args.dataset / "splits.json").read_text()))
    elif name == "train-data":
        from dasa.experiments import train_dataset
        train_dataset(c, args.dataset, args.output)
    elif name == "refine-data":
        from dasa.experiments import refine_dataset
        refine_dataset(c, args.dataset, args.candidates, args.output)
    elif name == "intent-recordings":
        from dasa.recorded import intent_recordings
        intent_recordings(c, args.dataset, args.output, args.participants, args.recordings)
    elif name == "representation-analyze":
        from dasa.representation import analyze
        analyze(c, args.recorded, args.participants, args.output)
    elif name == "representation-schedule":
        from dasa.study import write_representation_schedule
        write_representation_schedule(c, args.participants, args.output)
    elif name == "study-schedule":
        from dasa.study import write_schedule
        write_schedule(c, args.output)
    elif name == "study-analyze":
        from dasa.study import analyze
        analyze(c, args.trials, args.participants, args.protocol, args.tlx, args.output)
    elif name == "scene-manifest":
        from dasa.scenes import write_scene_manifest
        write_scene_manifest(c, args.output, args.count, args.scenario, args.seed)
    elif name == "calibrate":
        from dasa.robotics.calibration import calibrate_file
        calibrate_file(args.landmarks, args.output)
    elif name == "perceive":
        from dasa.robotics.perception import perceive_file
        perceive_file(args.rgbd, args.specification, args.output)
    elif name == "camera-capture":
        from dasa.robotics.camera import RealSenseCamera, detect_frame
        if args.samples < 1:
            raise ValueError("Capture count must be positive")
        device = RealSenseCamera(args.calibration, args.output, serial=args.serial)
        try:
            device.start()
            specification = json.loads(args.perception.read_text())
            from dasa.config import write_json
            for _ in range(args.samples):
                frame = device.capture()
                write_json(args.output / f"scene_{frame.sequence:09d}.json",
                           detect_frame(frame, device.calibration, specification, "snapshot"))
            write_json(args.output / "scene.json",
                       detect_frame(frame, device.calibration, specification, "snapshot"))
        finally:
            device.close()
    elif name == "hardware-config":
        from dasa.robotics.configuration import configure
        if args.feedback_delay is not None:
            c["task"]["feedback_delay"] = args.feedback_delay
        configure(c, args.host, args.calibration, args.scene, args.output, args.model,
                  "robotiq_socket", args.hardware_profile, args.perception, args.tool)
    elif name == "robot-read":
        from dasa.robotics.interfaces import record_robot
        record_robot(args.host, args.samples, args.output)
    elif name == "robot-session":
        from dasa.robotics.session import run_workstation
        run_workstation(c, args.hardware, args.models, args.output, args.execute,
                        args.condition, args.skill, autonomous_goal=args.goal, stage=args.stage)
    elif name == "hardware-trial":
        from dasa.robotics.experiment import run_experiment
        run_experiment(c, args.schedule, args.hardware, args.models, args.output,
                       args.execute, args.participant, args.trial_id, args.skill)
    elif name == "master-device":
        from dasa.robotics.master import run_master
        result = run_master(args.library, args.rotation, host=args.host, port=args.port,
                            device_index=args.device_index, frequency=args.frequency,
                            stiffness=args.stiffness, maxcommand=args.max_command,
                            maxfeedback=args.max_feedback, deadman_button=args.deadman_button,
                            grip_button=args.grip_button, feedback_timeout=args.feedback_timeout,
                            video_port=args.video_port, config=c, record_directory=args.output,
                            watchdog_unit_seconds=args.watchdog_unit_us * 1e-6)
        print(json.dumps(result, indent=2))
    elif name == "gripper":
        from dasa.robotics.interfaces import RobotiqSocket
        if (args.activate or args.open or args.close) and not args.execute:
            raise ValueError("Gripper actuation requires --execute")
        device = RobotiqSocket(args.host, args.port, args.execute)
        if args.activate:
            result = device.activate()
        elif args.open or args.close:
            result = {"command": device.command(bool(args.close)), "measured_completion": device.wait_until_stopped()}
        else:
            result = device.status()
        print(json.dumps(result, indent=2))
    elif name == "annotate-recording":
        from dasa.robotics.annotation import write_annotation
        write_annotation(args.recording, args.output, args.episode, args.operator,
                         args.goal, args.outcome, args.grasped_goal)
    elif name == "import-recording":
        from dasa.robotics.session import import_recording
        import_recording(args.recording, args.annotations, args.output)
    elif name == "measured-metrics":
        from dasa.robotics.analysis import export_metrics
        export_metrics(c, args.recordings, args.output)
    elif name == "generalization":
        from dasa.generalization import generalization
        generalization(c, args.prepared, args.candidates, args.output, args.scenes)
    elif name == "generalization-analyze":
        from dasa.generalization import aggregate_measured_evaluations
        aggregate_measured_evaluations(c, args.experiment, args.recordings, args.output)
    elif name == "refinement-schedule":
        from dasa.generalization import refinement_schedule
        refinement_schedule(c, args.prepared, args.operators, args.output)
    elif name == "validation-plan":
        from dasa.tuning import plan
        plan(c, args.prepared, args.output, args.parameters)
    elif name == "validation-select":
        from dasa.tuning import select
        select(c, args.plan, args.trials, args.output)
    elif name == "figures":
        from dasa.figures import build
        build(c, args.output)
    elif name == "recording-figures":
        from dasa.reporting import recording_figures
        recording_figures(args.recording, args.output)
    elif name == "learning-trajectories":
        from dasa.reporting import learning_trajectories
        learning_trajectories(c, args.recordings, args.scene_id, args.output)
    elif name == "report":
        from dasa.reporting import report
        report(c, args.output)
    elif name == "protocol":
        from dasa.reproducibility import export_protocol
        export_protocol(c, args.output)
    elif name == "audit-run":
        from dasa.reproducibility import audit_run
        print(json.dumps(audit_run(c, args.output), indent=2))


if __name__ == "__main__":
    main()
