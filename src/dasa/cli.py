import argparse
import json
from pathlib import Path

from dasa.config import load_config, seed_everything


def parser():
    p = argparse.ArgumentParser(prog="dasa")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--profile", help="YAML override merged into the base configuration")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "intent", "sharing", "refinement", "stress", "report", "port-bench",
                    "figures", "protocol", "audit-run"):
        s = sub.add_parser(command)
        s.add_argument("--output", type=Path, default=Path("artifacts/run"))
    s = sub.add_parser("scene-manifest")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--count", type=int, default=100)
    s.add_argument("--scenario", choices=["demonstrated", "novel", "wider", "extra_obstacles"], default="demonstrated")
    s.add_argument("--seed", type=int)
    s = sub.add_parser("perceive")
    s.add_argument("rgbd", type=Path)
    s.add_argument("specification", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("generalization")
    s.add_argument("prepared", type=Path)
    s.add_argument("candidates", type=Path)
    s.add_argument("--scenes", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("validation-sweep")
    s.add_argument("prepared", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--parameters", nargs="+")
    s = sub.add_parser("intent-recordings")
    s.add_argument("prepared", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("video")
    s.add_argument("trial", type=Path)
    s.add_argument("--output", type=Path, default=Path("artifacts/replay.gif"))
    s = sub.add_parser("validate-data")
    s.add_argument("dataset", type=Path)
    s = sub.add_parser("train-data")
    s.add_argument("dataset", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("refine-data")
    s.add_argument("dataset", type=Path)
    s.add_argument("candidates", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("study-schedule")
    s.add_argument("--output", type=Path, default=Path("artifacts/study_schedule.csv"))
    s = sub.add_parser("study-analyze")
    s.add_argument("trials", type=Path)
    s.add_argument("participants", type=Path)
    s.add_argument("protocol", type=Path)
    s.add_argument("--tlx", type=Path)
    s.add_argument("--output", type=Path, default=Path("artifacts/human_study"))
    s = sub.add_parser("calibrate")
    s.add_argument("landmarks", type=Path)
    s.add_argument("--output", type=Path, default=Path("artifacts/calibration.json"))
    s = sub.add_parser("robot-read")
    s.add_argument("--host", required=True)
    s.add_argument("--samples", type=int, default=100)
    s.add_argument("--output", type=Path, default=Path("artifacts/robot_state.json"))
    s = sub.add_parser("robot-replay")
    s.add_argument("trial", type=Path)
    s.add_argument("--models", type=Path, required=True)
    s.add_argument("--skill", type=Path)
    s.add_argument("--condition", default="C3")
    s.add_argument("--output", type=Path, default=Path("artifacts/robot_replay.json"))
    s = sub.add_parser("robot-session")
    s.add_argument("--hardware", type=Path, required=True)
    s.add_argument("--models", type=Path)
    s.add_argument("--skill", type=Path)
    s.add_argument("--condition", default="C3")
    s.add_argument("--execute", action="store_true", help="Enable RTDE speed commands and gripper actuation")
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("hardware-config")
    s.add_argument("--host", required=True)
    s.add_argument("--calibration", type=Path, required=True)
    s.add_argument("--scene", type=Path, required=True)
    s.add_argument("--model", choices=["UR10", "UR10e"], default="UR10")
    s.add_argument("--gripper", choices=["robotiq_socket", "external"], default="robotiq_socket")
    s.add_argument("--hardware-profile", type=Path)
    s.add_argument("--feedback-delay", type=float)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("hardware-trial")
    s.add_argument("schedule", type=Path)
    s.add_argument("--hardware", type=Path, required=True)
    s.add_argument("--models", type=Path, required=True)
    s.add_argument("--skill", type=Path)
    s.add_argument("--participant")
    s.add_argument("--trial-id")
    s.add_argument("--execute", action="store_true")
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("master-device")
    s.add_argument("--library", type=Path, required=True)
    s.add_argument("--rotation", type=Path, required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=55001)
    s.add_argument("--device-index", type=int, default=0)
    s.add_argument("--frequency", type=float, default=500.0)
    s.add_argument("--stiffness", type=float, default=100.0)
    s.add_argument("--max-command", type=float, default=12.0)
    s.add_argument("--max-feedback", type=float, default=2.0)
    s.add_argument("--deadman-button", type=int, default=0)
    s.add_argument("--grip-button", type=int, default=1)
    s.add_argument("--feedback-timeout", type=float, default=0.1)
    s = sub.add_parser("gripper")
    s.add_argument("--host", required=True)
    s.add_argument("--port", type=int, default=63352)
    s.add_argument("--activate", action="store_true")
    s.add_argument("--execute", action="store_true")
    s = sub.add_parser("annotate-recording")
    s.add_argument("recording", type=Path)
    s.add_argument("--episode", required=True)
    s.add_argument("--operator", required=True)
    s.add_argument("--goal", type=int, choices=range(3), required=True)
    s.add_argument("--grasped-goal", type=int, choices=range(3))
    s.add_argument("--outcome", choices=["success", "collision", "bounds", "wrong_object", "timeout", "aborted"], required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("measured-metrics")
    s.add_argument("recordings", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("import-recording")
    s.add_argument("recording", type=Path)
    s.add_argument("annotations", type=Path)
    s.add_argument("--output", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    config = load_config(args.config, args.profile)
    seed_everything(config["seed"])
    if args.command in {"prepare", "run", "train-data", "refine-data", "generalization", "validation-sweep", "intent-recordings"} and args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output already contains data; use a new --output path to preserve the existing run")
    if args.command == "video":
        from dasa.reporting import video
        video(args.trial, args.output)
    elif args.command == "validate-data":
        from dasa.data import validate_partitions
        validate_partitions(args.dataset, json.loads((args.dataset / "splits.json").read_text()))
        print("Demonstration schemas and operator-disjoint partitions verified")
    elif args.command == "train-data":
        from dasa.experiments import train_dataset
        train_dataset(config, args.dataset, args.output)
    elif args.command == "refine-data":
        from dasa.experiments import refine_dataset
        refine_dataset(config, args.dataset, args.candidates, args.output)
    elif args.command == "study-schedule":
        from dasa.study import write_schedule
        write_schedule(config, args.output)
    elif args.command == "study-analyze":
        from dasa.study import analyze
        analyze(config, args.trials, args.participants, args.protocol, args.tlx, args.output)
    elif args.command == "calibrate":
        from dasa.robotics.calibration import calibrate_file
        calibrate_file(args.landmarks, args.output)
    elif args.command == "robot-read":
        from dasa.robotics.interfaces import record_robot
        record_robot(args.host, args.samples, args.output)
    elif args.command == "robot-replay":
        from dasa.robotics.session import replay
        replay(config, args.models, args.trial, args.output, args.condition, args.skill)
    elif args.command == "robot-session":
        from dasa.robotics.session import run_workstation
        run_workstation(config, args.hardware, args.models, args.output, args.execute, args.condition, args.skill)
    elif args.command == "hardware-config":
        from dasa.robotics.configuration import configure
        if args.feedback_delay is not None:
            config["simulation"]["feedback_delay"] = args.feedback_delay
        configure(config, args.host, args.calibration, args.scene, args.output,
                  args.model, args.gripper, args.hardware_profile)
    elif args.command == "hardware-trial":
        from dasa.robotics.experiment import run_experiment
        run_experiment(config, args.schedule, args.hardware, args.models, args.output,
                       args.execute, args.participant, args.trial_id, args.skill)
    elif args.command == "master-device":
        from dasa.robotics.master import run_master
        print(json.dumps(run_master(args.library, args.rotation, host=args.host, port=args.port,
                         device_index=args.device_index, frequency=args.frequency,
                         stiffness=args.stiffness, maxcommand=args.max_command,
                         maxfeedback=args.max_feedback, deadman_button=args.deadman_button,
                         grip_button=args.grip_button, feedback_timeout=args.feedback_timeout), indent=2))
    elif args.command == "gripper":
        from dasa.robotics.interfaces import RobotiqSocket
        if args.activate and not args.execute:
            raise SystemExit("Gripper activation requires --execute")
        device = RobotiqSocket(args.host, args.port, args.execute)
        result = device.activate() if args.activate else device.status()
        print(json.dumps(result, indent=2))
    elif args.command == "annotate-recording":
        from dasa.robotics.annotation import write_annotation
        write_annotation(args.recording, args.output, args.episode, args.operator,
                         args.goal, args.outcome, args.grasped_goal)
    elif args.command == "measured-metrics":
        from dasa.robotics.analysis import export_metrics
        export_metrics(config, args.recordings, args.output)
    elif args.command == "import-recording":
        from dasa.robotics.session import import_recording
        import_recording(args.recording, args.annotations, args.output)
    elif args.command == "report":
        from dasa.reporting import report
        report(config, args.output)
    elif args.command == "figures":
        from dasa.figures import build
        result = build(config, args.output)
        print(json.dumps(result, indent=2))
    elif args.command == "protocol":
        from dasa.reproducibility import export_protocol
        export_protocol(config, args.output)
    elif args.command == "audit-run":
        from dasa.reproducibility import audit_run
        print(json.dumps(audit_run(config, args.output), indent=2))
    elif args.command == "scene-manifest":
        from dasa.scenes import write_scene_manifest
        write_scene_manifest(config, args.output, args.count, args.scenario, args.seed)
    elif args.command == "perceive":
        from dasa.robotics.perception import perceive_file
        perceive_file(args.rgbd, args.specification, args.output)
    elif args.command == "generalization":
        from dasa.generalization import generalization
        generalization(config, args.prepared, args.candidates, args.output, args.scenes)
    elif args.command == "validation-sweep":
        from dasa.tuning import sweep
        sweep(config, args.prepared, args.output, args.parameters)
    elif args.command == "intent-recordings":
        from dasa.recorded import intent_recordings
        intent_recordings(config, args.prepared, args.output)
    elif args.command == "port-bench":
        from dasa.portbench import run
        run(config, args.output)
    else:
        from dasa import experiments
        operations = {"prepare": experiments.prepare, "run": experiments.run_all,
                      "intent": experiments.intent_experiment, "sharing": experiments.sharing,
                      "refinement": experiments.refinement_experiment, "stress": experiments.stress_experiment}
        operations[args.command](config, args.output)


if __name__ == "__main__":
    main()
