"""Command-line entry points for reproducible experiment stages."""

import argparse
import json
from pathlib import Path

from dasa.config import load_config, seed_everything


def parser():
    p = argparse.ArgumentParser(prog="dasa")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--profile", help="YAML override merged into the base configuration")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "intent", "sharing", "refinement", "stress", "report", "port-bench"):
        s = sub.add_parser(command)
        s.add_argument("--output", type=Path, default=Path("artifacts/run"))
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
    s.add_argument("--condition", default="C3")
    s.add_argument("--output", type=Path, default=Path("artifacts/robot_replay.json"))
    s = sub.add_parser("robot-session")
    s.add_argument("--hardware", type=Path, required=True)
    s.add_argument("--models", type=Path, required=True)
    s.add_argument("--condition", default="C3")
    s.add_argument("--execute", action="store_true", help="Enable RTDE speed commands and gripper actuation")
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
    if args.command in {"prepare", "run", "train-data", "refine-data"} and args.output.exists() and any(args.output.iterdir()):
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
        replay(config, args.models, args.trial, args.output, args.condition)
    elif args.command == "robot-session":
        from dasa.robotics.session import run_workstation
        run_workstation(config, args.hardware, args.models, args.output, args.execute, args.condition)
    elif args.command == "import-recording":
        from dasa.robotics.session import import_recording
        import_recording(args.recording, args.annotations, args.output)
    elif args.command == "report":
        from dasa.reporting import report
        report(config, args.output)
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
