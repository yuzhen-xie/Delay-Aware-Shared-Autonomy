import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = []
    directories = ["src", "configs", "study", "scripts", ".github"]
    for directory in directories:
        files.extend(p for p in (root / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
                     and not p.name.endswith(".local.yaml")
                     and "local" not in p.relative_to(root).parts
                     and not any(part.endswith(".egg-info") for part in p.parts))
    files.extend(root / name for name in ("README.md", "pyproject.toml", "requirements.lock",
                                         "requirements-hardware.txt", ".gitignore"))
    files = sorted(set(files))
    checksums = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Bundle entry escapes the project root")
            name = str(path.relative_to(root)).replace("\\", "/")
            content = path.read_bytes()
            checksums[name] = hashlib.sha256(content).hexdigest()
            bundle.writestr(f"Delay-Aware-Shared-Autonomy/{name}", content)
        bundle.writestr("Delay-Aware-Shared-Autonomy/RELEASE_SHA256.json",
                        json.dumps(checksums, indent=2))
    print(f"Bundled {len(files)} files into {args.output}")


if __name__ == "__main__":
    main()
