#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${DASA_VENV:-/workspace/.venvs/dasa}"
export PIP_CACHE_DIR="${DASA_CACHE:-/workspace/.cache/pip}"
export MPLCONFIGDIR="${DASA_MPL_CACHE:-/tmp/dasa-matplotlib}"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
if [[ ! -x "$venv_dir/bin/python" ]]; then
  python3 -m venv "$venv_dir"
fi
cd "$repo_dir"
"$venv_dir/bin/python" -m pip install --require-hashes -r requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cpu
"$venv_dir/bin/python" -m pip install --no-deps --no-build-isolation -e .
"$venv_dir/bin/python" -m pip install --require-hashes --no-deps --only-binary=:all: \
  -r requirements-hardware.txt
"$venv_dir/bin/python" -c 'import rtde_receive, rtde_control; print("RTDE modules imported; no device connection attempted")'
"$venv_dir/bin/python" -m pip check
"$venv_dir/bin/python" -m pytest -q
