param([string]$EnvironmentPath = ".venv", [switch]$Hardware)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}
Invoke-Checked { py -3.12 -m venv $EnvironmentPath }
$Python = Join-Path $EnvironmentPath "Scripts\python.exe"
$env:MPLCONFIGDIR = Join-Path $env:TEMP "dasa-matplotlib"
Invoke-Checked { & $Python -m pip install --require-hashes -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu }
Invoke-Checked { & $Python -m pip install --no-deps --no-build-isolation -e . }
if ($Hardware) {
    Invoke-Checked { & $Python -m pip install --require-hashes --no-deps --only-binary=:all: -r requirements-hardware.txt }
    Invoke-Checked { & $Python -c 'import rtde_receive, rtde_control' }
}
Invoke-Checked { & $Python -m pip check }
Invoke-Checked { & $Python -m pytest -q }
