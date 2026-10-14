param([ValidateSet("smoke", "full")][string]$Profile = "smoke")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Python = ".venv\Scripts\python.exe"
$RunPath = "artifacts\" + $Profile + "-" + (Get-Date -Format "yyyyMMdd-HHmmss")
if ($Profile -eq "smoke") {
    & $Python -m dasa.cli --profile configs/smoke.yaml run --output $RunPath
} else {
    & $Python -m dasa.cli run --output $RunPath
}
if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $LASTEXITCODE" }
Write-Host "Report: $RunPath\report\index.html"
