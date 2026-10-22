param(
    [Parameter(Mandatory = $true)][string]$Schedule,
    [Parameter(Mandatory = $true)][string]$Hardware,
    [Parameter(Mandatory = $true)][string]$Models,
    [Parameter(Mandatory = $true)][string]$Participant,
    [Parameter(Mandatory = $true)][string]$TrialId,
    [Parameter(Mandatory = $true)][string]$Output,
    [switch]$Execute
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Python = ".venv\Scripts\python.exe"
$Arguments = @("-m", "dasa.cli", "hardware-trial", $Schedule, "--hardware", $Hardware,
               "--models", $Models, "--participant", $Participant, "--trial-id", $TrialId,
               "--output", $Output)
if ($Execute) { $Arguments += "--execute" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Acquisition failed: $LASTEXITCODE" }
