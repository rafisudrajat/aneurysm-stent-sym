# Run the virtual stenting pipeline for a given experiment directory.
#
# Usage:
#   .\scripts\run.ps1 "experiment\experiment 0"
#   .\scripts\run.ps1 "experiment\experiment 0" -Clean
#   .\scripts\run.ps1 "experiment\experiment 0" -SingleStent
#   .\scripts\run.ps1 "experiment\experiment 0" -Deploy -Pos outer
#
# Requires the virtual environment to be activated (.venv\Scripts\Activate.ps1).

param(
    [string]$ExperimentDir = "experiment\experiment 0",
    [switch]$Clean,
    [switch]$SingleStent,
    [switch]$Deploy,
    [string]$Pos = "both"
)

Set-Location (Split-Path -Parent $PSScriptRoot)

if ($Clean) {
    python run.py $ExperimentDir --clean
} elseif ($SingleStent) {
    python run.py $ExperimentDir --single-stent
} elseif ($Deploy) {
    python run.py $ExperimentDir --deploy --pos $Pos
} else {
    python run.py $ExperimentDir
}
