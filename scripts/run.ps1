# Run the full double-stent pipeline for a given experiment directory.
#
# Usage:
#   .\scripts\run.ps1 "experiment\experiment 0"
#   .\scripts\run.ps1 "experiment\experiment 0" -Clean
#   .\scripts\run.ps1 "experiment\experiment 0" -SingleStent
#
# Requires uv to be installed. See README for setup instructions.

param(
    [string]$ExperimentDir = "experiment\experiment 0",
    [switch]$Clean,
    [switch]$SingleStent
)

Set-Location (Split-Path -Parent $PSScriptRoot)

if ($Clean) {
    uv run stenting clean $ExperimentDir
} elseif ($SingleStent) {
    uv run stenting run $ExperimentDir --single-stent
} else {
    uv run stenting run $ExperimentDir
}
