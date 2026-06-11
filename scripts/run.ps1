# Run the full double-stent pipeline for a given experiment directory.
#
# Usage:
#   .\scripts\run.ps1 "experiment\experiment 0"
#   .\scripts\run.ps1 "experiment\experiment 0" --clean
#
# Requires uv to be installed. See README for setup instructions.

param(
    [string]$ExperimentDir = "experiment\experiment 0",
    [switch]$Clean
)

Set-Location (Split-Path -Parent $PSScriptRoot)

$args_list = @("run", "python", "run.py", "--experiment_dir", $ExperimentDir)
if ($Clean) {
    $args_list += "--clean"
}

& uv @args_list
