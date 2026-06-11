#!/usr/bin/env bash
# Run the virtual stenting pipeline for a given experiment directory.
#
# Usage:
#   ./scripts/run.sh "experiment/experiment 0"
#   ./scripts/run.sh "experiment/experiment 0" --clean
#   ./scripts/run.sh "experiment/experiment 0" --single-stent
#   ./scripts/run.sh "experiment/experiment 0" --deploy --pos outer
#
# Requires the virtual environment to be activated, or uv installed.

set -euo pipefail

EXPERIMENT_DIR="${1:-experiment/experiment 0}"
shift || true   # drop $1 so remaining args are passed through to run.py

cd "$(dirname "$0")/.."
python run.py "$EXPERIMENT_DIR" "$@"
