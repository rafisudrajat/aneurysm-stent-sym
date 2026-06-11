#!/usr/bin/env bash
# Run the full double-stent pipeline for a given experiment directory.
#
# Usage:
#   ./scripts/run.sh "experiment/experiment 0"
#   ./scripts/run.sh "experiment/experiment 0" --clean
#   ./scripts/run.sh "experiment/experiment 0" --single-stent
#
# Requires uv to be installed. See README for setup instructions.

set -euo pipefail

EXPERIMENT_DIR="${1:-experiment/experiment 0}"
shift || true   # drop $1 so remaining args (e.g. --clean) are passed through

cd "$(dirname "$0")/.."
uv run stenting run "$EXPERIMENT_DIR" "$@"
