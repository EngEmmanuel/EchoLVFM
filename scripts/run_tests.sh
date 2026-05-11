#!/usr/bin/env bash
# Run the EchoLVFM test suite.
#
# Usage:
#   bash scripts/run_tests.sh              # run all tests
#   bash scripts/run_tests.sh losses       # run loss tests only
#   bash scripts/run_tests.sh flows        # run flow tests only
#   bash scripts/run_tests.sh model        # run model tests only (requires diffusers)
#
# Must be run from the repo root:
#   cd /path/to/EchoLVFM
#   bash scripts/run_tests.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure the repo root is on PYTHONPATH so imports resolve correctly.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Default: run all tests. Accept optional filter argument.
FILTER="${1:-}"

if [ -z "$FILTER" ]; then
    echo "==> Running all tests"
    pytest tests/ -v
elif [ "$FILTER" = "losses" ]; then
    echo "==> Running loss tests"
    pytest tests/test_losses.py -v
elif [ "$FILTER" = "flows" ]; then
    echo "==> Running flow tests"
    pytest tests/test_flows.py -v
elif [ "$FILTER" = "model" ]; then
    echo "==> Running model tests (requires diffusers)"
    pytest tests/test_model.py -v
else
    echo "Unknown filter '$FILTER'. Valid options: losses, flows, model"
    echo "Running all tests instead."
    pytest tests/ -v
fi
