#!/usr/bin/env bash
# Watch a policy. With no checkpoint it plays the zero-action agent, which
# should simply stand -- the quickest check that a model change didn't break
# anything.
#
#   scripts/play.sh                                     # zero agent
#   scripts/play.sh --checkpoint-file logs/rsl_rl/rocky_velocity/<run>/model_3000.pt
#   scripts/play.sh --wandb-run-path axiboai/rocky/<run-id>
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${TASK:-Mjlab-Velocity-Flat-Rocky}"
if [[ $# -eq 0 ]]; then
  exec uv run play "${TASK}" --agent zero --num-envs 1
fi
exec uv run play "${TASK}" --num-envs 1 "$@"
