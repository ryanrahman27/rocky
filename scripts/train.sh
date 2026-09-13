#!/usr/bin/env bash
# Train Rocky's locomotion policy. Linux + NVIDIA GPU required.
#
#   scripts/train.sh                                  # flat ground, 4096 envs
#   scripts/train.sh Mjlab-Velocity-Rough-Rocky 8192 6000
#
# Everything after the first three positional args is passed through to mjlab,
# e.g. --env.scene.terrain.terrain-type plane
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${1:-Mjlab-Velocity-Flat-Rocky}"
NUM_ENVS="${2:-4096}"
ITERS="${3:-3000}"
shift $(( $# > 3 ? 3 : $# )) || true

echo "==> ${TASK}: ${NUM_ENVS} envs, ${ITERS} iterations"
exec uv run train "${TASK}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.max-iterations "${ITERS}" \
  "$@"
