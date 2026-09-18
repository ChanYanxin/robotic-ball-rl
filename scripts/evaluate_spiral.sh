#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 2
fi
python -m robotic_ball_rl.evaluate "$1" --seed 42 --steps 500
