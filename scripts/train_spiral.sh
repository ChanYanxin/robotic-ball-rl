#!/usr/bin/env bash
set -euo pipefail
python -m robotic_ball_rl.train --total-timesteps 100000 --n-envs 16 --seed 42
