# Underactuated Robotic Ball — Reinforcement Learning

Portfolio refactor of my September 2026 **Artificial Intelligence for Robotics** summer-school experiment at RWTH Aachen / IRT.

The project studies reinforcement-learning control of **Fizzy**, a single-motor underactuated robotic ball, using a supplied MuJoCo/Gymnasium simulator. My main experiment designed a new reward for tangential motion, trained a Soft Actor-Critic (SAC) policy, and evaluated the resulting behavior.

> **Scope note:** the original course simulator, robot model, notebooks, pretrained checkpoints, VMC baseline, and hardware adapter are not redistributed here. This repository contains my portfolio refactor, experiment artifacts, tests, and documentation. See [`docs/provenance.md`](docs/provenance.md).

## What I implemented

- a custom **tangential-velocity reward** for circular/spiral motion;
- a vectorized **SAC** training pipeline with 16 environments;
- observation/reward normalization with **VecNormalize**;
- deterministic evaluation and trajectory diagnostics;
- a reusable Gymnasium wrapper, configuration, unit tests, and CI;
- an explicit analysis of the **sim-to-real gap** and what additional sensing/calibration would be needed for physical deployment.

## Recorded training result

The SAC run used **16 parallel environments** and a **100,000-transition** training budget. The latest periodic training log (99,200 transitions) reports an SB3 rollout mean reward of **131**. This is a training statistic rather than an independent task-success metric.

![Training curve](results/training_curve.svg)

The saved 50-second deterministic rollout showed sustained tangential motion, but the radius grew and oscillated because the reward did not constrain radial error. That failure mode is useful: it directly motivates adding a radius-tracking term instead of treating the first reward as a finished controller.

## RL formulation

| Component | Experiment definition |
|---|---|
| Robot | Single-motor underactuated spherical robot |
| Simulator | MuJoCo, exposed through Gymnasium |
| Policy observation | 10-D IMU state: accelerometer (3), gyro (3), quaternion (4) |
| Action | One normalized motor command in `[-1, 1]` |
| Control rate | 10 Hz in the supplied environment |
| Algorithm | Soft Actor-Critic (Stable-Baselines3) |
| Reward | Cosine alignment between planar velocity and the local counter-clockwise tangent |
| Training | 100k transitions, 16 parallel envs, VecNormalize, `gamma=0.995` |

The reward implemented in [`rewards.py`](src/robotic_ball_rl/rewards.py) is:

```text
alpha = atan2(y, x)
tangent = [-sin(alpha), cos(alpha)]
reward = dot(tangent, velocity) / ||velocity||
```

A stationary robot receives zero reward. The design encourages circulation but does **not** penalize radial drift.

## Engineering structure

```text
robotic-ball-rl/
├── configs/                 # reproducible experiment settings
├── docs/                    # formulation, provenance, sim-to-real notes
├── results/                 # recorded training data and visualization
├── scripts/                 # thin train/evaluate entry points
├── src/robotic_ball_rl/
│   ├── rewards.py           # custom reward formulation
│   ├── wrappers.py          # Gymnasium reward wrapper
│   ├── external_env.py      # local-only adapter to the course simulator
│   ├── train.py             # vectorized SAC training pipeline
│   └── evaluate.py          # deterministic rollout + diagnostics
└── tests/                   # public unit tests independent of course code
```

## Run the public tests

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest -q
```

## Re-run the full experiment locally

Authorized summer-school participants can place the original `fizzy_modules` package on `PYTHONPATH`, then install the RL dependencies:

```bash
pip install -e ".[rl,dev]"
python -m robotic_ball_rl.train --total-timesteps 100000 --n-envs 16 --seed 42
python -m robotic_ball_rl.evaluate runs/<run-folder> --steps 500
```

The training script saves the policy and its matching `VecNormalize` statistics together so evaluation does not accidentally pair a model with incompatible normalization data.

## Sim-to-real boundary

The simulation can use world-frame position and velocity to compute the spiral reward even though the policy sees only IMU state. A physical implementation therefore needs external tracking or a different reward/task formulation. The saved summer-school notebook includes an optional real-robot interface scaffold, but the uploaded version does not contain a completed policy-transfer experiment; I therefore do **not** claim successful hardware deployment in this repository. See [`docs/sim2real.md`](docs/sim2real.md).

## Context

Fizzy is an open-source robotic-ball research platform for underactuated control and robot learning. Related research:

G. Spigler, S. Silenzi, B. Horstman, H. Vallery, *Intuitive Engineering as a Proxy Facilitates the Evolutionary Optimization of an Underactuated Robot Ball*, EARL @ RSS 2024.

- Project website: https://fizzyball.org/
- Paper: https://openreview.net/forum?id=n97siS87bF
