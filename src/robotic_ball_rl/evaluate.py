"""Evaluate a saved spiral policy and export trajectory diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .external_env import load_course_env_class
from .wrappers import SpiralRewardWrapper


def evaluate(run_dir: Path, *, seed: int = 42, steps: int = 500, output_dir: Path | None = None):
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cls = load_course_env_class()
    base = DummyVecEnv([
        lambda: TimeLimit(SpiralRewardWrapper(env_cls(render_mode=None)), max_episode_steps=steps)
    ])
    env = VecNormalize.load(run_dir / "vecnormalize.pkl", base)
    env.training = False
    env.norm_reward = False
    model = SAC.load(run_dir / "fizzy", device="cpu")

    rows = []
    try:
        env.seed(seed)
        obs = env.reset()
        for step_idx in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            info = infos[0]
            rows.append(
                {
                    "step": step_idx + 1,
                    "time_s": 0.1 * (step_idx + 1),
                    "x_m": float(info["x_position"]),
                    "y_m": float(info["y_position"]),
                    "reward": float(rewards[0]),
                }
            )
            if dones[0]:
                break
    finally:
        env.close()

    df = pd.DataFrame(rows)
    df["radius_m"] = np.hypot(df["x_m"], df["y_m"])
    df.to_csv(output_dir / "trajectory.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(df["x_m"], df["y_m"])
    ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Deterministic spiral-policy rollout")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_xy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(df["time_s"], df["radius_m"])
    ax.set(xlabel="time (s)", ylabel="radius from origin (m)", title="Radius over time")
    fig.tight_layout()
    fig.savefig(output_dir / "radius_vs_time.png", dpi=180)
    plt.close(fig)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    df = evaluate(args.run_dir, seed=args.seed, steps=args.steps, output_dir=args.output_dir)
    print(df.tail())


if __name__ == "__main__":
    main()
