"""Train SAC on the spiral reward using a locally supplied Fizzy environment."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from .external_env import make_spiral_env


def train(
    *,
    output_root: Path,
    total_timesteps: int = 100_000,
    n_envs: int = 16,
    seed: int = 42,
) -> Path:
    if n_envs not in (1, 2, 4, 8, 16):
        raise ValueError("n_envs must be one of 1, 2, 4, 8, 16")

    train_freq = 16 // n_envs
    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    run_dir = output_root / datetime.now().strftime("spiral_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    env = VecNormalize(
        VecMonitor(vec_cls([make_spiral_env for _ in range(n_envs)])),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    try:
        model = SAC(
            "MlpPolicy",
            env,
            train_freq=train_freq,
            gradient_steps=2,
            gamma=0.995,
            seed=seed,
            device="cpu",
            verbose=1,
            tensorboard_log=str(output_root / "tensorboard"),
        )
        model.learn(total_timesteps, tb_log_name=run_dir.name)
        model.save(run_dir / "fizzy")
        env.save(run_dir / "vecnormalize.pkl")
    finally:
        env.close()

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = train(
        output_root=args.output_root,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
