"""Adapter for the non-redistributed summer-school Fizzy environment."""

from __future__ import annotations

from importlib import import_module

from gymnasium.wrappers import TimeLimit

from .wrappers import SpiralRewardWrapper


def load_course_env_class():
    """Import the course environment when it is available on ``PYTHONPATH``.

    The public portfolio repository deliberately does not redistribute the
    summer-school simulator, model, notebooks, checkpoints, or UDP hardware
    adapter. Put the original course package in your local environment before
    running the full experiment.
    """

    try:
        module = import_module("fizzy_modules.fizzy_mujoco_optimized")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The external summer-school package 'fizzy_modules' was not found. "
            "See docs/provenance.md for the local-only setup." 
        ) from exc
    return module.FizzyMujocoEnvOptimized


def make_spiral_env(*, max_episode_steps: int = 200, render_mode=None):
    env_cls = load_course_env_class()
    base = env_cls(render_mode=render_mode)
    return TimeLimit(SpiralRewardWrapper(base), max_episode_steps=max_episode_steps)
