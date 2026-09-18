"""Gymnasium wrappers for reward-design experiments.

No course simulator source is bundled in this repository. These wrappers operate
on a compatible MuJoCo environment supplied separately by the user.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .rewards import tangential_alignment_reward


class SpiralRewardWrapper(gym.Wrapper):
    """Replace the original task reward with tangential-velocity alignment.

    Expected environment interface
    ------------------------------
    * observation is a dictionary containing ``"state"``;
    * ``env.unwrapped.data.joint(0).qpos[:2]`` exposes planar position;
    * ``env.unwrapped.data.joint(0).qvel[:2]`` exposes planar velocity.

    The wrapped policy receives only the ``state`` observation. An outer
    ``TimeLimit`` should terminate the fixed-duration episode.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("Expected a Dict observation space with a 'state' entry.")
        if "state" not in env.observation_space.spaces:
            raise KeyError("Observation space has no 'state' entry.")
        self.observation_space = env.observation_space["state"]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs["state"], info

    def step(self, action):
        observation, _base_reward, _terminated, _truncated, info = self.env.step(action)

        data = self.unwrapped.data
        position_xy = np.asarray(data.joint(0).qpos[:2], dtype=float)
        velocity_xy = np.asarray(data.joint(0).qvel[:2], dtype=float)
        reward = tangential_alignment_reward(position_xy, velocity_xy)

        # This experiment replaces the goal-reaching task. Episode length is
        # controlled by an outer gymnasium.wrappers.TimeLimit.
        terminated = False
        truncated = False
        return observation["state"], reward, terminated, truncated, info
