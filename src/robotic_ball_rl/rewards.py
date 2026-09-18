"""Reward functions used in the robotic-ball portfolio project.

The main experiment rewards velocity aligned with the local tangent of a circle
around the world-frame origin. Because the policy only observes onboard IMU
signals, position and planar velocity are privileged simulation signals used by
the reward during training, not policy inputs.
"""

from __future__ import annotations

import numpy as np


def tangential_alignment_reward(
    position_xy: np.ndarray,
    velocity_xy: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Return cosine alignment between planar velocity and the positive tangent.

    Parameters
    ----------
    position_xy:
        Robot position ``[x, y]`` in the world frame.
    velocity_xy:
        Robot planar velocity ``[vx, vy]`` in the world frame.
    eps:
        Minimum velocity norm below which reward is defined as zero.

    Notes
    -----
    At position angle ``alpha = atan2(y, x)``, the positive tangent is
    ``[-sin(alpha), cos(alpha)]``. The tangent already has unit norm, so the
    reward is the cosine of the velocity/tangent angle and lies in ``[-1, 1]``.
    """

    position = np.asarray(position_xy, dtype=float).reshape(2)
    velocity = np.asarray(velocity_xy, dtype=float).reshape(2)

    speed = float(np.linalg.norm(velocity))
    if not np.isfinite(speed) or speed <= eps:
        return 0.0

    alpha = float(np.arctan2(position[1], position[0]))
    tangent = np.array([-np.sin(alpha), np.cos(alpha)], dtype=float)
    return float(np.dot(tangent, velocity) / speed)
