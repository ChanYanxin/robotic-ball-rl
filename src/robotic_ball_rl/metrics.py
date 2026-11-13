"""Geometry-only diagnostics for robotic-ball simulation trajectories.

The metrics can be computed from saved x-y points without the original course
simulator, SAC checkpoint, or Stable-Baselines3 installation. They describe
behavior; they are NOT a task success rate or evidence of hardware deployment.
"""

from __future__ import annotations

import numpy as np


def trajectory_metrics(xy: np.ndarray, *, min_radius: float = 1e-6) -> dict:
    """Summarize planar motion relative to the world-frame origin.

    Parameters
    ----------
    xy:
        Array with shape (n_steps, 2), ordered in time; at least two points.
        Position coordinates must be in meters and finite.
    min_radius:
        Near-origin threshold below which angle-based diagnostics are
        undefined. Angular progress also assumes sufficiently dense samples
        that each inter-sample angular change is less than pi radians.

    Returns
    -------
    dict:
        JSON-safe values. Signed radial drift and signed CCW turn count are
        intentionally preserved. `net_turns_ccw` is None if the path touches
        the singular origin; `radius_cv` is None if mean radius is near zero.
    """
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("xy must have shape (n_steps >= 2, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("xy positions must all be finite")
    if min_radius <= 0 or not np.isfinite(min_radius):
        raise ValueError("min_radius must be finite and positive")

    radius = np.linalg.norm(points, axis=1)
    mean_radius = float(np.mean(radius))
    radius_cv = float(np.std(radius) / mean_radius) if mean_radius > min_radius else None
    if np.any(radius <= min_radius):
        net_turns_ccw = None
    else:
        angles = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
        net_turns_ccw = float((angles[-1] - angles[0]) / (2 * np.pi))

    return {
        "n_samples": int(len(points)),
        "mean_radius_m": mean_radius,
        "radius_std_m": float(np.std(radius)),
        "radius_cv": radius_cv,
        "radial_drift_m": float(radius[-1] - radius[0]),
        "radial_range_m": float(np.ptp(radius)),
        "path_length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
        "net_turns_ccw": net_turns_ccw,
    }
