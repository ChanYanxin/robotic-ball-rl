import numpy as np

from robotic_ball_rl.rewards import tangential_alignment_reward


def test_ccw_tangent_is_positive_one():
    assert np.isclose(tangential_alignment_reward([1.0, 0.0], [0.0, 2.0]), 1.0)


def test_clockwise_tangent_is_negative_one():
    assert np.isclose(tangential_alignment_reward([1.0, 0.0], [0.0, -2.0]), -1.0)


def test_radial_velocity_is_zero_reward():
    assert np.isclose(tangential_alignment_reward([1.0, 0.0], [3.0, 0.0]), 0.0)


def test_stationary_robot_has_zero_reward():
    assert tangential_alignment_reward([1.0, 2.0], [0.0, 0.0]) == 0.0
