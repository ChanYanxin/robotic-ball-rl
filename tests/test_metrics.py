import numpy as np
import pytest

from robotic_ball_rl.metrics import trajectory_metrics


def test_one_counterclockwise_circle_has_no_radial_drift():
    angles = np.linspace(0.0, 2.0 * np.pi, 101)
    xy = np.column_stack((2.0 * np.cos(angles), 2.0 * np.sin(angles)))
    result = trajectory_metrics(xy)

    assert result["n_samples"] == 101
    assert np.isclose(result["mean_radius_m"], 2.0)
    assert np.isclose(result["radius_std_m"], 0.0, atol=1e-12)
    assert np.isclose(result["radial_drift_m"], 0.0, atol=1e-12)
    assert np.isclose(result["net_turns_ccw"], 1.0)
    assert result["path_length_m"] > 12.5


def test_outward_spiral_has_positive_radial_drift():
    angles = np.linspace(0.0, np.pi, 21)
    radius = np.linspace(1.0, 2.5, 21)
    xy = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    result = trajectory_metrics(xy)

    assert np.isclose(result["radial_drift_m"], 1.5)
    assert np.isclose(result["radial_range_m"], 1.5)
    assert result["radius_cv"] > 0.0
    assert np.isclose(result["net_turns_ccw"], 0.5)


def test_clockwise_trajectory_has_negative_turn_count():
    angles = np.linspace(0.0, -np.pi, 21)
    xy = np.column_stack((np.cos(angles), np.sin(angles)))
    assert np.isclose(trajectory_metrics(xy)["net_turns_ccw"], -0.5)


def test_origin_does_not_produce_fake_angular_progress():
    result = trajectory_metrics(np.zeros((3, 2)))
    assert result["net_turns_ccw"] is None
    assert result["radius_cv"] is None


@pytest.mark.parametrize("xy", [np.zeros((1, 2)), np.zeros((3, 3)), [1, 2]])
def test_invalid_shapes_raise(xy):
    with pytest.raises(ValueError, match="shape"):
        trajectory_metrics(xy)


def test_nonfinite_positions_raise():
    with pytest.raises(ValueError, match="finite"):
        trajectory_metrics([[1.0, 0.0], [np.nan, 1.0]])
