from __future__ import annotations

import numpy as np

from racecar_ppo.features import agent_value, encode_observation, resample_lidar


def test_agent_value_handles_multi_agent_and_scalar_values() -> None:
    assert agent_value({"A": 1.5}, "A") == 1.5
    assert agent_value({"__all__": True}, "A") is True
    assert agent_value(2.0, "A") == 2.0


def test_lidar_is_resampled_and_clipped() -> None:
    result = resample_lidar([0.0, 5.0, np.inf, 20.0], count=4, max_range=10.0)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0, 1.0])


def test_observation_encoding_has_stable_bounded_shape() -> None:
    observation = {
        "A": {
            "lidar": np.linspace(0.0, 10.0, 1080),
            "velocity": [1, 2, 3, 4, 5, 6],
            "acceleration": [0, 1, 2, 3, 4, 5],
        }
    }
    encoded = encode_observation(observation, [0.2, -0.4], lidar_beams=64, lidar_range=10)
    assert encoded.shape == (78,)
    assert encoded.dtype == np.float32
    assert np.all(encoded >= -1.0)
    assert np.all(encoded <= 1.0)
    np.testing.assert_allclose(encoded[-2:], [0.2, -0.4])

