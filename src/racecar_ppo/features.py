"""Dependency-light feature extraction helpers used by the Gym wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def agent_value(value: Any, agent_id: str = "A", *, default: Any = None) -> Any:
    """Unwrap a Racecar Gym per-agent mapping without damaging sensor dicts."""
    if not isinstance(value, Mapping):
        return value
    if agent_id in value:
        return value[agent_id]
    if "__all__" in value:
        return value["__all__"]
    return default


def sensor_observation(observation: Any, agent_id: str = "A") -> Mapping[str, Any]:
    """Return the single car's sensor dictionary from either Racecar API shape."""
    if not isinstance(observation, Mapping):
        raise TypeError(f"Expected a mapping observation, got {type(observation).__name__}")
    nested = observation.get(agent_id)
    if isinstance(nested, Mapping):
        return nested
    return observation


def resample_lidar(values: Any, count: int, max_range: float) -> np.ndarray:
    """Evenly sample and normalize a LiDAR scan to a stable feature count."""
    lidar = np.asarray(values, dtype=np.float32).reshape(-1)
    if count <= 0:
        raise ValueError("count must be positive")
    if not np.isfinite(max_range) or max_range <= 0:
        raise ValueError("max_range must be finite and positive")
    if lidar.size == 0:
        return np.zeros(count, dtype=np.float32)

    lidar = np.nan_to_num(lidar, nan=max_range, posinf=max_range, neginf=0.0)
    positions = np.linspace(0, lidar.size - 1, num=count)
    sampled = np.interp(positions, np.arange(lidar.size), lidar)
    return np.clip(sampled / max_range, 0.0, 1.0).astype(np.float32)


def bounded_vector(values: Any, count: int, scale: float) -> np.ndarray:
    """Pad/truncate a sensor vector and smoothly map it to [-1, 1]."""
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    target = np.zeros(count, dtype=np.float32)
    copied = min(count, source.size)
    if copied:
        target[:copied] = np.nan_to_num(source[:copied], nan=0.0, posinf=scale, neginf=-scale)
    return np.tanh(target / scale).astype(np.float32)


def encode_observation(
    observation: Any,
    previous_action: Any,
    *,
    agent_id: str = "A",
    lidar_beams: int = 64,
    lidar_range: float = 10.0,
) -> np.ndarray:
    """Build the compact, track-agnostic feature vector consumed by PPO."""
    sensors = sensor_observation(observation, agent_id)
    lidar = resample_lidar(sensors.get("lidar", ()), lidar_beams, lidar_range)
    velocity = bounded_vector(sensors.get("velocity", ()), 6, scale=10.0)
    acceleration = bounded_vector(sensors.get("acceleration", ()), 6, scale=20.0)
    action = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    action_feature = np.zeros(2, dtype=np.float32)
    action_feature[: min(2, action.size)] = action[:2]
    return np.concatenate((lidar, velocity, acceleration, action_feature), dtype=np.float32)

