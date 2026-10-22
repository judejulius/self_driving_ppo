"""Gymnasium adapter from Racecar Gym's nested dictionaries to PPO vectors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from .features import agent_value, encode_observation


class RacecarPPOEnv(gym.Wrapper):
    """Expose a single Racecar Gym car as flat continuous Gymnasium spaces.

    The policy sees 64 normalized LiDAR beams, linear/angular velocity,
    acceleration, and its previous motor/steering command. It outputs
    ``[motor_or_speed, steering]`` in ``[-1, 1]``.
    """

    metadata: ClassVar[dict[str, list[str]]] = {
        "render_modes": ["rgb_array_follow", "human"]
    }

    def __init__(
        self,
        env: gym.Env,
        *,
        agent_id: str = "A",
        lidar_beams: int = 64,
        lidar_range: float | None = None,
        collision_penalty: float = 1.0,
        wrong_way_penalty: float = 0.25,
        steering_penalty: float = 0.002,
    ) -> None:
        super().__init__(env)
        self.agent_id = agent_id
        self.lidar_beams = lidar_beams
        self.lidar_range = lidar_range or self._infer_lidar_range(default=10.0)
        self.collision_penalty = collision_penalty
        self.wrong_way_penalty = wrong_way_penalty
        self.steering_penalty = steering_penalty
        self._previous_action = np.zeros(2, dtype=np.float32)

        feature_count = lidar_beams + 6 + 6 + 2
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(feature_count,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._previous_action.fill(0.0)
        observation, info = self.env.reset(**kwargs)
        return self._encode(observation), self._agent_info(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        observation, reward, terminated, truncated, info = self.env.step(
            self._native_action(action)
        )
        state = self._agent_info(info)
        base_reward = float(agent_value(reward, self.agent_id, default=0.0))
        collision = bool(state.get("wall_collision", False))
        wrong_way = bool(state.get("wrong_way", False))
        shaped_reward = (
            base_reward
            - self.collision_penalty * float(collision)
            - self.wrong_way_penalty * float(wrong_way)
            - self.steering_penalty * float(action[1] ** 2)
        )
        self._previous_action = action
        state.update(
            {
                "ppo/base_reward": base_reward,
                "ppo/shaped_reward": shaped_reward,
                "ppo/wall_collision": collision,
                "ppo/wrong_way": wrong_way,
            }
        )
        return (
            self._encode(observation),
            shaped_reward,
            self._done(terminated),
            self._done(truncated),
            state,
        )

    def _encode(self, observation: Any) -> np.ndarray:
        return encode_observation(
            observation,
            self._previous_action,
            agent_id=self.agent_id,
            lidar_beams=self.lidar_beams,
            lidar_range=self.lidar_range,
        )

    def _native_action(self, action: np.ndarray) -> Any:
        native_space = self.env.action_space
        outer = getattr(native_space, "spaces", {})
        agent_space = outer.get(self.agent_id, native_space) if isinstance(outer, Mapping) else native_space
        actuator_spaces = getattr(agent_space, "spaces", {})
        if not isinstance(actuator_spaces, Mapping):
            return action

        throttle_name = "motor" if "motor" in actuator_spaces else "speed"
        native = {
            throttle_name: self._box_value(actuator_spaces[throttle_name], action[0]),
            "steering": self._box_value(actuator_spaces["steering"], action[1]),
        }
        return {self.agent_id: native} if self.agent_id in outer else native

    @staticmethod
    def _box_value(space: gym.Space, value: float) -> np.ndarray:
        dtype = getattr(space, "dtype", np.float32)
        shape = getattr(space, "shape", (1,)) or (1,)
        return np.full(shape, value, dtype=dtype)

    def _done(self, value: Any) -> bool:
        if isinstance(value, Mapping):
            if self.agent_id in value:
                return bool(value[self.agent_id])
            if "__all__" in value:
                return bool(value["__all__"])
            return any(bool(item) for item in value.values())
        return bool(value)

    def _agent_info(self, info: Any) -> dict[str, Any]:
        if not isinstance(info, Mapping):
            return {"racecar_info": info}
        nested = info.get(self.agent_id)
        return dict(nested if isinstance(nested, Mapping) else info)

    def _infer_lidar_range(self, default: float) -> float:
        space: Any = self.env.observation_space
        spaces = getattr(space, "spaces", {})
        if isinstance(spaces, Mapping) and self.agent_id in spaces:
            space = spaces[self.agent_id]
            spaces = getattr(space, "spaces", {})
        lidar_space = spaces.get("lidar") if isinstance(spaces, Mapping) else None
        high = np.asarray(getattr(lidar_space, "high", ()), dtype=np.float32)
        finite = high[np.isfinite(high)]
        return float(finite.max()) if finite.size and finite.max() > 0 else default


def make_racecar_env(
    *,
    track: str = "Austria",
    scenario: str | Path | None = None,
    agent_id: str = "A",
) -> RacecarPPOEnv:
    """Create a registered Racecar Gym environment and apply the PPO adapter."""
    try:
        import racecar_gym.envs.gym_api  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Racecar Gym is not installed. Run the bootstrap script first."
        ) from exc

    # Racecar Gym treats its default ``human`` mode as a PyBullet GUI
    # connection. Use an RGB mode so PyBullet selects headless DIRECT mode;
    # training never calls render().
    kwargs: dict[str, Any] = {"render_mode": "rgb_array_follow"}
    if scenario:
        env_id = "SingleAgentRaceEnv-v0"
        kwargs["scenario"] = str(Path(scenario).resolve())
    else:
        # Racecar Gym registers filenames with ``str.capitalize()`` (for
        # example ``circle_ccw.yml`` becomes ``SingleAgentCircle_ccw-v0``).
        env_id = f"SingleAgent{track.capitalize()}-v0"

    return RacecarPPOEnv(gym.make(env_id, **kwargs), agent_id=agent_id)

