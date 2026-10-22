"""Tensor Optix PPO training orchestration."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import make_racecar_env


@dataclass(slots=True)
class TrainConfig:
    track: str = "Austria"
    scenario: str | None = None
    episodes: int = 300
    window_size: int = 1024
    early_episode: int = 5
    seed: int = 7
    output_dir: str = "artifacts"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_ppo_agent(env: Any) -> Any:
    """Build Tensor Optix's continuous Gaussian PPO agent.

    Tensor Optix's generic ``make_agent`` factory intentionally defaults Box
    action spaces to SAC, even when the algorithm label is PPO. Constructing
    the packaged continuous PPO class directly keeps this experiment PPO.
    """
    import torch
    from tensor_optix import HyperparamSet
    from tensor_optix.algorithms.torch_ppo_continuous import TorchGaussianPPOAgent
    from torch import nn

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    def network(output_dim: int, final_gain: float) -> nn.Sequential:
        model = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, output_dim),
        )
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(model[-1].weight, gain=final_gain)
        return model

    actor = network(2 * action_dim, final_gain=0.01)
    # Begin with moderate exploration instead of unit-variance steering.
    actor[-1].bias.data[action_dim:].fill_(-0.5)
    critic = network(1, final_gain=1.0)
    hyperparams = HyperparamSet(
        params={
            "learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "entropy_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_epochs": 10,
            "minibatch_size": 64,
            "max_grad_norm": 0.5,
        },
        episode_id=0,
    )
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=hyperparams.params["learning_rate"],
    )
    return TorchGaussianPPOAgent(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        action_dim=action_dim,
        hyperparams=hyperparams,
        device="auto",
    )


def _snapshot_callback(
    agent: Any, destination: Path, episode: int, metrics_path: Path
) -> Any:
    from tensor_optix import LoopCallback

    class SnapshotCallback(LoopCallback):
        def __init__(self) -> None:
            super().__init__()
            self.saved = False

        def on_episode_end(self, episode_id: int, eval_metrics: Any) -> None:
            if eval_metrics is None:
                return
            score = float(eval_metrics.primary_score)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"update": episode_id, "score": score}) + "\n")
            print(f"[PPO] update={episode_id:04d} score={score:+.5f}", flush=True)
            if not self.saved and episode_id >= episode:
                agent.save_weights(str(destination))
                self.saved = True

    return SnapshotCallback()


def policy_action(agent: Any, observation: np.ndarray) -> np.ndarray:
    """Return the deterministic policy mean, without exploration noise."""
    # Tensor Optix's continuous PPO ``act`` method samples from the Gaussian
    # policy. Evaluation uses the mean so checkpoints compare fairly.
    if all(hasattr(agent, name) for name in ("_actor", "_torch", "_device", "_action_dim")):
        torch = agent._torch
        obs = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32).to(
            agent._device
        )
        with torch.no_grad():
            output = agent._actor(obs)
            action = torch.tanh(output[:, : agent._action_dim])
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    action = agent.act(observation)
    if isinstance(action, tuple):
        action = action[0]
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(-1)


def train(config: TrainConfig) -> Path:
    """Train PPO and save the early snapshot plus Tensor Optix's best restored policy."""
    try:
        from tensor_optix import Optimizer
    except ImportError as exc:
        raise RuntimeError(
            "Tensor Optix is not installed. Run the bootstrap script first."
        ) from exc

    _set_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )

    early_path = output / "ppo_early"
    best_path = output / "ppo_best"
    metrics_path = output / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    train_env = make_racecar_env(track=config.track, scenario=config.scenario)
    agent = make_ppo_agent(train_env)
    agent.save_weights(str(early_path))
    callback = _snapshot_callback(agent, early_path, config.early_episode, metrics_path)

    optimizer = Optimizer(
        agent,
        train_env,
        window_size=config.window_size,
        max_episodes=config.episodes,
        callbacks=[callback],
        rollback_on_degradation=True,
        checkpoint_dir=str(output / "checkpoints"),
        dormant_threshold=10,
        min_episodes_before_dormant=min(50, config.episodes),
    )
    try:
        optimizer.run()
    finally:
        train_env.close()

    # Tensor Optix restores best-known weights on shutdown.
    agent.save_weights(str(best_path))
    return best_path


@dataclass(slots=True)
class EvalResult:
    steps: int
    total_reward: float
    start_progress: float
    end_progress: float
    lap: Any
    collisions: int
    first_collision: int | None
    mean_action: np.ndarray


def evaluate(
    checkpoint: str | Path,
    *,
    track: str = "Austria",
    scenario: str | None = None,
    steps: int = 2_000,
    seed: int = 7,
) -> EvalResult:
    """Run one deterministic rollout of a saved policy and summarize it."""
    env = make_racecar_env(track=track, scenario=scenario)
    try:
        agent = make_ppo_agent(env)
        agent.load_weights(str(checkpoint))
        observation, info = env.reset(seed=seed, options={"mode": "grid"})
        start = float(info.get("progress", 0.0))
        total_reward = 0.0
        collisions = 0
        first_collision: int | None = None
        actions: list[np.ndarray] = []
        step = 0
        for step in range(1, steps + 1):
            action = policy_action(agent, observation)
            actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info.get("wall_collision", False):
                collisions += 1
                first_collision = first_collision or step
            if terminated or truncated:
                break
    finally:
        env.close()
    return EvalResult(
        steps=step,
        total_reward=total_reward,
        start_progress=start,
        end_progress=float(info.get("progress", 0.0)),
        lap=info.get("lap"),
        collisions=collisions,
        first_collision=first_collision,
        mean_action=np.mean(actions, axis=0) if actions else np.zeros(2),
    )
