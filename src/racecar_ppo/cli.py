"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .training import TrainConfig, evaluate, train


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--track", default="Austria", help="Racecar Gym track name")
    parser.add_argument("--scenario", help="Path to a custom Racecar Gym YAML scenario")
    parser.add_argument("--seed", type=int, default=7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="racecar-ppo", description="Train and evaluate a Racecar Gym PPO policy"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="train a PPO driving policy")
    _common(train_parser)
    train_parser.add_argument("--output-dir", default="artifacts")
    train_parser.add_argument("--episodes", type=int, default=300)
    train_parser.add_argument("--window-size", type=int, default=1024)
    train_parser.add_argument("--early-episode", type=int, default=5)

    eval_parser = commands.add_parser("evaluate", help="run saved checkpoints deterministically")
    _common(eval_parser)
    eval_parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        default=[Path("artifacts/ppo_early"), Path("artifacts/ppo_best")],
        help="checkpoint directories (default: artifacts/ppo_early artifacts/ppo_best)",
    )
    eval_parser.add_argument("--steps", type=int, default=2_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        best = train(
            TrainConfig(
                track=args.track,
                scenario=args.scenario,
                episodes=args.episodes,
                window_size=args.window_size,
                early_episode=args.early_episode,
                seed=args.seed,
                output_dir=args.output_dir,
            )
        )
        print(f"Saved best policy to {best.resolve()}")
        return

    for checkpoint in args.checkpoints:
        result = evaluate(
            checkpoint, track=args.track, scenario=args.scenario, steps=args.steps, seed=args.seed
        )
        print(
            f"{checkpoint}: steps={result.steps} return={result.total_reward:.3f} "
            f"progress={result.start_progress:.3f}->{result.end_progress:.3f} "
            f"lap={result.lap} collisions={result.collisions} "
            f"first_collision={result.first_collision} mean_action={result.mean_action}"
        )


if __name__ == "__main__":
    main()
