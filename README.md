<div align="center">

# Racecar PPO

### Self-Driving Reinforcement Learning with PPO, Racecar Gym & Tensor Optix

Train a continuous-control PPO agent to drive a simulated race car using only LiDAR and motion sensors. Tensor Optix handles the training loop: adaptive hyperparameters, validation checkpoints, rollback on degradation, and best-policy restoration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/judejulius/racecar-ppo?style=social)](https://github.com/judejulius/racecar-ppo/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/judejulius/racecar-ppo?style=social)](https://github.com/judejulius/racecar-ppo/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/judejulius/racecar-ppo)](https://github.com/judejulius/racecar-ppo/issues)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.x-0081A5)](https://gymnasium.farama.org/)
[![PyBullet](https://img.shields.io/badge/PyBullet-3.25-4B8BBE)](https://pybullet.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)

---

**If you find this project useful, please consider giving it a star!** It helps others discover the project.

[**Give it a Star**](https://github.com/judejulius/racecar-ppo) &#11088;

</div>

---

## Table of Contents

- [Why Racecar PPO?](#why-racecar-ppo)
- [Features](#features)
  - [Continuous PPO Policy](#continuous-ppo-policy)
  - [Compact Sensor Observations](#compact-sensor-observations)
  - [Shaped Driving Reward](#shaped-driving-reward)
  - [Managed Training with Tensor Optix](#managed-training-with-tensor-optix)
  - [Deterministic Evaluation](#deterministic-evaluation)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Windows Setup](#windows-setup)
  - [Linux / macOS Setup](#linux--macos-setup)
  - [Smoke Test](#smoke-test)
- [Usage Guide](#usage-guide)
  - [Training](#training)
  - [Evaluating Checkpoints](#evaluating-checkpoints)
  - [Custom Tracks](#custom-tracks)
  - [Training Outputs](#training-outputs)
- [CLI Reference](#cli-reference)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Why Racecar PPO?

Getting an RL agent to drive a car in a physics simulator usually means fighting dependency conflicts, writing observation adapters, and babysitting training runs. **Racecar PPO** packages all of that into one command.

- **One-command setup**: Bootstrap scripts clone and patch the upstream simulators and resolve their conflicting dependency pins for you.
- **Real physics**: Racecar Gym runs on PyBullet with LiDAR, velocity and acceleration sensors at 100 Hz.
- **Hands-off training**: Tensor Optix decides when to checkpoint, when to roll back, and when training has converged.
- **Reproducible**: Seeded training and deterministic, noise-free evaluation of saved policies.
- **Small and readable**: About 600 lines of Python you can read in one sitting.

---

## Features

### Continuous PPO Policy

A Gaussian PPO agent (Tensor Optix's `TorchGaussianPPOAgent`) with separate 256x256 tanh actor and critic networks, orthogonal initialisation, and a reduced initial action variance for steadier early exploration.

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | `3e-4` |
| Clip ratio | `0.2` |
| Entropy coef | `0.01` |
| Value coef | `0.5` |
| Gamma / GAE lambda | `0.99` / `0.95` |
| Epochs / minibatch | `10` / `64` |
| Max grad norm | `0.5` |

---

### Compact Sensor Observations

Racecar Gym's nested multi-agent dictionaries are flattened into a single bounded 78-float vector:

| Feature | Size | Notes |
|---------|------|-------|
| LiDAR | 64 | Evenly resampled from the full scan, normalised to `[0, 1]` |
| Velocity | 6 | Linear + angular, bounded |
| Acceleration | 6 | Linear + angular, bounded |
| Previous action | 2 | Last `[motor, steering]` command |

The action is continuous `[motor_or_speed, steering]`, each clipped to `[-1, 1]`.

---

### Shaped Driving Reward

The Racecar Gym task reward (track progress) with small penalties added:

- **Wall collision**: `-1.0` per step in contact
- **Wrong way**: `-0.25` per step
- **Steering effort**: `-0.002 x steering^2` to discourage oscillation

---

### Managed Training with Tensor Optix

Training runs in fixed-size windows of simulator steps (1,024 by default). After each window, Tensor Optix:

- Scores the policy and **checkpoints** improvements
- **Rolls back** weights when performance degrades
- Enters a **dormant state** and stops early once it detects statistical convergence (after a 50-window warmup)
- **Restores the best-known policy** on shutdown

An early snapshot (`ppo_early`) is saved after a few windows, so you can compare it with the final policy (`ppo_best`).

---

### Deterministic Evaluation

The `evaluate` command runs each checkpoint from the same seeded grid start using the **policy mean**, with no sampling noise. It reports return, track progress, lap, collision count and the first collision step.

---

## Architecture

```text
Tensor Optix Optimizer        windows, scoring, checkpoints, rollback, convergence
        │  act / learn
        ▼
TorchGaussianPPOAgent         actor + critic MLPs (256x256, tanh)
        │  [motor, steering]          ▲  78-float observation
        ▼                             │
RacecarPPOEnv                 gymnasium.Wrapper: features.py + reward shaping
        │  dict actions               ▲  dict sensors (LiDAR, velocity, accel)
        ▼                             │
Racecar Gym  ──►  PyBullet physics @ 100 Hz
```

---

## Quick Start

### Prerequisites

| Requirement | Windows | Linux / macOS |
|-------------|---------|---------------|
| Git | Required | Required |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Required | Required |
| [Conda](https://docs.conda.io/) (Miniconda / Anaconda) | Required, for the prebuilt PyBullet | Not needed |
| Python 3.11 | Installed by Conda | Installed by uv |
| Internet access | First run downloads track assets | Same |

> **Why Conda on Windows?** Current PyBullet releases don't publish Windows wheels, so the bootstrap pulls a precompiled build from conda-forge.

### Windows Setup

```powershell
git clone https://github.com/judejulius/racecar-ppo.git
cd racecar-ppo
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

This creates a local `.conda-env\` with Python 3.11 and PyBullet, clones Racecar Gym and Tensor Optix into `.vendor\`, applies a small Python 3.11 compatibility patch, and installs everything.

### Linux / macOS Setup

```bash
git clone https://github.com/judejulius/racecar-ppo.git
cd racecar-ppo
bash scripts/bootstrap.sh
```

This creates a `.venv/` with uv and does the same clone, patch and install steps.

> **Why `--no-deps` for the upstream repos?** Racecar Gym pins Gymnasium 0.28 / NumPy 1.22, while Tensor Optix requires Gymnasium 1.x / NumPy 1.24+. The bootstrap installs a tested common set from `pyproject.toml`, then installs both repos without their conflicting pins.

### Smoke Test

Check the whole pipeline in about a minute:

```powershell
# Windows
.\.conda-env\python.exe -m racecar_ppo.cli train --track circle_ccw --episodes 3 --window-size 256
.\.conda-env\python.exe -m racecar_ppo.cli evaluate --track circle_ccw --steps 300
```

```bash
# Linux / macOS
uv run racecar-ppo train --track circle_ccw --episodes 3 --window-size 256
uv run racecar-ppo evaluate --track circle_ccw --steps 300
```

You should see `[PPO] update=... score=...` lines, then one summary line per checkpoint.

---

## Usage Guide

> On Windows, replace `uv run racecar-ppo` with `.\.conda-env\python.exe -m racecar_ppo.cli` in the examples below.

### Training

A useful run on the simple oval track:

```bash
uv run racecar-ppo train --track circle_ccw --episodes 160 --window-size 1024
```

On a real circuit (the default track is `Austria`):

```bash
uv run racecar-ppo train --episodes 300
```

`--episodes` counts **Tensor Optix training windows**, not laps. Each window is `--window-size` simulator steps. Training may stop early when Tensor Optix detects convergence; the best policy is always restored before saving.

### Evaluating Checkpoints

Compare the early snapshot with the best policy (default paths):

```bash
uv run racecar-ppo evaluate --track circle_ccw
```

Or pass any checkpoint directories:

```bash
uv run racecar-ppo evaluate --track circle_ccw --steps 5000 artifacts/ppo_best runs/exp2/ppo_best
```

Each checkpoint prints one line:

```
artifacts/ppo_best: steps=300 return=-160.505 progress=0.937->0.923 lap=1 collisions=162 first_collision=76 mean_action=[ 0.0054 -0.0025]
```

(That line comes from a 3-window smoke run, so it's expected to be bad. A properly trained policy should show positive return, increasing progress and few collisions.)

Always evaluate on the same `--track` (or `--scenario`) you trained on.

### Custom Tracks

Built-in tracks come from Racecar Gym (for example `Austria`, `Barcelona`, `Berlin`, `Columbia`, `Gbr`, `Montreal`, `Plechaty`, `Torino`, `Treitlstrasse`, `circle_cw`, `circle_ccw`). To use your own [Racecar Gym scenario](https://github.com/axelbr/racecar_gym#scenarios):

```bash
uv run racecar-ppo train --scenario path/to/scenario.yml --episodes 300
uv run racecar-ppo evaluate --scenario path/to/scenario.yml
```

### Training Outputs

```text
artifacts/
├── run_config.json         # Exact config used for the run
├── training_metrics.jsonl  # One {"update", "score"} line per scored window
├── checkpoints/            # Tensor Optix snapshots + manifest
├── ppo_early/              # Early-training policy (actor.pt, critic.pt)
└── ppo_best/               # Best restored policy (actor.pt, critic.pt)
```

Use `--output-dir` to keep multiple runs apart.

---

## CLI Reference

### `racecar-ppo train`

| Flag | Default | Description |
|------|---------|-------------|
| `--track` | `Austria` | Racecar Gym track name |
| `--scenario` | none | Path to a custom scenario YAML (overrides `--track`) |
| `--episodes` | `300` | Max Tensor Optix training windows |
| `--window-size` | `1024` | Simulator steps per window |
| `--early-episode` | `5` | Window after which `ppo_early` is snapshotted |
| `--seed` | `7` | Random seed |
| `--output-dir` | `artifacts` | Where outputs are written |

### `racecar-ppo evaluate`

| Flag | Default | Description |
|------|---------|-------------|
| `checkpoints...` | `artifacts/ppo_early artifacts/ppo_best` | Checkpoint directories to run |
| `--track` | `Austria` | Racecar Gym track name |
| `--scenario` | none | Path to a custom scenario YAML |
| `--steps` | `2000` | Max simulator steps per rollout |
| `--seed` | `7` | Reset seed |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| RL algorithm | PPO (continuous Gaussian policy) |
| Training orchestration | [Tensor Optix](https://github.com/sup3rus3r/tensor-optix) |
| Simulator | [Racecar Gym](https://github.com/axelbr/racecar_gym) on [PyBullet](https://pybullet.org/) |
| Env API | [Gymnasium](https://gymnasium.farama.org/) 1.x |
| Deep learning | [PyTorch](https://pytorch.org/) |
| Packaging | [uv](https://docs.astral.sh/uv/) + Hatchling, Conda on Windows |
| Testing / lint | pytest, ruff |

---

## Project Structure

```text
racecar-ppo/
├── src/racecar_ppo/
│   ├── cli.py          # `train` and `evaluate` commands
│   ├── env.py          # RacecarPPOEnv wrapper + reward shaping
│   ├── features.py     # LiDAR / motion feature encoding
│   └── training.py     # PPO agent, Tensor Optix loop, evaluation
├── tests/
│   └── test_features.py
├── scripts/
│   ├── bootstrap.ps1   # Windows setup (Conda + uv)
│   └── bootstrap.sh    # Linux / macOS setup (uv)
├── patches/
│   └── racecar-python311.patch   # Python 3.11 / NumPy compatibility for Racecar Gym
├── pyproject.toml
└── uv.lock
```

---

## Running Tests

```bash
uv run pytest            # Linux / macOS
```

```powershell
.\.conda-env\python.exe -m pytest   # Windows
```

The unit tests cover the observation encoder. The [smoke test](#smoke-test) is the end-to-end check, since it needs the simulator and downloaded tracks.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Racecar Gym is not installed` / `Tensor Optix is not installed` | Run the bootstrap script for your OS. |
| `Conda is required on Windows` | Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and make sure `conda` is on your `PATH`. |
| Scripts are blocked in PowerShell | Run `Set-ExecutionPolicy -Scope Process Bypass` in the same session first. |
| First run hangs or fails while downloading | Racecar Gym fetches track assets on first use, so check your network connection. |
| Lots of `starting thread` / `Thread TERMINATED` lines | That's PyBullet's own logging and it's harmless. |
| Gymnasium "share an object" warnings | They come from Racecar Gym's passive env checker and are harmless. |

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Run `uv run ruff check .` and `uv run pytest`
4. Commit your changes and open a pull request

Please open an issue first for larger changes, such as new algorithms or observation spaces.

---

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
Special thanks to [@sup3rus3r](https://github.com/sup3rus3r) for their work on [Tensor Optix](https://github.com/sup3rus3r/tensor-optix).

This project was inspired by their implementation, and their work served as a valuable reference during development.
