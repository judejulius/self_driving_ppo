#!/usr/bin/env bash
# Linux/macOS setup: uv venv + Racecar Gym + Tensor Optix (installed without their stale pins).
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
RACECAR_REF="${RACECAR_REF:-master}"
TENSOR_OPTIX_REF="${TENSOR_OPTIX_REF:-main}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor="$root/.vendor"
racecar="$vendor/racecar_gym"
tensor_optix="$vendor/tensor-optix"
cd "$root"
mkdir -p "$vendor"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

[ -d "$racecar/.git" ] || git clone https://github.com/axelbr/racecar_gym.git "$racecar"
[ -d "$tensor_optix/.git" ] || git clone https://github.com/sup3rus3r/tensor-optix.git "$tensor_optix"
git -C "$racecar" checkout "$RACECAR_REF"
git -C "$tensor_optix" checkout "$TENSOR_OPTIX_REF"

# Upstream Racecar Gym uses mutable dataclass defaults and NumPy aliases removed
# by modern Python/NumPy. Apply the compatibility patch once.
if ! grep -q "default_factory=VehicleSpec" "$racecar/racecar_gym/core/specs.py"; then
    git -C "$racecar" apply --unidiff-zero "$root/patches/racecar-python311.patch"
fi

uv sync --python "$PYTHON_VERSION" --extra dev
# The upstream repositories publish mutually incompatible old dependency pins.
uv pip install --python "$root/.venv" --no-deps --editable "$racecar" --editable "$tensor_optix"

echo "Ready. Smoke test: uv run racecar-ppo train --track circle_ccw --episodes 3 --window-size 256"
