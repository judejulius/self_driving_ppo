[CmdletBinding()]
param(
    [string]$Python = "3.11",
    [string]$RacecarRef = "master",
    [string]$TensorOptixRef = "main"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$vendorRoot = Join-Path $projectRoot ".vendor"
$racecarPath = Join-Path $vendorRoot "racecar_gym"
$tensorOptixPath = Join-Path $vendorRoot "tensor-optix"
$compatPatch = Join-Path $projectRoot "patches\racecar-python311.patch"
$env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)] [scriptblock]$Command,
        [Parameter(Mandatory = $true)] [string]$Label
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Set-Location $projectRoot
New-Item -ItemType Directory -Force -Path $vendorRoot | Out-Null

if (-not (Test-Path (Join-Path $racecarPath ".git"))) {
    Invoke-NativeChecked { git clone https://github.com/axelbr/racecar_gym.git $racecarPath } "Racecar Gym clone"
}
if (-not (Test-Path (Join-Path $tensorOptixPath ".git"))) {
    Invoke-NativeChecked { git clone https://github.com/sup3rus3r/tensor-optix.git $tensorOptixPath } "Tensor Optix clone"
}

Invoke-NativeChecked { git -C $racecarPath checkout $RacecarRef } "Racecar Gym checkout"
Invoke-NativeChecked { git -C $tensorOptixPath checkout $TensorOptixRef } "Tensor Optix checkout"

# Upstream Racecar Gym still uses mutable dataclass defaults and NumPy aliases
# removed by modern Python/NumPy. Apply the small compatibility patch once.
$specsFile = Join-Path $racecarPath "racecar_gym\core\specs.py"
if (-not (Select-String -LiteralPath $specsFile -SimpleMatch "default_factory=VehicleSpec" -Quiet)) {
    Invoke-NativeChecked {
        git -C $racecarPath apply --unidiff-zero $compatPatch
    } "Racecar Gym Python 3.11 compatibility patch"
}

if ($env:OS -eq "Windows_NT") {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "Conda is required on Windows because PyBullet does not publish current Windows wheels."
    }
    $condaEnv = Join-Path $projectRoot ".conda-env"
    $pythonExe = Join-Path $condaEnv "python.exe"
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        $env:CONDA_OVERRIDE_CUDA = "0"
        $env:CONDA_PKGS_DIRS = Join-Path $projectRoot ".conda-pkgs"
        Invoke-NativeChecked {
            conda create --solver libmamba --prefix $condaEnv --channel conda-forge `
                --override-channels --repodata-fn current_repodata.json `
                python=3.11 "pybullet=3.25=py311h3e17986_5" pip -y
        } "Conda PyBullet environment"
    }
    Invoke-NativeChecked {
        uv pip install --python $pythonExe --editable ".[dev]"
    } "Project dependency install"
    Invoke-NativeChecked {
        uv pip install --python $pythonExe --no-deps --editable $racecarPath --editable $tensorOptixPath
    } "Upstream editable installs"
    Write-Host "Ready. Smoke test: .\.conda-env\python.exe -m racecar_ppo.cli train --track circle_ccw --episodes 3 --window-size 256"
} else {
    Invoke-NativeChecked { uv sync --python $Python --extra dev } "Dependency sync"
    $venvPath = Join-Path $projectRoot ".venv"
    # The upstream repositories publish mutually incompatible old dependency pins.
    Invoke-NativeChecked {
        uv pip install --python $venvPath --no-deps --editable $racecarPath --editable $tensorOptixPath
    } "Upstream editable installs"
    Write-Host "Ready. Smoke test: uv run racecar-ppo train --track circle_ccw --episodes 3 --window-size 256"
}

