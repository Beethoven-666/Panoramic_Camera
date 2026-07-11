param(
    [switch]$ReuseSystemPackages,
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

if (-not (Test-Path -LiteralPath ".venv")) {
    if ($ReuseSystemPackages) {
        python -m venv .venv --system-site-packages
    } else {
        python -m venv .venv
    }
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install `
    torch==2.13.0+cu130 torchvision==0.28.0+cu130 `
    --index-url https://download.pytorch.org/whl/cu130
& $Python -m pip install -e ".[capture,test]"

if (-not (Test-Path -LiteralPath "third_party\UniStitch")) {
    git clone https://github.com/MmelodYy/UniStitch.git third_party/UniStitch
    git -C third_party/UniStitch checkout 78ebe7c07d516c591810337475ccdd4f2beff384
}

if (-not (Test-Path -LiteralPath "third_party\LightGlue")) {
    git clone https://github.com/cvg/LightGlue.git third_party/LightGlue
    git -C third_party/LightGlue checkout 746fac2c042e05d1865315b1413419f1c1e7ba55
}

if (-not $SkipModel) {
    & $Python scripts/download_unistitch_weights.py
}

Write-Host "Bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1"
