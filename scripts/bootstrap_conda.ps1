param(
    [switch]$SkipModel,
    [switch]$Recreate,
    [switch]$WithUnistitchDiagnostic
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Environment = Join-Path $Root ".conda"
Set-Location -LiteralPath $Root

if ($Recreate -and (Test-Path -LiteralPath $Environment)) {
    conda env remove --prefix $Environment --yes
}

if (-not (Test-Path -LiteralPath (Join-Path $Environment "python.exe"))) {
    conda env create --prefix $Environment --file environment.yml
}

conda run --prefix $Environment python -m pip install --upgrade pip setuptools wheel
conda run --prefix $Environment python -m pip install -e ".[capture,test]"

if ($WithUnistitchDiagnostic) {
    conda run --prefix $Environment python -m pip install `
        torch==2.13.0+cu130 torchvision==0.28.0+cu130 `
        --index-url https://download.pytorch.org/whl/cu130
    conda run --prefix $Environment python -m pip install -e ".[unistitch-diagnostic]"

    if (-not (Test-Path -LiteralPath "third_party\UniStitch")) {
        git clone https://github.com/MmelodYy/UniStitch.git third_party/UniStitch
        git -C third_party/UniStitch checkout 78ebe7c07d516c591810337475ccdd4f2beff384
    }

    if (-not (Test-Path -LiteralPath "third_party\LightGlue")) {
        git clone https://github.com/cvg/LightGlue.git third_party/LightGlue
        git -C third_party/LightGlue checkout 746fac2c042e05d1865315b1413419f1c1e7ba55
    }

    if (-not $SkipModel) {
        conda run --prefix $Environment python scripts/download_unistitch_weights.py
    }
}

Write-Host "Conda environment is ready: $Environment"
Write-Host "Activate with: conda activate $Environment"
