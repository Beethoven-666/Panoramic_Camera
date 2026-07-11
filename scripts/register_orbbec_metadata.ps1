param(
    [string]$Python
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

if (-not $Python) {
    $CondaPython = Join-Path $Root ".conda\python.exe"
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $CondaPython) {
        $Python = $CondaPython
    } elseif (Test-Path -LiteralPath $VenvPython) {
        $Python = $VenvPython
    } else {
        throw "No project Python found. Run bootstrap_conda.ps1 or bootstrap_windows.ps1 first."
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable does not exist: $Python"
}

$PackageDir = & $Python -c "import pathlib, pyorbbecsdk; print(pathlib.Path(pyorbbecsdk.__file__).resolve().parent)"
$Setup = Join-Path $PackageDir "shared\setup_env.py"
if (-not (Test-Path -LiteralPath $Setup)) {
    throw "Could not find the Orbbec environment setup script at $Setup"
}

& $Python $Setup
