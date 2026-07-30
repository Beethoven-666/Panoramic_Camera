param(
    [string]$PythonExe = 'D:\Panoramic_Camera\.conda\python.exe',
    [string]$SourceRoot = 'D:\open3d_cuda_build\Open3D-v0.19.0',
    [string]$BuildRoot = 'D:\open3d_cuda_build\Open3D-v0.19.0\build-cuda12.8-sm120-v3',
    [string]$CudaRoot = 'D:\open3d_cuda_build\cuda128-toolkit\Library',
    [string]$CudaArchitecture = '120',
    [int]$ParallelJobs = 2
)

$ErrorActionPreference = 'Stop'
$patchPath = Join-Path $PSScriptRoot 'patches\open3d-0.19-cuda13-cccl.patch'
$stdgpuPatchPath = Join-Path `
    $PSScriptRoot 'patches\stdgpu-cuda13-device-properties.patch'
$shutdownPatchPath = Join-Path `
    $PSScriptRoot 'patches\open3d-0.19-windows-cuda-shutdown.patch'
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python environment does not exist: $PythonExe"
}
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Visual Studio Build Tools are required'
}
if (-not (Test-Path -LiteralPath (Join-Path $CudaRoot 'bin\nvcc.exe'))) {
    throw "CUDA toolkit does not exist: $CudaRoot"
}
$cudaCompiler = Join-Path $CudaRoot 'bin\nvcc.exe'
$env:CUDA_PATH = $CudaRoot
$env:Path = "$(Join-Path $CudaRoot 'bin');$env:Path"
$env:NVCC_PREPEND_FLAGS = '--diag-suppress=221'
$installation = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) {
    throw 'Visual Studio C++ x64 tools are required'
}
$vsdev = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
$environmentLines = & cmd.exe /d /s /c `
    "`"$vsdev`" -arch=x64 -host_arch=x64 >nul && set"
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable(
            $matches[1], $matches[2], 'Process'
        )
    }
}

& $PythonExe -m pip install 'cmake>=3.29,<4' 'ninja>=1.11,<2'
$scripts = Join-Path (Split-Path -Parent $PythonExe) 'Scripts'
$cmake = Join-Path $scripts 'cmake.exe'
if (-not (Test-Path -LiteralPath $cmake)) {
    throw "CMake executable was not installed under $scripts"
}

if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot '.git'))) {
    $parent = Split-Path -Parent $SourceRoot
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    & git clone --depth 1 --branch v0.19.0 `
        https://github.com/isl-org/Open3D.git $SourceRoot
}
function Apply-Open3DPatchOnce([string]$Path, [string]$Marker, [string]$Label) {
    # Previous failed builds leave the source tree intentionally patched, while
    # their external stdgpu checkout is incomplete.  Detect the semantic marker
    # instead of relying on git's reverse-check, which cannot distinguish that
    # state from an offset patch in Open3D's vendored source.
    $sourceText = Get-Content -LiteralPath $Path -Raw
    if ($sourceText.Contains($Marker)) {
        return
    }
    & git -C $SourceRoot apply $patchPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Label compatibility patch cannot be applied"
    }
}
Apply-Open3DPatchOnce `
    (Join-Path $SourceRoot '3rdparty\stdgpu\stdgpu.cmake') `
    'OPEN3D_THRUST_INCLUDE_DIR' `
    'Open3D CUDA'
if (-not (Get-Content -LiteralPath (Join-Path $SourceRoot 'cpp\open3d\core\MemoryManagerCUDA.cpp') -Raw).Contains('cudaErrorCudartUnloading')) {
    & git -C $SourceRoot apply $shutdownPatchPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Open3D CUDA shutdown compatibility patch cannot be applied'
    }
}

# CUDA 12.8 diagnoses legacy libcu++/stdgpu `1e+300` float casts as
# diagnostic 221.  Open3D 0.19 treats all CUDA diagnostics as errors, so
# suppress this toolkit compatibility diagnostic at CMake configure time
# (NVCC_PREPEND_FLAGS alone is not propagated into generated Ninja rules).
$cudaCompatibilityFlags = '-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler --diag-suppress=221'

& $cmake -S $SourceRoot -B $BuildRoot -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /utf-8" `
    "-DCMAKE_CUDA_COMPILER=$cudaCompiler" `
    "-DPython3_EXECUTABLE=$PythonExe" `
    -DBUILD_PYTHON_MODULE=ON `
    -DBUILD_CUDA_MODULE=ON `
    $cudaCompatibilityFlags `
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitecture" `
    "-DCUDAToolkit_ROOT=$CudaRoot" `
    "-DOPEN3D_CUDA13_STDGPU_PATCH=$stdgpuPatchPath" `
    -DBUILD_COMMON_CUDA_ARCHS=OFF `
    -DDEVELOPER_BUILD=ON `
    -DBUILD_WITH_CUDA_STATIC=OFF `
    -DENABLE_CACHED_CUDA_MANAGER=ON `
    -DSTATIC_WINDOWS_RUNTIME=OFF `
    -DBUILD_GUI=OFF `
    -DBUILD_WEBRTC=OFF `
    -DBUILD_JUPYTER_EXTENSION=OFF `
    -DBUILD_EXAMPLES=OFF `
    -DBUILD_UNIT_TESTS=OFF `
    -DBUILD_BENCHMARKS=OFF `
    -DBUILD_ISPC_MODULE=OFF `
    -DBUILD_AZURE_KINECT=OFF `
    -DBUILD_LIBREALSENSE=OFF `
    -DBUILD_PYTORCH_OPS=OFF `
    -DBUILD_TENSORFLOW_OPS=OFF `
    -DBUNDLE_OPEN3D_ML=OFF
if ($LASTEXITCODE -ne 0) {
    throw "Open3D CUDA CMake configuration failed with exit code $LASTEXITCODE"
}
& $cmake --build $BuildRoot --target pip-package -j $ParallelJobs
if ($LASTEXITCODE -ne 0) {
    throw "Open3D CUDA build failed with exit code $LASTEXITCODE"
}

$wheel = Get-ChildItem -LiteralPath (Join-Path $BuildRoot 'lib') `
    -Filter 'open3d-*.whl' -Recurse |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $wheel) {
    throw 'Open3D CUDA build completed without producing a wheel'
}
Write-Output $wheel.FullName
