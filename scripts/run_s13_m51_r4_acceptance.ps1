[CmdletBinding()]
param(
    [string]$Acceptance = 'D:\central_strip_Panoramic_Camera\artifacts\S013_M5_1_r4_v6_component_chain_c2e',
    [string]$Candidate = 'D:\central_strip_Panoramic_Camera\configs\video_candidates\s013\S013_output_first_progressive_dense_central_slit_v6.yaml',
    [string]$Experiment = 'D:\Panoramic_Camera\.conda\Scripts\g305-video-experiment.exe',
    [string]$FastInput = 'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140',
    [string]$SlowInput = 'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033'
)

$ErrorActionPreference = 'Stop'

if (Test-Path -LiteralPath $Acceptance) {
    throw "Acceptance output must be new and absent: $Acceptance"
}
foreach ($required in @($Candidate, $Experiment, $FastInput, $SlowInput)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required acceptance input is missing: $required"
    }
}

$runs = @(
    @{
        Name = 'fast_direct'
        Input = $FastInput
        Extra = @('--trajectory-cache', (Join-Path $FastInput 'orbslam3_trajectory.json'))
    },
    @{
        Name = 'fast_ignore_pose'
        Input = $FastInput
        Extra = @('--ignore-pose')
    },
    @{
        Name = 'slow_direct'
        Input = $SlowInput
        Extra = @('--trajectory-cache', (Join-Path $SlowInput 'orbslam3_trajectory.json'))
    },
    @{
        Name = 'slow_ignore_pose'
        Input = $SlowInput
        Extra = @('--ignore-pose')
    }
)

foreach ($round in @('round_a', 'round_b')) {
    foreach ($run in $runs) {
        $output = Join-Path (Join-Path $Acceptance $round) $run.Name
        $arguments = @(
            $run.Input,
            '--output', $output,
            '--algorithm', 'candidate',
            '--candidate-config', $Candidate,
            '--report-level', 'full',
            '--artifact-level', 'audit'
        ) + $run.Extra
        & $Experiment @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Acceptance run failed: $round/$($run.Name) (exit $LASTEXITCODE)"
        }
        $pointer = Join-Path $output 'current_latest.json'
        if (-not (Test-Path -LiteralPath $pointer)) {
            throw "Acceptance run did not publish current_latest.json: $round/$($run.Name)"
        }
        $current = Get-Content -LiteralPath $pointer -Raw -Encoding utf8 | ConvertFrom-Json
        if ($current.stage -ne 'P2') {
            throw "Acceptance run did not stop at P2: $round/$($run.Name)"
        }
    }
}

Write-Output "Completed the 4 branches x 2 rounds matrix at $Acceptance"
Write-Output 'Create paired_timing_runs.json from the interleaved warm benchmark, then run:'
Write-Output "  python scripts/verify_s13_m51_r4_reproducible_seal.py --root `"$Acceptance`" --output `"$Acceptance\reproducible_seal.json`""
