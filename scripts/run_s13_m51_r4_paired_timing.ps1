[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineCandidate,
    [Parameter(Mandatory = $true)]
    [string]$Candidate,
    [Parameter(Mandatory = $true)]
    [string]$TimingOutput,
    [string]$FastInput = 'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140',
    [string]$TrajectoryCache = 'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140\orbslam3_trajectory.json',
    [string]$Experiment = 'D:\Panoramic_Camera\.conda\Scripts\g305-video-experiment.exe',
    [ValidateRange(5, 100)]
    [int]$PairCount = 5
)

$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $TimingOutput) {
    throw "Paired timing output must be new and absent: $TimingOutput"
}
$resultPath = Join-Path (Split-Path -Parent $TimingOutput) 'paired_timing_runs.json'
if (Test-Path -LiteralPath $resultPath) {
    throw "Paired timing result must be new and absent: $resultPath"
}
foreach ($required in @($BaselineCandidate, $Candidate, $FastInput, $TrajectoryCache, $Experiment)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required paired timing input is missing: $required"
    }
}
if ((Resolve-Path -LiteralPath $BaselineCandidate).Path -eq (Resolve-Path -LiteralPath $Candidate).Path) {
    throw 'Baseline and candidate configs must be distinct files from their respective commits.'
}

New-Item -ItemType Directory -Path $TimingOutput | Out-Null

function Invoke-TimedCandidate {
    param(
        [string]$Label,
        [string]$Config,
        [string]$RunRoot
    )
    & $Experiment `
        $FastInput `
        --output $RunRoot `
        --algorithm candidate `
        --candidate-config $Config `
        --trajectory-cache $TrajectoryCache `
        --report-level summary `
        --artifact-level minimal
    if ($LASTEXITCODE -ne 0) {
        throw "Paired timing run failed: $Label (exit $LASTEXITCODE)"
    }
    $pointer = Get-Content -LiteralPath (Join-Path $RunRoot 'current_latest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($pointer.stage -ne 'P2') {
        throw "Paired timing run did not stop at P2: $Label"
    }
    $performance = Join-Path $RunRoot (Join-Path $pointer.generation 'P2\performance.json')
    $document = Get-Content -LiteralPath $performance -Raw -Encoding utf8 | ConvertFrom-Json
    $seconds = [double]$document.total_m5
    if ([double]::IsNaN($seconds) -or [double]::IsInfinity($seconds) -or $seconds -le 0.0) {
        throw "Paired timing run has invalid total_m5: $Label"
    }
    return $seconds
}

# One cold run per implementation is intentionally excluded from the statistics.
[void](Invoke-TimedCandidate 'cold_baseline' $BaselineCandidate (Join-Path $TimingOutput 'cold_baseline'))
[void](Invoke-TimedCandidate 'cold_candidate' $Candidate (Join-Path $TimingOutput 'cold_candidate'))

$pairs = @()
for ($index = 0; $index -lt $PairCount; $index++) {
    $pairRoot = Join-Path $TimingOutput ('pair_{0:d2}' -f $index)
    if (($index % 2) -eq 0) {
        $order = 'v6-r1_then_v6-r2'
        $baseline = Invoke-TimedCandidate "pair_${index}_baseline" $BaselineCandidate (Join-Path $pairRoot 'baseline')
        $candidateSeconds = Invoke-TimedCandidate "pair_${index}_candidate" $Candidate (Join-Path $pairRoot 'candidate')
    }
    else {
        $order = 'v6-r2_then_v6-r1'
        $candidateSeconds = Invoke-TimedCandidate "pair_${index}_candidate" $Candidate (Join-Path $pairRoot 'candidate')
        $baseline = Invoke-TimedCandidate "pair_${index}_baseline" $BaselineCandidate (Join-Path $pairRoot 'baseline')
    }
    $pairs += [ordered]@{
        pair_id = ('pair-{0:d2}' -f $index)
        order = $order
        baseline_seconds = $baseline
        candidate_seconds = $candidateSeconds
    }
}

$result = [ordered]@{
    schema = 'gemini305-video-s13-m51-r4-paired-timing-runs/v1'
    pair_count = $pairs.Count
    baseline_config = (Resolve-Path -LiteralPath $BaselineCandidate).Path
    candidate_config = (Resolve-Path -LiteralPath $Candidate).Path
    cold_runs_excluded = $true
    pairs = $pairs
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding utf8
Write-Output $resultPath
