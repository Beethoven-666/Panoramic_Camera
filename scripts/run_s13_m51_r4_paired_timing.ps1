[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineCandidate,
    [Parameter(Mandatory = $true)]
    [string]$Candidate,
    [Parameter(Mandatory = $true)]
    [string]$BaselineSourceRoot,
    [Parameter(Mandatory = $true)]
    [string]$CandidateSourceRoot,
    [Parameter(Mandatory = $true)]
    [string]$TimingOutput,
    [string]$FastInput = 'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140',
    [string]$TrajectoryCache = 'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140\orbslam3_trajectory.json',
    [string]$Python = 'D:\Panoramic_Camera\.conda\python.exe',
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
foreach ($required in @($BaselineCandidate, $Candidate, $BaselineSourceRoot, $CandidateSourceRoot, $FastInput, $TrajectoryCache, $Python)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required paired timing input is missing: $required"
    }
}
if ((Resolve-Path -LiteralPath $BaselineCandidate).Path -eq (Resolve-Path -LiteralPath $Candidate).Path) {
    throw 'Baseline and candidate configs must be distinct files from their respective commits.'
}
foreach ($root in @($BaselineSourceRoot, $CandidateSourceRoot)) {
    $status = & git -C $root status --porcelain
    if ($LASTEXITCODE -ne 0 -or $status) {
        throw "Paired timing source root must be a clean Git worktree: $root"
    }
}
if (-not (Resolve-Path -LiteralPath $BaselineCandidate).Path.StartsWith((Resolve-Path -LiteralPath $BaselineSourceRoot).Path)) {
    throw 'Baseline config must be loaded from the baseline clean worktree.'
}
if (-not (Resolve-Path -LiteralPath $Candidate).Path.StartsWith((Resolve-Path -LiteralPath $CandidateSourceRoot).Path)) {
    throw 'Candidate config must be loaded from the candidate clean worktree.'
}
$baselineCommit = (& git -C $BaselineSourceRoot rev-parse HEAD).Trim()
$candidateCommit = (& git -C $CandidateSourceRoot rev-parse HEAD).Trim()
if ($baselineCommit -eq $candidateCommit) {
    throw 'Baseline and candidate source commits must be distinct.'
}

New-Item -ItemType Directory -Path $TimingOutput | Out-Null

function Invoke-TimedCandidate {
    param(
        [string]$Label,
        [string]$Config,
        [string]$SourceRoot,
        [string]$ExpectedCommit,
        [string]$ExpectedImplementation,
        [string]$RunRoot
    )
    $savedPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = (Join-Path $SourceRoot 'src')
    & $Python -c 'from panorama_demo.video_experiment import main; main()' `
        $FastInput `
        --output $RunRoot `
        --algorithm candidate `
        --candidate-config $Config `
        --trajectory-cache $TrajectoryCache `
        --report-level full `
        --artifact-level audit | Out-Host
    $exitCode = $LASTEXITCODE
    $env:PYTHONPATH = $savedPythonPath
    if ($exitCode -ne 0) {
        throw "Paired timing run failed: $Label (exit $exitCode)"
    }
    $pointer = Get-Content -LiteralPath (Join-Path $RunRoot 'current_latest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($pointer.stage -ne 'P2') {
        throw "Paired timing run did not stop at P2: $Label"
    }
    $performance = Join-Path $RunRoot (Join-Path $pointer.generation 'P2\performance.json')
    $completion = Join-Path $RunRoot (Join-Path $pointer.generation 'P2\P2_completion.json')
    $generationManifest = Join-Path $RunRoot (Join-Path $pointer.generation 'generation_manifest.json')
    $document = Get-Content -LiteralPath $performance -Raw -Encoding utf8 | ConvertFrom-Json
    $generation = Get-Content -LiteralPath $generationManifest -Raw -Encoding utf8 | ConvertFrom-Json
    $seconds = [double]$document.total_m5
    if ([double]::IsNaN($seconds) -or [double]::IsInfinity($seconds) -or $seconds -le 0.0) {
        throw "Paired timing run has invalid total_m5: $Label"
    }
    if ($generation.algorithm.working_tree_dirty -ne $false) {
        throw "Paired timing generation was not clean: $Label"
    }
    if ($generation.algorithm.source_commit -ne $ExpectedCommit -or $generation.algorithm.implementation_id -ne $ExpectedImplementation) {
        throw "Paired timing generation identity mismatch: $Label"
    }
    return [ordered]@{
        seconds = $seconds
        generation_id = $pointer.generation_id
        source_commit = $generation.algorithm.source_commit
        implementation_id = $generation.algorithm.implementation_id
        performance_sha256 = (Get-FileHash -LiteralPath $performance -Algorithm SHA256).Hash.ToLowerInvariant()
        completion_sha256 = (Get-FileHash -LiteralPath $completion -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

# One cold run per implementation is intentionally excluded from the statistics.
$baselineImplementation = 's013_output_first_progressive_dense_central_slit_m51_r3_component_local_ambiguity'
$candidateImplementation = 's013_output_first_progressive_dense_central_slit_m51_r4_component_chain_c2e'
[void](Invoke-TimedCandidate 'cold_baseline' $BaselineCandidate $BaselineSourceRoot $baselineCommit $baselineImplementation (Join-Path $TimingOutput 'cold_baseline'))
[void](Invoke-TimedCandidate 'cold_candidate' $Candidate $CandidateSourceRoot $candidateCommit $candidateImplementation (Join-Path $TimingOutput 'cold_candidate'))

$pairs = @()
for ($index = 0; $index -lt $PairCount; $index++) {
    $pairRoot = Join-Path $TimingOutput ('pair_{0:d2}' -f $index)
    if (($index % 2) -eq 0) {
        $order = 'v6-r1_then_v6-r2'
        $baseline = Invoke-TimedCandidate "pair_${index}_baseline" $BaselineCandidate $BaselineSourceRoot $baselineCommit $baselineImplementation (Join-Path $pairRoot 'baseline')
        $candidateRun = Invoke-TimedCandidate "pair_${index}_candidate" $Candidate $CandidateSourceRoot $candidateCommit $candidateImplementation (Join-Path $pairRoot 'candidate')
    }
    else {
        $order = 'v6-r2_then_v6-r1'
        $candidateRun = Invoke-TimedCandidate "pair_${index}_candidate" $Candidate $CandidateSourceRoot $candidateCommit $candidateImplementation (Join-Path $pairRoot 'candidate')
        $baseline = Invoke-TimedCandidate "pair_${index}_baseline" $BaselineCandidate $BaselineSourceRoot $baselineCommit $baselineImplementation (Join-Path $pairRoot 'baseline')
    }
    $pairs += [ordered]@{
        pair_id = ('pair-{0:d2}' -f $index)
        order = $order
        baseline_seconds = $baseline.seconds
        candidate_seconds = $candidateRun.seconds
        baseline_generation_id = $baseline.generation_id
        candidate_generation_id = $candidateRun.generation_id
        baseline_performance_sha256 = $baseline.performance_sha256
        candidate_performance_sha256 = $candidateRun.performance_sha256
        baseline_completion_sha256 = $baseline.completion_sha256
        candidate_completion_sha256 = $candidateRun.completion_sha256
    }
}

$result = [ordered]@{
    schema = 'gemini305-video-s13-m51-r4-paired-timing-runs/v1'
    pair_count = $pairs.Count
    baseline_config = (Resolve-Path -LiteralPath $BaselineCandidate).Path
    candidate_config = (Resolve-Path -LiteralPath $Candidate).Path
    cold_runs_excluded = $true
    baseline_source_commit = $baselineCommit
    candidate_source_commit = $candidateCommit
    baseline_working_tree_dirty = $false
    candidate_working_tree_dirty = $false
    baseline_implementation_id = $baselineImplementation
    candidate_implementation_id = $candidateImplementation
    pairs = $pairs
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding utf8
Write-Output $resultPath
