param(
    [ValidateSet('Status', 'Start', 'Resume', 'Stop')][string]$Action = 'Status',
    [string]$RunName = 'xiangqi-v3-20261015',
    [string]$Config = 'config/local-train.yaml'
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a simple directory name.' }
$projectDir = Split-Path -Parent $PSScriptRoot
$runRelative = "AlphaZero/runs/$RunName"
$runDirectory = Join-Path $projectDir $runRelative
if ($Action -in @('Status','Stop')) {
    & wsl --cd $projectDir --exec bash -c 'exec "$HOME/.venvs/xiangqi-alphazero/bin/python" -m AlphaZero.scripts.training_control "$@"' control $Action.ToLower() --run-dir $runRelative
    if ($LASTEXITCODE -ne 0) { throw 'Could not control this training run.' }
    if ($Action -eq 'Status' -and (Test-Path -LiteralPath "$runDirectory/metrics.jsonl")) {
        Get-Content -LiteralPath "$runDirectory/metrics.jsonl" -Tail 1
    }
    return
}
if ($Action -eq 'Start' -and (Test-Path -LiteralPath "$runDirectory/learner.pt")) { throw 'Run exists. Use Resume or choose another RunName.' }
if ($Action -eq 'Resume' -and -not (Test-Path -LiteralPath "$runDirectory/learner.pt")) { throw 'No learner checkpoint to resume.' }
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$trainArguments = @('--cd', $projectDir, '--exec', 'bash', 'scripts/train.sh', '--config', $Config, '--output-dir', $runRelative)
if ($Action -eq 'Resume') { $trainArguments += @('--resume', "$runRelative/learner.pt") }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$process = Start-Process -FilePath wsl.exe -ArgumentList $trainArguments -WindowStyle Hidden -RedirectStandardOutput "$runDirectory/training-$stamp.log" -RedirectStandardError "$runDirectory/training-$stamp.stderr.log" -PassThru
$process.Id | Set-Content -LiteralPath "$runDirectory/windows-process.pid"
Write-Output "Training process $($process.Id); dashboard: http://localhost:8765; output: $runDirectory"
