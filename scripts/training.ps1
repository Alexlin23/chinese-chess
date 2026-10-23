param(
    [ValidateSet('Status', 'Start', 'Resume', 'Stop')][string]$Action = 'Status',
    [string]$RunName,
    [string]$Config = 'config/local-train.yaml',
    [switch]$Performance,
    [int]$Simulations,
    [int]$PublishEvery,
    [int]$ReplayPositions,
    [int]$MaxGamePly,
    [int]$NoiseDepth,
    [int]$NewPositionsPerUpdate,
    [int]$Publications
)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$readConfig = 'from AlphaZero.train.config import AlphaZeroConfig; import sys,json; c=AlphaZeroConfig.from_yaml(sys.argv[1]); print(json.dumps(dict(directory=c.checkpoint_dir,port=c.telemetry_port)))'
$configOutput = & wsl --cd $projectDir --exec bash -c 'exec "$HOME/.venvs/xiangqi-alphazero/bin/python" -c "$1" "$2"' read-config $readConfig $Config
if ($LASTEXITCODE -ne 0) { throw 'Could not read training config.' }
$settings = ($configOutput | Select-Object -Last 1) | ConvertFrom-Json
if ($RunName) {
    if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a simple directory name.' }
    $runRelative = "AlphaZero/runs/$RunName"
} else {
    $runRelative = $settings.directory
}
if ($runRelative.StartsWith('/')) {
    $runDirectory = (& wsl --exec wslpath -w $runRelative).Trim()
} else {
    $runDirectory = Join-Path $projectDir $runRelative
}
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
$overrides = @{Simulations='--num-simulations'; PublishEvery='--checkpoint-interval'; ReplayPositions='--replay-window-positions'; MaxGamePly='--max-game-ply'; NoiseDepth='--search-noise-depth'; NewPositionsPerUpdate='--new-positions-per-update'; Publications='--stop-after-publications'}
foreach ($name in $overrides.Keys) {
    if ($PSBoundParameters.ContainsKey($name)) { $trainArguments += @($overrides[$name], [string]$PSBoundParameters[$name]) }
}
if ($Performance) { $trainArguments += '--performance' }
else { $trainArguments += '--no-performance' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$process = Start-Process -FilePath wsl.exe -ArgumentList $trainArguments -WindowStyle Hidden -RedirectStandardOutput "$runDirectory/training-$stamp.log" -RedirectStandardError "$runDirectory/training-$stamp.stderr.log" -PassThru
$process.Id | Set-Content -LiteralPath "$runDirectory/windows-process.pid"
Write-Output "Training process $($process.Id); dashboard: http://localhost:$($settings.port); output: $runDirectory"
