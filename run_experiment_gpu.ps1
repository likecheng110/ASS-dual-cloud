param(
    [string]$PythonExe = "python",
    [string]$EnvFile = "config\\windows_gpu_env.ps1",
    [string]$PreferredDevice = "",
    [int]$Repeats = 0,
    [string]$RepeatSeeds = "",
    [int]$SupplementarySeed = -1,
    [int]$EvalMaxSamples = 0,
    [string]$HeTaskFilter = "",
    [string]$TaskFilter = "",
    [string]$LoopbackTasks = "",
    [int]$DataLoaderWorkers = -1,
    [switch]$DisableLoopbackAudit,
    [switch]$ForceRetrainModels
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (Test-Path $EnvFile) {
    . $EnvFile
}

if ($PreferredDevice) {
    $env:ASS_DEVICE = $PreferredDevice
}
if ($Repeats -gt 0) {
    $env:ASS_EXP_REPEATS = [string]$Repeats
}
if ($RepeatSeeds) {
    $env:ASS_REPEAT_SEEDS = $RepeatSeeds
}
if ($SupplementarySeed -ge 0) {
    $env:ASS_SUPPLEMENTARY_SEED = [string]$SupplementarySeed
}
if ($EvalMaxSamples -gt 0) {
    $env:EVAL_MAX_SAMPLES = [string]$EvalMaxSamples
} elseif (-not $env:EVAL_MAX_SAMPLES) {
    $env:EVAL_MAX_SAMPLES = "5000"
}
if ($HeTaskFilter) {
    $env:RUN_HE_BASELINES = "1"
    $env:HE_TASK_FILTER = $HeTaskFilter
} elseif ($env:HE_TASK_FILTER -and -not (Test-Path $EnvFile)) {
    Remove-Item Env:HE_TASK_FILTER -ErrorAction SilentlyContinue
}
if ($DataLoaderWorkers -ge 0) {
    $env:ASS_DATALOADER_WORKERS = [string]$DataLoaderWorkers
}

if (-not $env:CONDA_DEFAULT_ENV) {
    $pythonParent = Split-Path -Parent $PythonExe
    if ($pythonParent) {
        $envParent = Split-Path -Parent $pythonParent
        if ($envParent -and (Split-Path -Leaf $envParent) -eq "envs") {
            $env:CONDA_DEFAULT_ENV = Split-Path -Leaf $pythonParent
        }
    }
}

if ($TaskFilter) {
    $env:TASK_FILTER = $TaskFilter
} elseif ($env:TASK_FILTER) {
    Remove-Item Env:TASK_FILTER -ErrorAction SilentlyContinue
}
if ($LoopbackTasks) {
    $env:LOOPBACK_TASKS = $LoopbackTasks
} elseif (-not $env:LOOPBACK_TASKS) {
    $env:LOOPBACK_TASKS = "MNIST,Medical,Digits"
}
if ($DisableLoopbackAudit.IsPresent) {
    $env:RUN_LOOPBACK_AUDIT = "0"
} elseif (-not $env:RUN_LOOPBACK_AUDIT) {
    $env:RUN_LOOPBACK_AUDIT = "1"
}

if ($ForceRetrainModels.IsPresent) {
    $env:FORCE_RETRAIN_MODELS = "1"
}

Write-Host "== ASS GPU Environment Check =="
Write-Host "Python executable: $PythonExe"
Write-Host "Conda env: $env:CONDA_DEFAULT_ENV"
Write-Host "Env file: $EnvFile"
Write-Host "Preferred device: $env:ASS_DEVICE"
Write-Host "DataLoader workers: $env:ASS_DATALOADER_WORKERS"
Write-Host "Repeat seeds: $env:ASS_REPEAT_SEEDS"
Write-Host "Supplementary seed: $env:ASS_SUPPLEMENTARY_SEED"
Write-Host "Loopback tasks: $env:LOOPBACK_TASKS"

& $PythonExe "core_code\check_gpu_env.py"
if ($LASTEXITCODE -ne 0) {
    throw "GPU environment check failed."
}

Write-Host ""
Write-Host "== Running ASS Experiment =="
& $PythonExe "core_code\run_experiment.py"
if ($LASTEXITCODE -ne 0) {
    throw "Experiment run failed."
}
