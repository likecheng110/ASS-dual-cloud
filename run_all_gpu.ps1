param(
    [string]$PythonExe = "python",
    [string]$EnvFile = "config\\windows_gpu_env.ps1",
    [string]$PreferredDevice = "cuda",
    [switch]$ForceRetrainModels
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "== ASS Full GPU Pipeline =="

& powershell -ExecutionPolicy Bypass -File ".\run_train_gpu.ps1" -PythonExe $PythonExe -EnvFile $EnvFile -PreferredDevice $PreferredDevice
if ($LASTEXITCODE -ne 0) {
    throw "GPU training stage failed."
}

Write-Host ""

if ($ForceRetrainModels.IsPresent) {
    & powershell -ExecutionPolicy Bypass -File ".\run_experiment_gpu.ps1" -PythonExe $PythonExe -EnvFile $EnvFile -PreferredDevice $PreferredDevice -ForceRetrainModels
} else {
    & powershell -ExecutionPolicy Bypass -File ".\run_experiment_gpu.ps1" -PythonExe $PythonExe -EnvFile $EnvFile -PreferredDevice $PreferredDevice
}

if ($LASTEXITCODE -ne 0) {
    throw "GPU experiment stage failed."
}
