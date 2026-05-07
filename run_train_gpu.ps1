param(
    [string]$PythonExe = "python",
    [string]$EnvFile = "config\\windows_gpu_env.ps1",
    [string]$PreferredDevice = ""
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

if (-not $env:CONDA_DEFAULT_ENV) {
    $pythonParent = Split-Path -Parent $PythonExe
    if ($pythonParent) {
        $envParent = Split-Path -Parent $pythonParent
        if ($envParent -and (Split-Path -Leaf $envParent) -eq "envs") {
            $env:CONDA_DEFAULT_ENV = Split-Path -Leaf $pythonParent
        }
    }
}

Write-Host "== ASS GPU Train =="
Write-Host "Python executable: $PythonExe"
Write-Host "Conda env: $env:CONDA_DEFAULT_ENV"
Write-Host "Env file: $EnvFile"
Write-Host "Preferred device: $env:ASS_DEVICE"
Write-Host "DataLoader workers: $env:ASS_DATALOADER_WORKERS"

& $PythonExe "core_code\check_gpu_env.py"
if ($LASTEXITCODE -ne 0) {
    throw "GPU environment check failed."
}

Write-Host ""
Write-Host "== Training Plain Models =="
& $PythonExe "core_code\train_plain.py"
if ($LASTEXITCODE -ne 0) {
    throw "Model training failed."
}
