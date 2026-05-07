param(
    [string]$PythonExe = "python",
    [string]$LogPath = "results\full_run_clean.log",
    [string]$CnnSeeds = "42,52,62",
    [int]$CnnEpochs = 45
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$envNames = @(
    "TASK_FILTER",
    "ASS_REPEAT_SEEDS",
    "ASS_SUPPLEMENTARY_SEED",
    "EVAL_MAX_SAMPLES",
    "HE_TASK_FILTER",
    "RUN_HE_BASELINES",
    "RUN_LOOPBACK_AUDIT",
    "LOOPBACK_TASKS",
    "LOOPBACK_METHODS",
    "LOOPBACK_EVAL_MAX_SAMPLES",
    "FORCE_RETRAIN_MODELS",
    "VALIDATION_CHECK_STRICT",
    "ASS_EXP_REPEATS"
)

foreach ($name in $envNames) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

. .\config\windows_gpu_env.ps1

$env:ASS_REPEAT_SEEDS = "42,52,62"
$env:EVAL_MAX_SAMPLES = "5000"
$env:HE_TASK_FILTER = "MNIST,Fashion,Medical"
$env:RUN_HE_BASELINES = "1"
$env:RUN_LOOPBACK_AUDIT = "1"
$env:LOOPBACK_TASKS = "MNIST,Medical,Digits"
$env:FORCE_RETRAIN_MODELS = "1"
$env:VALIDATION_CHECK_STRICT = "0"

$logFullPath = Join-Path $projectRoot $LogPath
if (Test-Path $logFullPath) {
    Remove-Item $logFullPath -Force
}
Write-Host "Writing log to: $logFullPath"
Write-Host "Python executable: $PythonExe"
Write-Host "Repeat seeds: $env:ASS_REPEAT_SEEDS"
Write-Host "Eval max samples: $env:EVAL_MAX_SAMPLES"
Write-Host "HE task filter: $env:HE_TASK_FILTER"
Write-Host "Loopback tasks: $env:LOOPBACK_TASKS"
Write-Host "Running experiment and redirecting all output to log file..."

$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $PythonExe "core_code\run_experiment.py" *> $logFullPath
$ErrorActionPreference = $previousErrorAction

if ($LASTEXITCODE -ne 0) {
    throw "Full clean experiment run failed."
}

Write-Host "Core experiment completed. Running post-processors and CNN calibration..."

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )
    Write-Host ""
    Write-Host "== $Name =="
    Add-Content -Path $logFullPath -Value ""
    Add-Content -Path $logFullPath -Value "== $Name =="
    & $Command[0] @($Command[1..($Command.Length - 1)]) 2>&1 | Tee-Object -FilePath $logFullPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed."
    }
}

$isicDir = "data\official_medical_images\ISIC2018\official_subset_512_seed42"
$isicLabels = Join-Path $isicDir "ISIC2018_Task3_Training_GroundTruth_subset_512_seed42.csv"

Invoke-Step "HE/PHE boundary postprocess" @($PythonExe, "core_code\experiments\postprocess_boundary_outputs.py", "--results-dir", "results")
Invoke-Step "Secure-CNN official ISIC mainline" @($PythonExe, "core_code\experiments\medical_image_experiment.py", "--dataset", "ISIC2018", "--isic-dir", $isicDir, "--isic-labels", $isicLabels, "--epochs", "3", "--eval-max-samples", "64", "--image-size", "112", "--pool-type", "max", "--force-retrain")
Invoke-Step "Transfer-learning ISIC CNN reference" @($PythonExe, "core_code\experiments\isic_transfer_reference.py", "--seeds", $CnnSeeds, "--epochs", "15")
Invoke-Step "Calibrated multi-seed ISIC CNN protocol" @($PythonExe, "core_code\experiments\isic_cnn_calibration.py", "--seeds", $CnnSeeds, "--epochs", "$CnnEpochs", "--model-variant", "enhanced", "--normalize", "imagenet", "--pool-type", "max", "--secure-eval", "--secure-scale-bits", "-1")
Invoke-Step "FGCS systems postprocess" @($PythonExe, "core_code\experiments\fgcs_systems_postprocess.py", "results")
Invoke-Step "Mainline workload integration" @($PythonExe, "core_code\experiments\integrate_mainline_workloads.py", "--results-dir", "results")
Invoke-Step "Secure operator correctness" @($PythonExe, "core_code\experiments\verify_correctness.py")
Invoke-Step "Publication anomaly audit" @($PythonExe, "core_code\experiments\publication_anomaly_audit.py", "--results-dir", "results")
Invoke-Step "SCI quality gate" @($PythonExe, "core_code\experiments\sci_quality_gate.py", "--results-dir", "results", "--strict")

Write-Host "Full pipeline completed. Tail of log:"
Get-Content $logFullPath -Tail 40
