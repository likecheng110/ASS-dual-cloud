# Reproducibility

This artifact provides executable code, helper scripts, and aggregate result files. Raw datasets, medical images, downloaded archives, trained weights, logs, and local-only artifacts are excluded from version control.

## Environment

Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

The experiments were organized for Python with PyTorch, torchvision, NumPy, pandas, scikit-learn, TenSEAL, and python-paillier. GPU execution is optional. On Windows, the helper scripts load `config/windows_gpu_env.ps1` when present.

## Data Placement

Datasets should be downloaded from their official sources and placed under local cache paths:

```text
data/
data/official_medical_images/
core_code/models/
```

These paths are ignored by Git to prevent accidental redistribution of datasets and model weights. See `DATASET_USE_NOTICE.md` for dataset sources and use boundaries.

## Full Pipeline

```powershell
powershell -ExecutionPolicy Bypass -File .\run_experiment_full_clean.ps1 -PythonExe python -CnnSeeds 42,52,62 -CnnEpochs 45
```

The script trains or refreshes task models when needed, runs ASS and baseline measurements, integrates the CNN workload evidence, rebuilds aggregate tables, and executes the final quality gates.

## Validation Commands

```powershell
python core_code\experiments\verify_correctness.py
python core_code\experiments\sci_quality_gate.py --results-dir results --strict
python core_code\experiments\publication_anomaly_audit.py --results-dir results
```

Expected validation status:

- Secure operator correctness: pass.
- Strict consistency gate: `PASS`.
- Publication anomaly audit: no critical findings; warnings document interpretation boundaries for HE/PHE, high-dimensional communication projections, and legacy ISIC operator benchmarks.

## Result Interpretation

The main reproducibility target is the consistency of system-level trade-offs across accuracy preservation, latency, communication, throughput, deployment, model exposure, and collusion-threshold metrics. HE/PHE rows are included to show the cost boundary of traditional encryption-based inference, not to serve as the primary accuracy evidence.
