# ASS-dual-cloud

ASS-dual-cloud contains the reference implementation and reproducibility package for a cloud-edge privacy-preserving inference system. The code evaluates secure MLP/FNN and secure-compatible CNN workloads under accuracy preservation, latency, communication, throughput, deployment, model-exposure, and collusion-threshold metrics.

## Repository Layout

```text
config/                 Windows GPU environment helper
core_code/              Training, secure inference, baselines, and experiment drivers
core_code/baselines/    ASS, two-cloud, three-share, SecureNN, CKKS, and Paillier baselines
core_code/experiments/  Reproduction, post-processing, quality gates, and artifact checks
data/                   Placeholder for locally downloaded datasets
results/                Aggregate result tables used by the artifact
scripts/                Dataset preparation helpers
```

The repository intentionally excludes raw datasets, medical images, downloaded archives, trained model weights, logs, and local-only artifacts.

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the full reproducibility pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_experiment_full_clean.ps1 -PythonExe python -CnnSeeds 42,52,62 -CnnEpochs 45
```

Run validation checks only:

```powershell
python core_code\experiments\verify_correctness.py
python core_code\experiments\sci_quality_gate.py --results-dir results --strict
python core_code\experiments\publication_anomaly_audit.py --results-dir results
```

## Main Artifact Files

- `REPRODUCIBILITY.md`: environment, data placement, and command checklist.
- `DATASET_USE_NOTICE.md`: public dataset sources, research-use basis, and redistribution boundary.
- `EXPERIMENT_SCOPE.md`: interpretation of HE/PHE baselines, CNN evidence, and simulator-based comparisons.
- `core_code/README.md`: runnable code entry points.
- `results/mainline_workloads.csv`: unified workload matrix for MLP/FNN and secure-compatible CNN experiments.
- `results/isic2018_cnn_calibration.md`: calibrated CNN workload validation and secure consistency.
- `results/isic2018_cnn_calibrated_system_summary.csv`: calibrated CNN system-level comparison.
- `results/isic2018_transfer_reference.md`: plaintext VGG16 transfer-learning reference for dataset learnability.

## Experimental Scope

ASS is evaluated as a privacy-preserving inference system rather than as a domain-specific classifier. Medical-image experiments are included to exercise secure CNN-style workloads and system trade-offs; they are not clinical validation experiments.

Homomorphic and partially homomorphic baselines are used as boundary comparisons for computational and communication overhead. Small-sample HE/PHE runs should be interpreted as engineering measurements, while the primary accuracy and workload-validity evidence is provided by the multi-seed plaintext, ASS, and calibrated secure-CNN result tables.

Formula-based SecureNN/Sonic/Cheetah/Delphi-style rows are system-level projections for comparison under documented assumptions. They are kept separate from direct executable ASS measurements in the result tables.
