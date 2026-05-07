# core_code

This directory contains the executable training, inference, baseline, and experiment code used by the artifact.

## Main Entry Points

- `run_experiment.py`: main experiment pipeline.
- `train_plain.py`: plaintext model training utilities.
- `experiments/verify_correctness.py`: secure operator correctness checks.
- `experiments/sci_quality_gate.py`: strict consistency gate over aggregate result files.
- `experiments/publication_anomaly_audit.py`: result-table anomaly and interpretation-boundary audit.
- `experiments/medical_image_experiment.py`: secure-compatible CNN operator and system benchmark on official medical images.
- `experiments/isic_cnn_calibration.py`: calibrated multi-seed secure-compatible CNN protocol.
- `experiments/isic_transfer_reference.py`: plaintext VGG16 transfer-learning reference.
- `experiments/fgcs_systems_postprocess.py`: system-level comparison post-processing.
- `experiments/integrate_mainline_workloads.py`: integration of MLP/FNN and CNN workload rows.

## Baselines

- `baselines/inference_ass.py`: ASS secure inference wrapper.
- `baselines/inference_2cloud.py`: two-cloud protocol wrapper.
- `baselines/inference_three_share.py`: three-share protocol wrapper.
- `baselines/inference_securenn.py`: SecureNN-style comparison.
- `baselines/inference_ckks.py`: direct CKKS boundary check.
- `baselines/inference_ckks_benchmark.py`: CKKS micro-benchmark used by the main pipeline.
- `baselines/inference_paillier.py`: Paillier/PHE boundary comparison.
- `baselines/secure_cnn_ops.py`: secure-compatible CNN operators.

The HE/PHE implementations are intentionally treated as boundary baselines. Runtime failures are surfaced as errors instead of being converted into successful zero-time rows.
