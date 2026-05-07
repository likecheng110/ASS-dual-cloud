# Experiment Scope

This artifact evaluates ASS as a privacy-preserving inference system for cloud-edge deployment. The experiments are organized around three questions:

1. Whether secure inference preserves the prediction behavior of the corresponding plaintext models.
2. How the protocols compare in latency, communication, throughput, deployment, model exposure, and collusion threshold.
3. Where homomorphic, partially homomorphic, and secure neural-network-style baselines become impractical or favorable under specific workload dimensions.

## Workload Dimensions

The result tables cover small tabular tasks, handwritten-digit tasks, Fashion-MNIST, synthetic long-tail workloads, stress settings, and a secure-compatible CNN medical-image workload. These workloads are used to test different input dimensions, model sizes, class distributions, and communication patterns.

## CNN Evidence

The ISIC 2018 experiments are workload-level evidence for secure CNN inference. The calibrated enhanced CNN is evaluated across stratified seeds, and ASS secure evaluation preserves the corresponding plaintext prediction behavior. The VGG16 transfer-learning result is a plaintext reference showing that the official subset contains learnable visual signal; it is not presented as a secure VGG16 implementation.

The legacy ISIC secure-CNN operator benchmark remains useful for system and operator timing, but the calibrated multi-seed CNN tables are the appropriate source for workload-validity claims.

## HE and PHE Baselines

CKKS and Paillier baselines are included to represent traditional encryption-based secure inference boundaries. Their role is to make the cost trade-off explicit: traditional HE/PHE can provide strong cryptographic protection, but often at substantial computational or communication cost for neural-network workloads.

Small-sample HE/PHE measurements are reported for overhead analysis. They should be read together with sample count, approximation status, and runtime limits.

## Projected System Comparisons

SecureNN, Sonic, Cheetah, and Delphi-style rows are included as documented system-level projections or simulators. They provide a comparison surface for deployment and communication analysis, while directly executable ASS measurements remain separated in the result tables.
