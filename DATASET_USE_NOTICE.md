# Dataset Use Notice

This repository does not redistribute raw datasets, medical images, trained weights, or downloaded archives. The code expects users to obtain datasets from their official sources and to comply with the corresponding dataset terms.

## Datasets Used in the Experiments

| Dataset | Role in this codebase | Public research-use basis | Redistribution in this repository |
| --- | --- | --- | --- |
| MNIST | Standard handwritten-digit benchmark for MLP/FNN secure inference | Public benchmark hosted by Yann LeCun, Corinna Cortes, and Christopher Burges, with public train/test files. Source: <https://yann.lecun.org/exdb/mnist/> | Not redistributed |
| Fashion-MNIST | Standard image-classification benchmark for non-medical secure inference | Public benchmark dataset released by Zalando Research under the MIT license. Source: <https://github.com/zalandoresearch/fashion-mnist> | Not redistributed |
| scikit-learn Digits | Lightweight digit workload and local sanity benchmark | Public dataset bundled through `sklearn.datasets.load_digits`; scikit-learn documents it as a copy of the UCI ML hand-written digits test set. Source: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html> | Not redistributed |
| ISIC 2018 | Medical-image workload for secure CNN feasibility and system-boundary evaluation | Public ISIC Challenge data. ISIC lists the 2018 challenge data under CC-BY-NC and gives required citations. Source: <https://challenge.isic-archive.com/data/> | Not redistributed |
| Synthetic medical/tabular workloads | Stress and long-tail protocol checks | Produced locally by the experiment scripts; no external personal or clinical data is embedded | Aggregate outputs only |

## Repository Boundary

The public artifact is limited to code, reproducibility scripts, aggregate result tables, and dataset-use documentation. Raw images, downloaded archives, dataset cache files, trained `.pth`/`.pt` weights, logs, and local-only artifacts are outside the repository boundary.

For ISIC 2018, publications using the artifact should cite the ISIC 2018 challenge and HAM10000 references required by the ISIC data page and describe the usage as non-commercial research benchmarking. Dataset redistribution rights remain governed by the official dataset sources.
