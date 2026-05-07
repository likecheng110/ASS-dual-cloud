# ISIC 2018 Transfer-Learning CNN Reference

This is a plaintext transfer-learning reference for workload validity. It is not claimed as a full secure VGG16 implementation.

| Seed | Test samples | Best epoch | Accuracy | Balanced accuracy | F1 | AUC | Confusion |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 104 | 12 | 0.7981 | 0.7981 | 0.8293 | 0.9197 | [[32, 20], [1, 51]] |
| 52 | 104 | 7 | 0.8269 | 0.8269 | 0.8448 | 0.8931 | [[37, 15], [3, 49]] |
| 62 | 104 | 14 | 0.7885 | 0.7885 | 0.8136 | 0.8439 | [[34, 18], [4, 48]] |

## Summary

- Accuracy mean/std: `0.8045` / `0.0200`
- Balanced accuracy mean/std: `0.8045` / `0.0200`
- F1 mean/std: `0.8292` / `0.0156`
- AUC mean/std: `0.8856` / `0.0385`

## Artifact Use

- Use this as a high-accuracy CNN reference showing that the ISIC subset has learnable image signal.
- Interpret it separately from secure VGG16 execution.
- Pair it with the enhanced secure-compatible CNN experiment for ASS layer support and system overhead.
