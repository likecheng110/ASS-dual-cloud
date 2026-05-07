# ISIC 2018 CNN Accuracy Calibration

This calibration run checks whether the official ISIC image branch starts from a meaningful plaintext CNN and whether ASS preserves the same prediction behavior under secure inference.

- Secure-compatible CNN variant: `enhanced`.
- Input normalization: `imagenet`.

| Seed | Test samples | Best epoch | Accuracy | Secure accuracy | Balanced accuracy | F1 | AUC | Secure ms/sample | Secure MB/sample | Confusion |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 104 | 33 | 0.7212 | 0.7212 | 0.7212 | 0.7680 | 0.8051 | 3.2563 | 8.8550 | [[27, 25], [4, 48]] |
| 52 | 104 | 28 | 0.8558 | 0.8558 | 0.8558 | 0.8673 | 0.9083 | 2.9405 | 8.8550 | [[40, 12], [3, 49]] |
| 62 | 104 | 23 | 0.7692 | 0.7692 | 0.7692 | 0.7895 | 0.8210 | 2.9261 | 8.8550 | [[35, 17], [7, 45]] |

## Summary

- Accuracy mean/std: `0.7821` / `0.0682`
- Balanced accuracy mean/std: `0.7821` / `0.0682`
- F1 mean/std: `0.8082` / `0.0522`
- AUC mean/std: `0.8448` / `0.0556`

## Calibrated Secure-CNN System Summary

| Method | Runs | Acc mean | Time ms/sample | Comm MB/sample | E_m | k_m |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Cloud-D | 3 | 0.7821 | 2.5848 | 1.3403 | 1 | 1 |
| ASS (Ours) | 3 | 0.7821 | 3.0409 | 8.8550 | 0 | 2 |
| Cheetah | 3 | 0.7821 | 4.1053 | 155.4761 | 1 | 1 |
| Delphi | 3 | 0.7821 | 8786.4009 | 859.0903 | 1 | 1 |
| Sonic | 3 | 0.7821 | 156.7497 | 26.5329 | 0 | 2 |

## Artifact Use

- Use this table to show that the secure-compatible CNN starts from a meaningful plaintext model and that ASS preserves the same prediction behavior under secure inference.
- Keep the secure-CNN table focused on Conv/ReLU/Pool/Linear support, latency, communication, and privacy exposure.
