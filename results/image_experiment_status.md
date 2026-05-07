# Secure CNN Medical Image Experiment Status

- Status: `COMPLETED_OFFICIAL_ISIC2018_SUBSET`
- Updated at: `2026-05-07`
- Mainline role: `secure_cnn_image_workload`
- Synthetic results generated: `false`

Official subset:

- Dataset: ISIC 2018 Task 3 Training.
- Official data page: https://challenge.isic-archive.com/data/
- Official ground-truth zip: https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip
- Official image source template: https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg
- Subset design: deterministic class-balanced subset, `sample_size=512`, `seed=42`.
- Positive classes: `AKIEC, BCC, MEL`.
- Negative classes: `BKL, DF, NV, VASC`.
- Downloaded images: `512`.
- Label rows: `512`.
- Class balance: `positive=256`, `negative=256`.

Experiment:

- Model: lightweight CNN, image size `112x112`, pool `max`.
- Split: `train=409`, `test=103`.
- Secure evaluation cap: `64` test samples.
- Plain accuracy: `0.53125`.
- Secure ASS accuracy: `0.53125`.
- Secure ASS time: `4.088803125 ms/sample`.
- Secure ASS communication: `3.8772201538085938 MB/sample`.
- Secure ASS total communication: `248.14208984375 MB over 64 samples`.
- Secure ASS rounds: `88`.

Outputs:

- `results/isic2018_image_sota_comparison.md`
- `results/isic2018_image_sota_comparison.csv`
- `results/isic2018_image_sota_comparison.json`
- `results/isic2018_official_subset_manifest.json`

ChestX-ray14 remains deferred for strict official-only provenance until NIH Box metadata CSV is available locally. Mirror metadata must be explicitly disclosed if used.
