import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16


CORE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CORE_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"

if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from experiments.medical_image_experiment import BinaryMedicalImageDataset  # noqa: E402
from runtime_config import configure_runtime_backend, dataloader_kwargs, detect_runtime_info  # noqa: E402


class ImageSubset(Dataset):
    def __init__(self, base: BinaryMedicalImageDataset, indices: Sequence[int], transform):
        self.samples = [base.samples[int(index)] for index in indices]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), torch.tensor(int(label), dtype=torch.long)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(labels: Sequence[int], seed: int, train_ratio: float):
    by_class: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)
    train_indices: List[int] = []
    test_indices: List[int] = []
    for label, indices in sorted(by_class.items()):
        rng = random.Random(seed + int(label))
        shuffled = list(indices)
        rng.shuffle(shuffled)
        split = int(len(shuffled) * train_ratio)
        train_indices.extend(shuffled[:split])
        test_indices.extend(shuffled[split:])
    random.Random(seed).shuffle(train_indices)
    random.Random(seed + 999).shuffle(test_indices)
    return train_indices, test_indices


def evaluate(model, loader, device):
    model.eval()
    targets: List[int] = []
    preds: List[int] = []
    probs: List[float] = []
    with torch.no_grad():
        for data, target in loader:
            logits = model(data.to(device, non_blocking=True))
            prob = logits.softmax(dim=1)[:, 1].detach().cpu().numpy()
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            probs.extend(float(value) for value in prob)
            preds.extend(int(value) for value in pred)
            targets.extend(int(value) for value in target.numpy())
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "f1": float(f1_score(targets, preds)),
        "auc": float(roc_auc_score(targets, probs)),
        "confusion": confusion_matrix(targets, preds).tolist(),
        "prediction_counts": {str(k): int(v) for k, v in Counter(preds).items()},
        "target_counts": {str(k): int(v) for k, v in Counter(targets).items()},
    }


def run_seed(args, seed: int, runtime_info):
    set_seed(seed)
    weights = VGG16_Weights.DEFAULT
    base = BinaryMedicalImageDataset(args.isic_dir, args.isic_labels, image_size=args.image_size, max_samples=0)
    labels = [int(base[index][1]) for index in range(len(base))]
    train_indices, test_indices = stratified_split(labels, seed, args.train_ratio)
    mean = weights.transforms().mean
    std = weights.transforms().std
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(args.rotation_degrees),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_loader = DataLoader(
        ImageSubset(base, train_indices, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        **dataloader_kwargs(),
    )
    test_loader = DataLoader(
        ImageSubset(base, test_indices, test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs(),
    )

    device = runtime_info["device"]
    model = vgg16(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(True),
        nn.Dropout(args.dropout),
        nn.Linear(512, 2),
    )
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_head, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best = None
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            for parameter in model.features[24:].parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_finetune, weight_decay=args.weight_decay)

        model.train()
        losses: List[float] = []
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        metrics = evaluate(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["loss"] = float(sum(losses) / max(len(losses), 1))
        if best is None or metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best = metrics

    assert best is not None
    train_counts = Counter(labels[index] for index in train_indices)
    test_counts = Counter(labels[index] for index in test_indices)
    return {
        "seed": seed,
        "model": "VGG16_ImageNet_transfer_reference",
        "dataset_size": len(base),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "train_class_0": int(train_counts.get(0, 0)),
        "train_class_1": int(train_counts.get(1, 0)),
        "test_class_0": int(test_counts.get(0, 0)),
        "test_class_1": int(test_counts.get(1, 0)),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "best_epoch": int(best["epoch"]),
        "accuracy": best["accuracy"],
        "balanced_accuracy": best["balanced_accuracy"],
        "f1": best["f1"],
        "auc": best["auc"],
        "confusion": best["confusion"],
        "prediction_counts": best["prediction_counts"],
        "target_counts": best["target_counts"],
        "scope": "plaintext_transfer_reference_not_full_secure_vgg16",
    }


def summarize(rows: List[Dict[str, object]]):
    payload = {}
    for metric in ("accuracy", "balanced_accuracy", "f1", "auc"):
        values = [float(row[metric]) for row in rows]
        payload[f"{metric}_mean"] = float(np.mean(values))
        payload[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return payload


def write_outputs(rows: List[Dict[str, object]], summary: Dict[str, object]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "isic2018_transfer_reference.csv"
    json_path = RESULTS_DIR / "isic2018_transfer_reference.json"
    md_path = RESULTS_DIR / "isic2018_transfer_reference.md"
    fieldnames = [
        "seed",
        "model",
        "dataset_size",
        "train_samples",
        "test_samples",
        "train_class_0",
        "train_class_1",
        "test_class_0",
        "test_class_1",
        "image_size",
        "epochs",
        "best_epoch",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auc",
        "confusion",
        "prediction_counts",
        "scope",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["confusion"] = json.dumps(output["confusion"])
            output["prediction_counts"] = json.dumps(output["prediction_counts"])
            writer.writerow({key: output.get(key) for key in fieldnames})
    json_path.write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# ISIC 2018 Transfer-Learning CNN Reference",
        "",
        "This is a plaintext transfer-learning reference for workload validity. It is not claimed as a full secure VGG16 implementation.",
        "",
        "| Seed | Test samples | Best epoch | Accuracy | Balanced accuracy | F1 | AUC | Confusion |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {test_samples} | {best_epoch} | {accuracy:.4f} | {balanced_accuracy:.4f} | {f1:.4f} | {auc:.4f} | {confusion} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Accuracy mean/std: `{summary['accuracy_mean']:.4f}` / `{summary['accuracy_std']:.4f}`",
            f"- Balanced accuracy mean/std: `{summary['balanced_accuracy_mean']:.4f}` / `{summary['balanced_accuracy_std']:.4f}`",
            f"- F1 mean/std: `{summary['f1_mean']:.4f}` / `{summary['f1_std']:.4f}`",
            f"- AUC mean/std: `{summary['auc_mean']:.4f}` / `{summary['auc_std']:.4f}`",
            "",
            "## Artifact Use",
            "",
            "- Use this as a high-accuracy CNN reference showing that the ISIC subset has learnable image signal.",
            "- Interpret it separately from secure VGG16 execution.",
            "- Pair it with the enhanced secure-compatible CNN experiment for ASS layer support, accuracy preservation, and system overhead.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def parse_args():
    default_isic_dir = PROJECT_DIR / "data" / "official_medical_images" / "ISIC2018" / "official_subset_512_seed42"
    default_labels = default_isic_dir / "ISIC2018_Task3_Training_GroundTruth_subset_512_seed42.csv"
    parser = argparse.ArgumentParser(description="ISIC transfer-learning CNN reference experiment.")
    parser.add_argument("--isic-dir", default=str(default_isic_dir))
    parser.add_argument("--isic-labels", default=str(default_labels))
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--unfreeze-epoch", type=int, default=8)
    parser.add_argument("--rotation-degrees", type=float, default=25.0)
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_info = detect_runtime_info()
    configure_runtime_backend(runtime_info)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    rows = [run_seed(args, seed, runtime_info) for seed in seeds]
    summary = summarize(rows)
    csv_path, json_path, md_path = write_outputs(rows, summary)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
