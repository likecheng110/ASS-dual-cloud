import argparse
import csv
import json
import os
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


CORE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CORE_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"

if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from experiments.medical_image_experiment import (  # noqa: E402
    BinaryMedicalImageDataset,
    _collect_layer_shapes,
    build_secure_compatible_cnn,
    run_ass_secure_cnn,
)
from baselines.sota_image_simulators import simulate_image_sota_baselines  # noqa: E402
from runtime_config import configure_runtime_backend, dataloader_kwargs, detect_runtime_info  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TransformSubset(Dataset):
    def __init__(self, base: BinaryMedicalImageDataset, indices: Sequence[int], transform):
        self.samples = [base.samples[int(index)] for index in indices]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), torch.tensor(int(label), dtype=torch.long)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(labels: Sequence[int], seed: int, train_ratio: float = 0.8):
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
            data = data.to(device, non_blocking=True)
            logits = model(data)
            prob = logits.softmax(dim=1)[:, 1].detach().cpu().numpy()
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            probs.extend(float(value) for value in prob)
            preds.extend(int(value) for value in pred)
            targets.extend(int(value) for value in target.numpy())
    accuracy = float(accuracy_score(targets, preds))
    balanced_accuracy = float(balanced_accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds))
    auc = float(roc_auc_score(targets, probs))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "auc": auc,
        "confusion": confusion_matrix(targets, preds).tolist(),
        "prediction_counts": {str(k): int(v) for k, v in Counter(preds).items()},
        "target_counts": {str(k): int(v) for k, v in Counter(targets).items()},
    }


def run_seed(args, seed: int, runtime_info):
    set_seed(seed)
    base_dataset = BinaryMedicalImageDataset(
        root_dir=args.isic_dir,
        labels_csv=args.isic_labels,
        image_size=args.image_size,
        max_samples=0,
    )
    labels = [int(base_dataset[index][1]) for index in range(len(base_dataset))]
    train_indices, test_indices = stratified_split(labels, seed=seed, train_ratio=args.train_ratio)

    train_steps = [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(args.rotation_degrees),
        transforms.ColorJitter(
            brightness=args.color_jitter,
            contrast=args.color_jitter,
            saturation=args.color_jitter * 0.7,
            hue=0.05,
        ),
        transforms.ToTensor(),
    ]
    test_steps = [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ]
    if args.normalize == "imagenet":
        train_steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        test_steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    train_transform = transforms.Compose(train_steps)
    test_transform = transforms.Compose(test_steps)
    train_loader = DataLoader(
        TransformSubset(base_dataset, train_indices, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        **dataloader_kwargs(),
    )
    test_loader = DataLoader(
        TransformSubset(base_dataset, test_indices, test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs(),
    )

    device = runtime_info["device"]
    model = build_secure_compatible_cnn(args.model_variant, image_size=args.image_size, pool_type=args.pool_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
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
        scheduler.step()

        metrics = evaluate(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["loss"] = float(sum(losses) / max(len(losses), 1))
        if best is None or metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best is not None
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)

    secure_metrics = {}
    if args.secure_eval:
        secure_acc, secure_time_s, secure_stats, secure_samples = run_ass_secure_cnn(
            model=model,
            loader=test_loader,
            device=device,
            image_size=args.image_size,
            max_eval_samples=args.secure_eval_samples,
            pool_type=args.pool_type,
            scale_bits=args.secure_scale_bits,
        )
        secure_metrics = {
            "secure_accuracy": float(secure_acc),
            "secure_eval_samples": int(secure_samples),
            "secure_time_ms_per_sample": float((secure_time_s / max(secure_samples, 1)) * 1000.0),
            "secure_comm_mb_per_sample": float((secure_stats.comm_bytes / (1024 ** 2)) / max(secure_samples, 1)),
            "secure_total_comm_mb": float(secure_stats.comm_bytes / (1024 ** 2)),
            "secure_rounds": int(secure_stats.rounds),
            "secure_scale_bits": int(args.secure_scale_bits),
        }

    train_counts = Counter(labels[index] for index in train_indices)
    test_counts = Counter(labels[index] for index in test_indices)
    row = {
        "seed": seed,
        "dataset_size": len(base_dataset),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "train_class_0": int(train_counts.get(0, 0)),
        "train_class_1": int(train_counts.get(1, 0)),
        "test_class_0": int(test_counts.get(0, 0)),
        "test_class_1": int(test_counts.get(1, 0)),
        "image_size": args.image_size,
        "pool_type": args.pool_type,
        "model_variant": args.model_variant,
        "normalize": args.normalize,
        "epochs": args.epochs,
        "best_epoch": int(best["epoch"]),
        "accuracy": best["accuracy"],
        "balanced_accuracy": best["balanced_accuracy"],
        "f1": best["f1"],
        "auc": best["auc"],
        "confusion": best["confusion"],
        "prediction_counts": best["prediction_counts"],
        "target_counts": best["target_counts"],
    }
    row.update(secure_metrics)
    return row


def summarize(rows: List[Dict[str, object]]):
    metrics = ["accuracy", "balanced_accuracy", "f1", "auc"]
    summary = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return summary


def build_system_rows(rows: List[Dict[str, object]], bitwidth: int = 32):
    system_rows: List[Dict[str, object]] = []
    for row in rows:
        if row.get("secure_accuracy") is None:
            continue
        simulated = simulate_image_sota_baselines(
            plain_acc=float(row["secure_accuracy"]),
            ass_time_ms=float(row["secure_time_ms_per_sample"]),
            ass_comm_mb=float(row["secure_comm_mb_per_sample"]),
            layers=_collect_layer_shapes(
                int(row["image_size"]),
                str(row["pool_type"]),
                str(row.get("model_variant", "lightweight")),
            ),
            bitwidth=bitwidth,
        )
        secure_samples = int(row.get("secure_eval_samples") or row["test_samples"])
        for item in simulated:
            system_rows.append(
                {
                    "Seed": row["seed"],
                    "Dataset": "ISIC2018",
                    "ImageSize": row["image_size"],
                    "PoolType": row["pool_type"],
                    "ModelVariant": row.get("model_variant", "lightweight"),
                    "TrainSamples": row["train_samples"],
                    "TestSamples": row["test_samples"],
                    "SecureEvalSamples": secure_samples,
                    "Method": item["Method"],
                    "Acc": item["Acc"],
                    "TimeMs": item["TimeMs"],
                    "CommMB": item["CommMB"],
                    "TotalCommMB": float(item.get("CommMB", 0.0)) * secure_samples,
                    "E_m": item["E_m"],
                    "k_m": item["k_m"],
                    "ComparisonScope": "same_cnn_prediction_behavior_formula_based_system_projection",
                }
            )
    return system_rows


def summarize_system_rows(system_rows: List[Dict[str, object]]):
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in system_rows:
        grouped.setdefault(str(row["Method"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for method, rows in sorted(grouped.items()):
        def values(key):
            return [float(row[key]) for row in rows if row.get(key) not in (None, "")]

        acc_values = values("Acc")
        time_values = values("TimeMs")
        comm_values = values("CommMB")
        total_comm_values = values("TotalCommMB")
        first = rows[0]
        summary_rows.append(
            {
                "Method": method,
                "Seeds": ",".join(str(row["Seed"]) for row in rows),
                "Runs": len(rows),
                "AccMean": float(np.mean(acc_values)) if acc_values else None,
                "AccStd": float(np.std(acc_values, ddof=1)) if len(acc_values) > 1 else 0.0,
                "TimeMsMean": float(np.mean(time_values)) if time_values else None,
                "TimeMsStd": float(np.std(time_values, ddof=1)) if len(time_values) > 1 else 0.0,
                "CommMBMean": float(np.mean(comm_values)) if comm_values else None,
                "CommMBStd": float(np.std(comm_values, ddof=1)) if len(comm_values) > 1 else 0.0,
                "TotalCommMBMean": float(np.mean(total_comm_values)) if total_comm_values else None,
                "E_m": first.get("E_m"),
                "k_m": first.get("k_m"),
                "ComparisonScope": first.get("ComparisonScope"),
            }
        )
    return summary_rows


def write_outputs(rows: List[Dict[str, object]], summary: Dict[str, object], system_rows=None, system_summary_rows=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "isic2018_cnn_calibration.csv"
    json_path = RESULTS_DIR / "isic2018_cnn_calibration.json"
    md_path = RESULTS_DIR / "isic2018_cnn_calibration.md"
    system_rows = system_rows or []
    system_summary_rows = system_summary_rows or []

    fieldnames = [
        "seed",
        "dataset_size",
        "train_samples",
        "test_samples",
        "train_class_0",
        "train_class_1",
        "test_class_0",
        "test_class_1",
        "image_size",
        "pool_type",
        "model_variant",
        "normalize",
        "epochs",
        "best_epoch",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auc",
        "secure_accuracy",
        "secure_eval_samples",
        "secure_time_ms_per_sample",
        "secure_comm_mb_per_sample",
        "secure_total_comm_mb",
        "secure_rounds",
        "secure_scale_bits",
        "confusion",
        "prediction_counts",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["confusion"] = json.dumps(out["confusion"])
            out["prediction_counts"] = json.dumps(out["prediction_counts"])
            writer.writerow({key: out.get(key) for key in fieldnames})

    payload = {"rows": rows, "summary": summary, "system_rows": system_rows, "system_summary": system_summary_rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if system_rows:
        system_by_seed_path = RESULTS_DIR / "isic2018_cnn_calibrated_system_by_seed.csv"
        fieldnames_system = [
            "Seed",
            "Dataset",
            "ImageSize",
            "PoolType",
            "ModelVariant",
            "TrainSamples",
            "TestSamples",
            "SecureEvalSamples",
            "Method",
            "Acc",
            "TimeMs",
            "CommMB",
            "TotalCommMB",
            "E_m",
            "k_m",
            "ComparisonScope",
        ]
        with system_by_seed_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames_system)
            writer.writeheader()
            writer.writerows(system_rows)

    if system_summary_rows:
        system_summary_path = RESULTS_DIR / "isic2018_cnn_calibrated_system_summary.csv"
        fieldnames_summary = [
            "Method",
            "Seeds",
            "Runs",
            "AccMean",
            "AccStd",
            "TimeMsMean",
            "TimeMsStd",
            "CommMBMean",
            "CommMBStd",
            "TotalCommMBMean",
            "E_m",
            "k_m",
            "ComparisonScope",
        ]
        with system_summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames_summary)
            writer.writeheader()
            writer.writerows(system_summary_rows)

    lines = [
        "# ISIC 2018 CNN Accuracy Calibration",
        "",
        "This calibration run checks whether the official ISIC image branch starts from a meaningful plaintext CNN and whether ASS preserves the same prediction behavior under secure inference.",
        "",
        f"- Secure-compatible CNN variant: `{rows[0].get('model_variant', 'unknown') if rows else 'unknown'}`.",
        f"- Input normalization: `{rows[0].get('normalize', 'unknown') if rows else 'unknown'}`.",
        "",
        "| Seed | Test samples | Best epoch | Accuracy | Secure accuracy | Balanced accuracy | F1 | AUC | Secure ms/sample | Secure MB/sample | Confusion |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {test_samples} | {best_epoch} | {accuracy:.4f} | {secure_accuracy} | {balanced_accuracy:.4f} | {f1:.4f} | {auc:.4f} | {secure_ms} | {secure_mb} | {confusion} |".format(
                seed=row["seed"],
                test_samples=row["test_samples"],
                best_epoch=row["best_epoch"],
                accuracy=row["accuracy"],
                secure_accuracy=f"{float(row['secure_accuracy']):.4f}" if row.get("secure_accuracy") is not None else "NA",
                balanced_accuracy=row["balanced_accuracy"],
                f1=row["f1"],
                auc=row["auc"],
                secure_ms=f"{float(row['secure_time_ms_per_sample']):.4f}" if row.get("secure_time_ms_per_sample") is not None else "NA",
                secure_mb=f"{float(row['secure_comm_mb_per_sample']):.4f}" if row.get("secure_comm_mb_per_sample") is not None else "NA",
                confusion=row["confusion"],
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
            "## Calibrated Secure-CNN System Summary",
            "",
        ]
    )
    if system_summary_rows:
        lines.extend(
            [
                "| Method | Runs | Acc mean | Time ms/sample | Comm MB/sample | E_m | k_m |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in system_summary_rows:
            lines.append(
                "| {Method} | {Runs} | {AccMean:.4f} | {TimeMsMean:.4f} | {CommMBMean:.4f} | {E_m} | {k_m} |".format(
                    **row
                )
            )
        lines.append("")
    else:
        lines.extend(["- Not generated. Run with `--secure-eval`.", ""])
    lines.extend(
        [
            "## Artifact Use",
            "",
            "- Use this table to show that the secure-compatible CNN starts from a meaningful plaintext model and that ASS preserves the same prediction behavior under secure inference.",
            "- Keep the secure-CNN table focused on Conv/ReLU/Pool/Linear support, latency, communication, and privacy exposure.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def parse_args():
    default_isic_dir = PROJECT_DIR / "data" / "official_medical_images" / "ISIC2018" / "official_subset_512_seed42"
    default_labels = default_isic_dir / "ISIC2018_Task3_Training_GroundTruth_subset_512_seed42.csv"
    parser = argparse.ArgumentParser(description="ISIC CNN accuracy calibration for artifact interpretation.")
    parser.add_argument("--isic-dir", default=str(default_isic_dir))
    parser.add_argument("--isic-labels", default=str(default_labels))
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--pool-type", choices=["avg", "max"], default="avg")
    parser.add_argument("--model-variant", choices=["lightweight", "enhanced"], default="enhanced")
    parser.add_argument("--normalize", choices=["none", "imagenet"], default="imagenet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rotation-degrees", type=float, default=25.0)
    parser.add_argument("--color-jitter", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--secure-eval", action="store_true", help="Run ASS secure-CNN evaluation on the best checkpoint.")
    parser.add_argument("--secure-eval-samples", type=int, default=0, help="0 means evaluate all test samples.")
    parser.add_argument("--secure-scale-bits", type=int, default=-1, help="-1 uses float-share simulation without fixed-point quantization.")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_info = detect_runtime_info()
    configure_runtime_backend(runtime_info)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    rows = [run_seed(args, seed, runtime_info) for seed in seeds]
    summary = summarize(rows)
    system_rows = build_system_rows(rows) if args.secure_eval else []
    system_summary_rows = summarize_system_rows(system_rows) if system_rows else []
    csv_path, json_path, md_path = write_outputs(rows, summary, system_rows, system_summary_rows)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
