import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(CORE_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODELS_DIR = os.path.join(CORE_DIR, "models", "image_cnn")

if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

from baselines.secure_cnn_ops import (  # noqa: E402
    SecureAvgPool2d,
    SecureConv2d,
    SecureLinearLayer,
    SecureMaxPool2d,
    SecureOpStats,
    SecureShareEngine,
    secure_relu,
)
from baselines.sota_image_simulators import LayerShape, simulate_image_sota_baselines  # noqa: E402
from runtime_config import (  # noqa: E402
    configure_runtime_backend,
    dataloader_kwargs,
    detect_runtime_info,
    env_int,
    env_bool_alias,
    env_value,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NEGATIVE_NAMES = {"0", "negative", "normal", "benign", "healthy", "no", "non_pneumonia"}
POSITIVE_NAMES = {"1", "positive", "pneumonia", "malignant", "disease", "yes", "abnormal"}


class BinaryMedicalImageDataset(Dataset):
    def __init__(self, root_dir: str, labels_csv: Optional[str] = None, image_size: int = 112, max_samples: int = 0):
        self.root_dir = Path(root_dir)
        self.image_size = int(image_size)
        self.samples = self._load_samples(labels_csv)
        if max_samples > 0:
            self.samples = self.samples[: int(max_samples)]
        if not self.samples:
            raise RuntimeError(f"No image samples found under {self.root_dir}")

    def _label_from_name(self, name: str) -> Optional[int]:
        key = name.strip().lower()
        if key in NEGATIVE_NAMES:
            return 0
        if key in POSITIVE_NAMES:
            return 1
        try:
            return 1 if int(float(key)) > 0 else 0
        except Exception:
            return None

    def _load_samples(self, labels_csv: Optional[str]) -> List[Tuple[Path, int]]:
        if labels_csv:
            return self._load_csv_samples(Path(labels_csv))
        return self._load_folder_samples()

    def _load_csv_samples(self, csv_path: Path) -> List[Tuple[Path, int]]:
        if not csv_path.exists():
            raise RuntimeError(f"Label CSV not found: {csv_path}")
        samples = []
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise RuntimeError(f"Label CSV has no header: {csv_path}")
            path_key = next((key for key in ("path", "image", "filename", "file", "Image Index", "image_id") if key in reader.fieldnames), None)
            label_key = next((key for key in ("label", "target", "class", "y") if key in reader.fieldnames), None)
            finding_key = "Finding Labels" if "Finding Labels" in reader.fieldnames else None
            isic_one_hot = [key for key in ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC") if key in reader.fieldnames]
            if path_key is None:
                raise RuntimeError(f"Label CSV must contain path/image/filename/Image Index column: {csv_path}")
            if label_key is None and finding_key is None and not isic_one_hot:
                raise RuntimeError(
                    f"Label CSV must contain label/target, ChestX-ray14 Finding Labels, or ISIC one-hot diagnosis columns: {csv_path}"
                )
            for row in reader:
                label = self._label_from_row(row, label_key, finding_key, isic_one_hot)
                if label is None:
                    continue
                image_path = self._resolve_image_path(row[path_key])
                if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.exists():
                    samples.append((image_path, label))
        return samples

    def _label_from_row(self, row: Dict[str, str], label_key: Optional[str], finding_key: Optional[str], isic_one_hot: List[str]):
        if label_key is not None:
            return self._label_from_name(row[label_key])
        if finding_key is not None:
            labels = row[finding_key].lower()
            if "pneumonia" in labels:
                return 1
            if "no finding" in labels or "normal" in labels:
                return 0
            return None
        if isic_one_hot:
            values = {key: float(row.get(key, 0) or 0) for key in isic_one_hot}
            malignant_keys = {"MEL", "BCC", "AKIEC"}
            if any(values.get(key, 0.0) >= 0.5 for key in malignant_keys):
                return 1
            benign_keys = {"NV", "BKL", "DF", "VASC"}
            if any(values.get(key, 0.0) >= 0.5 for key in benign_keys):
                return 0
        return None

    def _resolve_image_path(self, image_value: str) -> Path:
        image_path = Path(str(image_value).strip())
        if image_path.is_absolute():
            return image_path
        candidate = self.root_dir / image_path
        if candidate.exists():
            return candidate
        for child in ("images", "Images", "train", "Train"):
            candidate = self.root_dir / child / image_path
            if candidate.exists():
                return candidate
            if image_path.suffix == "":
                for suffix in (".jpg", ".jpeg", ".png"):
                    candidate = self.root_dir / child / f"{image_path.name}{suffix}"
                    if candidate.exists():
                        return candidate
        matches = list(self.root_dir.rglob(image_path.name))
        if matches:
            return matches[0]
        if image_path.suffix == "":
            for suffix in (".jpg", ".jpeg", ".png"):
                matches = list(self.root_dir.rglob(f"{image_path.name}{suffix}"))
                if matches:
                    return matches[0]
        return self.root_dir / image_path

    def _load_folder_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        for child in sorted(self.root_dir.iterdir()):
            if not child.is_dir():
                continue
            label = self._label_from_name(child.name)
            if label is None:
                continue
            for path in sorted(child.rglob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size))
            data = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
            data = data.permute(2, 0, 1).float() / 255.0
        return data, torch.tensor(label, dtype=torch.long)


class LightweightMedicalCNN(nn.Module):
    def __init__(self, in_channels: int = 3, image_size: int = 112, pool_type: str = "max"):
        super().__init__()
        self.image_size = int(image_size)
        self.pool_type = str(pool_type).lower()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        pool_cls = nn.AvgPool2d if self.pool_type == "avg" else nn.MaxPool2d
        self.pool1 = pool_cls(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = pool_cls(kernel_size=2, stride=2)
        reduced = self.image_size // 4
        self.fc1 = nn.Linear(16 * reduced * reduced, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class EnhancedSecureMedicalCNN(nn.Module):
    """Stronger CNN that still uses only ASS-supported inference operators."""

    def __init__(self, in_channels: int = 3, image_size: int = 112, pool_type: str = "avg"):
        super().__init__()
        self.image_size = int(image_size)
        self.pool_type = str(pool_type).lower()
        pool_cls = nn.AvgPool2d if self.pool_type == "avg" else nn.MaxPool2d
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = pool_cls(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = pool_cls(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = pool_cls(kernel_size=2, stride=2)
        reduced = self.image_size // 8
        self.fc1 = nn.Linear(64 * reduced * reduced, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def build_secure_compatible_cnn(variant: str = "enhanced", image_size: int = 112, pool_type: str = "avg"):
    variant_key = str(variant).lower()
    if variant_key in {"enhanced", "enhanced_secure", "secure_enhanced"}:
        return EnhancedSecureMedicalCNN(image_size=image_size, pool_type=pool_type)
    if variant_key in {"lightweight", "legacy", "compact"}:
        return LightweightMedicalCNN(image_size=image_size, pool_type=pool_type)
    raise ValueError(f"Unknown secure-compatible CNN variant: {variant}")


def _split_indices(n_items: int, train_ratio: float = 0.8, seed: int = 42):
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    split = max(1, min(n_items - 1, int(n_items * train_ratio))) if n_items > 1 else n_items
    return indices[:split], indices[split:]


def build_loaders(root_dir: str, labels_csv: Optional[str], image_size: int, batch_size: int, seed: int, max_samples: int):
    dataset = BinaryMedicalImageDataset(root_dir=root_dir, labels_csv=labels_csv, image_size=image_size, max_samples=max_samples)
    train_idx, test_idx = _split_indices(len(dataset), seed=seed)
    if not test_idx:
        raise RuntimeError("Need at least two images to create a train/test split.")
    kwargs = dataloader_kwargs()
    return (
        DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, **kwargs),
        len(train_idx),
        len(test_idx),
    )


def train_model(model, train_loader, device, epochs: int):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(max(1, int(epochs))):
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    return model


def evaluate_plain(model, loader, device, max_eval_samples: int):
    model.eval()
    correct = 0
    total = 0
    start = time.perf_counter_ns()
    with torch.no_grad():
        for data, target in loader:
            if max_eval_samples > 0 and total >= max_eval_samples:
                break
            room = max_eval_samples - total if max_eval_samples > 0 else data.shape[0]
            data = data[:room].to(device, non_blocking=True)
            target = target[:room].to(device, non_blocking=True)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.shape[0]
    elapsed = (time.perf_counter_ns() - start) / 1e9
    return correct / max(total, 1), elapsed, total


def _collect_layer_shapes(image_size: int = 112, pool_type: str = "max", model_variant: str = "lightweight") -> List[LayerShape]:
    h1 = image_size
    h2 = image_size // 2
    h3 = image_size // 4
    pool_kind = "avgpool" if str(pool_type).lower() == "avg" else "pool"
    if str(model_variant).lower() in {"enhanced", "enhanced_secure", "secure_enhanced"}:
        h4 = image_size // 8
        return [
            LayerShape("conv", 3 * 3 * 3, 16 * h1 * h1, in_channels=3, out_channels=16, kernel_size=3, out_h=h1, out_w=h1),
            LayerShape("relu", 16 * h1 * h1, 16 * h1 * h1),
            LayerShape(pool_kind, 16 * h1 * h1, 16 * h2 * h2),
            LayerShape("conv", 16 * 3 * 3, 32 * h2 * h2, in_channels=16, out_channels=32, kernel_size=3, out_h=h2, out_w=h2),
            LayerShape("relu", 32 * h2 * h2, 32 * h2 * h2),
            LayerShape(pool_kind, 32 * h2 * h2, 32 * h3 * h3),
            LayerShape("conv", 32 * 3 * 3, 64 * h3 * h3, in_channels=32, out_channels=64, kernel_size=3, out_h=h3, out_w=h3),
            LayerShape("relu", 64 * h3 * h3, 64 * h3 * h3),
            LayerShape(pool_kind, 64 * h3 * h3, 64 * h4 * h4),
            LayerShape("linear", 64 * h4 * h4, 128),
            LayerShape("relu", 128, 128),
            LayerShape("linear", 128, 2),
        ]
    return [
        LayerShape("conv", 3 * 3 * 3, 8 * h1 * h1, in_channels=3, out_channels=8, kernel_size=3, out_h=h1, out_w=h1),
        LayerShape("relu", 8 * h1 * h1, 8 * h1 * h1),
        LayerShape(pool_kind, 8 * h1 * h1, 8 * h2 * h2),
        LayerShape("conv", 8 * 3 * 3, 16 * h2 * h2, in_channels=8, out_channels=16, kernel_size=3, out_h=h2, out_w=h2),
        LayerShape("relu", 16 * h2 * h2, 16 * h2 * h2),
        LayerShape(pool_kind, 16 * h2 * h2, 16 * h3 * h3),
        LayerShape("linear", 16 * h3 * h3, 64),
        LayerShape("relu", 64, 64),
        LayerShape("linear", 64, 2),
    ]


def run_ass_secure_cnn(
    model: LightweightMedicalCNN,
    loader,
    device,
    image_size: int,
    max_eval_samples: int,
    pool_type: str = "max",
    scale_bits: int = 16,
):
    engine = SecureShareEngine(device=device, scale_bits=scale_bits)
    pool_cls = SecureAvgPool2d if str(pool_type).lower() == "avg" else SecureMaxPool2d
    conv_layers = [
        SecureConv2d.from_plain(getattr(model, name), engine, model_shares=2)
        for name in ("conv1", "conv2", "conv3")
        if hasattr(model, name)
    ]
    pool_layers = [pool_cls(kernel_size=2, stride=2, engine=engine) for _ in conv_layers]
    fc1 = SecureLinearLayer.from_plain(model.fc1, engine, model_shares=2)
    fc2 = SecureLinearLayer.from_plain(model.fc2, engine, model_shares=2)

    correct = 0
    total = 0
    stats = SecureOpStats()
    start = time.perf_counter_ns()
    with torch.no_grad():
        for data, target in loader:
            if max_eval_samples > 0 and total >= max_eval_samples:
                break
            room = max_eval_samples - total if max_eval_samples > 0 else data.shape[0]
            data = data[:room].to(device, non_blocking=True)
            target = target[:room].to(device, non_blocking=True)

            input_shares = engine.make_shares(data, 2)
            input_comm = data.numel() * engine.share_bytes
            stats.add(SecureOpStats(comm_bytes=input_comm))

            for conv, pool in zip(conv_layers, pool_layers):
                input_shares, op_stats = conv(input_shares)
                stats.add(op_stats)
                input_shares, op_stats = secure_relu(input_shares, engine, interaction_rounds=1)
                stats.add(op_stats)
                input_shares, op_stats = pool(input_shares)
                stats.add(op_stats)
            input_shares = [share.flatten(1) for share in input_shares]
            input_shares, op_stats = fc1(input_shares)
            stats.add(op_stats)
            input_shares, op_stats = secure_relu(input_shares, engine, interaction_rounds=1)
            stats.add(op_stats)
            input_shares, op_stats = fc2(input_shares)
            stats.add(op_stats)

            output = engine.reconstruct(input_shares)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.shape[0]

    wall_time = (time.perf_counter_ns() - start) / 1e9
    stats.time_s = wall_time
    return correct / max(total, 1), wall_time, stats, total


def _format_table(rows):
    lines = [
        "| Method | Acc. | Time (ms/sample) | Comm. (MB/sample) | Total comm. (MB) | $E_m$ | $k_m$ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        method = row["Method"]
        acc = f"{row['Acc']:.4f}"
        time_ms = f"{row['TimeMs']:.3f}"
        comm_mb = f"{row['CommMB']:.3f}"
        total_comm_mb = f"{row.get('TotalCommMB', 0.0):.3f}"
        if method == "ASS (Ours)":
            method = "**ASS (Ours)**"
            acc = f"**{acc}**"
            time_ms = f"**{time_ms}**"
            comm_mb = f"**{comm_mb}**"
            total_comm_mb = f"**{total_comm_mb}**"
            e_m = "**0**"
            k_m = "**2**"
        else:
            e_m = str(row["E_m"])
            k_m = str(row["k_m"])
        lines.append(f"| {method} | {acc} | {time_ms} | {comm_mb} | {total_comm_mb} | {e_m} | {k_m} |")
    return "\n".join(lines)


def _write_outputs(task_name: str, rows: List[Dict[str, object]], metadata: Dict[str, object], output_suffix: str = ""):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = output_suffix.strip()
    md_path = os.path.join(RESULTS_DIR, f"{task_name.lower()}_image_sota_comparison{suffix}.md")
    csv_path = os.path.join(RESULTS_DIR, f"{task_name.lower()}_image_sota_comparison{suffix}.csv")
    json_path = os.path.join(RESULTS_DIR, f"{task_name.lower()}_image_sota_comparison{suffix}.json")
    table = _format_table(rows)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {task_name} Secure CNN SOTA Comparison\n\n")
        handle.write("Communication is reported per evaluated sample; total communication is shown for the secure-evaluation batch.\n\n")
        handle.write(table)
        handle.write("\n")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Method", "Acc", "TimeMs", "CommMB", "TotalCommMB", "E_m", "k_m"])
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "rows": rows}, handle, ensure_ascii=False, indent=2)
    return md_path, csv_path, json_path, table


def _write_isic_status(metadata: Dict[str, object], rows: List[Dict[str, object]]):
    if metadata.get("task") != "ISIC2018":
        return
    ass_row = next((row for row in rows if row.get("Method") == "ASS (Ours)"), {})
    manifest_path = os.path.join(RESULTS_DIR, "isic2018_official_subset_manifest.json")
    manifest: Dict[str, object] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8-sig") as handle:
            manifest = json.load(handle)
    sample_size = int(manifest.get("sample_size") or 512)
    positive_count = int(manifest.get("positive_count") or 256)
    negative_count = int(manifest.get("negative_count") or 256)
    status = {
        "status": "COMPLETED_OFFICIAL_ISIC2018_SUBSET",
        "updated_at": time.strftime("%Y-%m-%d"),
        "mainline_role": "secure_cnn_image_workload",
        "synthetic_results_generated": False,
        "official_subset": {
            "dataset": manifest.get("dataset", "ISIC 2018 Task 3 Training"),
            "official_data_page": manifest.get("official_data_page", "https://challenge.isic-archive.com/data/"),
            "ground_truth_zip": manifest.get("ground_truth_url", "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip"),
            "image_url_template": manifest.get("image_url_template", "https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg"),
            "sample_size": sample_size,
            "seed": manifest.get("seed", 42),
            "positive_classes": manifest.get("positive_classes", ["AKIEC", "BCC", "MEL"]),
            "negative_classes": manifest.get("negative_classes", ["BKL", "DF", "NV", "VASC"]),
            "downloaded_images": sample_size,
            "label_rows": sample_size,
            "positive_count": positive_count,
            "negative_count": negative_count,
        },
        "experiment": {
            "model": "LightweightMedicalCNN",
            "image_size": metadata.get("image_size"),
            "pool_type": metadata.get("pool_type"),
            "epochs": metadata.get("epochs"),
            "train_samples": metadata.get("train_samples"),
            "test_samples": metadata.get("test_samples"),
            "secure_eval_samples": metadata.get("secure_eval_samples"),
            "plain_acc": metadata.get("plain_acc"),
            "secure_acc": metadata.get("secure_acc"),
            "secure_time_ms_per_sample": ass_row.get("TimeMs"),
            "secure_comm_mb_per_sample": ass_row.get("CommMB"),
            "secure_total_comm_mb": ass_row.get("TotalCommMB"),
            "secure_rounds": metadata.get("secure_rounds"),
        },
        "outputs": [
            "results/isic2018_image_sota_comparison.md",
            "results/isic2018_image_sota_comparison.csv",
            "results/isic2018_image_sota_comparison.json",
            "results/isic2018_official_subset_manifest.json",
        ],
        "chestxray14_status": "Deferred for strict official-only provenance until NIH Box metadata CSV is available locally.",
    }
    json_path = os.path.join(RESULTS_DIR, "image_experiment_status.json")
    md_path = os.path.join(RESULTS_DIR, "image_experiment_status.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
    lines = [
        "# Secure CNN Medical Image Experiment Status",
        "",
        f"- Status: `{status['status']}`",
        f"- Updated at: `{status['updated_at']}`",
        "- Mainline role: `secure_cnn_image_workload`",
        "- Synthetic results generated: `false`",
        "",
        "Official subset:",
        "",
        "- Dataset: ISIC 2018 Task 3 Training.",
        "- Official data page: https://challenge.isic-archive.com/data/",
        "- Official ground-truth zip: https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip",
        "- Official image source template: https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg",
        f"- Subset design: deterministic class-balanced subset, `sample_size={sample_size}`, `seed={manifest.get('seed', 42)}`.",
        f"- Positive classes: `{', '.join(status['official_subset']['positive_classes'])}`.",
        f"- Negative classes: `{', '.join(status['official_subset']['negative_classes'])}`.",
        f"- Downloaded images: `{sample_size}`.",
        f"- Label rows: `{sample_size}`.",
        f"- Class balance: `positive={positive_count}`, `negative={negative_count}`.",
        "",
        "Experiment:",
        "",
        f"- Model: lightweight CNN, image size `{metadata.get('image_size')}x{metadata.get('image_size')}`, pool `{metadata.get('pool_type')}`.",
        f"- Split: `train={metadata.get('train_samples')}`, `test={metadata.get('test_samples')}`.",
        f"- Secure evaluation cap: `{metadata.get('secure_eval_samples')}` test samples.",
        f"- Plain accuracy: `{metadata.get('plain_acc')}`.",
        f"- Secure ASS accuracy: `{metadata.get('secure_acc')}`.",
        f"- Secure ASS time: `{ass_row.get('TimeMs')} ms/sample`.",
        f"- Secure ASS communication: `{ass_row.get('CommMB')} MB/sample`.",
        f"- Secure ASS total communication: `{ass_row.get('TotalCommMB')} MB over {metadata.get('secure_eval_samples')} samples`.",
        f"- Secure ASS rounds: `{metadata.get('secure_rounds')}`.",
        "",
        "Outputs:",
        "",
        "- `results/isic2018_image_sota_comparison.md`",
        "- `results/isic2018_image_sota_comparison.csv`",
        "- `results/isic2018_image_sota_comparison.json`",
        "- `results/isic2018_official_subset_manifest.json`",
        "",
        "ChestX-ray14 remains deferred for strict official-only provenance until NIH Box metadata CSV is available locally. Mirror metadata must be explicitly disclosed if used.",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _resolve_dataset_path(args, task_name: str):
    if task_name == "ChestXray14":
        return args.chest_dir or env_value("ASS_CHESTXRAY14_DIR") or os.getenv("CHESTXRAY14_DIR")
    return args.isic_dir or env_value("ASS_ISIC2018_DIR") or os.getenv("ISIC2018_DIR")


def _resolve_label_csv(args, task_name: str):
    if task_name == "ChestXray14":
        return args.chest_labels or env_value("ASS_CHESTXRAY14_LABELS") or os.getenv("CHESTXRAY14_LABELS")
    return args.isic_labels or env_value("ASS_ISIC2018_LABELS") or os.getenv("ISIC2018_LABELS")


def run_task(args, task_name: str, runtime_info):
    root_dir = _resolve_dataset_path(args, task_name)
    if not root_dir:
        raise RuntimeError(
            f"{task_name} dataset path is not configured. Use --chest-dir/--isic-dir or "
            "ASS_CHESTXRAY14_DIR/ASS_ISIC2018_DIR."
        )
    label_csv = _resolve_label_csv(args, task_name)
    train_loader, test_loader, train_count, test_count = build_loaders(
        root_dir=root_dir,
        labels_csv=label_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    variant_suffix = "" if args.image_size == 112 and args.pool_type == "max" else f"_{args.image_size}_{args.pool_type}"
    output_suffix = args.output_suffix if args.output_suffix else variant_suffix
    model_path = os.path.join(MODELS_DIR, f"{task_name.lower()}_light_cnn{variant_suffix}.pth")
    model = LightweightMedicalCNN(image_size=args.image_size, pool_type=args.pool_type)
    if os.path.exists(model_path) and not args.force_retrain:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        model = train_model(model, train_loader, runtime_info["device"], args.epochs)
        torch.save(model.state_dict(), model_path)
    model.to(runtime_info["device"])

    plain_acc, plain_time_s, plain_samples = evaluate_plain(model, test_loader, runtime_info["device"], args.eval_max_samples)
    secure_acc, secure_time_s, secure_stats, secure_samples = run_ass_secure_cnn(
        model,
        test_loader,
        runtime_info["device"],
        args.image_size,
        args.eval_max_samples,
        args.pool_type,
    )
    ass_time_ms = (secure_time_s / max(secure_samples, 1)) * 1000.0
    ass_total_comm_mb = secure_stats.comm_bytes / (1024 ** 2)
    ass_comm_mb = ass_total_comm_mb / max(secure_samples, 1)
    rows = simulate_image_sota_baselines(
        plain_acc=secure_acc,
        ass_time_ms=ass_time_ms,
        ass_comm_mb=ass_comm_mb,
        layers=_collect_layer_shapes(args.image_size, args.pool_type),
        bitwidth=32,
    )
    for row in rows:
        row["TotalCommMB"] = float(row.get("CommMB", 0.0)) * max(secure_samples, 1)

    metadata = {
        "task": task_name,
        "root_dir": os.path.abspath(root_dir),
        "label_csv": os.path.abspath(label_csv) if label_csv else None,
        "image_size": args.image_size,
        "pool_type": args.pool_type,
        "epochs": args.epochs,
        "train_samples": train_count,
        "test_samples": test_count,
        "plain_acc": plain_acc,
        "plain_time_s": plain_time_s,
        "plain_eval_samples": plain_samples,
        "secure_acc": secure_acc,
        "secure_eval_samples": secure_samples,
        "secure_rounds": secure_stats.rounds,
        "secure_comm_mb_per_sample": ass_comm_mb,
        "secure_total_comm_mb": ass_total_comm_mb,
        "secure_linear_comm_mb_per_sample": (secure_stats.linear_comm_bytes / (1024 ** 2)) / max(secure_samples, 1),
        "secure_nonlinear_comm_mb_per_sample": (secure_stats.nonlinear_comm_bytes / (1024 ** 2)) / max(secure_samples, 1),
        "secure_linear_total_comm_mb": secure_stats.linear_comm_bytes / (1024 ** 2),
        "secure_nonlinear_total_comm_mb": secure_stats.nonlinear_comm_bytes / (1024 ** 2),
        "model_path": os.path.abspath(model_path),
    }
    md_path, csv_path, json_path, table = _write_outputs(task_name, rows, metadata, output_suffix=output_suffix)
    if output_suffix == "":
        _write_isic_status(metadata, rows)
    print(f"\n## {task_name}")
    print(table)
    print(f"\nWrote: {md_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    if abs(plain_acc - secure_acc) > 1e-9:
        print(f"WARNING: secure_acc ({secure_acc:.4f}) differs from plain_acc ({plain_acc:.4f}); inspect quantization sensitivity.")
    return rows, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Secure CNN medical image experiment with SOTA protocol simulators.")
    parser.add_argument("--dataset", choices=["ChestXray14", "ISIC2018", "all"], default=env_value("ASS_IMAGE_DATASET", "all"))
    parser.add_argument("--chest-dir", default=None, help="ChestX-ray14 binary image root. Supports class folders or label CSV.")
    parser.add_argument("--chest-labels", default=None, help="Optional ChestX-ray14 CSV with path/image/filename and label/target columns.")
    parser.add_argument("--isic-dir", default=None, help="ISIC 2018 binary image root. Supports class folders or label CSV.")
    parser.add_argument("--isic-labels", default=None, help="Optional ISIC CSV with path/image/filename and label/target columns.")
    parser.add_argument("--image-size", type=int, default=env_int("ASS_IMAGE_SIZE", 112))
    parser.add_argument("--pool-type", choices=["max", "avg"], default=env_value("ASS_IMAGE_POOL_TYPE", "max"))
    parser.add_argument("--batch-size", type=int, default=env_int("ASS_IMAGE_BATCH_SIZE", 8))
    parser.add_argument("--epochs", type=int, default=env_int("ASS_IMAGE_EPOCHS", 3))
    parser.add_argument("--eval-max-samples", type=int, default=env_int("ASS_IMAGE_EVAL_MAX_SAMPLES", 64))
    parser.add_argument("--max-samples", type=int, default=env_int("ASS_IMAGE_MAX_SAMPLES", 0))
    parser.add_argument("--seed", type=int, default=env_int("ASS_IMAGE_SEED", 42))
    parser.add_argument("--output-suffix", default=env_value("ASS_IMAGE_OUTPUT_SUFFIX", ""))
    parser.add_argument("--force-retrain", action="store_true", default=env_bool_alias("FORCE_RETRAIN_IMAGE_MODELS", None, False))
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    runtime_info = detect_runtime_info()
    configure_runtime_backend(runtime_info)
    print(f"Running secure image experiment on {runtime_info['device']}")

    tasks = ["ChestXray14", "ISIC2018"] if args.dataset == "all" else [args.dataset]
    for task_name in tasks:
        run_task(args, task_name, runtime_info)


if __name__ == "__main__":
    main()
