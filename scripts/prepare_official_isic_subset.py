import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


GROUND_TRUTH_ZIP_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip"
IMAGE_URL_TEMPLATE = "https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg"
OFFICIAL_DATA_PAGE = "https://challenge.isic-archive.com/data/"
LICENSE = "CC-BY-NC"
POSITIVE_CLASSES = {"MEL", "BCC", "AKIEC"}
NEGATIVE_CLASSES = {"NV", "BKL", "DF", "VASC"}


def download_file(url: str, destination: Path, retries: int = 3, timeout: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists, skip: {destination}")
        return

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    curl_path = shutil.which("curl.exe") or shutil.which("curl")
    if curl_path:
        cmd = [
            curl_path,
            "-L",
            "--fail",
            "--retry",
            str(max(0, retries)),
            "--retry-delay",
            "2",
            "--connect-timeout",
            str(max(5, timeout)),
            "--max-time",
            str(max(30, timeout * 2)),
            "-o",
            str(tmp_path),
            url,
        ]
        print(f"download with curl: {url}")
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(destination)
            return
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"curl failed with exit code {result.returncode}; falling back to urllib")

    for attempt in range(1, retries + 1):
        try:
            print(f"download [{attempt}/{retries}]: {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "ASS-review-experiment/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp_path.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(2 * attempt)


def extract_ground_truth(zip_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = target_dir / "ISIC2018_Task3_Training_GroundTruth.csv"
    if expected.exists():
        return expected

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)

    candidates = sorted(target_dir.rglob("ISIC2018_Task3_Training_GroundTruth.csv"))
    if not candidates:
        raise RuntimeError(f"ground-truth CSV not found after extracting {zip_path}")
    return candidates[0]


def read_ground_truth(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty ISIC ground-truth CSV: {csv_path}")
    missing = {"image", *POSITIVE_CLASSES, *NEGATIVE_CLASSES} - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"ISIC ground-truth CSV missing columns {sorted(missing)}: {csv_path}")
    return rows


def row_binary_label(row: Dict[str, str]) -> int:
    if any(float(row.get(key, 0) or 0) >= 0.5 for key in POSITIVE_CLASSES):
        return 1
    if any(float(row.get(key, 0) or 0) >= 0.5 for key in NEGATIVE_CLASSES):
        return 0
    raise ValueError(f"row has no recognized diagnosis flag: {row.get('image')}")


def balanced_sample(rows: Iterable[Dict[str, str]], sample_size: int, seed: int) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    positive = []
    negative = []
    for row in rows:
        label = row_binary_label(row)
        (positive if label == 1 else negative).append(row)

    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)

    target_pos = min(len(positive), sample_size // 2)
    target_neg = min(len(negative), sample_size - target_pos)
    if target_pos + target_neg < sample_size:
        remaining = positive[target_pos:] if target_pos < len(positive) else negative[target_neg:]
        need = sample_size - target_pos - target_neg
        selected = positive[:target_pos] + negative[:target_neg] + remaining[:need]
    else:
        selected = positive[:target_pos] + negative[:target_neg]
    rng.shuffle(selected)

    counts = {"positive": sum(row_binary_label(row) for row in selected)}
    counts["negative"] = len(selected) - counts["positive"]
    return selected, counts


def write_subset_csv(rows: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "image", "source_url", "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            image_id = row["image"]
            label = row_binary_label(row)
            writer.writerow(
                {
                    "path": f"images/{image_id}.jpg",
                    "label": label,
                    "image": image_id,
                    "source_url": IMAGE_URL_TEMPLATE.format(image_id=image_id),
                    "MEL": row.get("MEL", "0"),
                    "NV": row.get("NV", "0"),
                    "BCC": row.get("BCC", "0"),
                    "AKIEC": row.get("AKIEC", "0"),
                    "BKL": row.get("BKL", "0"),
                    "DF": row.get("DF", "0"),
                    "VASC": row.get("VASC", "0"),
                }
            )


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    md_path = path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Official ISIC 2018 Subset Manifest\n\n")
        for key, value in payload.items():
            handle.write(f"- {key}: {value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small official ISIC 2018 Task 3 subset for ASS experiments.")
    parser.add_argument("--target-root", default=r"data\official_medical_images\ISIC2018")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    target_root = Path(args.target_root)
    if not target_root.is_absolute():
        target_root = repo_root / target_root
    target_root = target_root.resolve()

    archive_dir = target_root / "archives"
    gt_zip = archive_dir / "ISIC2018_Task3_Training_GroundTruth.zip"
    gt_dir = target_root / "ground_truth"
    subset_dir = target_root / f"official_subset_{args.sample_size}_seed{args.seed}"
    images_dir = subset_dir / "images"
    labels_csv = subset_dir / f"ISIC2018_Task3_Training_GroundTruth_subset_{args.sample_size}_seed{args.seed}.csv"

    download_file(GROUND_TRUTH_ZIP_URL, gt_zip, retries=args.retries, timeout=args.timeout)
    ground_truth_csv = extract_ground_truth(gt_zip, gt_dir)
    rows = read_ground_truth(ground_truth_csv)
    selected, counts = balanced_sample(rows, sample_size=args.sample_size, seed=args.seed)

    for index, row in enumerate(selected, start=1):
        image_id = row["image"]
        image_path = images_dir / f"{image_id}.jpg"
        url = IMAGE_URL_TEMPLATE.format(image_id=image_id)
        print(f"image {index}/{len(selected)}: {image_id}")
        download_file(url, image_path, retries=args.retries, timeout=args.timeout)

    write_subset_csv(selected, labels_csv)
    manifest = {
        "dataset": "ISIC 2018 Task 3 Training",
        "official_data_page": OFFICIAL_DATA_PAGE,
        "ground_truth_url": GROUND_TRUTH_ZIP_URL,
        "image_url_template": IMAGE_URL_TEMPLATE,
        "license": LICENSE,
        "sample_size": len(selected),
        "seed": args.seed,
        "positive_classes": sorted(POSITIVE_CLASSES),
        "negative_classes": sorted(NEGATIVE_CLASSES),
        "positive_count": counts["positive"],
        "negative_count": counts["negative"],
        "subset_dir": str(subset_dir),
        "labels_csv": str(labels_csv),
    }
    write_manifest(subset_dir / "manifest.json", manifest)

    env_path = target_root.parent / "official_medical_image_env.ps1"
    env_text = (
        f'$env:ASS_ISIC2018_DIR = "{subset_dir}"\n'
        f'$env:ASS_ISIC2018_LABELS = "{labels_csv}"\n'
    )
    env_path.write_text(env_text, encoding="utf-8")

    print("\nPrepared official ISIC subset")
    print(f"subset_dir={subset_dir}")
    print(f"labels_csv={labels_csv}")
    print(f"positive={counts['positive']} negative={counts['negative']}")
    print(f"env_file={env_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
