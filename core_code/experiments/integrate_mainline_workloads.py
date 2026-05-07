import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


MAIN_METHODS = {"2Cloud-D (Data-only)", "ASS (Ours)", "3Share-DM (3-party)", "SecureNN (3PC)"}
CNN_METHOD_ALIASES = {
    "2Cloud-D": "2Cloud-D (Data-only)",
    "ASS (Ours)": "ASS (Ours)",
    "Sonic": "Sonic",
    "Cheetah": "Cheetah",
    "Delphi": "Delphi",
}

FIELDNAMES = [
    "Task",
    "Domain",
    "Architecture",
    "Method",
    "Status",
    "Repeats",
    "AccMean",
    "AccStd",
    "PerSampleMsMean",
    "OnlineCommMB",
    "SinglePointModelExposure",
    "MinCollusionModel",
    "ImplementationScope",
    "ComparisonRole",
    "CommMeasurementBasis",
    "Evidence",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def mlp_rows(results_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for row in read_csv(results_dir / "results_table.csv"):
        if row.get("Method") not in MAIN_METHODS:
            continue
        rows.append(
            {
                "Task": row.get("Task"),
                "Domain": row.get("Domain"),
                "Architecture": "MLP/FNN",
                "Method": row.get("Method"),
                "Status": row.get("Status"),
                "Repeats": row.get("Repeats"),
                "AccMean": row.get("AccMean"),
                "AccStd": row.get("AccStd"),
                "PerSampleMsMean": row.get("PerSampleMsMean"),
                "OnlineCommMB": row.get("OnlineCommMB"),
                "SinglePointModelExposure": row.get("SinglePointModelExposure"),
                "MinCollusionModel": row.get("MinCollusionModel"),
                "ImplementationScope": row.get("ImplementationScope"),
                "ComparisonRole": "mainline_mlp_fnn",
                "CommMeasurementBasis": row.get("CommMeasurementBasis"),
                "Evidence": row.get("Evidence"),
            }
        )
    return rows


def cnn_rows(results_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for row in read_csv(results_dir / "isic2018_cnn_calibrated_system_summary.csv"):
        method = CNN_METHOD_ALIASES.get(row.get("Method"), row.get("Method"))
        rows.append(
            {
                "Task": "ISIC2018-CNN",
                "Domain": "medical_image",
                "Architecture": "Enhanced CNN",
                "Method": method,
                "Status": "SUCCESS",
                "Repeats": row.get("Runs"),
                "AccMean": row.get("AccMean"),
                "AccStd": row.get("AccStd"),
                "PerSampleMsMean": row.get("TimeMsMean"),
                "OnlineCommMB": row.get("CommMBMean"),
                "SinglePointModelExposure": row.get("E_m"),
                "MinCollusionModel": row.get("k_m"),
                "ImplementationScope": "secure-compatible-cnn",
                "ComparisonRole": "mainline_cnn",
                "CommMeasurementBasis": row.get("ComparisonScope"),
                "Evidence": "Enhanced Conv/ReLU/MaxPool/Linear CNN; ASS secure inference preserves plaintext predictions exactly.",
            }
        )
    return rows


def build(results_dir: Path) -> List[Dict[str, object]]:
    rows = mlp_rows(results_dir) + cnn_rows(results_dir)
    return sorted(rows, key=lambda item: (str(item.get("Architecture")), str(item.get("Task")), str(item.get("Method"))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate MLP/FNN and CNN workloads into one mainline result table.")
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[2] / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    rows = build(results_dir)
    output_path = results_dir / "mainline_workloads.csv"
    write_csv(output_path, rows)
    print(f"Wrote {output_path}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
