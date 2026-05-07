import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROBABILITY_COLUMNS = {
    "Acc",
    "Accuracy",
    "AccMean",
    "AccStd",
    "AccCI95Low",
    "AccCI95High",
    "TailRecall",
    "PlainAcc",
    "SecureAcc",
    "GeneralMeanAcc",
    "MedicalMeanAcc",
}

NONNEGATIVE_COLUMNS = {
    "EvalSamples",
    "TimingSamples",
    "Repeats",
    "PerSampleMs",
    "PerSampleMsMean",
    "PerSampleMsStd",
    "TimeCI95Low",
    "TimeCI95High",
    "TimeMs",
    "ComputeMs",
    "TransferMs",
    "RoundTripMs",
    "ProjectedE2EMs",
    "Throughput",
    "ThroughputMean",
    "ProjectedQPS",
    "OnlineCommMB",
    "OfflineSetupMB",
    "CommMB",
    "LinearPct",
    "ReLUPct",
    "ComputeSharePct",
    "TransferSharePct",
    "RoundTripSharePct",
    "InteractionRounds",
    "Parties",
    "Batch",
    "Samples",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NONE", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _issue(severity: str, code: str, message: str, source: str, row: Optional[Dict[str, str]] = None):
    payload = {
        "severity": severity,
        "code": code,
        "source": source,
        "message": message,
    }
    if row:
        payload.update(
            {
                "task": row.get("Task"),
                "method": row.get("Method"),
                "status": row.get("Status"),
            }
        )
    return payload


def _audit_numeric_tables(results_dir: Path) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    for path in sorted(results_dir.glob("*.csv")):
        rows = _read_csv(path)
        for row_idx, row in enumerate(rows, start=2):
            label = f"{path.name}:{row_idx}"
            for column in sorted(PROBABILITY_COLUMNS | NONNEGATIVE_COLUMNS):
                if column not in row:
                    continue
                value = _float(row.get(column))
                if value is None:
                    continue
                if not math.isfinite(value):
                    issues.append(_issue("critical", "NON_FINITE", f"{column} is not finite: {row.get(column)}", label, row))
                    continue
                if column in PROBABILITY_COLUMNS and not 0.0 <= value <= 1.0:
                    issues.append(_issue("critical", "PROBABILITY_OUT_OF_RANGE", f"{column}={value} is outside [0,1].", label, row))
                if column in NONNEGATIVE_COLUMNS and value < 0.0:
                    issues.append(_issue("critical", "NEGATIVE_METRIC", f"{column}={value} is negative.", label, row))

            for low, high in (("AccCI95Low", "AccCI95High"), ("TimeCI95Low", "TimeCI95High")):
                low_value = _float(row.get(low))
                high_value = _float(row.get(high))
                if low_value is not None and high_value is not None and low_value > high_value:
                    issues.append(_issue("critical", "INVALID_CI", f"{low}={low_value} > {high}={high_value}.", label, row))

            status = str(row.get("Status") or "").upper()
            if status == "SUCCESS":
                for column in ("PerSampleMs", "PerSampleMsMean", "TimeMs"):
                    if column in row:
                        value = _float(row.get(column))
                        if value is not None and value <= 0.0:
                            issues.append(_issue("critical", "NONPOSITIVE_SUCCESS_TIME", f"{column}={value} for SUCCESS row.", label, row))
            if status == "APPROX":
                acc = _float(row.get("AccMean"))
                eval_samples = _float(row.get("EvalSamples"))
                if acc is not None and acc >= 0.999 and eval_samples is not None and eval_samples <= 5:
                    issues.append(
                        _issue(
                            "warning",
                            "APPROX_SMALL_SAMPLE_PERFECT_ACC",
                            "Approximate Paillier/PHE boundary run reports perfect accuracy on <=5 samples; cite only as engineering boundary timing, not accuracy evidence.",
                            label,
                            row,
                        )
                    )
    return issues


def _audit_fgcs_boundary(results_dir: Path) -> List[Dict[str, object]]:
    rows = _read_csv(results_dir / "fgcs_j2sp_vs_baselines_projection.csv")
    faster = [
        row
        for row in rows
        if _float(row.get("SecureNNOverASS")) is not None and _float(row.get("SecureNNOverASS")) < 1.0
    ]
    if not faster:
        return []
    tasks = sorted({str(row.get("Task")) for row in faster if row.get("Task")})
    scenarios = sorted({str(row.get("Scenario")) for row in faster if row.get("Scenario")})
    return [
        _issue(
            "warning",
            "FGCS_HIGH_DIMENSION_COMM_BOUNDARY",
            (
                f"SecureNN projects below ASS in {len(faster)}/{len(rows)} rows "
                f"(tasks={tasks}, scenarios={scenarios}). Present this as a high-dimensional communication boundary, not as an error."
            ),
            "fgcs_j2sp_vs_baselines_projection.csv",
        )
    ]


def _audit_isic_positioning(results_dir: Path) -> List[Dict[str, object]]:
    status = _read_json(results_dir / "image_experiment_status.json")
    experiment = status.get("experiment", {}) if isinstance(status, dict) else {}
    secure_acc = _float(experiment.get("secure_acc") if isinstance(experiment, dict) else None)
    synthetic = status.get("synthetic_results_generated") if isinstance(status, dict) else None
    calibration_rows = _read_csv(results_dir / "isic2018_cnn_calibration.csv")
    calibrated_system_rows = _read_csv(results_dir / "isic2018_cnn_calibrated_system_summary.csv")
    transfer_reference_rows = _read_csv(results_dir / "isic2018_transfer_reference.csv")
    calibration_acc = [_float(row.get("accuracy")) for row in calibration_rows]
    calibration_acc = [value for value in calibration_acc if value is not None]
    calibration_mean = sum(calibration_acc) / len(calibration_acc) if calibration_acc else None
    ass_system = next((row for row in calibrated_system_rows if row.get("Method") in {"ASS (Ours)", "ASS (Ours)"}), None)
    transfer_acc = [_float(row.get("accuracy")) for row in transfer_reference_rows]
    transfer_auc = [_float(row.get("auc")) for row in transfer_reference_rows]
    transfer_acc = [value for value in transfer_acc if value is not None]
    transfer_auc = [value for value in transfer_auc if value is not None]
    transfer_mean_acc = sum(transfer_acc) / len(transfer_acc) if transfer_acc else None
    transfer_mean_auc = sum(transfer_auc) / len(transfer_auc) if transfer_auc else None
    issues = []
    if synthetic is not False:
        issues.append(_issue("critical", "ISIC_SYNTHETIC_FLAG_NOT_FALSE", "ISIC status must explicitly report synthetic_results_generated=false.", "image_experiment_status.json"))
    if calibration_mean is None or calibration_mean < 0.75 or ass_system is None:
        issues.append(
            _issue(
                "critical",
                "ISIC_CALIBRATED_CNN_EVIDENCE_MISSING",
                "Calibrated multi-seed ISIC CNN evidence is missing or below the meaningful plaintext-model threshold; rerun isic_cnn_calibration.py with --model-variant enhanced --pool-type max --secure-eval.",
                "isic2018_cnn_calibration.csv",
            )
        )
    if transfer_mean_acc is None or transfer_mean_acc < 0.78 or transfer_mean_auc is None or transfer_mean_auc < 0.85:
        issues.append(
            _issue(
                "critical",
                "ISIC_TRANSFER_REFERENCE_WEAK_OR_MISSING",
                "Transfer-learning CNN reference is missing or too weak for artifact interpretation; rerun isic_transfer_reference.py.",
                "isic2018_transfer_reference.csv",
            )
        )
    if secure_acc is not None and secure_acc <= 0.55:
        issues.append(
            _issue(
                "warning",
                "ISIC_ACCURACY_NEAR_RANDOM",
                "Legacy official ISIC secure-CNN mainline accuracy is near the balanced binary random boundary; use the calibrated multi-seed CNN table for workload validity and keep the legacy table only as a systems/operator benchmark.",
                "image_experiment_status.json",
            )
        )
    return issues


def _to_markdown(payload: Dict[str, object]) -> str:
    lines = [
        "# Publication Anomaly Audit",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Results dir: `{payload['results_dir']}`",
        f"- Critical issues: `{payload['critical_count']}`",
        f"- Warnings: `{payload['warning_count']}`",
        f"- Status: `{payload['status']}`",
        "",
        "## Findings",
        "",
    ]
    findings = payload.get("findings", [])
    if not findings:
        lines.append("- No findings.")
    for item in findings:
        lines.append(
            f"- `{item['severity']}` `{item['code']}` `{item['source']}`: {item['message']}"
        )
        details = []
        for key in ("task", "method", "status"):
            if item.get(key):
                details.append(f"{key}={item[key]}")
        if details:
            lines.append(f"  Context: {', '.join(details)}")
    lines.extend(
        [
            "",
            "## Artifact Interpretation",
            "",
            "- Treat `critical=0` as the minimum consistency condition for numeric tables.",
            "- Treat warnings as interpretation constraints: they do not invalidate the experiment, but they should be framed carefully.",
        ]
    )
    return "\n".join(lines)


def run_audit(results_dir: Path) -> Dict[str, object]:
    findings = []
    findings.extend(_audit_numeric_tables(results_dir))
    findings.extend(_audit_fgcs_boundary(results_dir))
    findings.extend(_audit_isic_positioning(results_dir))
    critical_count = sum(1 for item in findings if item["severity"] == "critical")
    warning_count = sum(1 for item in findings if item["severity"] == "warning")
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results_dir": str(results_dir.resolve()),
        "critical_count": critical_count,
        "warning_count": warning_count,
        "status": "PASS_WITH_WARNINGS" if critical_count == 0 and warning_count else ("PASS" if critical_count == 0 else "FAIL"),
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit publication-facing result tables for anomalies.")
    default_results = Path(__file__).resolve().parents[2] / "results"
    parser.add_argument("--results-dir", default=str(default_results))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    payload = run_audit(results_dir)
    json_path = results_dir / "publication_anomaly_audit.json"
    md_path = results_dir / "publication_anomaly_audit.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.strict and payload["critical_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
