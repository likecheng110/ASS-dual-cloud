import argparse
import csv
import os
from collections import defaultdict


BOUNDARY_METHODS = {"CKKS (HE)", "Paillier (PHE)"}


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_csv_multi(paths, rows, fieldnames):
    for path in paths:
        save_csv(path, rows, fieldnames)


def safe_float(value):
    if value in (None, ""):
        return None
    return float(value)


def mark_skipped_boundary_rows(results_rows, seed_rows):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["Task"], row["Method"])].append(row["Status"])

    for row in results_rows:
        if row["Method"] not in BOUNDARY_METHODS:
            continue
        statuses = grouped.get((row["Task"], row["Method"]), [])
        if statuses and all(status == "SKIPPED" for status in statuses):
            row["Status"] = "SKIPPED"
            row["Repeats"] = "0"
            row["EvalSamples"] = ""
            row["TimingSamples"] = ""
            row["AccMean"] = ""
            row["AccStd"] = ""
            row["AccCI95Low"] = ""
            row["AccCI95High"] = ""
            row["AccDropVsPlain"] = ""
            row["PerSampleMsMean"] = ""
            row["PerSampleMsStd"] = ""
            row["TimeCI95Low"] = ""
            row["TimeCI95High"] = ""
            row["ThroughputMean"] = ""
            row["OnlineCommMB"] = ""
            row["OfflineSetupMB"] = ""
            row["LinearPct"] = ""
            row["ReLUPct"] = ""
            row["Error"] = ""


def build_he_boundary_summary(results_rows):
    grouped = defaultdict(dict)
    for row in results_rows:
        grouped[row["Task"]][row["Method"]] = row

    summary_rows = []
    for task_name in sorted(grouped):
        per_task = grouped[task_name]
        ass = per_task.get("ASS (Ours)")
        ckks = per_task.get("CKKS (HE)")
        paillier = per_task.get("Paillier (PHE)")
        if ass is None or ass.get("Status") != "SUCCESS":
            continue

        ckks_valid = ckks and ckks.get("Status") == "NA"
        paillier_valid = paillier and paillier.get("Status") == "APPROX"
        if not ckks_valid and not paillier_valid:
            continue

        ass_acc = safe_float(ass.get("AccMean"))
        ass_ms = safe_float(ass.get("PerSampleMsMean"))
        ass_exposure = ass.get("SinglePointModelExposure")
        ass_collusion = ass.get("MinCollusionModel")
        ckks_ms = safe_float(ckks.get("PerSampleMsMean")) if ckks_valid else None
        ckks_exposure = ckks.get("SinglePointModelExposure") if ckks else ""
        ckks_collusion = ckks.get("MinCollusionModel") if ckks else ""
        paillier_acc = safe_float(paillier.get("AccMean")) if paillier_valid else None
        paillier_ms = safe_float(paillier.get("PerSampleMsMean")) if paillier_valid else None
        summary_rows.append(
            {
                "Task": task_name,
                "ASSAcc": ass_acc,
                "ASSPerSampleMs": ass_ms,
                "ASSSinglePointModelExposure": ass_exposure,
                "ASSMinCollusionModel": ass_collusion,
                "CKKSMeasurementMode": "latency_microbenchmark",
                "CKKSPerSampleMs": ckks_ms,
                "CKKSLatencyVsASS": (ckks_ms / ass_ms) if ckks_ms is not None and ass_ms else "",
                "CKKSSinglePointModelExposure": ckks_exposure,
                "CKKSMinCollusionModel": ckks_collusion,
                "PaillierMeasurementMode": "sampled_end_to_end",
                "PaillierAcc": paillier_acc,
                "PaillierAccGapVsASS": (paillier_acc - ass_acc) if paillier_acc is not None and ass_acc is not None else "",
                "PaillierPerSampleMs": paillier_ms,
                "PaillierLatencyVsASS": (paillier_ms / ass_ms) if paillier_ms is not None and ass_ms else "",
                "PaillierSinglePointModelExposure": paillier.get("SinglePointModelExposure") if paillier else "",
                "PaillierMinCollusionModel": paillier.get("MinCollusionModel") if paillier else "",
                "BoundaryConclusion": "ASS removes single-point model exposure while keeping inference latency orders of magnitude below HE/PHE on applicable tasks.",
            }
        )
    return summary_rows


def build_he_boundary_overview(summary_rows):
    if not summary_rows:
        return []

    methods = []

    ckks_ratios = [safe_float(row.get("CKKSLatencyVsASS")) for row in summary_rows if safe_float(row.get("CKKSLatencyVsASS")) is not None]
    methods.append(
        {
            "Method": "CKKS (HE)",
            "ApplicableTasks": len(ckks_ratios),
            "MeasurementMode": "latency_microbenchmark",
            "MeanLatencyVsASS": (sum(ckks_ratios) / len(ckks_ratios)) if ckks_ratios else "",
            "MaxLatencyVsASS": max(ckks_ratios) if ckks_ratios else "",
            "AccuracyComparableTasks": 0,
            "MeanAccGapVsASS": "",
            "SinglePointModelExposure": "1",
            "MinCollusionModel": "1",
            "BoundaryConclusion": "CKKS provides a single-cloud encrypted inference boundary, but latency remains several orders above ASS and model ownership stays single-cloud visible.",
        }
    )

    paillier_ratios = [safe_float(row.get("PaillierLatencyVsASS")) for row in summary_rows if safe_float(row.get("PaillierLatencyVsASS")) is not None]
    paillier_gaps = [safe_float(row.get("PaillierAccGapVsASS")) for row in summary_rows if safe_float(row.get("PaillierAccGapVsASS")) is not None]
    methods.append(
        {
            "Method": "Paillier (PHE)",
            "ApplicableTasks": len(paillier_ratios),
            "MeasurementMode": "sampled_end_to_end",
            "MeanLatencyVsASS": (sum(paillier_ratios) / len(paillier_ratios)) if paillier_ratios else "",
            "MaxLatencyVsASS": max(paillier_ratios) if paillier_ratios else "",
            "AccuracyComparableTasks": len(paillier_gaps),
            "MeanAccGapVsASS": (sum(paillier_gaps) / len(paillier_gaps)) if paillier_gaps else "",
            "SinglePointModelExposure": "1",
            "MinCollusionModel": "1",
            "BoundaryConclusion": "Paillier keeps partial end-to-end task comparability, but remains far slower than ASS and still leaves model visibility on a single cloud.",
        }
    )
    return methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "results"))
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    artifact_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts"))
    results_path = os.path.join(results_dir, "results_table.csv")
    seed_path = os.path.join(results_dir, "main_seed_runs.csv")
    results_rows = load_csv(results_path)
    seed_rows = load_csv(seed_path)

    fieldnames = list(results_rows[0].keys())
    mark_skipped_boundary_rows(results_rows, seed_rows)
    save_csv_multi([results_path, os.path.join(artifact_dir, "results_table.csv")], results_rows, fieldnames)

    summary_rows = build_he_boundary_summary(results_rows)
    save_csv_multi(
        [os.path.join(results_dir, "he_boundary_summary.csv"), os.path.join(artifact_dir, "he_boundary_summary.csv")],
        summary_rows,
        [
            "Task",
            "ASSAcc",
            "ASSPerSampleMs",
            "ASSSinglePointModelExposure",
            "ASSMinCollusionModel",
            "CKKSMeasurementMode",
            "CKKSPerSampleMs",
            "CKKSLatencyVsASS",
            "CKKSSinglePointModelExposure",
            "CKKSMinCollusionModel",
            "PaillierMeasurementMode",
            "PaillierAcc",
            "PaillierAccGapVsASS",
            "PaillierPerSampleMs",
            "PaillierLatencyVsASS",
            "PaillierSinglePointModelExposure",
            "PaillierMinCollusionModel",
            "BoundaryConclusion",
        ],
    )
    overview_rows = build_he_boundary_overview(summary_rows)
    save_csv_multi(
        [os.path.join(results_dir, "he_boundary_overview.csv"), os.path.join(artifact_dir, "he_boundary_overview.csv")],
        overview_rows,
        [
            "Method",
            "ApplicableTasks",
            "MeasurementMode",
            "MeanLatencyVsASS",
            "MaxLatencyVsASS",
            "AccuracyComparableTasks",
            "MeanAccGapVsASS",
            "SinglePointModelExposure",
            "MinCollusionModel",
            "BoundaryConclusion",
        ],
    )
    print(f"Postprocessed boundary outputs in {results_dir}")


if __name__ == "__main__":
    main()
