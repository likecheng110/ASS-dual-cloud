import csv
import json
import os
import statistics
import sys
import time
import uuid
from typing import Dict, Iterable, List, Optional, Tuple


METHODS_FOR_SYSTEM_PROJECTION = {
    "2Cloud-D (Data-only)",
    "ASS (Ours)",
    "3Share-DM (3-party)",
    "SecureNN (3PC)",
}

NETWORK_SCENARIOS = [
    {
        "Scenario": "edge_lan",
        "Description": "co-located edge/cloudlet nodes",
        "RTTMs": 1.0,
        "BandwidthMbps": 1000.0,
    },
    {
        "Scenario": "campus_cloudlet",
        "Description": "campus or hospital cloudlet",
        "RTTMs": 5.0,
        "BandwidthMbps": 1000.0,
    },
    {
        "Scenario": "metropolitan_edge",
        "Description": "metropolitan edge federation",
        "RTTMs": 20.0,
        "BandwidthMbps": 200.0,
    },
    {
        "Scenario": "regional_cloud",
        "Description": "regional public cloud deployment",
        "RTTMs": 50.0,
        "BandwidthMbps": 100.0,
    },
    {
        "Scenario": "cross_region_cloud",
        "Description": "cross-region cloud collaboration",
        "RTTMs": 100.0,
        "BandwidthMbps": 50.0,
    },
    {
        "Scenario": "iot_gateway",
        "Description": "constrained IoT gateway uplink",
        "RTTMs": 30.0,
        "BandwidthMbps": 20.0,
    },
]


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NONE", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value) -> Optional[int]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", newline="") as existing, open(tmp_path, "r", encoding="utf-8", newline="") as candidate:
                if existing.read() == candidate.read():
                    os.remove(tmp_path)
                    return
        backup_path = f"{path}.bak-{int(time.time())}"
        if os.path.exists(path):
            os.replace(path, backup_path)
        os.replace(tmp_path, path)


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        backup_path = f"{path}.bak-{int(time.time())}"
        if os.path.exists(path):
            os.replace(path, backup_path)
        os.replace(tmp_path, path)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _mean(values: Iterable[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.mean(clean))


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _project_row(row: Dict[str, str], scenario: Dict[str, float]) -> Optional[Dict[str, object]]:
    compute_ms = _safe_float(row.get("PerSampleMsMean"))
    comm_mb = _safe_float(row.get("OnlineCommMB"))
    rounds = _safe_int(row.get("InteractionRounds"))
    if compute_ms is None or comm_mb is None or rounds is None:
        return None
    rtt_ms = float(scenario["RTTMs"])
    bandwidth_mbps = float(scenario["BandwidthMbps"])
    transfer_ms = (comm_mb * 8.0 * 1000.0) / bandwidth_mbps
    round_trip_ms = rounds * rtt_ms
    projected_ms = compute_ms + transfer_ms + round_trip_ms
    return {
        "Scenario": scenario["Scenario"],
        "ScenarioDescription": scenario["Description"],
        "Task": row.get("Task"),
        "Domain": row.get("Domain"),
        "Method": row.get("Method"),
        "Parties": _safe_int(row.get("Parties")),
        "InteractionRounds": rounds,
        "ComputeMs": compute_ms,
        "OnlineCommMB": comm_mb,
        "RTTMs": rtt_ms,
        "BandwidthMbps": bandwidth_mbps,
        "TransferMs": transfer_ms,
        "RoundTripMs": round_trip_ms,
        "ProjectedE2EMs": projected_ms,
        "ProjectedQPS": 1000.0 / projected_ms if projected_ms > 0 else None,
        "ComputeSharePct": (compute_ms / projected_ms) * 100.0 if projected_ms > 0 else None,
        "TransferSharePct": (transfer_ms / projected_ms) * 100.0 if projected_ms > 0 else None,
        "RoundTripSharePct": (round_trip_ms / projected_ms) * 100.0 if projected_ms > 0 else None,
    }


def build_deployment_projection(main_rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    projection_rows: List[Dict[str, object]] = []
    for row in main_rows:
        if row.get("Status") != "SUCCESS" or row.get("Method") not in METHODS_FOR_SYSTEM_PROJECTION:
            continue
        for scenario in NETWORK_SCENARIOS:
            projected = _project_row(row, scenario)
            if projected is not None:
                projection_rows.append(projected)

    by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in projection_rows:
        by_key[(row["Scenario"], row["Task"], row["Method"])] = row

    ratio_rows: List[Dict[str, object]] = []
    for scenario in [s["Scenario"] for s in NETWORK_SCENARIOS]:
        tasks = sorted({task for sc, task, _ in by_key if sc == scenario})
        for task in tasks:
            ass = by_key.get((scenario, task, "ASS (Ours)"))
            two = by_key.get((scenario, task, "2Cloud-D (Data-only)"))
            tri = by_key.get((scenario, task, "3Share-DM (3-party)"))
            sec = by_key.get((scenario, task, "SecureNN (3PC)"))
            if ass is None or two is None:
                continue
            ass_ms = ass["ProjectedE2EMs"]
            two_ms = two["ProjectedE2EMs"]
            tri_ms = tri["ProjectedE2EMs"] if tri else None
            sec_ms = sec["ProjectedE2EMs"] if sec else None
            ratio_rows.append(
                {
                    "Scenario": scenario,
                    "Task": task,
                    "Domain": ass["Domain"],
                    "ASSProjectedMs": ass_ms,
                    "TwoCloudProjectedMs": two_ms,
                    "ASSOverTwoCloud": ass_ms / two_ms if two_ms else None,
                    "ThreeShareProjectedMs": tri_ms,
                    "ThreeShareOverASS": tri_ms / ass_ms if tri_ms and ass_ms else None,
                    "SecureNNProjectedMs": sec_ms,
                    "SecureNNOverASS": sec_ms / ass_ms if sec_ms and ass_ms else None,
                    "ASSProjectedQPS": ass["ProjectedQPS"],
                    "ASSComputeSharePct": ass["ComputeSharePct"],
                    "ASSNetworkSharePct": (ass["TransferSharePct"] or 0.0) + (ass["RoundTripSharePct"] or 0.0),
                }
            )
    return projection_rows, ratio_rows


def build_throughput_summary(stress_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in stress_rows:
        if row.get("Error"):
            continue
        throughput = _safe_float(row.get("Throughput"))
        if throughput is None:
            continue
        grouped.setdefault(row.get("Method"), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for method, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: _safe_int(row.get("Batch")) or 0)
        first = rows[0]
        best = max(rows, key=lambda row: _safe_float(row.get("Throughput")) or 0.0)
        first_tp = _safe_float(first.get("Throughput"))
        best_tp = _safe_float(best.get("Throughput"))
        last = rows[-1]
        last_tp = _safe_float(last.get("Throughput"))
        summary_rows.append(
            {
                "Method": method,
                "MinBatch": _safe_int(first.get("Batch")),
                "MaxBatch": _safe_int(last.get("Batch")),
                "BestBatch": _safe_int(best.get("Batch")),
                "ThroughputAtMinBatch": first_tp,
                "ThroughputAtMaxBatch": last_tp,
                "BestThroughput": best_tp,
                "BestOverMinBatch": best_tp / first_tp if best_tp and first_tp else None,
                "MaxBatchOverMinBatch": last_tp / first_tp if last_tp and first_tp else None,
            }
        )
    return summary_rows


def build_image_summary(isic_result: Dict[str, object], isic_manifest: Dict[str, object]) -> List[Dict[str, object]]:
    if not isic_result or not isic_manifest:
        return []
    metadata = isic_result.get("metadata", {})
    rows = isic_result.get("rows", [])
    output: List[Dict[str, object]] = []
    for row in rows:
        output.append(
            {
                "Dataset": isic_manifest.get("dataset", "ISIC 2018 Task 3 Training"),
                "SampleSize": isic_manifest.get("sample_size"),
                "Seed": isic_manifest.get("seed"),
                "PositiveCount": isic_manifest.get("positive_count"),
                "NegativeCount": isic_manifest.get("negative_count"),
                "TrainSamples": metadata.get("train_samples"),
                "TestSamples": metadata.get("test_samples"),
                "SecureEvalSamples": metadata.get("secure_eval_samples"),
                "Method": row.get("Method"),
                "Acc": row.get("Acc"),
                "TimeMs": row.get("TimeMs"),
                "CommMB": row.get("CommMB"),
                "TotalCommMB": row.get("TotalCommMB"),
                "ModelExposure": row.get("E_m"),
                "ModelShareThreshold": row.get("k_m"),
            }
        )
    return output


def build_report(
    projection_rows: List[Dict[str, object]],
    ratio_rows: List[Dict[str, object]],
    throughput_rows: List[Dict[str, object]],
    image_rows: List[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# FGCS-Oriented Systems Evidence Report")
    lines.append("")
    lines.append("## Scope Alignment")
    lines.append("")
    lines.append("- Distributed and collaborative computing: two-cloud, three-share, and SecureNN-style multi-party deployments.")
    lines.append("- Cloud/edge/IoT infrastructures: projected end-to-end latency under edge LAN, cloudlet, regional cloud, cross-region cloud, and constrained IoT gateway links.")
    lines.append("- High-performance and scalable computing: batch-throughput scaling from 256 to 2048 samples.")
    lines.append("- Security and privacy in future systems: data/model split exposure, collusion threshold, and communication-round evidence.")
    lines.append("- Data-intensive applications: official ISIC 2018 secure-CNN workload as a main image benchmark branch.")
    lines.append("")

    ass_ratios = [row["ASSOverTwoCloud"] for row in ratio_rows if row.get("ASSOverTwoCloud") is not None]
    three_ratios = [row["ThreeShareOverASS"] for row in ratio_rows if row.get("ThreeShareOverASS") is not None]
    secure_ratios = [row["SecureNNOverASS"] for row in ratio_rows if row.get("SecureNNOverASS") is not None]
    secure_better_rows = [
        row for row in ratio_rows
        if row.get("SecureNNOverASS") is not None and row["SecureNNOverASS"] < 1.0
    ]
    secure_better_tasks = sorted({row.get("Task") for row in secure_better_rows if row.get("Task")})
    lines.append("## Deployment Projection Summary")
    lines.append("")
    lines.append(f"- Deployment rows: `{len(projection_rows)}`.")
    lines.append(f"- ASS / 2Cloud projected latency mean: `{_fmt(_mean(ass_ratios))}`.")
    lines.append(f"- 3Share / ASS projected latency mean: `{_fmt(_mean(three_ratios))}`.")
    lines.append(f"- SecureNN / ASS projected latency mean: `{_fmt(_mean(secure_ratios))}`.")
    lines.append(
        f"- Boundary cases where SecureNN projects below ASS: `{len(secure_better_rows)}/{len(secure_ratios)}` "
        f"rows, tasks=`{', '.join(secure_better_tasks) if secure_better_tasks else 'none'}`."
    )
    lines.append("")
    lines.append("| Scenario | Task | ASS ms | ASS/2Cloud | 3Share/ASS | SecureNN/ASS | ASS network share |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    for row in ratio_rows:
        if row["Task"] not in {"MNIST", "Medical", "Digits", "Liver"}:
            continue
        lines.append(
            "| {Scenario} | {Task} | {ass_ms} | {ass_two} | {tri_ass} | {sec_ass} | {net} |".format(
                Scenario=row["Scenario"],
                Task=row["Task"],
                ass_ms=_fmt(row.get("ASSProjectedMs"), 3),
                ass_two=_fmt(row.get("ASSOverTwoCloud"), 3),
                tri_ass=_fmt(row.get("ThreeShareOverASS"), 3),
                sec_ass=_fmt(row.get("SecureNNOverASS"), 3),
                net=_fmt(row.get("ASSNetworkSharePct"), 2),
            )
        )
    lines.append("")

    lines.append("## Batch Throughput Scaling")
    lines.append("")
    lines.append("| Method | Min batch | Max batch | Best batch | Best throughput | Best/min |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for row in throughput_rows:
        lines.append(
            f"| {row['Method']} | {row['MinBatch']} | {row['MaxBatch']} | {row['BestBatch']} | "
            f"{_fmt(row.get('BestThroughput'), 1)} | {_fmt(row.get('BestOverMinBatch'), 3)} |"
        )
    lines.append("")

    if image_rows:
        ass_image = next((row for row in image_rows if row.get("Method") == "ASS (Ours)"), None)
        lines.append("## Official ISIC 2018 Secure-CNN Mainline Workload")
        lines.append("")
        lines.append(
            f"- Official subset size: `{image_rows[0].get('SampleSize')}` "
            f"(`positive={image_rows[0].get('PositiveCount')}`, `negative={image_rows[0].get('NegativeCount')}`)."
        )
        if ass_image:
            lines.append(
                f"- ASS secure CNN: acc=`{_fmt(ass_image.get('Acc'))}`, "
                f"time=`{_fmt(ass_image.get('TimeMs'), 4)} ms/sample`, "
                f"comm=`{_fmt(ass_image.get('CommMB'), 4)} MB/sample` "
                f"(`{_fmt(ass_image.get('TotalCommMB'), 4)} MB total`)."
            )
        lines.append("")
    lines.append("## Recommended FGCS Framing")
    lines.append("")
    lines.append("- Present ASS as a future cloud/edge privacy-preserving inference system, not only as a cryptographic primitive.")
    lines.append("- Lead with model-split security and deployment-projected latency/communication under multi-cloud and edge scenarios.")
    lines.append("- Treat the official ISIC secure-CNN workload as the main image branch, with explicit scope that it is a systems workload rather than a clinical diagnostic benchmark.")
    lines.append("- State the MNIST/Fashion projection boundary explicitly: ASS buys model non-exposure at extra communication cost on high-dimensional inputs.")
    lines.append("- Keep synthetic-data fallback and CKKS payload guards in the reproducibility checklist.")
    return "\n".join(lines)


def run(results_dir: str) -> Dict[str, object]:
    main_rows = _load_csv(os.path.join(results_dir, "results_table.csv"))
    stress_rows = _load_csv(os.path.join(results_dir, "stress_results.csv"))
    isic_result = _load_json(os.path.join(results_dir, "isic2018_image_sota_comparison.json"))
    isic_manifest = _load_json(os.path.join(results_dir, "isic2018_official_subset_manifest.json"))

    projection_rows, ratio_rows = build_deployment_projection(main_rows)
    throughput_rows = build_throughput_summary(stress_rows)
    image_rows = build_image_summary(isic_result, isic_manifest)

    _write_csv(
        os.path.join(results_dir, "fgcs_deployment_projection.csv"),
        projection_rows,
        [
            "Scenario",
            "ScenarioDescription",
            "Task",
            "Domain",
            "Method",
            "Parties",
            "InteractionRounds",
            "ComputeMs",
            "OnlineCommMB",
            "RTTMs",
            "BandwidthMbps",
            "TransferMs",
            "RoundTripMs",
            "ProjectedE2EMs",
            "ProjectedQPS",
            "ComputeSharePct",
            "TransferSharePct",
            "RoundTripSharePct",
        ],
    )
    _write_csv(
        os.path.join(results_dir, "fgcs_j2sp_vs_baselines_projection.csv"),
        ratio_rows,
        [
            "Scenario",
            "Task",
            "Domain",
            "ASSProjectedMs",
            "TwoCloudProjectedMs",
            "ASSOverTwoCloud",
            "ThreeShareProjectedMs",
            "ThreeShareOverASS",
            "SecureNNProjectedMs",
            "SecureNNOverASS",
            "ASSProjectedQPS",
            "ASSComputeSharePct",
            "ASSNetworkSharePct",
        ],
    )
    _write_csv(
        os.path.join(results_dir, "fgcs_throughput_scaling_summary.csv"),
        throughput_rows,
        [
            "Method",
            "MinBatch",
            "MaxBatch",
            "BestBatch",
            "ThroughputAtMinBatch",
            "ThroughputAtMaxBatch",
            "BestThroughput",
            "BestOverMinBatch",
            "MaxBatchOverMinBatch",
        ],
    )
    _write_csv(
        os.path.join(results_dir, "fgcs_isic2018_system_summary.csv"),
        image_rows,
        [
            "Dataset",
            "SampleSize",
            "Seed",
            "PositiveCount",
            "NegativeCount",
            "TrainSamples",
            "TestSamples",
            "SecureEvalSamples",
            "Method",
            "Acc",
            "TimeMs",
            "CommMB",
            "TotalCommMB",
            "ModelExposure",
            "ModelShareThreshold",
        ],
    )

    payload = {
        "network_scenarios": NETWORK_SCENARIOS,
        "deployment_projection_rows": len(projection_rows),
        "ratio_rows": len(ratio_rows),
        "throughput_rows": len(throughput_rows),
        "image_rows": len(image_rows),
        "mean_j2sp_over_2cloud": _mean(row.get("ASSOverTwoCloud") for row in ratio_rows),
        "mean_3share_over_j2sp": _mean(row.get("ThreeShareOverASS") for row in ratio_rows),
        "mean_securenn_over_j2sp": _mean(row.get("SecureNNOverASS") for row in ratio_rows),
        "securenn_faster_than_j2sp_rows": len([
            row for row in ratio_rows
            if row.get("SecureNNOverASS") is not None and row["SecureNNOverASS"] < 1.0
        ]),
        "securenn_faster_than_j2sp_tasks": sorted({
            row.get("Task")
            for row in ratio_rows
            if row.get("SecureNNOverASS") is not None and row["SecureNNOverASS"] < 1.0 and row.get("Task")
        }),
    }
    _write_json(os.path.join(results_dir, "fgcs_systems_summary.json"), payload)
    _atomic_write_text(
        os.path.join(results_dir, "fgcs_systems_report.md"),
        build_report(projection_rows, ratio_rows, throughput_rows, image_rows),
    )
    return payload


def main() -> int:
    default_results = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    results_dir = sys.argv[1] if len(sys.argv) > 1 else default_results
    payload = run(os.path.abspath(results_dir))
    print("FGCS systems postprocess completed.")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
