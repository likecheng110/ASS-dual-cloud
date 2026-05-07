# -*- coding: utf-8 -*-
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


MAIN_REQUIRED_METHODS = [
    "2Cloud-D (Data-only)",
    "ASS (Ours)",
    "3Share-DM (3-party)",
    "SecureNN (3PC)",
]

SPLIT_METHODS = [
    "2Cloud-D (Data-only)",
    "ASS (Ours)",
    "3Share-DM (3-party)",
]

HE_METHODS = ["CKKS (HE)", "Paillier (PHE)"]
MEDICAL_TASKS = {"Medical", "Diabetes", "Heart", "Liver"}
GENERAL_TASKS = {"MNIST", "Fashion", "Wine", "Digits"}
METHOD_ALIASES = {
    "ASS (Ours)": "ASS (Ours)",
}


@dataclass
class GateCheck:
    cid: str
    title: str
    critical: bool
    passed: bool
    detail: str


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NONE", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    as_float = _safe_float(value)
    if as_float is None:
        return None
    return int(as_float)


def _load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        method = row.get("Method")
        if method in METHOD_ALIASES:
            row["Method"] = METHOD_ALIASES[method]
    return rows


def _load_json(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _mean(values: Sequence[float]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(statistics.mean(valid))


def _is_finite_number(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def _pair_rows(main_rows: List[Dict[str, str]], method_a: str, method_b: str) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
    by_key = {}
    for row in main_rows:
        task = row.get("Task")
        method = row.get("Method")
        if not task or not method:
            continue
        by_key[(task, method)] = row
    pairs = []
    tasks = sorted({task for task, method in by_key if method in {method_a, method_b}})
    for task in tasks:
        row_a = by_key.get((task, method_a))
        row_b = by_key.get((task, method_b))
        if row_a and row_b:
            pairs.append((row_a, row_b))
    return pairs


def _security_lookup(rows: List[Dict[str, str]], method: str) -> Dict[str, str]:
    for row in rows:
        if row.get("Method") == method:
            return row
    return {}


def _build_checks(results_dir: str) -> Tuple[List[GateCheck], Dict[str, object]]:
    paths = {
        "main": os.path.join(results_dir, "results_table.csv"),
        "mainline": os.path.join(results_dir, "mainline_workloads.csv"),
        "main_seed": os.path.join(results_dir, "main_seed_runs.csv"),
        "split": os.path.join(results_dir, "split_ablation.csv"),
        "round": os.path.join(results_dir, "round_ablation.csv"),
        "transfer": os.path.join(results_dir, "transfer_results.csv"),
        "longtail": os.path.join(results_dir, "long_tail_results.csv"),
        "security": os.path.join(results_dir, "security_metrics.csv"),
        "plain": os.path.join(results_dir, "plaintext_reference.csv"),
        "plain_seed": os.path.join(results_dir, "plaintext_seed_runs.csv"),
        "loopback": os.path.join(results_dir, "loopback_comm_audit.csv"),
    }
    optional_paths = {
        "raw": os.path.join(results_dir, "results_raw.json"),
    }

    main_rows = _load_csv(paths["main"])
    mainline_rows = _load_csv(paths["mainline"])
    main_seed_rows = _load_csv(paths["main_seed"])
    split_rows = _load_csv(paths["split"])
    round_rows = _load_csv(paths["round"])
    transfer_rows = _load_csv(paths["transfer"])
    longtail_rows = _load_csv(paths["longtail"])
    security_rows = _load_csv(paths["security"])
    plain_rows = _load_csv(paths["plain"])
    plain_seed_rows = _load_csv(paths["plain_seed"])
    loopback_rows = _load_csv(paths["loopback"])
    raw_payload = _load_json(optional_paths["raw"])

    checks: List[GateCheck] = []
    summary: Dict[str, object] = {}

    missing_files = [name for name, path in paths.items() if not os.path.exists(path)]
    empty_files = [name for name, rows in {
        "main": main_rows,
        "mainline": mainline_rows,
        "split": split_rows,
        "round": round_rows,
        "transfer": transfer_rows,
        "longtail": longtail_rows,
        "security": security_rows,
        "plain": plain_rows,
        "main_seed": main_seed_rows,
        "plain_seed": plain_seed_rows,
    }.items() if not rows]

    checks.append(
        GateCheck(
            cid="C0",
            title="Public result file completeness",
            critical=True,
            passed=(not missing_files and not empty_files),
            detail=f"missing={missing_files or '[]'}, empty={empty_files or '[]'}",
        )
    )

    main_required_rows = [
        row for row in main_rows if row.get("Method") in MAIN_REQUIRED_METHODS
    ]
    failed_rows = [row for row in main_required_rows if row.get("Status") == "FAILED"]
    repeat_values = [_safe_int(row.get("Repeats")) for row in main_required_rows if row.get("Status") == "SUCCESS"]
    repeat_values = [v for v in repeat_values if v is not None]
    min_repeats = min(repeat_values) if repeat_values else None

    checks.append(
        GateCheck(
            cid="C1",
            title="Main experiment stability",
            critical=True,
            passed=(len(failed_rows) == 0 and (min_repeats is not None and min_repeats >= 3)),
            detail=f"failed={len(failed_rows)}, min_repeats={min_repeats}",
        )
    )

    mainline_tasks = sorted({row.get("Task") for row in mainline_rows if row.get("Task")})
    mainline_architectures = sorted({row.get("Architecture") for row in mainline_rows if row.get("Architecture")})
    mainline_cnn_ass = next(
        (
            row
            for row in mainline_rows
            if row.get("Task") == "ISIC2018-CNN"
            and row.get("Architecture") == "Enhanced CNN"
            and row.get("Method") == "ASS (Ours)"
        ),
        None,
    )
    checks.append(
        GateCheck(
            cid="C1E",
            title="Mainline workload table integrates CNN",
            critical=True,
            passed=(
                len(mainline_rows) >= 37
                and "MLP/FNN" in mainline_architectures
                and "Enhanced CNN" in mainline_architectures
                and "ISIC2018-CNN" in mainline_tasks
                and mainline_cnn_ass is not None
                and _safe_float(mainline_cnn_ass.get("AccMean")) is not None
                and _safe_float(mainline_cnn_ass.get("AccMean")) >= 0.75
            ),
            detail=(
                f"rows={len(mainline_rows)}, tasks={mainline_tasks}, "
                f"architectures={mainline_architectures}, cnn_ass={mainline_cnn_ass is not None}"
            ),
        )
    )

    per_method_seeds = {}
    for method in MAIN_REQUIRED_METHODS:
        per_method_seeds[method] = sorted(
            {
                row.get("Seed")
                for row in main_seed_rows
                if row.get("Method") == method and row.get("Status") == "SUCCESS" and row.get("Seed")
            }
        )
    all_seed_values = sorted({seed for seeds in per_method_seeds.values() for seed in seeds})
    checks.append(
        GateCheck(
            cid="C1B",
            title="Main multi-seed retraining coverage",
            critical=True,
            passed=(len(all_seed_values) >= 3 and all(len(per_method_seeds[method]) >= 3 for method in MAIN_REQUIRED_METHODS)),
            detail=f"all_seeds={all_seed_values}, per_method={per_method_seeds}",
        )
    )

    project_dir = os.path.abspath(os.path.join(results_dir, ".."))
    data_loader_path = os.path.join(project_dir, "core_code", "data_loader.py")
    data_loader_source = ""
    if os.path.exists(data_loader_path):
        with open(data_loader_path, "r", encoding="utf-8") as handle:
            data_loader_source = handle.read()
    synthetic_env_enabled = os.getenv("ASS_ALLOW_SYNTHETIC_DATA", "").strip().lower() in {"1", "true", "yes", "on"}
    raw_config = raw_payload.get("config", {}) if isinstance(raw_payload, dict) else {}
    recorded_allow_synthetic = raw_config.get("allow_synthetic_data") if isinstance(raw_config, dict) else None
    source_denies_synthetic_by_default = (
        "ASS_ALLOW_SYNTHETIC_DATA" in data_loader_source
        and "Synthetic fallback is disabled for reproducible experiments" in data_loader_source
    )
    checks.append(
        GateCheck(
            cid="C1C",
            title="Formal data loading forbids silent synthetic fallback",
            critical=True,
            passed=(
                source_denies_synthetic_by_default
                and not synthetic_env_enabled
                and recorded_allow_synthetic in {False, None}
            ),
            detail=(
                f"source_denies_synthetic_by_default={source_denies_synthetic_by_default}, "
                f"ASS_ALLOW_SYNTHETIC_DATA_enabled={synthetic_env_enabled}, "
                f"recorded_allow_synthetic={recorded_allow_synthetic}"
            ),
        )
    )

    ckks_rows = [
        ("main", row)
        for row in main_rows
        if row.get("Method") == "CKKS (HE)"
    ] + [
        ("seed", row)
        for row in main_seed_rows
        if row.get("Method") == "CKKS (HE)"
    ]
    ckks_invalid = []
    ckks_success_like_count = 0
    for source, row in ckks_rows:
        status = row.get("Status")
        if status in {"NA", "SUCCESS", "APPROX"}:
            ckks_success_like_count += 1
            ms = _safe_float(row.get("PerSampleMsMean"))
            timing_samples = _safe_int(row.get("TimingSamples"))
            error_text = str(row.get("Error") or "").strip()
            if ms is None or ms <= 0 or timing_samples is None or timing_samples <= 0 or error_text:
                ckks_invalid.append(
                    {
                        "source": source,
                        "task": row.get("Task"),
                        "status": status,
                        "ms": ms,
                        "timing_samples": timing_samples,
                        "error": error_text,
                    }
                )
        elif status == "FAILED" and not str(row.get("Error") or "").strip():
            ckks_invalid.append({"source": source, "task": row.get("Task"), "status": status, "error": ""})
    checks.append(
        GateCheck(
            cid="C1D",
            title="CKKS payload integrity guard",
            critical=True,
            passed=(len(ckks_invalid) == 0),
            detail=f"ckks_rows={len(ckks_rows)}, success_like={ckks_success_like_count}, invalid={ckks_invalid}",
        )
    )

    sanity_sources = {
        "main": main_rows,
        "main_seed": main_seed_rows,
        "split": split_rows,
        "round": round_rows,
        "transfer": transfer_rows,
        "longtail": longtail_rows,
        "stress": _load_csv(os.path.join(results_dir, "stress_results.csv")),
        "fgcs_projection": _load_csv(os.path.join(results_dir, "fgcs_deployment_projection.csv")),
        "fgcs_ratio": _load_csv(os.path.join(results_dir, "fgcs_j2sp_vs_baselines_projection.csv")),
        "fgcs_throughput": _load_csv(os.path.join(results_dir, "fgcs_throughput_scaling_summary.csv")),
        "fgcs_image": _load_csv(os.path.join(results_dir, "fgcs_isic2018_system_summary.csv")),
        "isic": _load_csv(os.path.join(results_dir, "isic2018_image_sota_comparison.csv")),
    }
    probability_cols = {
        "Acc",
        "Accuracy",
        "AccMean",
        "AccStd",
        "AccCI95Low",
        "AccCI95High",
        "TailRecall",
        "GeneralMeanAcc",
        "MedicalMeanAcc",
        "PlainAcc",
        "SecureAcc",
    }
    nonnegative_cols = {
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
    positive_when_success = {
        "PerSampleMs",
        "PerSampleMsMean",
        "TimeMs",
        "ComputeMs",
        "ProjectedE2EMs",
        "Throughput",
        "ThroughputMean",
        "ProjectedQPS",
    }
    success_like_statuses = {"", "SUCCESS", "NA", "APPROX"}
    sanity_issues = []
    for source_name, rows in sanity_sources.items():
        for row_index, row in enumerate(rows, start=2):
            status = str(row.get("Status") or "").strip().upper()
            row_label = {
                "source": source_name,
                "line": row_index,
                "task": row.get("Task"),
                "method": row.get("Method"),
            }
            for col in sorted(probability_cols | nonnegative_cols):
                if col not in row:
                    continue
                value = _safe_float(row.get(col))
                if value is None:
                    continue
                if not _is_finite_number(value):
                    sanity_issues.append({**row_label, "column": col, "value": row.get(col), "reason": "non_finite"})
                    continue
                if col in probability_cols and not (0.0 <= value <= 1.0):
                    sanity_issues.append({**row_label, "column": col, "value": value, "reason": "probability_out_of_range"})
                if col in nonnegative_cols and value < 0.0:
                    sanity_issues.append({**row_label, "column": col, "value": value, "reason": "negative_metric"})
                if col in positive_when_success and status in success_like_statuses and value <= 0.0:
                    sanity_issues.append({**row_label, "column": col, "value": value, "reason": "nonpositive_success_metric"})
            acc_low = _safe_float(row.get("AccCI95Low"))
            acc_high = _safe_float(row.get("AccCI95High"))
            if acc_low is not None and acc_high is not None and acc_low > acc_high:
                sanity_issues.append({**row_label, "column": "AccCI95", "value": f"{acc_low}>{acc_high}", "reason": "invalid_interval"})
            time_low = _safe_float(row.get("TimeCI95Low"))
            time_high = _safe_float(row.get("TimeCI95High"))
            if time_low is not None and time_high is not None and time_low > time_high:
                sanity_issues.append({**row_label, "column": "TimeCI95", "value": f"{time_low}>{time_high}", "reason": "invalid_interval"})
    checks.append(
        GateCheck(
            cid="C7",
            title="Numeric sanity guard for publication tables",
            critical=True,
            passed=(len(sanity_issues) == 0),
            detail=f"issues={sanity_issues[:20]}, total={len(sanity_issues)}",
        )
    )

    scope_rows = [row for row in main_rows if row.get("Method") in MAIN_REQUIRED_METHODS]
    scope_complete = all(row.get("ImplementationScope") and row.get("CommMeasurementBasis") for row in scope_rows)
    scope_values = sorted({(row.get("Method"), row.get("ImplementationScope"), row.get("CommMeasurementBasis")) for row in scope_rows})
    checks.append(
        GateCheck(
            cid="C2",
            title="Baseline scope annotation completeness",
            critical=True,
            passed=(len(scope_rows) > 0 and scope_complete),
            detail=f"scopes={scope_values}",
        )
    )

    ass_2cloud_pairs = _pair_rows(main_rows, "ASS (Ours)", "2Cloud-D (Data-only)")
    acc_deltas = []
    latency_ratios = []
    comm_ratios = []
    for ass_row, two_row in ass_2cloud_pairs:
        ass_acc = _safe_float(ass_row.get("AccMean"))
        two_acc = _safe_float(two_row.get("AccMean"))
        if ass_acc is not None and two_acc is not None:
            acc_deltas.append(abs(ass_acc - two_acc))
        ass_ms = _safe_float(ass_row.get("PerSampleMsMean"))
        two_ms = _safe_float(two_row.get("PerSampleMsMean"))
        if ass_ms is not None and two_ms is not None and two_ms > 0:
            latency_ratios.append(ass_ms / two_ms)
        ass_comm = _safe_float(ass_row.get("OnlineCommMB"))
        two_comm = _safe_float(two_row.get("OnlineCommMB"))
        if ass_comm is not None and two_comm is not None and two_comm > 0:
            comm_ratios.append(ass_comm / two_comm)

    sec_ass = _security_lookup(security_rows, "ASS (Ours)")
    sec_2cloud = _security_lookup(security_rows, "2Cloud-D (Data-only)")
    ass_model_exposure = _safe_int(sec_ass.get("SinglePointModelExposure"))
    cloud_model_exposure = _safe_int(sec_2cloud.get("SinglePointModelExposure"))
    ass_collusion = _safe_int(sec_ass.get("MinCollusionModel"))
    cloud_collusion = _safe_int(sec_2cloud.get("MinCollusionModel"))
    mean_abs_acc_delta = _mean(acc_deltas)
    mean_latency_ratio = _mean(latency_ratios)
    mean_comm_ratio = _mean(comm_ratios)

    checks.append(
        GateCheck(
            cid="H1",
            title="Claim 1: ASS vs 2Cloud-D accuracy equivalence and model protection",
            critical=True,
            passed=(
                mean_abs_acc_delta is not None
                and mean_abs_acc_delta <= 0.01
                and ass_model_exposure == 0
                and cloud_model_exposure == 1
                and ass_collusion is not None
                and cloud_collusion is not None
                and ass_collusion > cloud_collusion
            ),
            detail=(
                f"mean_abs_acc_delta={mean_abs_acc_delta}, "
                f"ass/two_latency_ratio={mean_latency_ratio}, "
                f"ass/two_comm_ratio={mean_comm_ratio}, "
                f"model_exposure(ass,two)=({ass_model_exposure},{cloud_model_exposure}), "
                f"min_collusion_model(ass,two)=({ass_collusion},{cloud_collusion})"
            ),
        )
    )

    transfer_ass = next((row for row in transfer_rows if row.get("Method") == "ASS (Ours)"), None)
    general_tasks_found = sorted(
        {row.get("Task") for row in main_rows if row.get("Domain") == "general" and row.get("Method") == "ASS (Ours)" and row.get("Status") == "SUCCESS"}
    )
    medical_tasks_found = sorted(
        {row.get("Task") for row in main_rows if row.get("Domain") == "medical" and row.get("Method") == "ASS (Ours)" and row.get("Status") == "SUCCESS"}
    )
    general_mean_acc = _safe_float(transfer_ass.get("GeneralMeanAcc")) if transfer_ass else None
    medical_mean_acc = _safe_float(transfer_ass.get("MedicalMeanAcc")) if transfer_ass else None
    acc_domain_gap = _safe_float(transfer_ass.get("AccDomainGap")) if transfer_ass else None

    checks.append(
        GateCheck(
            cid="H2",
            title="Claim 2: general-to-medical transfer coverage",
            critical=True,
            passed=(
                transfer_ass is not None
                and len(set(general_tasks_found) & GENERAL_TASKS) >= 4
                and len(set(medical_tasks_found) & MEDICAL_TASKS) >= 4
                and general_mean_acc is not None
                and medical_mean_acc is not None
            ),
            detail=(
                f"general_tasks={general_tasks_found}, medical_tasks={medical_tasks_found}, "
                f"general_mean_acc={general_mean_acc}, medical_mean_acc={medical_mean_acc}, "
                f"acc_domain_gap={acc_domain_gap}"
            ),
        )
    )

    round_ratio_detail = {}
    round_pass = True
    for method in SPLIT_METHODS:
        method_rows = [row for row in round_rows if row.get("Method") == method]
        by_task_round = {}
        for row in method_rows:
            task = row.get("Task")
            rounds = _safe_int(row.get("Rounds"))
            if task is None or rounds is None:
                continue
            by_task_round[(task, rounds)] = row
        latency_rr = []
        comm_rr = []
        for task in {key[0] for key in by_task_round.keys()}:
            row_r1 = by_task_round.get((task, 1))
            row_r4 = by_task_round.get((task, 4))
            if not row_r1 or not row_r4:
                continue
            ms1 = _safe_float(row_r1.get("PerSampleMs"))
            ms4 = _safe_float(row_r4.get("PerSampleMs"))
            comm1 = _safe_float(row_r1.get("OnlineCommMB"))
            comm4 = _safe_float(row_r4.get("OnlineCommMB"))
            if ms1 is not None and ms4 is not None and ms1 > 0:
                latency_rr.append(ms4 / ms1)
            if comm1 is not None and comm4 is not None and comm1 > 0:
                comm_rr.append(comm4 / comm1)
        avg_latency_rr = _mean(latency_rr)
        avg_comm_rr = _mean(comm_rr)
        round_ratio_detail[method] = {
            "avg_r4_over_r1_latency": avg_latency_rr,
            "avg_r4_over_r1_comm": avg_comm_rr,
            "pairs": len(latency_rr),
        }
        method_pass = (
            avg_latency_rr is not None
            and avg_comm_rr is not None
            and avg_latency_rr > 1.0
            and avg_comm_rr > 1.0
        )
        round_pass = round_pass and method_pass

    checks.append(
        GateCheck(
            cid="H3",
            title="Claim 3: one-round interaction efficiency",
            critical=True,
            passed=round_pass,
            detail=json.dumps(round_ratio_detail, ensure_ascii=False),
        )
    )

    ass_3share_pairs = _pair_rows(main_rows, "ASS (Ours)", "3Share-DM (3-party)")
    latency_3share_over_ass = []
    comm_3share_over_ass = []
    for ass_row, tri_row in ass_3share_pairs:
        ass_ms = _safe_float(ass_row.get("PerSampleMsMean"))
        tri_ms = _safe_float(tri_row.get("PerSampleMsMean"))
        if ass_ms is not None and tri_ms is not None and ass_ms > 0:
            latency_3share_over_ass.append(tri_ms / ass_ms)
        ass_comm = _safe_float(ass_row.get("OnlineCommMB"))
        tri_comm = _safe_float(tri_row.get("OnlineCommMB"))
        if ass_comm is not None and tri_comm is not None and ass_comm > 0:
            comm_3share_over_ass.append(tri_comm / ass_comm)
    avg_latency_3_over_ass = _mean(latency_3share_over_ass)
    avg_comm_3_over_ass = _mean(comm_3share_over_ass)
    sec_3share = _security_lookup(security_rows, "3Share-DM (3-party)")
    collusion_3 = _safe_int(sec_3share.get("MinCollusionModel"))

    checks.append(
        GateCheck(
            cid="H4",
            title="Claim 4: three-party extension security-cost trade-off",
            critical=True,
            passed=(
                collusion_3 is not None
                and ass_collusion is not None
                and collusion_3 > ass_collusion
                and avg_latency_3_over_ass is not None
                and avg_latency_3_over_ass > 1.0
                and avg_comm_3_over_ass is not None
                and avg_comm_3_over_ass > 1.0
            ),
            detail=(
                f"min_collusion_model(3share,ass)=({collusion_3},{ass_collusion}), "
                f"avg_latency_ratio(3share/ass)={avg_latency_3_over_ass}, "
                f"avg_comm_ratio(3share/ass)={avg_comm_3_over_ass}"
            ),
        )
    )

    ass_med = None
    for row in main_rows:
        if row.get("Task") == "Medical" and row.get("Method") == "ASS (Ours)":
            ass_med = _safe_float(row.get("PerSampleMsMean"))
            if ass_med is not None:
                break
    he_rows = [row for row in main_rows if row.get("Method") in HE_METHODS and row.get("Status") in {"NA", "APPROX", "SUCCESS"}]
    ckks_med = next((row for row in he_rows if row.get("Task") == "Medical" and row.get("Method") == "CKKS (HE)"), None)
    paillier_med = next((row for row in he_rows if row.get("Task") == "Medical" and row.get("Method") == "Paillier (PHE)"), None)
    ckks_ms = _safe_float(ckks_med.get("PerSampleMsMean")) if ckks_med else None
    paillier_ms = _safe_float(paillier_med.get("PerSampleMsMean")) if paillier_med else None
    ckks_ratio = (ckks_ms / ass_med) if (ckks_ms is not None and ass_med is not None and ass_med > 0) else None
    paillier_ratio = (paillier_ms / ass_med) if (paillier_ms is not None and ass_med is not None and ass_med > 0) else None

    checks.append(
        GateCheck(
            cid="H5",
            title="Claim 5: HE/PHE engineering boundary",
            critical=False,
            passed=(
                ass_med is not None
                and ckks_ms is not None
                and paillier_ms is not None
                and ckks_ratio is not None
                and paillier_ratio is not None
                and ckks_ratio > 1.0
                and paillier_ratio > 1.0
            ),
            detail=(
                f"ass_med_ms={ass_med}, ckks_med_ms={ckks_ms}, paillier_med_ms={paillier_ms}, "
                f"ckks/ass={ckks_ratio}, paillier/ass={paillier_ratio}"
            ),
        )
    )

    tail_rows_ass = [row for row in longtail_rows if row.get("Method") == "ASS (Ours)" and row.get("Task") in MEDICAL_TASKS]
    tail_drop_records = []
    for task in MEDICAL_TASKS:
        r1 = next((row for row in tail_rows_ass if row.get("Task") == task and row.get("Ratio") == "1:1"), None)
        r19 = next((row for row in tail_rows_ass if row.get("Task") == task and row.get("Ratio") == "19:1"), None)
        if not r1 or not r19:
            continue
        tail1 = _safe_float(r1.get("TailRecall"))
        tail19 = _safe_float(r19.get("TailRecall"))
        if tail1 is None or tail19 is None:
            continue
        tail_drop_records.append((task, tail1, tail19, tail19 - tail1))
    has_tail_drop = any(delta < 0 for _, _, _, delta in tail_drop_records)

    checks.append(
        GateCheck(
            cid="C3",
            title="Medical long-tail evidence availability",
            critical=True,
            passed=(len(tail_drop_records) >= 1 and has_tail_drop),
            detail=f"tail_records={tail_drop_records}",
        )
    )

    valid_loopback_rows = [
        row for row in loopback_rows
        if row.get("AuditMode") == "localhost-loopback" and _safe_float(row.get("LoopbackReplayMB")) is not None
    ]
    audited_pairs = sorted({(row.get("Task"), row.get("Method")) for row in valid_loopback_rows})
    checks.append(
        GateCheck(
            cid="C4",
            title="Local loopback communication audit availability",
            critical=False,
            passed=len(valid_loopback_rows) >= 1,
            detail=f"audited_pairs={audited_pairs}",
        )
    )

    image_status_path = os.path.join(results_dir, "image_experiment_status.json")
    isic_result_path = os.path.join(results_dir, "isic2018_image_sota_comparison.json")
    isic_result_csv_path = os.path.join(results_dir, "isic2018_image_sota_comparison.csv")
    isic_manifest_path = os.path.join(results_dir, "isic2018_official_subset_manifest.json")
    isic_calibration_path = os.path.join(results_dir, "isic2018_cnn_calibration.csv")
    isic_calibrated_system_path = os.path.join(results_dir, "isic2018_cnn_calibrated_system_summary.csv")
    isic_transfer_reference_path = os.path.join(results_dir, "isic2018_transfer_reference.csv")
    image_status = _load_json(image_status_path)
    isic_result = _load_json(isic_result_path)
    isic_result_csv_rows = _load_csv(isic_result_csv_path)
    isic_manifest = _load_json(isic_manifest_path)
    isic_calibration_rows = _load_csv(isic_calibration_path)
    isic_calibrated_system_rows = _load_csv(isic_calibrated_system_path)
    isic_transfer_reference_rows = _load_csv(isic_transfer_reference_path)
    isic_metadata = isic_result.get("metadata", {}) if isinstance(isic_result, dict) else {}
    isic_rows = isic_result.get("rows", []) if isinstance(isic_result, dict) else []
    if not isic_rows:
        isic_rows = isic_result_csv_rows
    ass_image_row = next(
        (
            row
            for row in isic_rows
            if METHOD_ALIASES.get(row.get("Method"), row.get("Method")) == "ASS (Ours)"
        ),
        None,
    )
    public_operator_pass = (
        image_status.get("status") == "COMPLETED_OFFICIAL_ISIC2018_SUBSET"
        and image_status.get("synthetic_results_generated") is False
        and ass_image_row is not None
        and _safe_float(ass_image_row.get("TimeMs")) is not None
        and _safe_float(ass_image_row.get("CommMB")) is not None
    )
    private_manifest_pass = (
        isic_manifest.get("sample_size") == 512
        and isic_manifest.get("positive_count") == 256
        and isic_manifest.get("negative_count") == 256
        and _safe_int(isic_metadata.get("train_samples")) == 409
        and _safe_int(isic_metadata.get("test_samples")) == 103
    )
    checks.append(
        GateCheck(
            cid="C5",
            title="Legacy ISIC2018 secure-CNN operator benchmark availability",
            critical=False,
            passed=public_operator_pass,
            detail=(
                f"status={image_status.get('status')}, "
                f"sample_size={isic_manifest.get('sample_size')}, "
                f"pos={isic_manifest.get('positive_count')}, neg={isic_manifest.get('negative_count')}, "
                f"train={isic_metadata.get('train_samples')}, test={isic_metadata.get('test_samples')}, "
                f"ass_row_present={ass_image_row is not None}, "
                f"private_manifest_pass={private_manifest_pass}"
            ),
        )
    )

    calibration_acc_values = [
        _safe_float(row.get("accuracy"))
        for row in isic_calibration_rows
        if _safe_float(row.get("accuracy")) is not None
    ]
    calibration_secure_acc_values = [
        _safe_float(row.get("secure_accuracy"))
        for row in isic_calibration_rows
        if _safe_float(row.get("secure_accuracy")) is not None
    ]
    calibration_seed_values = sorted({row.get("seed") for row in isic_calibration_rows if row.get("seed")})
    calibration_mean_acc = _mean(calibration_acc_values)
    secure_plain_diffs = [
        abs(float(row.get("accuracy")) - float(row.get("secure_accuracy")))
        for row in isic_calibration_rows
        if _safe_float(row.get("accuracy")) is not None and _safe_float(row.get("secure_accuracy")) is not None
    ]
    max_secure_plain_diff = max(secure_plain_diffs) if secure_plain_diffs else None
    calibrated_ass_row = next(
        (
            row
            for row in isic_calibrated_system_rows
            if METHOD_ALIASES.get(row.get("Method"), row.get("Method")) == "ASS (Ours)"
        ),
        None,
    )
    checks.append(
        GateCheck(
            cid="C5B",
            title="ISIC2018 calibrated CNN accuracy and secure consistency",
            critical=True,
            passed=(
                len(calibration_seed_values) >= 3
                and calibration_mean_acc is not None
                and calibration_mean_acc >= 0.75
                and len(calibration_secure_acc_values) >= 3
                and max_secure_plain_diff is not None
                and max_secure_plain_diff <= 1e-9
                and calibrated_ass_row is not None
                and _safe_float(calibrated_ass_row.get("TimeMsMean")) is not None
                and _safe_float(calibrated_ass_row.get("CommMBMean")) is not None
            ),
            detail=(
                f"seeds={calibration_seed_values}, mean_acc={calibration_mean_acc}, "
                f"max_secure_plain_diff={max_secure_plain_diff}, calibrated_ass_row={calibrated_ass_row is not None}"
            ),
        )
    )

    transfer_reference_acc_values = [
        _safe_float(row.get("accuracy"))
        for row in isic_transfer_reference_rows
        if _safe_float(row.get("accuracy")) is not None
    ]
    transfer_reference_auc_values = [
        _safe_float(row.get("auc"))
        for row in isic_transfer_reference_rows
        if _safe_float(row.get("auc")) is not None
    ]
    transfer_reference_seed_values = sorted({row.get("seed") for row in isic_transfer_reference_rows if row.get("seed")})
    transfer_reference_mean_acc = _mean(transfer_reference_acc_values)
    transfer_reference_mean_auc = _mean(transfer_reference_auc_values)
    checks.append(
        GateCheck(
            cid="C5C",
            title="ISIC2018 transfer-learning CNN reference strength",
            critical=True,
            passed=(
                len(transfer_reference_seed_values) >= 3
                and transfer_reference_mean_acc is not None
                and transfer_reference_mean_acc >= 0.78
                and transfer_reference_mean_auc is not None
                and transfer_reference_mean_auc >= 0.85
            ),
            detail=(
                f"seeds={transfer_reference_seed_values}, mean_acc={transfer_reference_mean_acc}, "
                f"mean_auc={transfer_reference_mean_auc}"
            ),
        )
    )

    fgcs_summary_path = os.path.join(results_dir, "fgcs_systems_summary.json")
    fgcs_report_path = os.path.join(results_dir, "fgcs_systems_report.md")
    fgcs_projection_path = os.path.join(results_dir, "fgcs_deployment_projection.csv")
    fgcs_ratio_path = os.path.join(results_dir, "fgcs_j2sp_vs_baselines_projection.csv")
    fgcs_throughput_path = os.path.join(results_dir, "fgcs_throughput_scaling_summary.csv")
    fgcs_image_path = os.path.join(results_dir, "fgcs_isic2018_system_summary.csv")
    fgcs_summary = _load_json(fgcs_summary_path)
    fgcs_projection = _load_csv(fgcs_projection_path)
    fgcs_ratio = _load_csv(fgcs_ratio_path)
    fgcs_throughput = _load_csv(fgcs_throughput_path)
    fgcs_image = _load_csv(fgcs_image_path)
    fgcs_pass = (
        os.path.exists(fgcs_report_path)
        and len(fgcs_projection) >= 100
        and len(fgcs_ratio) >= 40
        and len(fgcs_throughput) >= 4
        and len(fgcs_image) >= 5
        and _safe_int(fgcs_summary.get("deployment_projection_rows")) == len(fgcs_projection)
        and _safe_int(fgcs_summary.get("ratio_rows")) == len(fgcs_ratio)
    )
    checks.append(
        GateCheck(
            cid="C6",
            title="FGCS systems-oriented evidence package",
            critical=False,
            passed=fgcs_pass,
            detail=(
                f"projection={len(fgcs_projection)}, ratio={len(fgcs_ratio)}, "
                f"throughput={len(fgcs_throughput)}, image={len(fgcs_image)}, "
                f"report_exists={os.path.exists(fgcs_report_path)}"
            ),
        )
    )

    critical_checks = [c for c in checks if c.critical]
    passed_critical = [c for c in critical_checks if c.passed]
    all_checks_passed = all(c.passed for c in checks)
    critical_pass = len(passed_critical) == len(critical_checks)

    summary.update(
        {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results_dir": os.path.abspath(results_dir),
            "total_checks": len(checks),
            "passed_checks": len([c for c in checks if c.passed]),
            "critical_checks": len(critical_checks),
            "passed_critical_checks": len(passed_critical),
            "critical_pass": critical_pass,
            "all_checks_passed": all_checks_passed,
            "review_status": "PASS" if critical_pass else "NEEDS_REVIEW",
            "checks": [
                {
                    "id": c.cid,
                    "title": c.title,
                    "critical": c.critical,
                    "passed": c.passed,
                    "detail": c.detail,
                }
                for c in checks
            ],
        }
    )
    return checks, summary


def _to_markdown(summary: Dict[str, object]) -> str:
    checks = summary["checks"]
    lines = []
    lines.append("# Experiment Consistency Gate")
    lines.append("")
    lines.append(f"- Generated at: `{summary['generated_at']}`")
    lines.append(f"- Results dir: `{summary['results_dir']}`")
    lines.append(f"- Critical pass: `{summary['critical_pass']}`")
    lines.append(f"- Review status: `{summary['review_status']}`")
    lines.append("- Use: artifact-level consistency gate for aggregate result files.")
    lines.append("")
    lines.append("| ID | Check | Critical | Result | Detail |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for item in checks:
        flag = "PASS" if item["passed"] else "REVIEW"
        critical = "yes" if item["critical"] else "no"
        lines.append(
            f"| {item['id']} | {item['title']} | {critical} | {flag} | {item['detail']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- `PASS`: the evidence chain is consistent with the artifact scope.")
    lines.append("- `REVIEW`: at least one check needs experiment rerun or scope correction.")
    lines.append("- This gate does not replace research ethics, peer review, or manual result inspection.")
    return "\n".join(lines)


def _write_outputs(results_dir: str, summary: Dict[str, object]):
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "sci_quality_gate.json")
    md_path = os.path.join(results_dir, "sci_quality_gate.md")
    _atomic_write_text(json_path, json.dumps(summary, ensure_ascii=False, indent=2))
    _atomic_write_text(md_path, _to_markdown(summary))
    return json_path, md_path


def _atomic_write_text(path: str, text: str):
    tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        backup_path = f"{path}.bak-{int(time.time())}"
        if os.path.exists(path):
            os.replace(path, backup_path)
        os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Experiment consistency gate")
    default_results = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    parser.add_argument("--results-dir", default=default_results, help="结果目录（默认 FGCS_Privacy_Inference/results）")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any critical check fails")
    args = parser.parse_args()

    checks, summary = _build_checks(args.results_dir)
    json_path, md_path = _write_outputs(args.results_dir, summary)

    print("Experiment consistency audit completed.")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Critical pass: {summary['critical_pass']}")
    print(f"Review status: {summary['review_status']}")

    if args.strict and not summary["critical_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
