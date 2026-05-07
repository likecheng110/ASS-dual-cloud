import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CORE_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
ARTIFACTS_DIR = os.path.join(CORE_DIR, "artifacts")
MODELS_DIR = os.path.join(CORE_DIR, "models")

if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

from baselines.inference_2cloud import run_2cloud_inference
from baselines.inference_ass import run_ass_inference
from baselines.inference_paillier import run_paillier_inference
from baselines.inference_securenn import run_securenn_inference
from baselines.inference_three_share import run_three_share_inference
from baselines.shared_core import load_task_model, load_task_model_share, run_shared_protocol_inference
from data_loader import (
    get_diabetes_data,
    get_digits_data,
    get_fashion_mnist_data,
    get_heart_disease_data,
    get_liver_data,
    get_medical_data,
    get_mnist_data,
    get_wine_data,
)
from experiments.loopback_comm_audit import TRACEABLE_METHODS, audit_traceable_protocol
from runtime_config import (
    configure_runtime_backend,
    dataloader_kwargs,
    detect_runtime_info,
    env_bool,
    env_bool_alias,
    env_csv_list,
    env_csv_list_alias,
    env_int,
    env_int_alias,
)
from train_plain import MODEL_SCHEMA_VERSION, evaluate, train_task_model


def atomic_write_text(path: str, text: str, encoding: str = "utf-8", newline=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding=encoding, newline=newline) as handle:
        handle.write(text)
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        backup_path = f"{path}.bak-{int(time.time())}"
        if os.path.exists(path):
            os.replace(path, backup_path)
        os.replace(tmp_path, path)


def atomic_write_json(path: str, payload):
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


DATASETS = [
    {"name": "MNIST", "loader": get_mnist_data, "model_file": "mnist_mlp.pth", "input_shape": (784,), "domain": "general", "is_medical": False, "round_ablation": True, "long_tail": False},
    {"name": "Fashion", "loader": get_fashion_mnist_data, "model_file": "fashion_mnist_mlp.pth", "input_shape": (784,), "domain": "general", "is_medical": False, "round_ablation": True, "long_tail": False},
    {"name": "Medical", "loader": get_medical_data, "model_file": "medical_mlp.pth", "input_shape": (30,), "domain": "medical", "is_medical": True, "round_ablation": True, "long_tail": True},
    {"name": "Wine", "loader": get_wine_data, "model_file": "wine_mlp.pth", "input_shape": (13,), "domain": "general", "is_medical": False, "round_ablation": True, "long_tail": False},
    {"name": "Diabetes", "loader": get_diabetes_data, "model_file": "diabetes_mlp.pth", "input_shape": (10,), "domain": "medical", "is_medical": True, "round_ablation": True, "long_tail": True},
    {"name": "Heart", "loader": get_heart_disease_data, "model_file": "heart_mlp.pth", "input_shape": (13,), "domain": "medical", "is_medical": True, "round_ablation": True, "long_tail": True},
    {"name": "Digits", "loader": get_digits_data, "model_file": "digits_mlp.pth", "input_shape": (64,), "domain": "general", "is_medical": False, "round_ablation": True, "long_tail": False},
    {"name": "Liver", "loader": get_liver_data, "model_file": "liver_mlp.pth", "input_shape": (6,), "domain": "medical", "is_medical": True, "round_ablation": True, "long_tail": True},
]


PROTOCOLS = {
    "ASS (Ours)": {
        "category": "ours",
        "parties": 2,
        "data_split": 1,
        "model_split": 1,
        "interaction_rounds": 1,
        "single_point_data_exposure": 0,
        "single_point_model_exposure": 0,
        "min_collusion_input": 2,
        "min_collusion_model": 2,
        "comm_type": "Estimated + LoopbackAudit",
        "implementation_scope": "share-simulator",
        "comparison_role": "main",
        "comm_measurement_basis": "analytic_estimate_with_localhost_replay",
        "evidence": "2-share data split + 2-share model split; one-round non-linear interaction simulation",
    },
    "2Cloud-D (Data-only)": {
        "category": "ref23",
        "parties": 2,
        "data_split": 1,
        "model_split": 0,
        "interaction_rounds": 1,
        "single_point_data_exposure": 0,
        "single_point_model_exposure": 1,
        "min_collusion_input": 2,
        "min_collusion_model": 1,
        "comm_type": "Estimated + LoopbackAudit",
        "implementation_scope": "share-simulator",
        "comparison_role": "main",
        "comm_measurement_basis": "analytic_estimate_with_localhost_replay",
        "evidence": "Reference-style baseline: only input data are split, model remains visible on each cloud",
    },
    "3Share-DM (3-party)": {
        "category": "ablation",
        "parties": 3,
        "data_split": 1,
        "model_split": 1,
        "interaction_rounds": 1,
        "single_point_data_exposure": 0,
        "single_point_model_exposure": 0,
        "min_collusion_input": 3,
        "min_collusion_model": 3,
        "comm_type": "Estimated + LoopbackAudit",
        "implementation_scope": "share-simulator",
        "comparison_role": "main",
        "comm_measurement_basis": "analytic_estimate_with_localhost_replay",
        "evidence": "3-share extension for data and model; evaluates the cost of adding one more party",
    },
    "SecureNN (3PC)": {
        "category": "baseline",
        "parties": 3,
        "data_split": 1,
        "model_split": 1,
        "interaction_rounds": 4,
        "single_point_data_exposure": 0,
        "single_point_model_exposure": 0,
        "min_collusion_input": 2,
        "min_collusion_model": 2,
        "comm_type": "Estimated",
        "implementation_scope": "simulator",
        "comparison_role": "baseline",
        "comm_measurement_basis": "task_level_proxy",
        "evidence": "Three-party SecureNN-style simulator with multi-round DReLU overhead",
    },
    "CKKS (HE)": {
        "category": "he",
        "parties": 2,
        "data_split": 0,
        "model_split": 0,
        "interaction_rounds": 0,
        "single_point_data_exposure": 0,
        "single_point_model_exposure": 1,
        "min_collusion_input": 2,
        "min_collusion_model": 1,
        "comm_type": "Ciphertext-heavy",
        "implementation_scope": "micro-benchmark",
        "comparison_role": "boundary",
        "comm_measurement_basis": "micro_benchmark_projection",
        "evidence": "Micro-benchmark projection for single-cloud CKKS inference",
    },
    "Paillier (PHE)": {
        "category": "phe",
        "parties": 2,
        "data_split": 0,
        "model_split": 0,
        "interaction_rounds": 0,
        "single_point_data_exposure": 0,
        "single_point_model_exposure": 1,
        "min_collusion_input": 2,
        "min_collusion_model": 1,
        "comm_type": "Estimated",
        "implementation_scope": "sampled-client-aided",
        "comparison_role": "boundary",
        "comm_measurement_basis": "sampled_execution_with_time_cap",
        "evidence": "Single-cloud partially homomorphic baseline with client-aided non-linearity",
    },
}


ROUND_METHODS = ["2Cloud-D (Data-only)", "ASS (Ours)", "3Share-DM (3-party)"]
LONG_TAIL_METHODS = ["2Cloud-D (Data-only)", "ASS (Ours)", "3Share-DM (3-party)"]
MAIN_PROTOCOL_ORDER = ["2Cloud-D (Data-only)", "ASS (Ours)", "3Share-DM (3-party)", "SecureNN (3PC)", "CKKS (HE)", "Paillier (PHE)"]
MAIN_REPEAT_METHODS = {"ASS (Ours)", "2Cloud-D (Data-only)", "3Share-DM (3-party)", "SecureNN (3PC)"}
MODEL_SPLIT_NECESSITY_METHODS = ["2Cloud-D (Data-only)", "ASS (Ours)"]
MEDICAL_LONGTAIL_DIR = os.path.join(MODELS_DIR, "medical_longtail")


class _SilentProgress:
    def update(self, _n=1):
        return

    def set_postfix_str(self, _text):
        return

    def close(self):
        return


class _TextProgress:
    def __init__(self, total: int, desc: str, unit: str):
        self.total = max(0, int(total))
        self.desc = desc
        self.unit = unit
        self.current = 0
        self._next_report = 0
        if self.total > 0:
            print(f"{self.desc}: 0/{self.total} {self.unit} (0%)")

    def update(self, n=1):
        if self.total <= 0:
            return
        self.current = min(self.total, self.current + max(0, int(n)))
        pct = int((self.current * 100) / self.total)
        if self.current >= self.total or pct >= self._next_report:
            print(f"{self.desc}: {self.current}/{self.total} {self.unit} ({pct}%)")
            while self._next_report <= pct:
                self._next_report += 5

    def set_postfix_str(self, _text):
        return

    def close(self):
        if self.total > 0 and self.current < self.total:
            pct = int((self.current * 100) / self.total)
            print(f"{self.desc}: {self.current}/{self.total} {self.unit} ({pct}%)")


def _make_progress(total: int, desc: str, unit: str):
    if env_bool("ASS_DISABLE_PROGRESS", False):
        return _SilentProgress()
    if _tqdm is not None:
        return _tqdm(total=max(0, int(total)), desc=desc, unit=unit, dynamic_ncols=True, leave=False)
    return _TextProgress(total=total, desc=desc, unit=unit)


def _mean_std(values: List[float]):
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _ci95(
    mean_value: Optional[float],
    std_value: Optional[float],
    n: int,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
):
    if mean_value is None or std_value is None or n < 2:
        return None, None
    margin = 1.96 * (std_value / math.sqrt(n))
    low = mean_value - margin
    high = mean_value + margin
    if lower_bound is not None:
        low = max(lower_bound, low)
    if upper_bound is not None:
        high = min(upper_bound, high)
    return low, high


def _mean_layer(layers: List[Dict[str, float]]):
    valid_layers = [layer for layer in layers if layer and "Linear" in layer and "ReLU" in layer]
    if not valid_layers:
        return {"Linear": None, "ReLU": None}
    return {
        "Linear": float(sum(layer["Linear"] for layer in valid_layers) / len(valid_layers)),
        "ReLU": float(sum(layer["ReLU"] for layer in valid_layers) / len(valid_layers)),
    }


def _batch_size_for_task(task_name: str) -> int:
    if task_name in {"MNIST", "Fashion"}:
        return 2048
    if task_name == "Digits":
        return 512
    return 256


def _load_test_loader(loader_func, batch_size: int, eval_upsample: int):
    try:
        loaded = loader_func(batch_size=batch_size, upsample_factor=eval_upsample)
    except TypeError:
        loaded = loader_func(batch_size=batch_size)
    if len(loaded) == 3:
        _, test_loader, _ = loaded
    else:
        _, test_loader = loaded
    return test_loader


def _cap_eval_loader(test_loader, max_samples: int):
    if max_samples <= 0:
        return test_loader
    dataset = test_loader.dataset
    keep = min(len(dataset), max_samples)
    xs = []
    ys = []
    for idx in range(keep):
        data, target = dataset[idx]
        xs.append(data)
        ys.append(int(target))
    x_tensor = torch.stack(xs)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    subset = TensorDataset(x_tensor, y_tensor)
    kwargs = {}
    return DataLoader(subset, batch_size=test_loader.batch_size, shuffle=False, **dataloader_kwargs())


def _per_sample_ms(total_time_s: float, samples: int) -> float:
    return (total_time_s / max(samples, 1)) * 1000.0


def _throughput(per_sample_ms: Optional[float]):
    if per_sample_ms is None or per_sample_ms <= 0:
        return None
    return 1000.0 / per_sample_ms


def _balanced_accuracy(targets, predictions):
    labels = sorted(set(int(x) for x in targets.tolist()))
    recalls = []
    for label in labels:
        positives = targets == label
        denom = positives.sum()
        if denom == 0:
            continue
        recalls.append(float(((predictions == label) & positives).sum()) / float(denom))
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


def _macro_f1(targets, predictions):
    labels = sorted(set(int(x) for x in targets.tolist()))
    f1_values = []
    for label in labels:
        true_positive = float(((predictions == label) & (targets == label)).sum())
        false_positive = float(((predictions == label) & (targets != label)).sum())
        false_negative = float(((predictions != label) & (targets == label)).sum())
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append((2.0 * precision * recall) / (precision + recall))
    if not f1_values:
        return 0.0
    return float(sum(f1_values) / len(f1_values))


def _tail_recall(targets, predictions, tail_label: int):
    positives = targets == tail_label
    denom = positives.sum()
    if denom == 0:
        return 0.0
    return float(((predictions == tail_label) & positives).sum()) / float(denom)


def _result_error(task_name: str, method_name: str, exc: Exception):
    profile = PROTOCOLS[method_name]
    return {
        "Task": task_name, "Method": method_name, "Status": "FAILED", "EvalSamples": None, "TimingSamples": None, "Repeats": 0,
        "AccMean": None, "AccStd": None, "AccCI95Low": None, "AccCI95High": None, "AccDropVsPlain": None,
        "PerSampleMsMean": None, "PerSampleMsStd": None, "TimeCI95Low": None, "TimeCI95High": None, "ThroughputMean": None,
        "OnlineCommMB": None, "OfflineSetupMB": None, "LinearPct": None, "ReLUPct": None,
        "SeedMode": "runtime_error", "SeedsUsed": "", "ImplementationScope": profile["implementation_scope"],
        "ComparisonRole": profile["comparison_role"], "CommMeasurementBasis": profile["comm_measurement_basis"],
        "Parties": profile["parties"], "InteractionRounds": profile["interaction_rounds"], "DataSplit": profile["data_split"],
        "ModelSplit": profile["model_split"], "SinglePointDataExposure": profile["single_point_data_exposure"],
        "SinglePointModelExposure": profile["single_point_model_exposure"], "MinCollusionInput": profile["min_collusion_input"],
        "MinCollusionModel": profile["min_collusion_model"], "CommType": profile["comm_type"], "Evidence": profile["evidence"],
        "Error": str(exc),
    }


def train_models_for_seed(seed: int, runtime_info, dataset_cfgs=None):
    print(f">>> Retraining plain models with seed={seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_plan = dataset_cfgs or DATASETS
    os.makedirs(MODELS_DIR, exist_ok=True)
    progress = _make_progress(total=len(train_plan), desc="Train models", unit="model")
    trained_files = []

    try:
        for dataset_cfg in train_plan:
            progress.set_postfix_str(dataset_cfg["name"])
            loader_result = dataset_cfg["loader"]()
            if len(loader_result) == 2:
                train_loader, test_loader = loader_result
                input_dim = dataset_cfg["input_shape"][0]
            else:
                train_loader, test_loader, input_dim = loader_result

            model = train_task_model(dataset_cfg["name"], train_loader, input_dim=input_dim, device=runtime_info["device"], desc_prefix=f"[seed={seed}]")
            accuracy = evaluate(model, test_loader, device=runtime_info["device"])
            model_path = os.path.join(MODELS_DIR, dataset_cfg["model_file"])
            torch.save(model.state_dict(), model_path)
            trained_files.append(dataset_cfg["model_file"])
            print(f"    [{dataset_cfg['name']}] plain acc={accuracy:.4f}")
            progress.update(1)
    finally:
        progress.close()

    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "seed": int(seed),
        "model_dir": os.path.abspath(MODELS_DIR),
        "device": str(runtime_info["device"]),
        "runtime": dict(runtime_info, device=str(runtime_info["device"])),
        "models": trained_files,
    }
    atomic_write_json(os.path.join(MODELS_DIR, "model_manifest.json"), manifest)


def ensure_models(force_retrain: bool = False, runtime_info=None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
    expected_files = [dataset["model_file"] for dataset in DATASETS]

    needs_training = force_retrain
    if not os.path.exists(manifest_path):
        needs_training = True
    else:
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("schema_version") != MODEL_SCHEMA_VERSION:
                needs_training = True
        except Exception:
            needs_training = True

    for filename in expected_files:
        if not os.path.exists(os.path.join(MODELS_DIR, filename)):
            needs_training = True
            break

    if not needs_training:
        return

    print(">>> Plain models are missing or outdated. Retraining now...")
    if runtime_info is None:
        subprocess.run([sys.executable, os.path.join(CORE_DIR, "train_plain.py")], cwd=CORE_DIR, check=True)
    else:
        train_models_for_seed(env_int("ASS_TRAIN_SEED", 42), runtime_info, DATASETS)


def collect_plain_reference(dataset_cfg, device, eval_upsample: int, eval_max_samples: int):
    model_path = os.path.join(MODELS_DIR, dataset_cfg["model_file"])
    model = load_task_model(dataset_cfg["name"], dataset_cfg["input_shape"], model_path, device=device)
    test_loader = _cap_eval_loader(
        _load_test_loader(dataset_cfg["loader"], _batch_size_for_task(dataset_cfg["name"]), eval_upsample),
        eval_max_samples,
    )
    total = 0
    correct = 0
    wall_start = time.perf_counter_ns()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1).to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    wall_time = (time.perf_counter_ns() - wall_start) / 1e9
    return {
        "Task": dataset_cfg["name"],
        "Domain": dataset_cfg["domain"],
        "PlainAcc": correct / max(total, 1),
        "PlainPerSampleMs": _per_sample_ms(wall_time, total),
        "EvalSamples": total,
    }


def aggregate_plain_seed_runs(plain_seed_rows: List[Dict[str, object]]):
    grouped = {}
    for row in plain_seed_rows:
        grouped.setdefault((row["Task"], row["Domain"]), []).append(row)

    aggregated = []
    for (task_name, domain), rows in grouped.items():
        acc_values = [row["PlainAcc"] for row in rows if row["PlainAcc"] is not None]
        time_values = [row["PlainPerSampleMs"] for row in rows if row["PlainPerSampleMs"] is not None]
        acc_mean, acc_std = _mean_std(acc_values)
        time_mean, time_std = _mean_std(time_values)
        seeds = [str(row["Seed"]) for row in rows]
        aggregated.append(
            {
                "Task": task_name,
                "Domain": domain,
                "PlainAcc": acc_mean,
                "PlainAccStd": acc_std,
                "PlainPerSampleMs": time_mean,
                "PlainPerSampleMsStd": time_std,
                "EvalSamples": rows[0]["EvalSamples"],
                "Repeats": len(rows),
                "SeedsUsed": ",".join(seeds),
            }
        )
    return sorted(aggregated, key=lambda row: row["Task"])


def aggregate_main_seed_results(seed_rows: List[Dict[str, object]], plain_reference_map):
    grouped = {}
    for row in seed_rows:
        grouped.setdefault((row["Task"], row["Method"]), []).append(row)

    aggregated = []
    for (task_name, method_name), rows in grouped.items():
        profile = PROTOCOLS[method_name]
        success_like = [row for row in rows if row["Status"] in {"SUCCESS", "NA", "APPROX"}]
        skipped_only = rows and all(row["Status"] == "SKIPPED" for row in rows)
        if skipped_only:
            aggregated.append(
                {
                    "Task": task_name,
                    "Domain": rows[0]["Domain"],
                    "Method": method_name,
                    "Status": "SKIPPED",
                    "EvalSamples": None,
                    "TimingSamples": None,
                    "Repeats": 0,
                    "AccMean": None,
                    "AccStd": None,
                    "AccCI95Low": None,
                    "AccCI95High": None,
                    "AccDropVsPlain": None,
                    "PerSampleMsMean": None,
                    "PerSampleMsStd": None,
                    "TimeCI95Low": None,
                    "TimeCI95High": None,
                    "ThroughputMean": None,
                    "OnlineCommMB": None,
                    "OfflineSetupMB": None,
                    "LinearPct": None,
                    "ReLUPct": None,
                    "SeedMode": "multi_seed_train_repeats",
                    "SeedsUsed": ",".join(str(row["Seed"]) for row in rows),
                    "ImplementationScope": profile["implementation_scope"],
                    "ComparisonRole": profile["comparison_role"],
                    "CommMeasurementBasis": profile["comm_measurement_basis"],
                    "Parties": profile["parties"],
                    "InteractionRounds": profile["interaction_rounds"],
                    "DataSplit": profile["data_split"],
                    "ModelSplit": profile["model_split"],
                    "SinglePointDataExposure": profile["single_point_data_exposure"],
                    "SinglePointModelExposure": profile["single_point_model_exposure"],
                    "MinCollusionInput": profile["min_collusion_input"],
                    "MinCollusionModel": profile["min_collusion_model"],
                    "CommType": profile["comm_type"],
                    "Evidence": profile["evidence"],
                    "Error": None,
                }
            )
            continue
        if not success_like:
            source_row = rows[0]
            result = _result_error(task_name, method_name, RuntimeError(source_row.get("Error") or "no successful seed run"))
            result["Domain"] = source_row["Domain"]
            result["SeedMode"] = "multi_seed_train_repeats"
            result["SeedsUsed"] = ",".join(str(row["Seed"]) for row in rows)
            aggregated.append(result)
            continue

        source_row = success_like[0]
        acc_values = [row["AccMean"] for row in success_like if row["AccMean"] is not None]
        time_values = [row["PerSampleMsMean"] for row in success_like if row["PerSampleMsMean"] is not None]
        comm_values = [row["OnlineCommMB"] for row in success_like if row["OnlineCommMB"] is not None]
        offline_values = [row["OfflineSetupMB"] for row in success_like if row["OfflineSetupMB"] is not None]
        layer = _mean_layer(
            [{"Linear": row["LinearPct"], "ReLU": row["ReLUPct"]} for row in success_like if row["LinearPct"] is not None and row["ReLUPct"] is not None]
        )

        acc_mean, acc_std = _mean_std(acc_values)
        time_mean, time_std = _mean_std(time_values)
        acc_ci_low, acc_ci_high = _ci95(acc_mean, acc_std, len(acc_values), lower_bound=0.0, upper_bound=1.0)
        time_ci_low, time_ci_high = _ci95(time_mean, time_std, len(time_values), lower_bound=0.0)
        seeds = [str(row["Seed"]) for row in success_like]
        aggregated.append(
            {
                "Task": task_name,
                "Domain": source_row["Domain"],
                "Method": method_name,
                "Status": source_row["Status"],
                "EvalSamples": source_row["EvalSamples"],
                "TimingSamples": source_row["TimingSamples"],
                "Repeats": len(success_like),
                "AccMean": acc_mean,
                "AccStd": acc_std,
                "AccCI95Low": acc_ci_low,
                "AccCI95High": acc_ci_high,
                "AccDropVsPlain": (plain_reference_map[task_name]["PlainAcc"] - acc_mean) if acc_mean is not None else None,
                "PerSampleMsMean": time_mean,
                "PerSampleMsStd": time_std,
                "TimeCI95Low": time_ci_low,
                "TimeCI95High": time_ci_high,
                "ThroughputMean": _throughput(time_mean),
                "OnlineCommMB": float(sum(comm_values) / len(comm_values)) if comm_values else None,
                "OfflineSetupMB": float(sum(offline_values) / len(offline_values)) if offline_values else None,
                "LinearPct": layer["Linear"],
                "ReLUPct": layer["ReLU"],
                "SeedMode": "multi_seed_train_repeats",
                "SeedsUsed": ",".join(seeds),
                "ImplementationScope": profile["implementation_scope"],
                "ComparisonRole": profile["comparison_role"],
                "CommMeasurementBasis": profile["comm_measurement_basis"],
                "Parties": profile["parties"],
                "InteractionRounds": profile["interaction_rounds"],
                "DataSplit": profile["data_split"],
                "ModelSplit": profile["model_split"],
                "SinglePointDataExposure": profile["single_point_data_exposure"],
                "SinglePointModelExposure": profile["single_point_model_exposure"],
                "MinCollusionInput": profile["min_collusion_input"],
                "MinCollusionModel": profile["min_collusion_model"],
                "CommType": profile["comm_type"],
                "Evidence": profile["evidence"],
                "Error": None,
            }
        )
    return sorted(aggregated, key=lambda row: (row["Task"], MAIN_PROTOCOL_ORDER.index(row["Method"])))


def build_he_boundary_summary(main_results):
    rows = []
    ass_rows = {row["Task"]: row for row in main_results if row["Method"] == "ASS (Ours)" and row["Status"] == "SUCCESS"}
    grouped = {}
    for row in main_results:
        grouped.setdefault(row["Task"], {})[row["Method"]] = row

    for task_name, per_task in sorted(grouped.items()):
        ass_row = ass_rows.get(task_name)
        ckks_row = per_task.get("CKKS (HE)")
        paillier_row = per_task.get("Paillier (PHE)")
        if ass_row is None:
            continue
        if not ckks_row and not paillier_row:
            continue

        ckks_ms = ckks_row["PerSampleMsMean"] if ckks_row and ckks_row["Status"] == "NA" else None
        paillier_ms = paillier_row["PerSampleMsMean"] if paillier_row and paillier_row["Status"] == "APPROX" else None
        ass_ms = ass_row["PerSampleMsMean"]
        rows.append(
            {
                "Task": task_name,
                "ASSPerSampleMs": ass_ms,
                "CKKSStatus": ckks_row["Status"] if ckks_row else None,
                "CKKSPerSampleMs": ckks_ms,
                "CKKSvsASS": (ckks_ms / ass_ms) if ckks_ms is not None and ass_ms else None,
                "PaillierStatus": paillier_row["Status"] if paillier_row else None,
                "PaillierPerSampleMs": paillier_ms,
                "PailliervsASS": (paillier_ms / ass_ms) if paillier_ms is not None and ass_ms else None,
                "SummaryNote": "Only tasks with valid HE/PHE boundary outputs should be used for direct timing comparison.",
            }
        )
    return rows


def validate_ckks_payload(payload):
    if payload is None:
        raise RuntimeError("CKKS micro-benchmark returned no payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"CKKS micro-benchmark returned invalid payload: {payload!r}")

    status = str(payload.get("Status") or "").strip().upper()
    if status in {"FAILED", "ERROR"}:
        raise RuntimeError(payload.get("Error") or payload.get("Comm") or f"CKKS payload status={status}")
    for error_key in ("Error", "error", "Exception", "exception"):
        error_value = payload.get(error_key)
        if error_value:
            raise RuntimeError(f"CKKS micro-benchmark error payload: {error_value}")

    comm_value = payload.get("Comm")
    if isinstance(comm_value, str) and comm_value.strip().lower().startswith("error:"):
        raise RuntimeError(comm_value)

    try:
        ckks_time = float(payload.get("Time"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"CKKS micro-benchmark returned non-numeric timing: {payload.get('Time')!r}") from exc
    if ckks_time <= 0:
        raise RuntimeError(f"CKKS micro-benchmark returned non-positive timing: {ckks_time}")
    return ckks_time


def execute_method_once(method_name: str, dataset_cfg, eval_upsample: int, eval_max_samples: int, runtime_device, he_cfg):
    task_name = dataset_cfg["name"]
    model_path = os.path.join(MODELS_DIR, dataset_cfg["model_file"])
    test_loader = _cap_eval_loader(
        _load_test_loader(dataset_cfg["loader"], _batch_size_for_task(task_name), eval_upsample),
        eval_max_samples,
    )

    if method_name == "ASS (Ours)":
        acc, total_time, comm_mb, layer, samples, offline_mb = run_ass_inference(model_path, test_loader, dataset_cfg["input_shape"], task_name=task_name, device=runtime_device)
        return {"status": "SUCCESS", "acc": acc, "per_sample_ms": _per_sample_ms(total_time, samples), "online_comm_mb": comm_mb, "offline_setup_mb": offline_mb, "layer": layer, "eval_samples": len(test_loader.dataset), "timing_samples": samples}

    if method_name == "2Cloud-D (Data-only)":
        acc, total_time, comm_mb, layer, samples, offline_mb = run_2cloud_inference(model_path, test_loader, dataset_cfg["input_shape"], task_name=task_name, device=runtime_device)
        return {"status": "SUCCESS", "acc": acc, "per_sample_ms": _per_sample_ms(total_time, samples), "online_comm_mb": comm_mb, "offline_setup_mb": offline_mb, "layer": layer, "eval_samples": len(test_loader.dataset), "timing_samples": samples}

    if method_name == "3Share-DM (3-party)":
        acc, total_time, comm_mb, layer, samples, offline_mb = run_three_share_inference(model_path, test_loader, dataset_cfg["input_shape"], task_name=task_name, device=runtime_device)
        return {"status": "SUCCESS", "acc": acc, "per_sample_ms": _per_sample_ms(total_time, samples), "online_comm_mb": comm_mb, "offline_setup_mb": offline_mb, "layer": layer, "eval_samples": len(test_loader.dataset), "timing_samples": samples}

    if method_name == "SecureNN (3PC)":
        acc, avg_time_s, comm_mb, layer, timing_samples, offline_mb = run_securenn_inference(model_path, test_loader, dataset_cfg["input_shape"], task_name=task_name, device=runtime_device)
        return {"status": "SUCCESS", "acc": acc, "per_sample_ms": avg_time_s * 1000.0, "online_comm_mb": comm_mb, "offline_setup_mb": offline_mb, "layer": layer, "eval_samples": len(test_loader.dataset), "timing_samples": timing_samples}

    if method_name == "CKKS (HE)":
        if not he_cfg["run_he"] or (he_cfg["task_filter"] and task_name not in he_cfg["task_filter"]):
            return {"status": "SKIPPED", "acc": None, "per_sample_ms": None, "online_comm_mb": None, "offline_setup_mb": None, "layer": None, "eval_samples": None, "timing_samples": None}
        script_path = os.path.join(CORE_DIR, "baselines", "inference_ckks_benchmark.py")
        proc = subprocess.run([sys.executable, script_path, task_name, model_path], cwd=CORE_DIR, capture_output=True, text=True, timeout=he_cfg["ckks_timeout_seconds"])
        if proc.returncode != 0 or "JSON_START" not in proc.stdout:
            raise RuntimeError(proc.stderr.strip() or "CKKS micro-benchmark failed")
        payload = json.loads(proc.stdout.split("JSON_START")[-1].strip())
        ckks_time = validate_ckks_payload(payload)
        return {"status": "NA", "acc": None, "per_sample_ms": ckks_time * 1000.0, "online_comm_mb": None, "offline_setup_mb": None, "layer": payload.get("Layer"), "eval_samples": None, "timing_samples": 1}

    if method_name == "Paillier (PHE)":
        if not he_cfg["run_he"] or (he_cfg["task_filter"] and task_name not in he_cfg["task_filter"]):
            return {"status": "SKIPPED", "acc": None, "per_sample_ms": None, "online_comm_mb": None, "offline_setup_mb": None, "layer": None, "eval_samples": None, "timing_samples": None}
        sampled_loader = _cap_eval_loader(_load_test_loader(dataset_cfg["loader"], 1, eval_upsample), he_cfg["paillier_eval_cap"])
        acc, avg_time_s, layer, eval_samples = run_paillier_inference(
            model_path=model_path,
            test_loader=sampled_loader,
            input_shape=dataset_cfg["input_shape"],
            is_medical=dataset_cfg["is_medical"],
            task_name=task_name,
            progress_prefix=f"[{task_name}] Paillier",
            progress_interval=he_cfg["paillier_progress_interval"],
            max_eval_samples=he_cfg["paillier_eval_cap"],
            max_total_seconds=he_cfg["paillier_max_seconds"],
            key_bits=he_cfg["paillier_key_bits"],
        )
        return {"status": "APPROX", "acc": acc, "per_sample_ms": avg_time_s * 1000.0, "online_comm_mb": 5.0 if dataset_cfg["is_medical"] else 50.0, "offline_setup_mb": None, "layer": layer, "eval_samples": eval_samples, "timing_samples": eval_samples}

    raise ValueError(f"Unsupported method: {method_name}")


def aggregate_method_results(method_name: str, task_name: str, plain_acc: float, run_rows: List[Dict[str, object]]):
    profile = PROTOCOLS[method_name]
    status_values = [row["status"] for row in run_rows]
    final_status = next((value for value in status_values if value != "SUCCESS"), "SUCCESS")
    if final_status == "FAILED":
        raise RuntimeError("aggregate_method_results received failed rows")
    if final_status == "SKIPPED":
        return {
            "Task": task_name, "Method": method_name, "Status": "SKIPPED", "EvalSamples": None, "TimingSamples": None, "Repeats": 0,
            "AccMean": None, "AccStd": None, "AccCI95Low": None, "AccCI95High": None, "AccDropVsPlain": None,
            "PerSampleMsMean": None, "PerSampleMsStd": None, "TimeCI95Low": None, "TimeCI95High": None, "ThroughputMean": None,
            "OnlineCommMB": None, "OfflineSetupMB": None, "LinearPct": None, "ReLUPct": None,
            "SeedMode": "single_seed_reference", "SeedsUsed": "", "ImplementationScope": profile["implementation_scope"],
            "ComparisonRole": profile["comparison_role"], "CommMeasurementBasis": profile["comm_measurement_basis"],
            "Parties": profile["parties"], "InteractionRounds": profile["interaction_rounds"], "DataSplit": profile["data_split"],
            "ModelSplit": profile["model_split"], "SinglePointDataExposure": profile["single_point_data_exposure"],
            "SinglePointModelExposure": profile["single_point_model_exposure"], "MinCollusionInput": profile["min_collusion_input"],
            "MinCollusionModel": profile["min_collusion_model"], "CommType": profile["comm_type"], "Evidence": profile["evidence"], "Error": None,
        }

    acc_values = [row["acc"] for row in run_rows if row["acc"] is not None]
    time_values = [row["per_sample_ms"] for row in run_rows if row["per_sample_ms"] is not None]
    comm_values = [row["online_comm_mb"] for row in run_rows if row["online_comm_mb"] is not None]
    offline_values = [row["offline_setup_mb"] for row in run_rows if row["offline_setup_mb"] is not None]
    layer = _mean_layer([row["layer"] for row in run_rows if row["layer"] is not None])

    acc_mean, acc_std = _mean_std(acc_values)
    time_mean, time_std = _mean_std(time_values)
    acc_ci_low, acc_ci_high = _ci95(acc_mean, acc_std, len(acc_values), lower_bound=0.0, upper_bound=1.0)
    time_ci_low, time_ci_high = _ci95(time_mean, time_std, len(time_values), lower_bound=0.0)

    return {
        "Task": task_name, "Method": method_name, "Status": final_status, "EvalSamples": run_rows[0]["eval_samples"], "TimingSamples": run_rows[0]["timing_samples"], "Repeats": len(run_rows),
        "AccMean": acc_mean, "AccStd": acc_std, "AccCI95Low": acc_ci_low, "AccCI95High": acc_ci_high, "AccDropVsPlain": (plain_acc - acc_mean) if acc_mean is not None else None,
        "PerSampleMsMean": time_mean, "PerSampleMsStd": time_std, "TimeCI95Low": time_ci_low, "TimeCI95High": time_ci_high, "ThroughputMean": _throughput(time_mean),
        "OnlineCommMB": float(sum(comm_values) / len(comm_values)) if comm_values else None, "OfflineSetupMB": float(sum(offline_values) / len(offline_values)) if offline_values else None,
        "LinearPct": layer["Linear"], "ReLUPct": layer["ReLU"], "SeedMode": "single_seed_reference", "SeedsUsed": "",
        "ImplementationScope": profile["implementation_scope"], "ComparisonRole": profile["comparison_role"],
        "CommMeasurementBasis": profile["comm_measurement_basis"], "Parties": profile["parties"], "InteractionRounds": profile["interaction_rounds"],
        "DataSplit": profile["data_split"], "ModelSplit": profile["model_split"], "SinglePointDataExposure": profile["single_point_data_exposure"],
        "SinglePointModelExposure": profile["single_point_model_exposure"], "MinCollusionInput": profile["min_collusion_input"], "MinCollusionModel": profile["min_collusion_model"],
        "CommType": profile["comm_type"], "Evidence": profile["evidence"], "Error": None,
    }


def run_main_matrix(dataset_cfgs, plain_reference_map, runtime_info, config):
    results = []
    he_cfg = {
        "run_he": config["run_he"],
        "task_filter": config["he_task_filter"],
        "ckks_timeout_seconds": config["ckks_timeout_seconds"],
        "paillier_eval_cap": config["paillier_eval_cap"],
        "paillier_max_seconds": config["paillier_max_seconds"],
        "paillier_key_bits": config["paillier_key_bits"],
        "paillier_progress_interval": config["paillier_progress_interval"],
    }
    total_steps = len(dataset_cfgs) * sum(config["main_repeats"] if method_name in MAIN_REPEAT_METHODS else 1 for method_name in MAIN_PROTOCOL_ORDER)
    progress = _make_progress(total=total_steps, desc="Main matrix", unit="run")

    try:
        for dataset_cfg in dataset_cfgs:
            task_name = dataset_cfg["name"]
            print(f"\n>>> Running main matrix for {task_name}")
            plain_acc = plain_reference_map[task_name]["PlainAcc"]
            for method_name in MAIN_PROTOCOL_ORDER:
                repeats = config["main_repeats"] if method_name in MAIN_REPEAT_METHODS else 1
                run_rows = []
                try:
                    for repeat_idx in range(repeats):
                        progress.set_postfix_str(f"{task_name} | {method_name} | {repeat_idx + 1}/{repeats}")
                        run_rows.append(execute_method_once(method_name, dataset_cfg, config["eval_upsample"], config["eval_max_samples"], runtime_info["device"], he_cfg))
                        progress.update(1)
                    aggregated = aggregate_method_results(method_name, task_name, plain_acc, run_rows)
                except Exception as exc:
                    missing_steps = max(0, repeats - len(run_rows))
                    if missing_steps:
                        progress.update(missing_steps)
                    aggregated = _result_error(task_name, method_name, exc)
                aggregated["Domain"] = dataset_cfg["domain"]
                results.append(aggregated)
    finally:
        progress.close()
    return results


def run_multi_seed_main_matrix(dataset_cfgs, runtime_info, config):
    repeat_seeds = config["repeat_seeds"]
    if not repeat_seeds:
        raise ValueError("run_multi_seed_main_matrix requires repeat_seeds")

    seed_rows = []
    plain_seed_rows = []
    reference_seed = repeat_seeds[-1]
    total_steps = len(repeat_seeds) * len(dataset_cfgs)
    progress = _make_progress(total=total_steps, desc="Multi-seed retraining", unit="task")

    try:
        for seed in repeat_seeds:
            train_models_for_seed(seed, runtime_info, dataset_cfgs)
            plain_reference_map = {}
            for dataset_cfg in dataset_cfgs:
                progress.set_postfix_str(f"seed={seed} | {dataset_cfg['name']}")
                row = collect_plain_reference(dataset_cfg, runtime_info["device"], config["eval_upsample"], config["eval_max_samples"])
                row["Seed"] = seed
                plain_seed_rows.append(row)
                plain_reference_map[row["Task"]] = row
                progress.update(1)

            seed_config = dict(config)
            seed_config["main_repeats"] = 1
            if seed != reference_seed:
                seed_config["run_he"] = False
            seed_result_rows = run_main_matrix(dataset_cfgs, plain_reference_map, runtime_info, seed_config)
            for row in seed_result_rows:
                row["Seed"] = seed
                seed_rows.append(row)
    finally:
        progress.close()

    plain_rows = aggregate_plain_seed_runs(plain_seed_rows)
    plain_reference_map = {row["Task"]: row for row in plain_rows}
    main_results = aggregate_main_seed_results(seed_rows, plain_reference_map)
    return plain_rows, plain_seed_rows, main_results, seed_rows


def run_round_ablation(dataset_cfgs, runtime_info, config):
    rows = []
    total_steps = sum(1 for cfg in dataset_cfgs if cfg["round_ablation"]) * len(config["round_values"]) * len(ROUND_METHODS)
    progress = _make_progress(total=total_steps, desc="Round ablation", unit="run")
    try:
        for dataset_cfg in dataset_cfgs:
            if not dataset_cfg["round_ablation"]:
                continue
            task_name = dataset_cfg["name"]
            model_path = os.path.join(MODELS_DIR, dataset_cfg["model_file"])
            base_loader = _cap_eval_loader(_load_test_loader(dataset_cfg["loader"], _batch_size_for_task(task_name), config["eval_upsample"]), config["round_eval_cap"])
            for rounds in config["round_values"]:
                for method_name in ROUND_METHODS:
                    progress.set_postfix_str(f"{task_name} | {method_name} | rounds={rounds}")
                    try:
                        if method_name == "ASS (Ours)":
                            acc, total_time, comm_mb, _, samples, _ = run_ass_inference(model_path, base_loader, dataset_cfg["input_shape"], task_name=task_name, interaction_rounds=rounds, device=runtime_info["device"])
                        elif method_name == "2Cloud-D (Data-only)":
                            acc, total_time, comm_mb, _, samples, _ = run_2cloud_inference(model_path, base_loader, dataset_cfg["input_shape"], task_name=task_name, interaction_rounds=rounds, device=runtime_info["device"])
                        else:
                            acc, total_time, comm_mb, _, samples, _ = run_three_share_inference(model_path, base_loader, dataset_cfg["input_shape"], task_name=task_name, interaction_rounds=rounds, device=runtime_info["device"])
                        rows.append({"Task": task_name, "Method": method_name, "Rounds": rounds, "Acc": acc, "PerSampleMs": _per_sample_ms(total_time, samples), "OnlineCommMB": comm_mb * max(rounds, 1), "EvalSamples": len(base_loader.dataset), "Error": None})
                    except Exception as exc:
                        rows.append({"Task": task_name, "Method": method_name, "Rounds": rounds, "Acc": None, "PerSampleMs": None, "OnlineCommMB": None, "EvalSamples": None, "Error": str(exc)})
                    finally:
                        progress.update(1)
    finally:
        progress.close()
    return rows


def build_long_tail_loader(test_loader, ratio: int, seed: int = 42):
    dataset = test_loader.dataset
    xs = []
    ys = []
    for idx in range(len(dataset)):
        data, target = dataset[idx]
        xs.append(data)
        ys.append(int(target))
    x_tensor = torch.stack(xs)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    unique_labels = y_tensor.unique(sorted=True).tolist()
    if len(unique_labels) != 2:
        return test_loader, None

    label_a, label_b = unique_labels
    idx_a = (y_tensor == label_a).nonzero(as_tuple=False).view(-1).numpy()
    idx_b = (y_tensor == label_b).nonzero(as_tuple=False).view(-1).numpy()
    majority_idx, minority_idx = (idx_a, idx_b) if len(idx_a) >= len(idx_b) else (idx_b, idx_a)
    tail_label = label_b if len(idx_a) >= len(idx_b) else label_a

    rng = np.random.RandomState(seed + ratio)
    if ratio <= 1:
        keep_major = min(len(majority_idx), len(minority_idx))
        keep_minor = keep_major
    else:
        keep_major = len(majority_idx)
        keep_minor = max(1, keep_major // ratio)

    major_pick = rng.choice(majority_idx, size=keep_major, replace=keep_major > len(majority_idx))
    minor_pick = rng.choice(minority_idx, size=keep_minor, replace=keep_minor > len(minority_idx))
    selected = np.concatenate([major_pick, minor_pick])
    rng.shuffle(selected)
    subset = TensorDataset(x_tensor[selected], y_tensor[selected])
    return DataLoader(subset, batch_size=test_loader.batch_size, shuffle=False), int(tail_label)


def run_long_tail_analysis(dataset_cfgs, runtime_info, config):
    rows = []
    total_steps = sum(1 for cfg in dataset_cfgs if cfg["long_tail"]) * len(config["longtail_ratios"]) * len(LONG_TAIL_METHODS)
    progress = _make_progress(total=total_steps, desc="Long-tail analysis", unit="run")
    try:
        for dataset_cfg in dataset_cfgs:
            if not dataset_cfg["long_tail"]:
                continue
            task_name = dataset_cfg["name"]
            model_path = os.path.join(MODELS_DIR, dataset_cfg["model_file"])
            base_loader = _cap_eval_loader(_load_test_loader(dataset_cfg["loader"], _batch_size_for_task(task_name), config["eval_upsample"]), config["longtail_eval_cap"])
            for ratio in config["longtail_ratios"]:
                long_tail_loader, tail_label = build_long_tail_loader(base_loader, ratio)
                for method_name in LONG_TAIL_METHODS:
                    progress.set_postfix_str(f"{task_name} | {method_name} | ratio={ratio}:1")
                    if method_name == "2Cloud-D (Data-only)":
                        result = run_shared_protocol_inference(model_path, long_tail_loader, dataset_cfg["input_shape"], task_name=task_name, data_shares=2, model_shares=1, interaction_rounds=1, device=runtime_info["device"], return_predictions=True)
                    elif method_name == "ASS (Ours)":
                        result = run_shared_protocol_inference(model_path, long_tail_loader, dataset_cfg["input_shape"], task_name=task_name, data_shares=2, model_shares=2, interaction_rounds=1, device=runtime_info["device"], return_predictions=True)
                    else:
                        result = run_shared_protocol_inference(model_path, long_tail_loader, dataset_cfg["input_shape"], task_name=task_name, data_shares=3, model_shares=3, interaction_rounds=1, device=runtime_info["device"], return_predictions=True)

                    predictions = np.array(result["Predictions"])
                    targets = np.array(result["Targets"])
                    rows.append(
                        {
                            "Task": task_name,
                            "Method": method_name,
                            "Ratio": f"{ratio}:1" if ratio > 1 else "1:1",
                            "Acc": result["Acc"],
                            "BalancedAcc": _balanced_accuracy(targets, predictions),
                            "MacroF1": _macro_f1(targets, predictions),
                            "TailRecall": _tail_recall(targets, predictions, tail_label),
                            "PerSampleMs": _per_sample_ms(result["Time"], result["Samples"]),
                            "OnlineCommMB": result["Comm"],
                            "Samples": result["Samples"],
                        }
                    )
                    progress.update(1)
    finally:
        progress.close()
    return rows


def _ratio_label(ratio: int) -> str:
    return "1:1" if int(ratio) <= 1 else f"{int(ratio)}:1"


def _prediction_metrics(targets, predictions, tail_label: Optional[int] = None):
    targets_np = np.asarray(targets, dtype=np.int64)
    predictions_np = np.asarray(predictions, dtype=np.int64)
    acc = float((predictions_np == targets_np).mean()) if len(targets_np) > 0 else 0.0
    return {
        "Acc": acc,
        "BalancedAcc": _balanced_accuracy(targets_np, predictions_np),
        "MacroF1": _macro_f1(targets_np, predictions_np),
        "TailRecall": _tail_recall(targets_np, predictions_np, tail_label) if tail_label is not None else None,
        "Samples": int(len(targets_np)),
    }


def _evaluate_model_metrics(model, test_loader, device, tail_label: Optional[int] = None):
    predictions = []
    targets = []
    total = 0
    wall_start = time.perf_counter_ns()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1).to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())
            total += data.size(0)
    wall_time = (time.perf_counter_ns() - wall_start) / 1e9
    metrics = _prediction_metrics(targets, predictions, tail_label=tail_label)
    metrics["PerSampleMs"] = _per_sample_ms(wall_time, total)
    return metrics


def _load_medical_train_ratio_bundle(dataset_cfg, ratio: int, batch_size: int, upsample_factor: int, seed: int):
    loaded = dataset_cfg["loader"](
        batch_size=batch_size,
        upsample_factor=upsample_factor,
        train_ratio=ratio,
        seed=seed,
        return_metadata=True,
    )
    train_loader, test_loader, input_dim, meta = loaded
    return train_loader, test_loader, input_dim, meta


def _medical_longtail_model_paths(task_name: str, ratio: int):
    safe_task = task_name.lower()
    model_path = os.path.join(MEDICAL_LONGTAIL_DIR, f"{safe_task}_r{int(ratio)}.pth")
    meta_path = os.path.join(MEDICAL_LONGTAIL_DIR, f"{safe_task}_r{int(ratio)}.json")
    return model_path, meta_path


def ensure_medical_longtail_models(dataset_cfgs, runtime_info, config):
    if not config["run_model_split_necessity"]:
        return {}

    os.makedirs(MEDICAL_LONGTAIL_DIR, exist_ok=True)
    model_records = {}
    medical_cfgs = [cfg for cfg in dataset_cfgs if cfg["is_medical"] and cfg["long_tail"]]
    total_steps = len(medical_cfgs) * len(config["necessity_ratios"])
    progress = _make_progress(total=total_steps, desc="Medical long-tail training", unit="model")

    try:
        for dataset_cfg in medical_cfgs:
            task_name = dataset_cfg["name"]
            for ratio in config["necessity_ratios"]:
                progress.set_postfix_str(f"{task_name} | {_ratio_label(ratio)}")
                model_path, meta_path = _medical_longtail_model_paths(task_name, ratio)

                if (not config["force_retrain_longtail_models"]) and os.path.exists(model_path) and os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as handle:
                        model_records[(task_name, int(ratio))] = json.load(handle)
                    progress.update(1)
                    continue

                train_loader, test_loader, input_dim, meta = _load_medical_train_ratio_bundle(
                    dataset_cfg,
                    ratio=ratio,
                    batch_size=64,
                    upsample_factor=1,
                    seed=config["necessity_seed"],
                )
                model = train_task_model(
                    task_name,
                    train_loader,
                    input_dim=input_dim,
                    device=runtime_info["device"],
                    desc_prefix=f"{task_name} {_ratio_label(ratio)}",
                )
                plain_acc = evaluate(model, test_loader, device=runtime_info["device"])
                torch.save(model.state_dict(), model_path)

                record = {
                    "Task": task_name,
                    "TrainRatio": _ratio_label(ratio),
                    "TrainRatioValue": int(ratio),
                    "ModelPath": model_path,
                    "InputDim": int(input_dim),
                    "PlainAcc": float(plain_acc),
                    "MajorityLabel": meta.get("majority_label"),
                    "MinorityLabel": meta.get("minority_label"),
                    "TrainCountsBefore": meta.get("counts_before"),
                    "TrainCountsAfter": meta.get("counts_after"),
                }
                with open(meta_path, "w", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                model_records[(task_name, int(ratio))] = record
                progress.update(1)
    finally:
        progress.close()

    return model_records


def _run_model_split_protocol(method_name: str, dataset_cfg, model_path: str, test_loader, runtime_device):
    if method_name == "2Cloud-D (Data-only)":
        return run_shared_protocol_inference(
            model_path=model_path,
            test_loader=test_loader,
            input_shape=dataset_cfg["input_shape"],
            task_name=dataset_cfg["name"],
            data_shares=2,
            model_shares=1,
            interaction_rounds=1,
            device=runtime_device,
            return_predictions=True,
        )
    if method_name == "ASS (Ours)":
        return run_shared_protocol_inference(
            model_path=model_path,
            test_loader=test_loader,
            input_shape=dataset_cfg["input_shape"],
            task_name=dataset_cfg["name"],
            data_shares=2,
            model_shares=2,
            interaction_rounds=1,
            device=runtime_device,
            return_predictions=True,
        )
    raise ValueError(f"Unsupported model-split necessity method: {method_name}")


def run_model_split_necessity(dataset_cfgs, runtime_info, config):
    if not config["run_model_split_necessity"]:
        return []

    rows = []
    model_records = ensure_medical_longtail_models(dataset_cfgs, runtime_info, config)
    medical_cfgs = [cfg for cfg in dataset_cfgs if cfg["is_medical"] and cfg["long_tail"]]
    total_steps = len(medical_cfgs) * len(config["necessity_ratios"]) * len(MODEL_SPLIT_NECESSITY_METHODS)
    progress = _make_progress(total=total_steps, desc="Model-split necessity", unit="run")

    try:
        for dataset_cfg in medical_cfgs:
            task_name = dataset_cfg["name"]
            for ratio in config["necessity_ratios"]:
                record = model_records[(task_name, int(ratio))]
                _, test_loader, _, meta = _load_medical_train_ratio_bundle(
                    dataset_cfg,
                    ratio=ratio,
                    batch_size=_batch_size_for_task(task_name),
                    upsample_factor=1,
                    seed=config["necessity_seed"],
                )
                tail_label = meta.get("minority_label")
                model_path = record["ModelPath"]

                for method_name in MODEL_SPLIT_NECESSITY_METHODS:
                    progress.set_postfix_str(f"{task_name} | {_ratio_label(ratio)} | {method_name}")
                    protocol_result = _run_model_split_protocol(method_name, dataset_cfg, model_path, test_loader, runtime_info["device"])
                    protocol_metrics = _prediction_metrics(protocol_result["Targets"], protocol_result["Predictions"], tail_label=tail_label)
                    protocol_metrics["PerSampleMs"] = _per_sample_ms(protocol_result["Time"], protocol_result["Samples"])

                    if method_name == "2Cloud-D (Data-only)":
                        cloud_model = load_task_model(task_name, dataset_cfg["input_shape"], model_path, device=runtime_info["device"])
                        cloud_mode = "full_model"
                        cloud_executable = 1
                        cloud_note = "Single cloud owns the full plaintext model."
                    else:
                        cloud_model = load_task_model_share(
                            task_name,
                            dataset_cfg["input_shape"],
                            model_path,
                            model_shares=2,
                            share_index=0,
                            share_seed=config["necessity_seed"],
                            device=runtime_info["device"],
                        )
                        cloud_mode = "single_share_proxy"
                        cloud_executable = 0
                        cloud_note = "Single cloud sees only one additive parameter share."

                    cloud_metrics = _evaluate_model_metrics(cloud_model, test_loader, runtime_info["device"], tail_label=tail_label)

                    rows.append(
                        {
                            "Task": task_name,
                            "TrainRatio": _ratio_label(ratio),
                            "TrainRatioValue": int(ratio),
                            "Method": method_name,
                            "MajorityLabel": meta.get("majority_label"),
                            "MinorityLabel": tail_label,
                            "TrainMajorityCount": meta.get("counts_after", {}).get(str(meta.get("majority_label")), meta.get("counts_after", {}).get(meta.get("majority_label"))),
                            "TrainMinorityCount": meta.get("counts_after", {}).get(str(tail_label), meta.get("counts_after", {}).get(tail_label)),
                            "ProtocolAcc": protocol_metrics["Acc"],
                            "ProtocolBalancedAcc": protocol_metrics["BalancedAcc"],
                            "ProtocolMacroF1": protocol_metrics["MacroF1"],
                            "ProtocolTailRecall": protocol_metrics["TailRecall"],
                            "ProtocolPerSampleMs": protocol_metrics["PerSampleMs"],
                            "ProtocolOnlineCommMB": protocol_result["Comm"],
                            "SingleCloudExecutable": cloud_executable,
                            "SingleCloudMode": cloud_mode,
                            "SingleCloudAcc": cloud_metrics["Acc"],
                            "SingleCloudBalancedAcc": cloud_metrics["BalancedAcc"],
                            "SingleCloudMacroF1": cloud_metrics["MacroF1"],
                            "SingleCloudTailRecall": cloud_metrics["TailRecall"],
                            "SingleCloudPerSampleMs": cloud_metrics["PerSampleMs"],
                            "ExposureGapAcc": protocol_metrics["Acc"] - cloud_metrics["Acc"],
                            "ExposureGapTailRecall": (protocol_metrics["TailRecall"] - cloud_metrics["TailRecall"]) if protocol_metrics["TailRecall"] is not None and cloud_metrics["TailRecall"] is not None else None,
                            "PlainAcc": record["PlainAcc"],
                            "ModelPath": model_path,
                            "CloudNote": cloud_note,
                        }
                    )
                    progress.update(1)
    finally:
        progress.close()

    return rows


def build_model_split_necessity_summary(rows):
    summary_rows = []
    keys = sorted({(row["TrainRatio"], row["Method"]) for row in rows})
    for train_ratio, method_name in keys:
        matched = [row for row in rows if row["TrainRatio"] == train_ratio and row["Method"] == method_name]
        summary_rows.append(
            {
                "TrainRatio": train_ratio,
                "Method": method_name,
                "Tasks": len(matched),
                "MeanProtocolAcc": statistics.mean(row["ProtocolAcc"] for row in matched),
                "MeanProtocolTailRecall": statistics.mean(row["ProtocolTailRecall"] for row in matched if row["ProtocolTailRecall"] is not None),
                "MeanSingleCloudAcc": statistics.mean(row["SingleCloudAcc"] for row in matched),
                "MeanSingleCloudTailRecall": statistics.mean(row["SingleCloudTailRecall"] for row in matched if row["SingleCloudTailRecall"] is not None),
                "MeanExposureGapAcc": statistics.mean(row["ExposureGapAcc"] for row in matched),
                "MeanExposureGapTailRecall": statistics.mean(row["ExposureGapTailRecall"] for row in matched if row["ExposureGapTailRecall"] is not None),
                "ExecutableShare": statistics.mean(row["SingleCloudExecutable"] for row in matched),
            }
        )
    return summary_rows


def build_transfer_summary(main_results):
    rows = []
    methods = ["2Cloud-D (Data-only)", "ASS (Ours)", "3Share-DM (3-party)", "SecureNN (3PC)"]
    for method_name in methods:
        method_rows = [row for row in main_results if row["Method"] == method_name and row["Status"] == "SUCCESS"]
        general_rows = [row for row in method_rows if row["Domain"] == "general"]
        medical_rows = [row for row in method_rows if row["Domain"] == "medical"]
        if not general_rows or not medical_rows:
            continue
        rows.append(
            {
                "Method": method_name,
                "GeneralMeanAcc": statistics.mean(row["AccMean"] for row in general_rows if row["AccMean"] is not None),
                "MedicalMeanAcc": statistics.mean(row["AccMean"] for row in medical_rows if row["AccMean"] is not None),
                "GeneralMeanLatencyMs": statistics.mean(row["PerSampleMsMean"] for row in general_rows if row["PerSampleMsMean"] is not None),
                "MedicalMeanLatencyMs": statistics.mean(row["PerSampleMsMean"] for row in medical_rows if row["PerSampleMsMean"] is not None),
                "AccDomainGap": statistics.mean(row["AccMean"] for row in medical_rows if row["AccMean"] is not None) - statistics.mean(row["AccMean"] for row in general_rows if row["AccMean"] is not None),
                "LatencyDomainGap": statistics.mean(row["PerSampleMsMean"] for row in medical_rows if row["PerSampleMsMean"] is not None) - statistics.mean(row["PerSampleMsMean"] for row in general_rows if row["PerSampleMsMean"] is not None),
            }
        )
    return rows


def build_split_ablation(main_results):
    rows = []
    for row in main_results:
        if row["Method"] not in {"2Cloud-D (Data-only)", "ASS (Ours)", "3Share-DM (3-party)"}:
            continue
        rows.append({"Task": row["Task"], "Method": row["Method"], "Status": row["Status"], "AccMean": row["AccMean"], "AccDropVsPlain": row["AccDropVsPlain"], "PerSampleMsMean": row["PerSampleMsMean"], "OnlineCommMB": row["OnlineCommMB"], "OfflineSetupMB": row["OfflineSetupMB"], "SinglePointModelExposure": row["SinglePointModelExposure"], "MinCollusionModel": row["MinCollusionModel"]})
    return rows


def build_security_metrics():
    rows = []
    for method_name, profile in PROTOCOLS.items():
        rows.append(
            {
                "Method": method_name,
                "Parties": profile["parties"],
                "DataSplit": profile["data_split"],
                "ModelSplit": profile["model_split"],
                "InteractionRounds": profile["interaction_rounds"],
                "SinglePointDataExposure": profile["single_point_data_exposure"],
                "SinglePointModelExposure": profile["single_point_model_exposure"],
                "MinCollusionInput": profile["min_collusion_input"],
                "MinCollusionModel": profile["min_collusion_model"],
                "ImplementationScope": profile["implementation_scope"],
                "ComparisonRole": profile["comparison_role"],
                "CommMeasurementBasis": profile["comm_measurement_basis"],
                "Evidence": profile["evidence"],
            }
        )
    return rows


def run_loopback_comm_audit(dataset_cfgs, runtime_info, config):
    if not config["run_loopback_audit"]:
        return []

    selected_tasks = set(config["loopback_tasks"]) if config["loopback_tasks"] else {cfg["name"] for cfg in dataset_cfgs}
    selected_methods = [method for method in config["loopback_methods"] if method in TRACEABLE_METHODS]
    selected_cfgs = [cfg for cfg in dataset_cfgs if cfg["name"] in selected_tasks]
    rows = []
    total_steps = len(selected_cfgs) * len(selected_methods)
    progress = _make_progress(total=total_steps, desc="Loopback comm audit", unit="run")

    try:
        for dataset_cfg in selected_cfgs:
            task_name = dataset_cfg["name"]
            model_path = os.path.join(MODELS_DIR, dataset_cfg["model_file"])
            test_loader = _cap_eval_loader(
                _load_test_loader(dataset_cfg["loader"], _batch_size_for_task(task_name), config["eval_upsample"]),
                config["loopback_eval_cap"],
            )
            for method_name in selected_methods:
                progress.set_postfix_str(f"{task_name} | {method_name}")
                try:
                    audit_row = audit_traceable_protocol(
                        method_name=method_name,
                        model_path=model_path,
                        test_loader=test_loader,
                        input_shape=dataset_cfg["input_shape"],
                        task_name=task_name,
                        device=runtime_info["device"],
                    )
                    audit_row["SeedContext"] = config["supplementary_seed"]
                    rows.append(audit_row)
                except Exception as exc:
                    rows.append(
                        {
                            "Task": task_name,
                            "Method": method_name,
                            "Samples": None,
                            "EstimatedOnlineCommMB": None,
                            "TraceEvents": None,
                            "LoopbackReplayMB": None,
                            "LoopbackElapsedMs": None,
                            "LoopbackThroughputMBps": None,
                            "AuditMode": "localhost-loopback",
                            "AuditNote": f"FAILED: {exc}",
                            "SeedContext": config["supplementary_seed"],
                        }
                    )
                finally:
                    progress.update(1)
    finally:
        progress.close()
    return rows


def run_stress_test(runtime_info, config):
    if not config["run_stress"]:
        return []
    task_cfg = next(dataset for dataset in DATASETS if dataset["name"] == "MNIST")
    model_path = os.path.join(MODELS_DIR, task_cfg["model_file"])
    rows = []
    total_steps = len(config["stress_batches"]) * 4
    progress = _make_progress(total=total_steps, desc="Stress test", unit="run")
    try:
        for batch_size in config["stress_batches"]:
            test_loader = _cap_eval_loader(_load_test_loader(task_cfg["loader"], batch_size, eval_upsample=1), config["stress_eval_cap"])
            for method_name in ["2Cloud-D (Data-only)", "ASS (Ours)", "3Share-DM (3-party)", "SecureNN (3PC)"]:
                progress.set_postfix_str(f"batch={batch_size} | {method_name}")
                try:
                    if method_name == "ASS (Ours)":
                        _, total_time, _, _, samples, _ = run_ass_inference(model_path, test_loader, task_cfg["input_shape"], task_name="MNIST", device=runtime_info["device"])
                        throughput = samples / max(total_time, 1e-9)
                    elif method_name == "2Cloud-D (Data-only)":
                        _, total_time, _, _, samples, _ = run_2cloud_inference(model_path, test_loader, task_cfg["input_shape"], task_name="MNIST", device=runtime_info["device"])
                        throughput = samples / max(total_time, 1e-9)
                    elif method_name == "3Share-DM (3-party)":
                        _, total_time, _, _, samples, _ = run_three_share_inference(model_path, test_loader, task_cfg["input_shape"], task_name="MNIST", device=runtime_info["device"])
                        throughput = samples / max(total_time, 1e-9)
                    else:
                        _, avg_time_s, _, _, timing_samples, _ = run_securenn_inference(model_path, test_loader, task_cfg["input_shape"], task_name="MNIST", device=runtime_info["device"])
                        throughput = 1.0 / max(avg_time_s, 1e-9)
                        samples = timing_samples
                    rows.append({"Method": method_name, "Batch": batch_size, "Throughput": throughput, "Samples": samples, "Error": None})
                except Exception as exc:
                    rows.append({"Method": method_name, "Batch": batch_size, "Throughput": None, "Samples": None, "Error": str(exc)})
                finally:
                    progress.update(1)
    finally:
        progress.close()
    return rows


def save_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        backup_path = f"{path}.bak-{int(time.time())}"
        if os.path.exists(path):
            os.replace(path, backup_path)
        os.replace(tmp_path, path)


def fmt_pct(value):
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def fmt_float(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def generate_results_report(runtime_info, plain_rows, plain_seed_rows, main_results, main_seed_rows, transfer_rows, longtail_rows, necessity_rows, necessity_summary_rows, security_rows, he_boundary_rows, loopback_rows, stress_rows, config):
    lines = []
    lines.append("# ASS Experiment Report")
    lines.append("")
    lines.append("## 1. Experiment Pipeline")
    lines.append("- The pipeline runs plaintext references, secure-inference baselines, ablations, transfer tests, long-tail tests, HE/PHE comparisons, and local loopback communication checks.")
    lines.append("- Timing is reported as per-sample latency for consistent comparison across methods.")
    lines.append("- Dataset preprocessing includes validation for label mappings and task-specific input shapes.")
    lines.append("")
    lines.append("## 2. Runtime Environment")
    if runtime_info["use_cuda"]:
        lines.append(f"- Device: CUDA, GPU = {runtime_info['gpu_name'] or 'Unknown GPU'}.")
    else:
        lines.append("- Device: CPU.")
    lines.append(f"- Python：{runtime_info['python_executable']}")
    lines.append(f"- Conda environment：{runtime_info['conda_env']}")
    lines.append(f"- Torch：{runtime_info['torch_version']}，CUDA Build：{runtime_info['torch_cuda_version']}")
    lines.append(f"- Requested device：{runtime_info['requested_device']}")
    lines.append(f"- Main repeats：{config['main_repeats']}.")
    lines.append(f"- Main statistics mode：{'multi-seed retraining' if config['repeat_seeds'] else 'single-seed inference repeats'}.")
    if config["repeat_seeds"]:
        lines.append(f"- Training seeds：{','.join(str(seed) for seed in config['repeat_seeds'])}.")
    lines.append(f"- Supplementary seed：{config['supplementary_seed']}.")
    lines.append(f"- Evaluation sample cap：{config['eval_max_samples']}.")
    lines.append(f"- DataLoader workers：{os.getenv('ASS_DATALOADER_WORKERS', '0')}。")
    lines.append("")
    lines.append("## 3. Experiment Coverage")
    lines.append("1. Data-only two-cloud inference versus joint data/model splitting: `split_ablation.csv`.")
    lines.append("2. Transfer from general datasets to medical datasets: `transfer_results.csv`.")
    lines.append("3. One-round versus multi-round protocol cost: `round_ablation.csv`.")
    lines.append("4. Three-share extension cost: `3Share-DM (3-party)` in the main result table.")
    lines.append("5. HE/PHE comparisons: `CKKS (HE)` and `Paillier (PHE)` in the main result table.")
    lines.append("")
    lines.append("## 4. Plaintext Reference Accuracy")
    lines.append("| Task | Domain | Plain Acc | Plain Acc Std | Plain Time/sample (ms) | Eval Samples | Seeds |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in plain_rows:
        lines.append(f"| {row['Task']} | {row['Domain']} | {fmt_pct(row['PlainAcc'])} | {fmt_float(row.get('PlainAccStd'))} | {fmt_float(row['PlainPerSampleMs'])} | {row['EvalSamples']} | {row.get('SeedsUsed', '-') } |")
    lines.append("")
    lines.append("## 5. Main Result Summary")
    lines.append("| Task | Method | Status | Acc | Repeats | Seed Mode | Scope | Comm Basis | Time/sample (ms) | Online Comm (MB) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in main_results:
        lines.append(f"| {row['Task']} | {row['Method']} | {row['Status']} | {fmt_pct(row['AccMean'])} | {row['Repeats']} | {row['SeedMode']} | {row['ImplementationScope']} | {row['CommMeasurementBasis']} | {fmt_float(row['PerSampleMsMean'])} | {fmt_float(row['OnlineCommMB'])} |")
    lines.append("")
    if main_seed_rows:
        lines.append("## 5A. Main Result Seed Details")
        lines.append("| Seed | Task | Method | Status | Acc | Time/sample (ms) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for row in main_seed_rows:
            lines.append(f"| {row['Seed']} | {row['Task']} | {row['Method']} | {row['Status']} | {fmt_pct(row['AccMean'])} | {fmt_float(row['PerSampleMsMean'])} |")
    lines.append("")
    lines.append("## 6. Transfer and Long-Tail Tests")
    lines.append("| Method | General Mean Acc | Medical Mean Acc | General Latency (ms) | Medical Latency (ms) | Acc Gap |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in transfer_rows:
        lines.append(f"| {row['Method']} | {fmt_pct(row['GeneralMeanAcc'])} | {fmt_pct(row['MedicalMeanAcc'])} | {fmt_float(row['GeneralMeanLatencyMs'])} | {fmt_float(row['MedicalMeanLatencyMs'])} | {fmt_float(row['AccDomainGap'])} |")
    lines.append("")
    lines.append("| Task | Ratio | Method | Acc | Balanced Acc | Macro-F1 | Tail Recall | Time/sample (ms) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in longtail_rows:
        lines.append(f"| {row['Task']} | {row['Ratio']} | {row['Method']} | {fmt_pct(row['Acc'])} | {fmt_float(row['BalancedAcc'])} | {fmt_float(row['MacroF1'])} | {fmt_float(row['TailRecall'])} | {fmt_float(row['PerSampleMs'])} |")
    lines.append("")
    if necessity_summary_rows:
        lines.append("## 6A. Medical Model-Split Necessity Under Long-Tail Training")
        lines.append("| Train Ratio | Method | Mean Protocol Acc | Mean Single-Cloud Acc | Mean Protocol Tail Recall | Mean Single-Cloud Tail Recall | Executable Share |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for row in necessity_summary_rows:
            lines.append(f"| {row['TrainRatio']} | {row['Method']} | {fmt_pct(row['MeanProtocolAcc'])} | {fmt_pct(row['MeanSingleCloudAcc'])} | {fmt_float(row['MeanProtocolTailRecall'])} | {fmt_float(row['MeanSingleCloudTailRecall'])} | {fmt_float(row['ExecutableShare'])} |")
        lines.append("")
        lines.append("| Task | Train Ratio | Method | Protocol Acc | Single-Cloud Acc | Exposure Gap | Single-Cloud Mode |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for row in necessity_rows:
            lines.append(f"| {row['Task']} | {row['TrainRatio']} | {row['Method']} | {fmt_pct(row['ProtocolAcc'])} | {fmt_pct(row['SingleCloudAcc'])} | {fmt_float(row['ExposureGapAcc'])} | {row['SingleCloudMode']} |")
        lines.append("")
    lines.append("## 7. Security Metrics")
    lines.append("| Method | Scope | Role | Comm Basis | Parties | Data Split | Model Split | Single-Point Model Exposure | Min Collusion Model |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in security_rows:
        lines.append(f"| {row['Method']} | {row['ImplementationScope']} | {row['ComparisonRole']} | {row['CommMeasurementBasis']} | {row['Parties']} | {row['DataSplit']} | {row['ModelSplit']} | {row['SinglePointModelExposure']} | {row['MinCollusionModel']} |")
    if he_boundary_rows:
        lines.append("")
        lines.append("## 7A. HE/PHE Boundary Comparison")
        lines.append("| Task | ASS (ms) | CKKS Status | CKKS (ms) | CKKS / ASS | Paillier Status | Paillier (ms) | Paillier / ASS |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for row in he_boundary_rows:
            lines.append(f"| {row['Task']} | {fmt_float(row['ASSPerSampleMs'])} | {row['CKKSStatus'] or '-'} | {fmt_float(row['CKKSPerSampleMs'])} | {fmt_float(row['CKKSvsASS'])} | {row['PaillierStatus'] or '-'} | {fmt_float(row['PaillierPerSampleMs'])} | {fmt_float(row['PailliervsASS'])} |")
    if loopback_rows:
        lines.append("")
        lines.append("## 8. Local Loopback Communication Check")
        lines.append("| Task | Method | Estimated Comm (MB) | Replay MB | Replay Time (ms) | Throughput (MB/s) | Seed |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for row in loopback_rows:
            lines.append(f"| {row['Task']} | {row['Method']} | {fmt_float(row['EstimatedOnlineCommMB'])} | {fmt_float(row['LoopbackReplayMB'])} | {fmt_float(row['LoopbackElapsedMs'])} | {fmt_float(row['LoopbackThroughputMBps'])} | {row.get('SeedContext', '-')} |")
    if stress_rows:
        lines.append("")
        lines.append("## 9. Stress Test")
        lines.append("| Method | Batch | Throughput | Samples |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for row in stress_rows:
            lines.append(f"| {row['Method']} | {row['Batch']} | {fmt_float(row['Throughput'], 1)} | {row['Samples'] if row['Samples'] is not None else '-'} |")
    lines.append("")
    lines.append("## 10. Suggested Plots")
    lines.append("- Plot 1: accuracy, latency, and communication from `split_ablation.csv`.")
    lines.append("- Plot 2: one-round versus four-round latency and communication from `round_ablation.csv`.")
    lines.append("- Plot 3: general-to-medical transfer statistics from `transfer_results.csv`.")
    lines.append("- Plot 4: tail recall and balanced accuracy under increasing long-tail ratios from `long_tail_results.csv`.")
    lines.append("- Plot 5: HE/PHE boundary comparison from `he_boundary_summary.csv`.")
    lines.append("- Plot 6: estimated communication versus local loopback replay cost from `loopback_comm_audit.csv`.")
    lines.append("")
    lines.append("## 11. Result Notes")
    lines.append("- Main means and variances should prioritize multi-seed retraining when repeat seeds are configured.")
    lines.append("- HE/PHE rows are best interpreted through `he_boundary_summary.csv` because not every task is equally suitable for every backend.")
    lines.append("- `loopback_comm_audit.csv` is a local `localhost` replay check, not a cross-host packet capture.")
    lines.append("- `SecureNN` is included as a simulator-style comparison and has a different implementation scope from ASS.")
    lines.append("")
    lines.append("## 12. 局限性与诚实披露")
    lines.append("- 当前主线包含轻量 MLP/FNN 表格任务与官方 ISIC 2018 secure CNN 图像任务；ISIC 分支用于系统工作负载和安全 CNN 算子验证，不应写成临床诊断 SOTA。")
    lines.append("- 多 seed 重训目前主要覆盖主结果矩阵；长尾、轮次与压力测试默认在固定种子上复核。")
    lines.append("- 单机 loopback 只能反映本机传输栈开销，不能替代跨主机网络时延、抓包和部署级验证。")
    lines.append("- HE / PHE 的当前实现仍偏边界比较与采样执行，不应写成与主线方法完全同口径的系统对照。")
    return "\n".join(lines)


def write_outputs(runtime_info, config, plain_rows, plain_seed_rows, main_results, main_seed_rows, split_rows, round_rows, transfer_rows, longtail_rows, necessity_rows, necessity_summary_rows, security_rows, loopback_rows, stress_rows):
    runtime_dump = dict(runtime_info)
    runtime_dump["device"] = str(runtime_dump.get("device"))
    config_dump = dict(config)
    if isinstance(config_dump.get("he_task_filter"), set):
        config_dump["he_task_filter"] = sorted(config_dump["he_task_filter"])
    he_boundary_rows = build_he_boundary_summary(main_results)
    raw_payload = {
        "runtime": runtime_dump,
        "config": config_dump,
        "plain_reference": plain_rows,
        "plain_seed_runs": plain_seed_rows,
        "main_results": main_results,
        "main_seed_runs": main_seed_rows,
        "split_ablation": split_rows,
        "round_ablation": round_rows,
        "transfer_results": transfer_rows,
        "long_tail_results": longtail_rows,
        "model_split_necessity": necessity_rows,
        "model_split_necessity_summary": necessity_summary_rows,
        "security_metrics": security_rows,
        "he_boundary_summary": he_boundary_rows,
        "loopback_comm_audit": loopback_rows,
        "stress_results": stress_rows,
    }

    for output_dir in (RESULTS_DIR, ARTIFACTS_DIR):
        os.makedirs(output_dir, exist_ok=True)
        atomic_write_json(os.path.join(output_dir, "results_raw.json"), raw_payload)

        save_csv(os.path.join(output_dir, "plaintext_reference.csv"), plain_rows, ["Task", "Domain", "PlainAcc", "PlainAccStd", "PlainPerSampleMs", "PlainPerSampleMsStd", "EvalSamples", "Repeats", "SeedsUsed"])
        save_csv(os.path.join(output_dir, "plaintext_seed_runs.csv"), plain_seed_rows, ["Seed", "Task", "Domain", "PlainAcc", "PlainPerSampleMs", "EvalSamples"])
        save_csv(os.path.join(output_dir, "results_table.csv"), main_results, ["Task", "Domain", "Method", "Status", "EvalSamples", "TimingSamples", "Repeats", "AccMean", "AccStd", "AccCI95Low", "AccCI95High", "AccDropVsPlain", "PerSampleMsMean", "PerSampleMsStd", "TimeCI95Low", "TimeCI95High", "ThroughputMean", "OnlineCommMB", "OfflineSetupMB", "LinearPct", "ReLUPct", "SeedMode", "SeedsUsed", "ImplementationScope", "ComparisonRole", "CommMeasurementBasis", "Parties", "InteractionRounds", "DataSplit", "ModelSplit", "SinglePointDataExposure", "SinglePointModelExposure", "MinCollusionInput", "MinCollusionModel", "CommType", "Evidence", "Error"])
        save_csv(os.path.join(output_dir, "main_seed_runs.csv"), main_seed_rows, ["Seed", "Task", "Domain", "Method", "Status", "EvalSamples", "TimingSamples", "Repeats", "AccMean", "AccStd", "AccCI95Low", "AccCI95High", "AccDropVsPlain", "PerSampleMsMean", "PerSampleMsStd", "TimeCI95Low", "TimeCI95High", "ThroughputMean", "OnlineCommMB", "OfflineSetupMB", "LinearPct", "ReLUPct", "SeedMode", "SeedsUsed", "ImplementationScope", "ComparisonRole", "CommMeasurementBasis", "Parties", "InteractionRounds", "DataSplit", "ModelSplit", "SinglePointDataExposure", "SinglePointModelExposure", "MinCollusionInput", "MinCollusionModel", "CommType", "Evidence", "Error"])
        save_csv(os.path.join(output_dir, "split_ablation.csv"), split_rows, ["Task", "Method", "Status", "AccMean", "AccDropVsPlain", "PerSampleMsMean", "OnlineCommMB", "OfflineSetupMB", "SinglePointModelExposure", "MinCollusionModel"])
        save_csv(os.path.join(output_dir, "round_ablation.csv"), round_rows, ["Task", "Method", "Rounds", "Acc", "PerSampleMs", "OnlineCommMB", "EvalSamples", "Error"])
        save_csv(os.path.join(output_dir, "transfer_results.csv"), transfer_rows, ["Method", "GeneralMeanAcc", "MedicalMeanAcc", "GeneralMeanLatencyMs", "MedicalMeanLatencyMs", "AccDomainGap", "LatencyDomainGap"])
        save_csv(os.path.join(output_dir, "long_tail_results.csv"), longtail_rows, ["Task", "Method", "Ratio", "Acc", "BalancedAcc", "MacroF1", "TailRecall", "PerSampleMs", "OnlineCommMB", "Samples"])
        save_csv(os.path.join(output_dir, "model_split_necessity.csv"), necessity_rows, ["Task", "TrainRatio", "TrainRatioValue", "Method", "MajorityLabel", "MinorityLabel", "TrainMajorityCount", "TrainMinorityCount", "ProtocolAcc", "ProtocolBalancedAcc", "ProtocolMacroF1", "ProtocolTailRecall", "ProtocolPerSampleMs", "ProtocolOnlineCommMB", "SingleCloudExecutable", "SingleCloudMode", "SingleCloudAcc", "SingleCloudBalancedAcc", "SingleCloudMacroF1", "SingleCloudTailRecall", "SingleCloudPerSampleMs", "ExposureGapAcc", "ExposureGapTailRecall", "PlainAcc", "ModelPath", "CloudNote"])
        save_csv(os.path.join(output_dir, "model_split_necessity_summary.csv"), necessity_summary_rows, ["TrainRatio", "Method", "Tasks", "MeanProtocolAcc", "MeanProtocolTailRecall", "MeanSingleCloudAcc", "MeanSingleCloudTailRecall", "MeanExposureGapAcc", "MeanExposureGapTailRecall", "ExecutableShare"])
        save_csv(os.path.join(output_dir, "security_metrics.csv"), security_rows, ["Method", "Parties", "DataSplit", "ModelSplit", "InteractionRounds", "SinglePointDataExposure", "SinglePointModelExposure", "MinCollusionInput", "MinCollusionModel", "ImplementationScope", "ComparisonRole", "CommMeasurementBasis", "Evidence"])
        save_csv(os.path.join(output_dir, "he_boundary_summary.csv"), he_boundary_rows, ["Task", "ASSPerSampleMs", "CKKSStatus", "CKKSPerSampleMs", "CKKSvsASS", "PaillierStatus", "PaillierPerSampleMs", "PailliervsASS", "SummaryNote"])
        save_csv(os.path.join(output_dir, "loopback_comm_audit.csv"), loopback_rows, ["Task", "Method", "Samples", "EstimatedOnlineCommMB", "TraceEvents", "LoopbackReplayMB", "LoopbackElapsedMs", "LoopbackThroughputMBps", "AuditMode", "AuditNote", "SeedContext"])
        save_csv(os.path.join(output_dir, "stress_results.csv"), stress_rows, ["Method", "Batch", "Throughput", "Samples", "Error"])

    report_path = os.path.join(RESULTS_DIR, "source_report_README.md")
    atomic_write_text(report_path, generate_results_report(runtime_info, plain_rows, plain_seed_rows, main_results, main_seed_rows, transfer_rows, longtail_rows, necessity_rows, necessity_summary_rows, security_rows, he_boundary_rows, loopback_rows, stress_rows, config))


def run_validation_check(config):
    if not config["run_validation_check"]:
        return
    script_path = os.path.join(CORE_DIR, "experiments", "validation_check.py")
    if not os.path.exists(script_path):
        print("WARNING: validation check script not found, skip.")
        return
    cmd = [sys.executable, script_path, "--results-dir", RESULTS_DIR]
    if config["validation_check_strict"]:
        cmd.append("--strict")
    print("\n>>> Step 9: validation check")
    proc = subprocess.run(cmd, cwd=CORE_DIR, check=False)
    if proc.returncode != 0:
        message = f"Validation check failed with exit code {proc.returncode}."
        if config["validation_check_strict"]:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


def main():
    repeat_seed_env = os.getenv("ASS_REPEAT_SEEDS")
    config = {
        "main_repeats": max(1, env_int("ASS_EXP_REPEATS", 3)),
        "repeat_seeds": [int(value) for value in env_csv_list("ASS_REPEAT_SEEDS", "")] if repeat_seed_env else [],
        "supplementary_seed": env_int("ASS_SUPPLEMENTARY_SEED", 42),
        "eval_upsample": max(1, env_int("EVAL_UPSAMPLE_FACTOR", 1)),
        "eval_max_samples": env_int("EVAL_MAX_SAMPLES", 5000),
        "run_he": env_bool("RUN_HE_BASELINES", True),
        "he_task_filter": set(env_csv_list("HE_TASK_FILTER", "")) if os.getenv("HE_TASK_FILTER") else None,
        "ckks_timeout_seconds": max(10, env_int("CKKS_TIMEOUT_SECONDS", 90)),
        "paillier_eval_cap": max(1, env_int("PAILLIER_MAX_SAMPLES", 30)),
        "paillier_max_seconds": max(10, env_int("PAILLIER_MAX_SECONDS", 120)),
        "paillier_key_bits": max(512, env_int("PAILLIER_KEY_BITS", 1024)),
        "paillier_progress_interval": max(1, env_int("PAILLIER_PROGRESS_INTERVAL", 1)),
        "round_values": [int(value) for value in env_csv_list("ROUND_VALUES", "1,4")],
        "round_eval_cap": max(1, env_int("ROUND_EVAL_MAX_SAMPLES", 2000)),
        "longtail_ratios": [int(value) for value in env_csv_list("LONGTAIL_RATIOS", "1,9,19")],
        "longtail_eval_cap": max(1, env_int("LONGTAIL_EVAL_MAX_SAMPLES", 3000)),
        "run_model_split_necessity": env_bool("RUN_MODEL_SPLIT_NECESSITY", True),
        "necessity_ratios": [int(value) for value in env_csv_list("MODEL_SPLIT_NECESSITY_RATIOS", "1,4,9,19")],
        "necessity_seed": env_int("MODEL_SPLIT_NECESSITY_SEED", 42),
        "force_retrain_longtail_models": env_bool("FORCE_RETRAIN_LONGTAIL_MODELS", False) or env_bool("FORCE_RETRAIN_MODELS", False),
        "run_stress": env_bool("RUN_STRESS_TEST", True),
        "stress_batches": [int(value) for value in env_csv_list("ASS_STRESS_BATCHES", "256,512,1024,2048")],
        "stress_eval_cap": max(1, env_int("STRESS_EVAL_MAX_SAMPLES", 4096)),
        "force_retrain_models": env_bool("FORCE_RETRAIN_MODELS", False),
        "run_loopback_audit": env_bool("RUN_LOOPBACK_AUDIT", True),
        "loopback_tasks": env_csv_list("LOOPBACK_TASKS", "MNIST,Medical,Digits"),
        "loopback_methods": env_csv_list("LOOPBACK_METHODS", "2Cloud-D (Data-only),ASS (Ours),3Share-DM (3-party)"),
        "loopback_eval_cap": max(1, env_int("LOOPBACK_EVAL_MAX_SAMPLES", 512)),
        "run_validation_check": env_bool("RUN_VALIDATION_CHECK", True),
        "validation_check_strict": env_bool("VALIDATION_CHECK_STRICT", False),
        "allow_synthetic_data": env_bool("ASS_ALLOW_SYNTHETIC_DATA", False),
        "experiment_allow_synthetic_data": env_bool("ASS_EXPERIMENT_ALLOW_SYNTHETIC_DATA", False),
    }
    if config["allow_synthetic_data"] and not config["experiment_allow_synthetic_data"]:
        raise RuntimeError(
            "Synthetic fallback is enabled. Formal run_experiment.py refuses synthetic fallback by default. "
            "Unset ASS_ALLOW_SYNTHETIC_DATA for formal experiment runs, or set "
            "ASS_EXPERIMENT_ALLOW_SYNTHETIC_DATA=1 only for demo/debug runs."
        )
    if config["repeat_seeds"] and not os.getenv("ASS_SUPPLEMENTARY_SEED"):
        config["supplementary_seed"] = config["repeat_seeds"][-1]

    task_filter = set(env_csv_list("TASK_FILTER", "")) if os.getenv("TASK_FILTER") else None
    dataset_cfgs = [cfg for cfg in DATASETS if task_filter is None or cfg["name"] in task_filter]

    runtime_info = detect_runtime_info()
    print(f"Running on {'CUDA' if runtime_info['use_cuda'] else 'CPU'}")
    print(f"Python: {runtime_info['python_executable']}")
    print(f"Conda env: {runtime_info['conda_env']}")
    print(f"Torch: {runtime_info['torch_version']} | CUDA build: {runtime_info['torch_cuda_version']}")
    print(f"Requested device: {runtime_info['requested_device']}")
    if runtime_info["gpu_name"]:
        print(f"Detected GPU: {runtime_info['gpu_name']} (count={runtime_info['gpu_count']})")
    if runtime_info["warning"]:
        print(f"WARNING: {runtime_info['warning']}")

    configure_runtime_backend(runtime_info)

    if config["repeat_seeds"]:
        print("\n>>> Step 1: Multi-seed retraining and plaintext reference evaluation")
        plain_rows, plain_seed_rows, main_results, main_seed_rows = run_multi_seed_main_matrix(dataset_cfgs, runtime_info, config)
        plain_reference_map = {row["Task"]: row for row in plain_rows}
        if config["supplementary_seed"] != config["repeat_seeds"][-1]:
            train_models_for_seed(config["supplementary_seed"], runtime_info, dataset_cfgs)
    else:
        ensure_models(force_retrain=config["force_retrain_models"], runtime_info=runtime_info)
        plain_seed_rows = []
        main_seed_rows = []
        plain_rows = []
        plain_reference_map = {}
        print("\n>>> Step 1: Plaintext reference evaluation")
        plain_progress = _make_progress(total=len(dataset_cfgs), desc="Plain reference", unit="task")
        try:
            for dataset_cfg in dataset_cfgs:
                plain_progress.set_postfix_str(dataset_cfg["name"])
                row = collect_plain_reference(dataset_cfg, runtime_info["device"], config["eval_upsample"], config["eval_max_samples"])
                row["Repeats"] = 1
                row["SeedsUsed"] = str(config["supplementary_seed"])
                plain_rows.append(row)
                plain_reference_map[row["Task"]] = row
                print(f"    [{row['Task']}] Plain Acc = {row['PlainAcc']:.4f}, Time/sample = {row['PlainPerSampleMs']:.4f} ms")
                plain_progress.update(1)
        finally:
            plain_progress.close()

        print("\n>>> Step 2: Main comparison matrix")
        main_results = run_main_matrix(dataset_cfgs, plain_reference_map, runtime_info, config)

    print("\n>>> Step 3: Round ablation")
    round_rows = run_round_ablation(dataset_cfgs, runtime_info, config)

    print("\n>>> Step 4: Long-tail analysis")
    longtail_rows = run_long_tail_analysis(dataset_cfgs, runtime_info, config)

    print("\n>>> Step 5: Model-split necessity under medical long-tail training")
    necessity_rows = run_model_split_necessity(dataset_cfgs, runtime_info, config)
    necessity_summary_rows = build_model_split_necessity_summary(necessity_rows)

    print("\n>>> Step 6: Transfer and security summaries")
    transfer_rows = build_transfer_summary(main_results)
    split_rows = build_split_ablation(main_results)
    security_rows = build_security_metrics()

    print("\n>>> Step 7: Loopback communication audit")
    loopback_rows = run_loopback_comm_audit(dataset_cfgs, runtime_info, config)

    print("\n>>> Step 8: Stress test")
    stress_rows = run_stress_test(runtime_info, config)

    print("\n>>> Step 9: Writing results")
    write_outputs(runtime_info, config, plain_rows, plain_seed_rows, main_results, main_seed_rows, split_rows, round_rows, transfer_rows, longtail_rows, necessity_rows, necessity_summary_rows, security_rows, loopback_rows, stress_rows)
    run_validation_check(config)
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
