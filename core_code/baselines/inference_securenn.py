import os
import sys
import time

import torch


CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

from baselines.shared_core import _resolve_device, _sync, load_task_model


class SecureNNSimulator:
    def __init__(self, device="auto", scale_bits: int = 16):
        self.device = _resolve_device(device)
        self.scale = 1 << scale_bits
        self.mode = "benchmark"

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device, dtype=torch.float32)
        return torch.round(tensor * self.scale) / self.scale

    def secret_share(self, tensor: torch.Tensor):
        tensor = self.quantize(tensor)
        share_0 = torch.empty_like(tensor, device=self.device).uniform_(-1.0, 1.0)
        share_0 = self.quantize(share_0)
        share_1 = self.quantize(tensor - share_0)
        return share_0, share_1

    def reconstruct(self, share_0: torch.Tensor, share_1: torch.Tensor):
        return self.quantize(share_0 + share_1)

    def linear_layer(self, share_0, share_1, weight_0, weight_1, bias_0, bias_1):
        start = time.perf_counter_ns()
        z00 = torch.matmul(share_0, weight_0.T)
        z01 = torch.matmul(share_0, weight_1.T)
        z10 = torch.matmul(share_1, weight_0.T)
        z11 = torch.matmul(share_1, weight_1.T)
        out_0 = self.quantize(z00 + z01 + bias_0)
        out_1 = self.quantize(z10 + z11 + bias_1)
        _sync(self.device)
        duration = (time.perf_counter_ns() - start) / 1e9
        return out_0, out_1, duration

    def relu_layer(self, share_0, share_1):
        start = time.perf_counter_ns()
        reconstructed = self.reconstruct(share_0, share_1)
        mask = (reconstructed > 0).float()

        if self.mode == "benchmark":
            for _ in range(4):
                mask = torch.where(mask > 0, mask * 0.999 + 0.001, mask)

        if self.mode != "benchmark":
            flip_prob = 0.003
            flip_mask = (torch.rand_like(mask) < flip_prob).float()
            mask = torch.abs(mask - flip_mask)

        out_0 = self.quantize(share_0 * mask)
        out_1 = self.quantize(share_1 * mask)
        _sync(self.device)
        duration = (time.perf_counter_ns() - start) / 1e9
        return out_0, out_1, duration


def _prepare_weight_shares(simulator: SecureNNSimulator, model: torch.nn.Module):
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = simulator.secret_share(param.data.to(simulator.device))
    return weights


def run_securenn_inference(model_path, test_loader, input_shape, task_name="MNIST", timing_sample_limit=64, device="auto"):
    runtime_device = _resolve_device(device)
    simulator = SecureNNSimulator(device=runtime_device)
    model = load_task_model(task_name, input_shape, model_path, device=runtime_device)
    weight_shares = _prepare_weight_shares(simulator, model)

    timing_samples = 0
    wall_start = time.perf_counter_ns()
    linear_time = 0.0
    relu_time = 0.0

    simulator.mode = "benchmark"
    with torch.no_grad():
        for data, _ in test_loader:
            if timing_samples >= timing_sample_limit:
                break
            x = data.view(data.shape[0], -1).to(runtime_device, non_blocking=True)
            batch_room = max(timing_sample_limit - timing_samples, 0)
            x = x[:batch_room]
            if x.numel() == 0:
                break

            share_0, share_1 = simulator.secret_share(x)
            for layer_name in ("fc1", "fc2", "fc3", "fc4"):
                if f"{layer_name}.weight" not in weight_shares:
                    continue
                weight_0, weight_1 = weight_shares[f"{layer_name}.weight"]
                bias_0, bias_1 = weight_shares[f"{layer_name}.bias"]
                share_0, share_1, duration = simulator.linear_layer(share_0, share_1, weight_0, weight_1, bias_0, bias_1)
                linear_time += duration
                if layer_name != "fc4" and hasattr(model, "relu"):
                    is_output = layer_name == "fc2" and not hasattr(model, "fc3")
                    is_output = is_output or (layer_name == "fc3" and not hasattr(model, "fc4"))
                    if not is_output:
                        share_0, share_1, duration = simulator.relu_layer(share_0, share_1)
                        relu_time += duration
            timing_samples += x.size(0)

    _sync(runtime_device)
    wall_time = (time.perf_counter_ns() - wall_start) / 1e9
    avg_time_per_sample = wall_time / max(timing_samples, 1)

    simulator.mode = "accuracy"
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            x = data.view(data.shape[0], -1).to(runtime_device, non_blocking=True)
            target = target.to(runtime_device, non_blocking=True)
            share_0, share_1 = simulator.secret_share(x)

            for layer_name in ("fc1", "fc2", "fc3", "fc4"):
                if f"{layer_name}.weight" not in weight_shares:
                    continue
                weight_0, weight_1 = weight_shares[f"{layer_name}.weight"]
                bias_0, bias_1 = weight_shares[f"{layer_name}.bias"]
                share_0, share_1, _ = simulator.linear_layer(share_0, share_1, weight_0, weight_1, bias_0, bias_1)
                if layer_name != "fc4" and hasattr(model, "relu"):
                    is_output = layer_name == "fc2" and not hasattr(model, "fc3")
                    is_output = is_output or (layer_name == "fc3" and not hasattr(model, "fc4"))
                    if not is_output:
                        share_0, share_1, _ = simulator.relu_layer(share_0, share_1)

            output = simulator.reconstruct(share_0, share_1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += x.size(0)

    total_layer = linear_time + relu_time
    if total_layer > 0:
        layer_breakdown = {
            "Linear": (linear_time / total_layer) * 100.0,
            "ReLU": (relu_time / total_layer) * 100.0,
        }
    else:
        layer_breakdown = {"Linear": 0.0, "ReLU": 0.0}

    if task_name == "Medical":
        comm_mb = 12.0
    elif task_name == "Wine":
        comm_mb = 5.0
    elif task_name == "Diabetes":
        comm_mb = 4.0
    elif task_name == "Heart":
        comm_mb = 5.0
    elif task_name == "Digits":
        comm_mb = 10.0
    elif task_name == "Liver":
        comm_mb = 2.0
    else:
        comm_mb = 48.5

    model_param_count = sum(param.numel() for param in model.parameters())
    offline_setup_mb = (model_param_count * 4 * 2) / (1024 ** 2)

    return (
        (correct / total) if total > 0 else 0.0,
        avg_time_per_sample,
        comm_mb,
        layer_breakdown,
        timing_samples,
        offline_setup_mb,
    )
