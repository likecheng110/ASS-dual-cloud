import os
import sys
import time
from typing import Dict, List, Tuple

import torch


CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

from train_plain import (
    Diabetes_MLP,
    Digits_MLP,
    FashionMNIST_MLP,
    Heart_MLP,
    Liver_MLP,
    Medical_MLP,
    MNIST_MLP,
    Wine_MLP,
)


def _resolve_device(device="auto"):
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _create_task_model(task_name: str, input_shape: Tuple[int, ...]):
    if task_name == "Medical":
        model = Medical_MLP(input_shape[0])
    elif task_name == "Wine":
        model = Wine_MLP(input_shape[0])
    elif task_name == "Diabetes":
        model = Diabetes_MLP(input_shape[0])
    elif task_name == "Heart":
        model = Heart_MLP(input_shape[0])
    elif task_name == "Digits":
        model = Digits_MLP()
    elif task_name == "Liver":
        model = Liver_MLP(input_shape[0])
    elif task_name in ("Fashion", "FashionMNIST"):
        model = FashionMNIST_MLP()
    else:
        model = MNIST_MLP()
    return model


def load_task_model(task_name: str, input_shape: Tuple[int, ...], model_path: str, device="cpu"):
    model = _create_task_model(task_name, input_shape)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(_resolve_device(device))
    model.eval()
    return model


def load_task_model_share(
    task_name: str,
    input_shape: Tuple[int, ...],
    model_path: str,
    model_shares: int = 2,
    share_index: int = 0,
    share_seed: int = 42,
    device="cpu",
):
    runtime_device = _resolve_device(device)
    model = load_task_model(task_name, input_shape, model_path, device=runtime_device)
    engine = FixedPointShareEngine(device=runtime_device)

    with torch.random.fork_rng(devices=[runtime_device] if runtime_device.type == "cuda" else []):
        torch.manual_seed(int(share_seed))
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(int(share_seed))
        prepared_layers = engine.prepare_linear_layers(model, model_shares)

    share_model = _create_task_model(task_name, input_shape)
    share_state = share_model.state_dict()
    for layer_name, weight_shares, bias_shares in prepared_layers:
        share_state[f"{layer_name}.weight"] = weight_shares[share_index].detach().cpu()
        share_state[f"{layer_name}.bias"] = bias_shares[share_index].detach().cpu()
    share_model.load_state_dict(share_state)
    share_model.to(runtime_device)
    share_model.eval()
    return share_model


class FixedPointShareEngine:
    def __init__(self, device="auto", scale_bits: int = 16):
        self.device = _resolve_device(device)
        self.scale = 1 << scale_bits

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device, dtype=torch.float32)
        return torch.round(tensor * self.scale) / self.scale

    def make_shares(self, tensor: torch.Tensor, num_shares: int) -> List[torch.Tensor]:
        tensor = self.quantize(tensor)
        if num_shares <= 1:
            return [tensor]

        shares: List[torch.Tensor] = []
        accum = torch.zeros_like(tensor, device=self.device)
        for _ in range(num_shares - 1):
            rand_share = torch.empty_like(tensor, device=self.device).uniform_(-1.0, 1.0)
            rand_share = self.quantize(rand_share)
            shares.append(rand_share)
            accum = accum + rand_share
        shares.append(self.quantize(tensor - accum))
        return shares

    def reconstruct(self, shares: List[torch.Tensor]) -> torch.Tensor:
        if not shares:
            raise ValueError("shares must not be empty")
        total = torch.zeros_like(shares[0], device=self.device)
        for share in shares:
            total = total + share.to(self.device)
        return self.quantize(total)

    def prepare_linear_layers(self, model: torch.nn.Module, model_shares: int):
        layers = []
        for layer_name in ("fc1", "fc2", "fc3", "fc4"):
            if not hasattr(model, layer_name):
                continue
            layer = getattr(model, layer_name)
            weight = self.quantize(layer.weight.data)
            bias = self.quantize(layer.bias.data)
            weight_shares = self.make_shares(weight, model_shares)
            bias_shares = self.make_shares(bias, model_shares)
            layers.append((layer_name, weight_shares, bias_shares))
        return layers

    def linear_layer(self, input_shares: List[torch.Tensor], weight_shares: List[torch.Tensor], bias_shares: List[torch.Tensor]):
        start = time.perf_counter_ns()
        output = None
        for x_share in input_shares:
            for w_share in weight_shares:
                partial = torch.matmul(x_share, w_share.T)
                output = partial if output is None else output + partial
        for b_share in bias_shares:
            output = output + b_share
        output = self.quantize(output)
        _sync(self.device)
        duration = (time.perf_counter_ns() - start) / 1e9

        output_shares = self.make_shares(output, len(input_shares))
        output_terms = len(input_shares) * len(weight_shares)
        comm_bytes = output.numel() * 8 * max(output_terms - 1, 0)
        comm_bytes += output.numel() * 8 * max(len(input_shares) - 1, 0)
        return output_shares, duration, comm_bytes

    def relu_layer(self, input_shares: List[torch.Tensor], interaction_rounds: int):
        rounds = max(1, int(interaction_rounds))
        start = time.perf_counter_ns()
        plaintext = self.reconstruct(input_shares)
        working = plaintext
        for _ in range(rounds):
            sign_mask = (working > 0).float()
            sign_shares = self.make_shares(sign_mask, len(input_shares))
            working = self.reconstruct(sign_shares)
        activated = self.quantize(torch.clamp(plaintext, min=0))
        _sync(self.device)
        duration = (time.perf_counter_ns() - start) / 1e9

        output_shares = self.make_shares(activated, len(input_shares))
        comm_bytes = plaintext.numel() * 8 * len(input_shares) * rounds
        return output_shares, duration, comm_bytes


def run_shared_protocol_inference(
    model_path: str,
    test_loader,
    input_shape: Tuple[int, ...],
    task_name: str = "MNIST",
    data_shares: int = 2,
    model_shares: int = 2,
    interaction_rounds: int = 1,
    device="auto",
    return_predictions: bool = False,
    return_comm_trace: bool = False,
) -> Dict[str, object]:
    runtime_device = _resolve_device(device)
    engine = FixedPointShareEngine(device=runtime_device)
    model = load_task_model(task_name, input_shape, model_path, device=runtime_device)
    prepared_layers = engine.prepare_linear_layers(model, model_shares)

    correct = 0
    total = 0
    total_time = 0.0
    linear_time = 0.0
    relu_time = 0.0
    online_comm_bytes = 0.0

    predictions = []
    targets = []
    comm_trace_bytes = []

    model_param_count = sum(param.numel() for param in model.parameters())
    offline_setup_mb = (model_param_count * 4 * max(model_shares - 1, 0)) / (1024 ** 2)

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1).to(runtime_device, non_blocking=True)
            target = target.to(runtime_device, non_blocking=True)

            batch_start = time.perf_counter_ns()
            input_shares = engine.make_shares(data, data_shares)
            input_share_bytes = data.numel() * 8 * max(data_shares - 1, 0)
            online_comm_bytes += input_share_bytes
            if return_comm_trace and input_share_bytes > 0:
                comm_trace_bytes.append(int(input_share_bytes))

            for layer_idx, (_, weight_shares, bias_shares) in enumerate(prepared_layers):
                input_shares, duration, comm_bytes = engine.linear_layer(input_shares, weight_shares, bias_shares)
                linear_time += duration
                online_comm_bytes += comm_bytes
                if return_comm_trace and comm_bytes > 0:
                    comm_trace_bytes.append(int(comm_bytes))

                is_last_layer = layer_idx == len(prepared_layers) - 1
                if not is_last_layer:
                    input_shares, duration, comm_bytes = engine.relu_layer(input_shares, interaction_rounds)
                    relu_time += duration
                    online_comm_bytes += comm_bytes
                    if return_comm_trace and comm_bytes > 0:
                        comm_trace_bytes.append(int(comm_bytes))

            output = engine.reconstruct(input_shares)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

            if return_predictions:
                predictions.extend(pred.detach().cpu().tolist())
                targets.extend(target.detach().cpu().tolist())

            _sync(runtime_device)
            total_time += (time.perf_counter_ns() - batch_start) / 1e9

    total_layer_time = linear_time + relu_time
    if total_layer_time > 0:
        layer_breakdown = {
            "Linear": (linear_time / total_layer_time) * 100.0,
            "ReLU": (relu_time / total_layer_time) * 100.0,
        }
    else:
        layer_breakdown = {"Linear": 0.0, "ReLU": 0.0}

    result = {
        "Acc": (correct / total) if total > 0 else 0.0,
        "Time": total_time,
        "Comm": online_comm_bytes / (1024 ** 2),
        "OfflineSetupMB": offline_setup_mb,
        "Layer": layer_breakdown,
        "Samples": total,
    }
    if return_predictions:
        result["Predictions"] = predictions
        result["Targets"] = targets
    if return_comm_trace:
        result["CommTraceBytes"] = comm_trace_bytes
    return result
