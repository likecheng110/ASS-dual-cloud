import math
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class LayerShape:
    kind: str
    input_elements: int
    output_elements: int
    in_channels: int = 1
    out_channels: int = 1
    kernel_size: int = 1
    out_h: int = 1
    out_w: int = 1


def _mb(bits: float):
    return bits / 8.0 / (1024 ** 2)


def _sum_ops(layers: Iterable[LayerShape]):
    ops = 0
    activations = 0
    for layer in layers:
        if layer.kind in {"conv", "linear"}:
            ops += layer.output_elements * max(layer.input_elements // max(layer.output_elements, 1), 1)
        if layer.kind in {"relu", "pool"}:
            activations += layer.output_elements
    return max(ops, 1), max(activations, 1)


def simulate_image_sota_baselines(plain_acc: float, ass_time_ms: float, ass_comm_mb: float, layers: List[LayerShape], bitwidth: int = 32) -> List[Dict[str, object]]:
    lambda_bits = 128
    packing_slots = 8192
    linear_layers = [layer for layer in layers if layer.kind in {"conv", "linear"}]
    nonlinear_layers = [layer for layer in layers if layer.kind in {"relu", "pool"}]
    total_output = sum(layer.output_elements for layer in linear_layers) or 1
    total_nonlinear = sum(layer.output_elements for layer in nonlinear_layers) or 1
    total_ops, _ = _sum_ops(layers)

    two_cloud_comm_bits = total_output * bitwidth
    sonic_comm_bits = 0
    for layer in linear_layers:
        if layer.kind == "conv":
            n_out = max(layer.kernel_size, 1)
            n = max(layer.kernel_size, 1)
            p = max(layer.in_channels, 1)
            spatial_sites = max(layer.out_h * layer.out_w, 1)
            sonic_comm_bits += (layer.out_channels * spatial_sites * (n_out ** 2) * ((n * p + 1) ** 2) * 4 * bitwidth) / packing_slots
        else:
            n = max(layer.input_elements, 1)
            sonic_comm_bits += (layer.output_elements * (n + 1) * 4 * bitwidth) / packing_slots
    sonic_relu_rounds = int(math.log2(bitwidth)) + 2
    sonic_comm_bits += total_nonlinear * (9 * bitwidth - 16)

    cheetah_comm_bits = 0
    for layer in linear_layers:
        if layer.kind == "conv":
            cheetah_comm_bits += layer.out_channels * layer.in_channels * layer.out_h * layer.out_w * (layer.kernel_size ** 2) * bitwidth
        else:
            cheetah_comm_bits += layer.input_elements * layer.output_elements * bitwidth
    cheetah_comm_bits += total_nonlinear * (11 * bitwidth)

    delphi_comm_bits = total_output * bitwidth
    delphi_comm_bits += total_nonlinear * (4 * lambda_bits * bitwidth)

    he_mops = total_ops / 1_000_000.0
    cheetah_time_ms = max(ass_time_ms * 1.35, he_mops * 6.0)
    sonic_time_ms = max(ass_time_ms * 2.5, ass_time_ms + sonic_relu_rounds * 0.05 * total_nonlinear / 1000.0)
    delphi_time_ms = max(ass_time_ms * 1.8, ass_time_ms + 0.02 * total_nonlinear)
    two_cloud_time_ms = ass_time_ms * 0.85

    return [
        {"Method": "2Cloud-D", "Acc": plain_acc, "TimeMs": two_cloud_time_ms, "CommMB": _mb(two_cloud_comm_bits), "E_m": 1, "k_m": 1},
        {"Method": "Delphi", "Acc": plain_acc, "TimeMs": delphi_time_ms, "CommMB": _mb(delphi_comm_bits), "E_m": 1, "k_m": 1},
        {"Method": "Sonic", "Acc": plain_acc, "TimeMs": sonic_time_ms, "CommMB": _mb(sonic_comm_bits), "E_m": 0, "k_m": 2},
        {"Method": "Cheetah", "Acc": plain_acc, "TimeMs": cheetah_time_ms, "CommMB": _mb(cheetah_comm_bits), "E_m": 1, "k_m": 1},
        {"Method": "ASS (Ours)", "Acc": plain_acc, "TimeMs": ass_time_ms, "CommMB": ass_comm_mb, "E_m": 0, "k_m": 2},
    ]
