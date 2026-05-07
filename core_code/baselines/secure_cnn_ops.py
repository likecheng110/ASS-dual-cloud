import math
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _as_pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def secure_im2col(x_share: torch.Tensor, kernel_size, stride=1, padding=0):
    return F.unfold(x_share, kernel_size=_as_pair(kernel_size), stride=_as_pair(stride), padding=_as_pair(padding))


@dataclass
class SecureOpStats:
    time_s: float = 0.0
    comm_bytes: float = 0.0
    rounds: int = 0
    linear_time_s: float = 0.0
    nonlinear_time_s: float = 0.0
    linear_comm_bytes: float = 0.0
    nonlinear_comm_bytes: float = 0.0

    def add(self, other: "SecureOpStats"):
        self.time_s += other.time_s
        self.comm_bytes += other.comm_bytes
        self.rounds += other.rounds
        self.linear_time_s += other.linear_time_s
        self.nonlinear_time_s += other.nonlinear_time_s
        self.linear_comm_bytes += other.linear_comm_bytes
        self.nonlinear_comm_bytes += other.nonlinear_comm_bytes
        return self


class SecureShareEngine:
    def __init__(self, device="auto", scale_bits: int = 16, bitwidth: int = 32):
        if isinstance(device, torch.device):
            self.device = device
        elif device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.scale_bits = int(scale_bits)
        self.scale = None if self.scale_bits < 0 else 1 << self.scale_bits
        self.bitwidth = int(bitwidth)
        self.share_bytes = max(1, self.bitwidth // 8)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def quantize(self, tensor: torch.Tensor):
        tensor = tensor.to(self.device, dtype=torch.float32)
        if self.scale is None:
            return tensor
        return torch.round(tensor * self.scale) / self.scale

    def make_shares(self, tensor: torch.Tensor, num_shares: int) -> List[torch.Tensor]:
        tensor = self.quantize(tensor)
        if num_shares <= 1:
            return [tensor]
        shares = []
        accum = torch.zeros_like(tensor, device=self.device)
        for _ in range(num_shares - 1):
            rand_share = torch.empty_like(tensor, device=self.device).uniform_(-1.0, 1.0)
            rand_share = self.quantize(rand_share)
            shares.append(rand_share)
            accum = accum + rand_share
        shares.append(self.quantize(tensor - accum))
        return shares

    def reconstruct(self, shares: Sequence[torch.Tensor]):
        if not shares:
            raise ValueError("shares must not be empty")
        total = torch.zeros_like(shares[0], device=self.device)
        for share in shares:
            total = total + share.to(self.device)
        return self.quantize(total)


class SecureLinearLayer:
    def __init__(self, weight_shares, bias_shares, engine: SecureShareEngine):
        self.weight_shares = weight_shares
        self.bias_shares = bias_shares
        self.engine = engine

    @classmethod
    def from_plain(cls, layer: torch.nn.Linear, engine: SecureShareEngine, model_shares: int = 2):
        weight = engine.quantize(layer.weight.data)
        bias = engine.quantize(layer.bias.data)
        return cls(engine.make_shares(weight, model_shares), engine.make_shares(bias, model_shares), engine)

    def __call__(self, input_shares: List[torch.Tensor]):
        start = time.perf_counter_ns()
        output = None
        for x_share in input_shares:
            x_share = x_share.view(x_share.shape[0], -1)
            for w_share in self.weight_shares:
                partial = torch.matmul(x_share, w_share.T)
                output = partial if output is None else output + partial
        for b_share in self.bias_shares:
            output = output + b_share
        output = self.engine.quantize(output)
        self.engine.sync()
        duration = (time.perf_counter_ns() - start) / 1e9
        output_shares = self.engine.make_shares(output, len(input_shares))

        output_terms = len(input_shares) * len(self.weight_shares)
        comm_bytes = output.numel() * self.engine.share_bytes * max(output_terms - 1, 0)
        stats = SecureOpStats(
            time_s=duration,
            comm_bytes=comm_bytes,
            rounds=1,
            linear_time_s=duration,
            linear_comm_bytes=comm_bytes,
        )
        return output_shares, stats


class SecureConv2d:
    def __init__(self, weight_shares, bias_shares, stride, padding, engine: SecureShareEngine):
        self.weight_shares = weight_shares
        self.bias_shares = bias_shares
        self.stride = _as_pair(stride)
        self.padding = _as_pair(padding)
        self.engine = engine
        self.out_channels = weight_shares[0].shape[0]
        self.kernel_size = weight_shares[0].shape[-2:]

    @classmethod
    def from_plain(cls, layer: torch.nn.Conv2d, engine: SecureShareEngine, model_shares: int = 2):
        weight = engine.quantize(layer.weight.data)
        bias = engine.quantize(layer.bias.data)
        return cls(engine.make_shares(weight, model_shares), engine.make_shares(bias, model_shares), layer.stride, layer.padding, engine)

    def __call__(self, input_shares: List[torch.Tensor]):
        if not input_shares:
            raise ValueError("input_shares must not be empty")

        batch, _, height, width = input_shares[0].shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        out_h = ((height + 2 * ph - kh) // sh) + 1
        out_w = ((width + 2 * pw - kw) // sw) + 1

        start = time.perf_counter_ns()
        unfolded_shares = [
            secure_im2col(x_share, self.kernel_size, stride=self.stride, padding=self.padding)
            for x_share in input_shares
        ]
        output = None
        for x_col_share in unfolded_shares:
            for w_share in self.weight_shares:
                w_col_share = w_share.view(w_share.shape[0], -1)
                partial = torch.einsum("bkl,ok->bol", x_col_share, w_col_share)
                output = partial if output is None else output + partial
        for b_share in self.bias_shares:
            output = output + b_share.view(1, -1, 1)
        output = self.engine.quantize(output).view(batch, self.out_channels, out_h, out_w)
        self.engine.sync()
        duration = (time.perf_counter_ns() - start) / 1e9
        output_shares = self.engine.make_shares(output, len(input_shares))

        output_terms = len(input_shares) * len(self.weight_shares)
        comm_bytes = output.numel() * self.engine.share_bytes * max(output_terms - 1, 0)
        stats = SecureOpStats(
            time_s=duration,
            comm_bytes=comm_bytes,
            rounds=1,
            linear_time_s=duration,
            linear_comm_bytes=comm_bytes,
        )
        return output_shares, stats


class SecureMaxPool2d:
    def __init__(self, kernel_size=2, stride=None, padding=0, engine: SecureShareEngine = None):
        self.kernel_size = _as_pair(kernel_size)
        self.stride = _as_pair(stride if stride is not None else kernel_size)
        self.padding = _as_pair(padding)
        self.engine = engine or SecureShareEngine()

    def __call__(self, input_shares: List[torch.Tensor]):
        if not input_shares:
            raise ValueError("input_shares must not be empty")

        batch, channels, height, width = input_shares[0].shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        out_h = ((height + 2 * ph - kh) // sh) + 1
        out_w = ((width + 2 * pw - kw) // sw) + 1
        window = kh * kw
        depth = int(math.ceil(math.log2(max(window, 1))))

        start = time.perf_counter_ns()
        plaintext = self.engine.reconstruct(input_shares)
        unfolded = F.unfold(plaintext, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        unfolded = unfolded.view(batch, channels, window, -1)
        pooled = unfolded.max(dim=2).values.view(batch, channels, out_h, out_w)
        pooled = self.engine.quantize(pooled)
        self.engine.sync()
        duration = (time.perf_counter_ns() - start) / 1e9
        output_shares = self.engine.make_shares(pooled, len(input_shares))

        compare_count = pooled.numel() * max(window - 1, 0)
        compare_bytes = compare_count * len(input_shares) * self.engine.share_bytes
        stats = SecureOpStats(
            time_s=duration,
            comm_bytes=compare_bytes,
            rounds=depth,
            nonlinear_time_s=duration,
            nonlinear_comm_bytes=compare_bytes,
        )
        return output_shares, stats


class SecureAvgPool2d:
    def __init__(self, kernel_size=2, stride=None, padding=0, engine: SecureShareEngine = None):
        self.kernel_size = _as_pair(kernel_size)
        self.stride = _as_pair(stride if stride is not None else kernel_size)
        self.padding = _as_pair(padding)
        self.engine = engine or SecureShareEngine()

    def __call__(self, input_shares: List[torch.Tensor]):
        if not input_shares:
            raise ValueError("input_shares must not be empty")

        start = time.perf_counter_ns()
        output_shares = [
            self.engine.quantize(
                F.avg_pool2d(share, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
            )
            for share in input_shares
        ]
        self.engine.sync()
        duration = (time.perf_counter_ns() - start) / 1e9
        stats = SecureOpStats(time_s=duration)
        return output_shares, stats


def secure_relu(input_shares: List[torch.Tensor], engine: SecureShareEngine, interaction_rounds: int = 1):
    start = time.perf_counter_ns()
    plaintext = engine.reconstruct(input_shares)
    activated = engine.quantize(torch.clamp(plaintext, min=0))
    engine.sync()
    duration = (time.perf_counter_ns() - start) / 1e9
    output_shares = engine.make_shares(activated, len(input_shares))
    rounds = max(1, int(interaction_rounds))
    comm_bytes = plaintext.numel() * len(input_shares) * engine.share_bytes * rounds
    stats = SecureOpStats(
        time_s=duration,
        comm_bytes=comm_bytes,
        rounds=rounds,
        nonlinear_time_s=duration,
        nonlinear_comm_bytes=comm_bytes,
    )
    return output_shares, stats
