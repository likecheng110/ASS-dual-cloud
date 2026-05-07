import os
import sys

import torch
import torch.nn.functional as F


CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

from baselines.secure_cnn_ops import (  # noqa: E402
    SecureAvgPool2d,
    SecureConv2d,
    SecureLinearLayer,
    SecureMaxPool2d,
    SecureShareEngine,
    secure_relu,
)


def _assert_close(name, actual, expected, atol=2e-4):
    max_diff = torch.max(torch.abs(actual - expected)).item()
    print(f"{name}: max_diff={max_diff:.8f}")
    if max_diff > atol:
        raise AssertionError(f"{name} mismatch: max_diff={max_diff} > {atol}")


def verify_conv_layer(device="cpu"):
    torch.manual_seed(42)
    engine = SecureShareEngine(device=device)
    layer = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1).to(engine.device).eval()
    x = torch.randn(2, 3, 8, 8, device=engine.device)

    secure_layer = SecureConv2d.from_plain(layer, engine, model_shares=2)
    output_shares, stats = secure_layer(engine.make_shares(x, 2))
    actual = engine.reconstruct(output_shares)
    expected = engine.quantize(layer(engine.quantize(x)))

    _assert_close("SecureConv2d", actual, expected)
    assert stats.rounds == 1
    assert stats.comm_bytes > 0


def verify_linear_layer(device="cpu"):
    torch.manual_seed(43)
    engine = SecureShareEngine(device=device)
    layer = torch.nn.Linear(16, 5).to(engine.device).eval()
    x = torch.randn(3, 16, device=engine.device)

    secure_layer = SecureLinearLayer.from_plain(layer, engine, model_shares=2)
    output_shares, stats = secure_layer(engine.make_shares(x, 2))
    actual = engine.reconstruct(output_shares)
    expected = engine.quantize(layer(engine.quantize(x)))

    _assert_close("SecureLinearLayer", actual, expected)
    assert stats.rounds == 1
    assert stats.comm_bytes > 0


def verify_relu_layer(device="cpu"):
    torch.manual_seed(44)
    engine = SecureShareEngine(device=device)
    x = torch.randn(2, 4, 5, 5, device=engine.device)

    output_shares, stats = secure_relu(engine.make_shares(x, 2), engine, interaction_rounds=2)
    actual = engine.reconstruct(output_shares)
    expected = engine.quantize(torch.clamp(engine.quantize(x), min=0))

    _assert_close("secure_relu", actual, expected)
    assert stats.rounds == 2
    assert stats.comm_bytes > 0


def verify_pool_layers(device="cpu"):
    torch.manual_seed(45)
    engine = SecureShareEngine(device=device)
    x = torch.randn(2, 3, 8, 8, device=engine.device)
    shares = engine.make_shares(x, 2)

    max_pool = SecureMaxPool2d(kernel_size=2, stride=2, engine=engine)
    max_shares, max_stats = max_pool(shares)
    _assert_close(
        "SecureMaxPool2d",
        engine.reconstruct(max_shares),
        engine.quantize(F.max_pool2d(engine.quantize(x), kernel_size=2, stride=2)),
    )
    assert max_stats.rounds > 0
    assert max_stats.comm_bytes > 0

    avg_pool = SecureAvgPool2d(kernel_size=2, stride=2, engine=engine)
    avg_shares, avg_stats = avg_pool(shares)
    _assert_close(
        "SecureAvgPool2d",
        engine.reconstruct(avg_shares),
        engine.quantize(F.avg_pool2d(engine.quantize(x), kernel_size=2, stride=2)),
    )
    assert avg_stats.rounds == 0
    assert avg_stats.comm_bytes == 0


def main():
    requested = os.getenv("ASS_VERIFY_DEVICE", "cpu").strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ASS_VERIFY_DEVICE=cuda requested but CUDA is unavailable.")
    device = "cuda" if requested == "cuda" else "cpu"

    verify_conv_layer(device=device)
    verify_linear_layer(device=device)
    verify_relu_layer(device=device)
    verify_pool_layers(device=device)
    print("All current secure operator checks passed.")


if __name__ == "__main__":
    main()
