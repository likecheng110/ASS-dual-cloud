import torch
import time
import tenseal as ts
import numpy as np


def _load_linear_weights(state_dict):
    required = ["fc1.weight", "fc1.bias"]
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise RuntimeError(f"CKKS direct inference requires an MLP state dict; missing {missing}")

    w1 = state_dict["fc1.weight"].numpy().T.astype(np.float64)
    b1 = state_dict["fc1.bias"].numpy().astype(np.float64)

    if "fc2.weight" not in state_dict:
        return w1, b1, None, None

    w2 = state_dict["fc2.weight"].numpy().T.astype(np.float64)
    b2 = state_dict["fc2.bias"].numpy().astype(np.float64)
    return w1, b1, w2, b2


def run_ckks_inference(model_path, data_loader, input_shape, is_medical=False):
    """Run a tiny direct CKKS sanity check.

    The publication pipeline uses ``inference_ckks_benchmark.py`` instead because
    full direct CKKS inference is too fragile and slow for the complete matrix.
    This legacy helper must fail loudly; returning zero timings would pollute a
    result table with a false successful measurement.
    """

    try:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
    except Exception as exc:
        raise RuntimeError(f"CKKS context initialization failed: {exc}") from exc

    state_dict = torch.load(model_path, map_location="cpu")
    w1, b1, w2, b2 = _load_linear_weights(state_dict)

    num_samples = 1
    total_time = 0
    correct = 0
    count = 0

    for images, labels in data_loader:
        if count >= num_samples:
            break

        for i in range(len(images)):
            if count >= num_samples:
                break

            img = images[i].view(-1)
            label = labels[i]

            start = time.perf_counter_ns()

            try:
                enc_x = ts.ckks_vector(context, img.tolist())
                enc_hidden = enc_x.matmul(w1.tolist())
                enc_hidden.add(b1.tolist())
                enc_hidden.square_()

                if w2 is not None:
                    enc_out = enc_hidden.matmul(w2.tolist())
                    enc_out.add(b2.tolist())
                else:
                    enc_out = enc_hidden

                plain_out = enc_out.decrypt()
                pred = np.argmax(plain_out)

                if pred == label.item():
                    correct += 1

            except Exception as exc:
                raise RuntimeError(f"CKKS direct inference failed on sample {count}: {exc}") from exc

            end = time.perf_counter_ns()
            total_time += (end - start) / 1e9
            count += 1

    if count == 0:
        raise RuntimeError("CKKS direct inference received an empty data loader")

    avg_time = total_time / count if count > 0 else 0
    acc = correct / count if count > 0 else 0

    return acc, avg_time, 0
