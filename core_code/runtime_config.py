import os
import sys

import torch


def env_value(name, default=None, legacy_name=None):
    value = os.getenv(name)
    if value is None and legacy_name:
        value = os.getenv(legacy_name)
    return default if value is None else value


def env_int(name, default):
    try:
        return int(env_value(name, str(default)))
    except Exception:
        return default


def env_int_alias(name, legacy_name, default):
    try:
        return int(env_value(name, str(default), legacy_name=legacy_name))
    except Exception:
        return default


def env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_bool_alias(name, legacy_name, default):
    raw = env_value(name, None, legacy_name=legacy_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_csv_list(name, default):
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def env_csv_list_alias(name, legacy_name, default):
    raw = env_value(name, default, legacy_name=legacy_name)
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_requested_device():
    return env_value("ASS_DEVICE", "auto").strip().lower()


def detect_runtime_info():
    requested = get_requested_device()
    cuda_available = torch.cuda.is_available()
    if requested == "cuda":
        use_cuda = cuda_available
    elif requested == "cpu":
        use_cuda = False
    else:
        use_cuda = cuda_available

    device = torch.device("cuda" if use_cuda else "cpu")
    gpu_name = None
    gpu_count = 0
    if cuda_available:
        try:
            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 0
        if gpu_count > 0:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None

    warning = None
    if requested == "cuda" and not cuda_available:
        warning = "ASS_DEVICE=cuda was requested, but torch.cuda.is_available() is False."

    return {
        "device": device,
        "use_cuda": use_cuda,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "requested_device": requested,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "conda_env": os.getenv("CONDA_DEFAULT_ENV"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": gpu_count,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "warning": warning,
    }


def configure_runtime_backend(runtime_info):
    if runtime_info.get("use_cuda"):
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def dataloader_kwargs():
    workers = env_int("ASS_DATALOADER_WORKERS", 0)
    kwargs = {"num_workers": max(0, workers)}
    if torch.cuda.is_available():
        kwargs["pin_memory"] = True
    if workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs
