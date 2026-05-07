import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import urllib.request
import gzip
import shutil
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from runtime_config import dataloader_kwargs

# ... (existing functions) ...

def _allow_synthetic_data():
    raw = os.getenv("ASS_ALLOW_SYNTHETIC_DATA", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _synthetic_fallback(dataset_name, exc, X, y, batch_size, upsample_factor=1, train_ratio=1, seed=42, return_metadata=False):
    message = f"{dataset_name} data unavailable: {exc}"
    if not _allow_synthetic_data():
        raise RuntimeError(
            f"{message}. Synthetic fallback is disabled for reproducible experiments. "
            "Set ASS_ALLOW_SYNTHETIC_DATA=1 only for demo/debug runs."
        ) from exc
    print(f"{message}. ASS_ALLOW_SYNTHETIC_DATA=1, using synthetic fallback.")
    return _process_data(X, y, batch_size, upsample_factor=upsample_factor, train_ratio=train_ratio, seed=seed, return_metadata=return_metadata)


def _safe_classification_split(X, y, test_size=0.2, random_state=42):
    try:
        if len(np.unique(y)) > 1:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except Exception:
        pass
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _resolve_existing_data_dir(default_dir, filenames, fallback_parts):
    if all(os.path.exists(os.path.join(default_dir, name)) for name in filenames):
        return default_dir
    fallback_dir = os.path.join(os.path.dirname(os.path.dirname(default_dir)), *fallback_parts)
    if all(os.path.exists(os.path.join(fallback_dir, name)) for name in filenames):
        return fallback_dir
    return default_dir


def _label_counts(y):
    values, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(values.tolist(), counts.tolist())}


def _apply_binary_train_ratio(X, y, train_ratio=1, seed=42):
    X = np.asarray(X)
    y = np.asarray(y)
    counts_before = _label_counts(y)
    meta = {
        "requested_ratio": int(train_ratio),
        "applied_ratio": "original",
        "majority_label": None,
        "minority_label": None,
        "counts_before": counts_before,
        "counts_after": counts_before,
    }

    unique = sorted(np.unique(y).tolist())
    if len(unique) != 2:
        return X, y, meta

    majority_label = max(unique, key=lambda label: counts_before[int(label)])
    minority_label = min(unique, key=lambda label: counts_before[int(label)])
    meta["majority_label"] = int(majority_label)
    meta["minority_label"] = int(minority_label)

    ratio = max(1, int(train_ratio))
    rng = np.random.RandomState(seed + ratio * 17)
    majority_idx = np.where(y == majority_label)[0]
    minority_idx = np.where(y == minority_label)[0]

    if ratio <= 1:
        keep_major = min(len(majority_idx), len(minority_idx))
        keep_minor = keep_major
    else:
        keep_major = min(len(majority_idx), len(minority_idx) * ratio)
        keep_minor = max(1, keep_major // ratio)

    majority_pick = rng.choice(majority_idx, size=keep_major, replace=False)
    minority_pick = rng.choice(minority_idx, size=keep_minor, replace=False)
    selected = np.concatenate([majority_pick, minority_pick])
    rng.shuffle(selected)

    X_selected = X[selected]
    y_selected = y[selected]
    counts_after = _label_counts(y_selected)
    meta["counts_after"] = counts_after
    meta["applied_ratio"] = f"{counts_after.get(int(majority_label), 0)}:{counts_after.get(int(minority_label), 0)}"
    return X_selected, y_selected, meta


def _build_loader_bundle(X_train, y_train, X_test, y_test, batch_size, upsample_factor=1):
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    repeat_factor = max(1, int(upsample_factor))
    X_test_u = X_test_t.repeat(repeat_factor, 1)
    y_test_u = y_test_t.repeat(repeat_factor)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_u, y_test_u)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dataloader_kwargs()),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dataloader_kwargs()),
        X_train.shape[1],
    )


def get_digits_data(batch_size=64, upsample_factor=50):
    print("Loading UCI Digits Data (General Baseline)...")
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0 # UCI digits are 0-16
    y = data.target
    
    X_train, X_test, y_train, y_test = _safe_classification_split(X, y, test_size=0.2, random_state=42)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Standard small dataset upsampling for benchmarking
    X_test_u = X_test_t.repeat(upsample_factor, 1)
    y_test_u = y_test_t.repeat(upsample_factor)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_u, y_test_u)
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dataloader_kwargs()), DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dataloader_kwargs()), 64 # 8x8 images

def get_liver_data(batch_size=64, upsample_factor=100, train_ratio=1, seed=42, return_metadata=False):
    print("Loading Liver Disorders Data (BUPA)...")
    # This is a classic medical dataset often used in ML papers
    # Source: https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    file_path = os.path.join(data_dir, 'liver.csv')
    
    if not os.path.exists(file_path):
        try:
            urllib.request.urlretrieve(url, file_path)
        except Exception as exc:
            return _synthetic_fallback(
                "Liver",
                exc,
                np.random.randn(345, 6),
                np.random.randint(0, 2, 345),
                batch_size,
                upsample_factor=upsample_factor,
                train_ratio=train_ratio,
                seed=seed,
                return_metadata=return_metadata,
            )
            
    try:
        df = pd.read_csv(file_path, header=None)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values
        # The local BUPA copy uses labels {1, 2}; map them to {0, 1}.
        y = (y > 1).astype(int)
        return _process_data(X, y, batch_size, upsample_factor=upsample_factor, train_ratio=train_ratio, seed=seed, return_metadata=return_metadata)
    except Exception as exc:
        return _synthetic_fallback(
            "Liver",
            exc,
            np.random.randn(345, 6),
            np.random.randint(0, 2, 345),
            batch_size,
            upsample_factor=upsample_factor,
            train_ratio=train_ratio,
            seed=seed,
            return_metadata=return_metadata,
        )


# Removed torchvision dependency due to environment issues

# Manual MNIST Downloader
def download_and_parse_mnist(data_dir='./data'):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    paths = {}
    for file in files:
        gz_path = os.path.join(data_dir, file)
        raw_path = os.path.join(data_dir, file.replace('.gz', ''))
        paths[file] = raw_path
        
        need_download = True
        if os.path.exists(gz_path):
            try:
                with gzip.open(gz_path, 'rb') as f:
                    f.read(1)
                need_download = False
            except:
                print(f"Corrupted file found: {file}. Re-downloading...")
                os.remove(gz_path)
        
        if need_download:
            print(f"Downloading {file}...")
            try:
                mirror_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
                urllib.request.urlretrieve(mirror_url + file, gz_path)
            except Exception as e:
                print(f"Mirror download failed: {e}. Trying primary...")
                urllib.request.urlretrieve(base_url + file, gz_path)
            
        if not os.path.exists(raw_path):
            print(f"Extracting {file}...")
            try:
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(raw_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                print(f"Extraction failed for {file}: {e}")
                if os.path.exists(gz_path): os.remove(gz_path)
                raise e
                    
    return paths

# Manual Fashion-MNIST Downloader
def download_and_parse_fashion_mnist(data_dir='./data/fashion'):
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    paths = {}
    for file in files:
        gz_path = os.path.join(data_dir, file)
        raw_path = os.path.join(data_dir, file.replace('.gz', ''))
        paths[file] = raw_path
        
        need_download = True
        if os.path.exists(gz_path):
            try:
                with gzip.open(gz_path, 'rb') as f:
                    f.read(1)
                need_download = False
            except:
                print(f"Corrupted file found: {file}. Re-downloading...")
                os.remove(gz_path)
        
        if need_download:
            print(f"Downloading {file}...")
            try:
                urllib.request.urlretrieve(base_url + file, gz_path)
            except Exception as e:
                print(f"Download failed: {e}")
                raise e
            
        if not os.path.exists(raw_path):
            print(f"Extracting {file}...")
            try:
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(raw_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                print(f"Extraction failed for {file}: {e}")
                if os.path.exists(gz_path): os.remove(gz_path)
                raise e
                    
    return paths

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.astype(np.int64)

def get_mnist_data(batch_size=64):
    print("Downloading/Loading Real MNIST Data (Manual)...")
    try:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
        required_files = [
            'train-images-idx3-ubyte',
            'train-labels-idx1-ubyte',
            't10k-images-idx3-ubyte',
            't10k-labels-idx1-ubyte',
        ]
        default_dir = data_dir
        data_dir = _resolve_existing_data_dir(default_dir, required_files, ['archive', '论文＋代码', 'Model_2pc', 'data'])
        if data_dir == default_dir:
            download_and_parse_mnist(data_dir)
        
        x_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        y_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        x_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        
        # Normalize to [-1, 1] for better convergence if needed, but [0, 1] is fine for simple MLP
        # PyTorch transforms.Normalize((0.5,), (0.5,)) maps [0,1] -> [-1,1]
        # Here we have [0,1]. Let's keep it [0,1] or map manually.
        # Let's map to [-1, 1] to match standard practice
        x_train = (x_train - 0.5) / 0.5
        x_test = (x_test - 0.5) / 0.5
        
        x_train_t = torch.from_numpy(x_train)
        y_train_t = torch.from_numpy(y_train)
        x_test_t = torch.from_numpy(x_test)
        y_test_t = torch.from_numpy(y_test)
        
        print(f"  Train: {x_train_t.shape}, Test: {x_test_t.shape}")
        
        train_ds = TensorDataset(x_train_t, y_train_t)
        test_ds = TensorDataset(x_test_t, y_test_t)
        
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dataloader_kwargs()), DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dataloader_kwargs())
        
    except Exception as e:
        print(f"Error manually loading MNIST: {e}")
        x = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,))
        if not _allow_synthetic_data():
            raise RuntimeError(
                "MNIST data unavailable. Synthetic fallback is disabled for reproducible experiments. "
                "Set ASS_ALLOW_SYNTHETIC_DATA=1 only for demo/debug runs."
            ) from e
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, **dataloader_kwargs()), DataLoader(ds, batch_size=batch_size, **dataloader_kwargs())

def get_fashion_mnist_data(batch_size=64):
    print("Loading Fashion-MNIST Data (Manual)...")
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fashion')
        download_and_parse_fashion_mnist(data_dir)
        
        x_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        y_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        x_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        
        # Normalize [-1, 1]
        x_train = (x_train - 0.5) / 0.5
        x_test = (x_test - 0.5) / 0.5
        
        x_train_t = torch.from_numpy(x_train)
        y_train_t = torch.from_numpy(y_train)
        x_test_t = torch.from_numpy(x_test)
        y_test_t = torch.from_numpy(y_test)
        
        print(f"  Train: {x_train_t.shape}, Test: {x_test_t.shape}")
        
        train_ds = TensorDataset(x_train_t, y_train_t)
        test_ds = TensorDataset(x_test_t, y_test_t)
        
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dataloader_kwargs()), DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dataloader_kwargs())
    except Exception as e:
        print(f"Error loading Fashion-MNIST: {e}")
        x = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,))
        if not _allow_synthetic_data():
            raise RuntimeError(
                "Fashion-MNIST data unavailable. Synthetic fallback is disabled for reproducible experiments. "
                "Set ASS_ALLOW_SYNTHETIC_DATA=1 only for demo/debug runs."
            ) from e
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, **dataloader_kwargs()), DataLoader(ds, batch_size=batch_size, **dataloader_kwargs())

def get_medical_data(batch_size=64, upsample_factor=90, train_ratio=1, seed=42, return_metadata=False):
    print("Loading Real Medical Data (Breast Cancer) with Upsampling...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = _safe_classification_split(X, y, test_size=0.2, random_state=42)
    
    X_train_raw, y_train_raw, meta = _apply_binary_train_ratio(X_train_raw, y_train_raw, train_ratio=train_ratio, seed=seed)
    train_loader, test_loader, input_dim = _build_loader_bundle(X_train_raw, y_train_raw, X_test_raw, y_test_raw, batch_size, upsample_factor=upsample_factor)
    if return_metadata:
        return train_loader, test_loader, input_dim, meta
    return train_loader, test_loader, input_dim

def get_wine_data(batch_size=64, upsample_factor=200):
    print("Loading Wine Data (Chemical Analysis)...")
    data = load_wine()
    X = data.data
    y = data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = _safe_classification_split(X, y, test_size=0.2, random_state=42)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    X_test_u = X_test_t.repeat(upsample_factor, 1)
    y_test_u = y_test_t.repeat(upsample_factor)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_u, y_test_u)
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dataloader_kwargs()), DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dataloader_kwargs()), X.shape[1]

def get_heart_disease_data(batch_size=64, upsample_factor=100, train_ratio=1, seed=42, return_metadata=False):
    print("Loading Heart Disease Data (UCI)...")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        # Download if not exists
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        file_path = os.path.join(data_dir, 'heart.csv')
        
        if not os.path.exists(file_path):
            print("Downloading Heart Disease Dataset...")
            try:
                # Use urllib to download
                with urllib.request.urlopen(url) as response, open(file_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            except Exception as e:
                return _synthetic_fallback(
                    "Heart",
                    e,
                    np.random.randn(303, 13),
                    np.random.randint(0, 2, 303),
                    batch_size,
                    upsample_factor=upsample_factor,
                    train_ratio=train_ratio,
                    seed=seed,
                    return_metadata=return_metadata,
                )

        # Load and process
        # UCI format: 14 columns, last is target, missing values are '?'
        try:
            df = pd.read_csv(file_path, header=None, na_values='?')
        except Exception as exc:
             return _synthetic_fallback(
                 "Heart",
                 exc,
                 np.random.randn(303, 13),
                 np.random.randint(0, 2, 303),
                 batch_size,
                 upsample_factor=upsample_factor,
                 train_ratio=train_ratio,
                 seed=seed,
                 return_metadata=return_metadata,
             )
             
        df = df.dropna()
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values
        # Binarize target: 0 is healthy, 1-4 is disease -> 0 vs 1
        y = (y > 0).astype(int)
        
        return _process_data(X, y, batch_size, upsample_factor=upsample_factor, train_ratio=train_ratio, seed=seed, return_metadata=return_metadata)
        
    except Exception as e:
        print(f"Error in Heart Disease Loader: {e}")
        return _synthetic_fallback(
            "Heart",
            e,
            np.random.randn(100, 13),
            np.random.randint(0, 2, 100),
            batch_size,
            upsample_factor=upsample_factor,
            train_ratio=train_ratio,
            seed=seed,
            return_metadata=return_metadata,
        )

def _process_data(X, y, batch_size, upsample_factor=100, train_ratio=1, seed=42, return_metadata=False):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = _safe_classification_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train, meta = _apply_binary_train_ratio(X_train, y_train, train_ratio=train_ratio, seed=seed)
    train_loader, test_loader, input_dim = _build_loader_bundle(X_train, y_train, X_test, y_test, batch_size, upsample_factor=upsample_factor)
    if return_metadata:
        return train_loader, test_loader, input_dim, meta
    return train_loader, test_loader, input_dim

def get_diabetes_data(batch_size=64, upsample_factor=100, train_ratio=1, seed=42, return_metadata=False):
    print("Loading Diabetes Data (Chronic Disease Risk)...")
    data = load_diabetes()
    X = data.data
    y = data.target
    
    median_val = np.median(y)
    y_bin = (y > median_val).astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = _safe_classification_split(X, y_bin, test_size=0.2, random_state=42)
    
    X_train, y_train, meta = _apply_binary_train_ratio(X_train, y_train, train_ratio=train_ratio, seed=seed)
    train_loader, test_loader, input_dim = _build_loader_bundle(X_train, y_train, X_test, y_test, batch_size, upsample_factor=upsample_factor)
    if return_metadata:
        return train_loader, test_loader, input_dim, meta
    return train_loader, test_loader, input_dim
