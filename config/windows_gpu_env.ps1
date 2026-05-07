$env:PYTHONUTF8 = "1"

# Device selection: auto / cuda / cpu
$env:ASS_DEVICE = "cuda"

# DataLoader workers used by train_plain.py, run_experiment.py and all loaders
$env:ASS_DATALOADER_WORKERS = "0"

# Main experiment defaults
$env:ASS_EXP_REPEATS = "3"
$env:ASS_REPEAT_SEEDS = "42,52,62"
$env:EVAL_MAX_SAMPLES = "5000"
$env:RUN_HE_BASELINES = "1"
$env:HE_TASK_FILTER = "MNIST,Fashion,Medical"
$env:PAILLIER_MAX_SAMPLES = "5"
$env:PAILLIER_MAX_SECONDS = "600"
$env:PAILLIER_PROGRESS_INTERVAL = "1"
$env:RUN_MODEL_SPLIT_NECESSITY = "1"
$env:MODEL_SPLIT_NECESSITY_RATIOS = "1,4,9,19"
$env:RUN_LOOPBACK_AUDIT = "1"
$env:LOOPBACK_TASKS = "MNIST,Medical,Digits"
$env:LOOPBACK_METHODS = "2Cloud-D (Data-only),ASS (Ours),3Share-DM (3-party)"
$env:LOOPBACK_EVAL_MAX_SAMPLES = "512"

# Uncomment if you want to restrict tasks during debug
# $env:TASK_FILTER = "Medical,Digits,Liver"

# Uncomment for full retraining before the final run
# $env:FORCE_RETRAIN_MODELS = "1"
# $env:FORCE_RETRAIN_LONGTAIL_MODELS = "1"
