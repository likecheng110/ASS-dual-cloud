from baselines.shared_core import run_shared_protocol_inference


def run_2cloud_inference(model_path, test_loader, input_shape, task_name="MNIST", interaction_rounds=1, device="auto"):
    result = run_shared_protocol_inference(
        model_path=model_path,
        test_loader=test_loader,
        input_shape=input_shape,
        task_name=task_name,
        data_shares=2,
        model_shares=1,
        interaction_rounds=interaction_rounds,
        device=device,
    )
    return (
        result["Acc"],
        result["Time"],
        result["Comm"],
        result["Layer"],
        result["Samples"],
        result["OfflineSetupMB"],
    )
