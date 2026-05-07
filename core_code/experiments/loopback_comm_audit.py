import socket
import threading
import time
from typing import Dict, Iterable, List, Tuple

from baselines.shared_core import run_shared_protocol_inference


TRACEABLE_METHODS: Dict[str, Tuple[int, int]] = {
    "2Cloud-D (Data-only)": (2, 1),
    "ASS (Ours)": (2, 2),
    "3Share-DM (3-party)": (3, 3),
}


def _recv_exact(sock: socket.socket, expected_bytes: int):
    remaining = max(0, int(expected_bytes))
    while remaining > 0:
        chunk = sock.recv(min(65536, remaining))
        if not chunk:
            break
        remaining -= len(chunk)


def replay_loopback_bytes(trace_bytes: Iterable[int], chunk_size: int = 65536) -> Dict[str, float]:
    trace = [max(0, int(size)) for size in trace_bytes if int(size) > 0]
    total_bytes = int(sum(trace))
    if total_bytes <= 0:
        return {
            "LoopbackReplayMB": 0.0,
            "LoopbackElapsedMs": 0.0,
            "LoopbackThroughputMBps": None,
            "TraceEvents": 0,
        }

    sender, receiver = socket.socketpair()
    buffer = b"\0" * max(1024, int(chunk_size))
    recv_holder = {"done": False}

    def _reader():
        try:
            _recv_exact(receiver, total_bytes)
        finally:
            recv_holder["done"] = True
            receiver.close()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    start = time.perf_counter_ns()
    try:
        for size in trace:
            remaining = size
            while remaining > 0:
                payload = buffer[: min(len(buffer), remaining)]
                sender.sendall(payload)
                remaining -= len(payload)
        sender.shutdown(socket.SHUT_WR)
        reader.join()
    finally:
        sender.close()

    elapsed_s = (time.perf_counter_ns() - start) / 1e9
    replay_mb = total_bytes / (1024 ** 2)
    throughput = (replay_mb / elapsed_s) if elapsed_s > 0 else None
    return {
        "LoopbackReplayMB": replay_mb,
        "LoopbackElapsedMs": elapsed_s * 1000.0,
        "LoopbackThroughputMBps": throughput,
        "TraceEvents": len(trace),
    }


def audit_traceable_protocol(
    method_name: str,
    model_path: str,
    test_loader,
    input_shape,
    task_name: str,
    device="auto",
) -> Dict[str, object]:
    if method_name not in TRACEABLE_METHODS:
        raise ValueError(f"Unsupported traceable method: {method_name}")

    data_shares, model_shares = TRACEABLE_METHODS[method_name]
    result = run_shared_protocol_inference(
        model_path=model_path,
        test_loader=test_loader,
        input_shape=input_shape,
        task_name=task_name,
        data_shares=data_shares,
        model_shares=model_shares,
        interaction_rounds=1,
        device=device,
        return_comm_trace=True,
    )
    loopback = replay_loopback_bytes(result.get("CommTraceBytes", []))
    return {
        "Task": task_name,
        "Method": method_name,
        "Samples": result["Samples"],
        "EstimatedOnlineCommMB": result["Comm"],
        "TraceEvents": loopback["TraceEvents"],
        "LoopbackReplayMB": loopback["LoopbackReplayMB"],
        "LoopbackElapsedMs": loopback["LoopbackElapsedMs"],
        "LoopbackThroughputMBps": loopback["LoopbackThroughputMBps"],
        "AuditMode": "localhost-loopback",
        "AuditNote": "Single-machine socket replay of protocol-equivalent message sizes; not a cross-host network capture.",
    }
