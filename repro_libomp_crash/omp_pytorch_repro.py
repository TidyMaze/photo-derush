#!/usr/bin/env python3
"""Minimal repro harness to exercise libtorch/libomp on macOS.

This script performs simple PyTorch CPU operations in multiple threads to try
and trigger EXC_BAD_ACCESS in libomp as observed in the app. It deliberately
avoids Qt, multiprocessing, and the rest of the app stack.

If this doesn't reproduce the crash in your environment, switch the TRY_YOLO
flag to True to run the direct YOLOv8 detection call from your existing
`run_yolov8_direct.py` (copy it into this folder first).
"""

import os
import time
import argparse

# Set conservative single-threaded env vars similar to what we used in repro
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "compact")


def do_cpu_work(size: int = 512):
    """Perform a few tensor ops resembling the kernels we saw in traces."""
    try:
        import torch
    except Exception as e:
        print("Failed to import torch:", e)
        raise
    # allocate random tensors and run copy / diag_embed-ish ops
    x = torch.randn(size, size, dtype=torch.float32)
    y = torch.zeros_like(x)
    y.copy_(x)
    # diag_embed-like workload
    v = torch.randn(size, dtype=torch.float32)
    z = torch.diag_embed(v)
    return (x.sum().item(), y.sum().item(), z.sum().item())


def main(iterations: int, threads: int, batch_size: int, try_yolo: bool):
    print(f"Starting repro: iterations={iterations} threads={threads} batch_size={batch_size} try_yolo={try_yolo}")

    if try_yolo:
        # If user wants to try YOLO-based workload, import and call it here.
        # Expect the user to place a copy of their `run_yolov8_direct.py` in
        # this folder that exposes a `run_iterations(iterations)` function.
        try:
            import run_yolov8_direct as yolo_runner
        except Exception as e:
            print("Failed to import run_yolov8_direct.py in this folder:", e)
            return 1
        print("Running YOLO direct runner (from run_yolov8_direct.py)")
        yolo_runner.run_iterations(iterations)
        return 0

    # Use ThreadPoolExecutor to emulate concurrency that may stress libomp
    import concurrent.futures

    total = 0
    start_time = time.time()
    for it in range(1, iterations + 1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
            futures = [ex.submit(do_cpu_work, batch_size) for _ in range(threads)]
            for f in concurrent.futures.as_completed(futures):
                try:
                    res = f.result()
                    total += sum(res)
                except Exception as e:
                    print(f"Worker raised: {e}")
                    raise
        if it % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {it}/{iterations} elapsed={elapsed:.1f}s total={total:.1f}")
    print("Done")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000, help="Number of loop iterations")
    parser.add_argument("--threads", type=int, default=4, help="Number of concurrent threads in each iteration")
    parser.add_argument("--batch-size", type=int, default=512, help="Tensor dimension for square tensors")
    parser.add_argument("--try-yolo", action="store_true", help="If set, import and run run_yolov8_direct.py in this folder")
    args = parser.parse_args()
    raise SystemExit(main(args.iterations, args.threads, args.batch_size, args.try_yolo))
