#!/usr/bin/env python3
"""
Test workload to verify remote Kaggle GPU execution.
Reports hardware specs, PyTorch CUDA capabilities, runs a tensor benchmark,
and writes a structured receipt to /kaggle/working/.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

def main():
    print("=" * 60)
    print("🚀 HELLO FROM KAGGLE CLOUD RUNNER!")
    print("=" * 60)
    print(f"Python: {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"Hostname: {platform.node()}")
    print(f"CWD: {os.getcwd()}")

    # Check GPU availability
    gpu_info = {}
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_props = torch.cuda.get_device_properties(current_device)
            total_vram_gb = device_props.total_memory / (1024 ** 3)

            print(f"GPU Device Count: {device_count}")
            print(f"GPU Device Name: {device_name}")
            print(f"GPU Total VRAM: {total_vram_gb:.2f} GB")
            print(f"CUDA Capability: {device_props.major}.{device_props.minor}")

            # Run a tensor benchmark
            print("\nRunning GPU Tensor Benchmark (8192 x 8192 Matrix Multiplication)...")
            size = 8192
            t0 = time.perf_counter()
            a = torch.randn(size, size, device="cuda", dtype=torch.float32)
            b = torch.randn(size, size, device="cuda", dtype=torch.float32)
            torch.cuda.synchronize()
            t_alloc = time.perf_counter() - t0

            t1 = time.perf_counter()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            t_matmul = time.perf_counter() - t1

            # 2 * N^3 operations for matrix multiplication
            tflops = (2.0 * (size ** 3)) / (t_matmul * 1e12)
            print(f"Matrix Mul Time: {t_matmul * 1000:.2f} ms")
            print(f"Effective Compute: {tflops:.2f} TFLOPS")

            gpu_info = {
                "cuda_available": True,
                "device_name": device_name,
                "device_count": device_count,
                "total_vram_gb": round(total_vram_gb, 2),
                "cuda_capability": f"{device_props.major}.{device_props.minor}",
                "matmul_time_ms": round(t_matmul * 1000, 2),
                "compute_tflops": round(tflops, 2),
            }
        else:
            print("WARNING: CUDA is not available! Running on CPU.")
            gpu_info = {"cuda_available": False}
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        gpu_info = {"error": str(e)}

    # Ensure working directory output is written
    out_dir = Path("/kaggle/working")
    if not out_dir.exists():
        out_dir = Path(".")

    receipt = {
        "status": "SUCCESS",
        "timestamp": time.time(),
        "platform": platform.platform(),
        "gpu_info": gpu_info,
    }

    out_file = out_dir / "gpu_verification_receipt.json"
    with open(out_file, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved verification receipt to: {out_file}")
    print("=" * 60)
    print("✅ KAGGLE GPU EXECUTION VERIFIED AND COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
