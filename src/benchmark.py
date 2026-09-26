import csv
import time
from statistics import mean, median

import torch


# ============================================================
# Benchmark configuration
# ============================================================

MATRIX_SIZES = [1024, 2048, 4096, 8192]

WARMUP_ITERATIONS = 5
BENCHMARK_ITERATIONS = 20

CPU_DEVICE = "cpu"
GPU_DEVICE = "cuda"


# ============================================================
# Benchmark one matrix size
# ============================================================

def benchmark_matrix_size(N):

    print(f"\nMatrix size: {N} x {N}")
    print(f"Warm-up iterations: {WARMUP_ITERATIONS}")
    print(f"Benchmark iterations: {BENCHMARK_ITERATIONS}")

    # --------------------------------------------------------
    # Create matrices on CPU
    # --------------------------------------------------------

    A_cpu = torch.randn(N, N, device=CPU_DEVICE)
    B_cpu = torch.randn(N, N, device=CPU_DEVICE)

    # --------------------------------------------------------
    # Copy the same matrices to GPU
    # --------------------------------------------------------

    A_gpu = A_cpu.to(GPU_DEVICE)
    B_gpu = B_cpu.to(GPU_DEVICE)

    # --------------------------------------------------------
    # CPU benchmark
    # --------------------------------------------------------

    cpu_times = []

    for _ in range(BENCHMARK_ITERATIONS):

        start = time.perf_counter()

        C_cpu = A_cpu @ B_cpu

        end = time.perf_counter()

        cpu_times.append(end - start)

    # --------------------------------------------------------
    # GPU warm-up
    # --------------------------------------------------------

    for _ in range(WARMUP_ITERATIONS):

        C_gpu = A_gpu @ B_gpu

    # Wait until all warm-up GPU operations finish
    torch.cuda.synchronize()

    # --------------------------------------------------------
    # GPU benchmark
    # --------------------------------------------------------

    gpu_times = []

    for _ in range(BENCHMARK_ITERATIONS):

        # Make sure previous GPU work has finished
        torch.cuda.synchronize()

        start = time.perf_counter()

        C_gpu = A_gpu @ B_gpu

        # GPU operations are asynchronous.
        # Synchronization ensures the multiplication is finished
        # before we stop the timer.
        torch.cuda.synchronize()

        end = time.perf_counter()

        gpu_times.append(end - start)

    # --------------------------------------------------------
    # Result validation
    # --------------------------------------------------------

    # Copy GPU result back to CPU
    gpu_result_cpu = C_gpu.cpu()

    results_match = torch.allclose(
        C_cpu,
        gpu_result_cpu,
        rtol=1e-3,
        atol=1e-3
    )

    max_absolute_error = torch.max(
        torch.abs(C_cpu - gpu_result_cpu)
    ).item()

    # --------------------------------------------------------
    # Calculate timing statistics
    # --------------------------------------------------------

    cpu_average = mean(cpu_times)
    cpu_median = median(cpu_times)
    cpu_minimum = min(cpu_times)

    gpu_average = mean(gpu_times)
    gpu_median = median(gpu_times)
    gpu_minimum = min(gpu_times)

    # --------------------------------------------------------
    # Calculate GFLOPS
    # --------------------------------------------------------

    # Matrix multiplication:

    # C = A × B

    # For N × N matrices:
    #
    # Approximately 2 × N³ floating-point operations
    #
    # 1 GFLOP = 1 billion floating-point operations

    flops = 2 * (N ** 3)

    cpu_gflops = flops / cpu_average / 1e9
    gpu_gflops = flops / gpu_average / 1e9

    # --------------------------------------------------------
    # Calculate GPU speedup
    # --------------------------------------------------------

    gpu_speedup = cpu_average / gpu_average

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nResults")
    print("-------")

    print(f"Results match: {results_match}")
    print(f"Max absolute error: {max_absolute_error:.8e}")

    print(f"CPU average: {cpu_average * 1000:.3f} ms")
    print(f"CPU median:  {cpu_median * 1000:.3f} ms")
    print(f"CPU minimum: {cpu_minimum * 1000:.3f} ms")

    print(f"GPU average: {gpu_average * 1000:.3f} ms")
    print(f"GPU median:  {gpu_median * 1000:.3f} ms")
    print(f"GPU minimum: {gpu_minimum * 1000:.3f} ms")

    print(f"CPU throughput: {cpu_gflops:.2f} GFLOPS")
    print(f"GPU throughput: {gpu_gflops:.2f} GFLOPS")

    print(f"GPU speedup: {gpu_speedup:.2f}x")

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "matrix_size": N,

        "cpu_average_ms": cpu_average * 1000,
        "cpu_median_ms": cpu_median * 1000,
        "cpu_minimum_ms": cpu_minimum * 1000,

        "gpu_average_ms": gpu_average * 1000,
        "gpu_median_ms": gpu_median * 1000,
        "gpu_minimum_ms": gpu_minimum * 1000,

        "cpu_gflops": cpu_gflops,
        "gpu_gflops": gpu_gflops,

        "gpu_speedup": gpu_speedup,

        "results_match": results_match,
        "max_absolute_error": max_absolute_error,
    }


# ============================================================
# Main benchmark
# ============================================================

def main():

    print("=" * 60)
    print("AMD GPU Matrix Multiplication Benchmark")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"HIP version: {torch.version.hip}")

    print(f"GPU available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: AMD GPU not detected.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU capability: {torch.cuda.get_device_capability(0)}")

    print(f"\nMatrix sizes: {MATRIX_SIZES}")
    print(f"Warm-up iterations: {WARMUP_ITERATIONS}")
    print(f"Benchmark iterations: {BENCHMARK_ITERATIONS}")

    print(f"\nCPU threads: {torch.get_num_threads()}")
    print(f"CPU inter-op threads: {torch.get_num_interop_threads()}")

    # --------------------------------------------------------
    # Run benchmarks
    # --------------------------------------------------------

    all_results = []

    for N in MATRIX_SIZES:

        result = benchmark_matrix_size(N)

        all_results.append(result)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = "results/benchmark.csv"

    fieldnames = [
        "matrix_size",

        "cpu_average_ms",
        "cpu_median_ms",
        "cpu_minimum_ms",

        "gpu_average_ms",
        "gpu_median_ms",
        "gpu_minimum_ms",

        "cpu_gflops",
        "gpu_gflops",

        "gpu_speedup",

        "results_match",
        "max_absolute_error",
    ]

    with open(output_file, "w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)

    print(f"Results saved to: {output_file}")


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
    main()
