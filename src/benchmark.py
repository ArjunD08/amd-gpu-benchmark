import time
import statistics
import csv

import torch


def benchmark_matrix_size(N):
    ITERATIONS = 20

    print(f"\nMatrix size: {N} x {N}")
    print(f"Iterations: {ITERATIONS}")

    # -------------------------
    # Create input matrices
    # -------------------------

    A_cpu = torch.randn(N, N, device="cpu")
    B_cpu = torch.randn(N, N, device="cpu")

    A_gpu = A_cpu.to("cuda")
    B_gpu = B_cpu.to("cuda")

    # -------------------------
    # CPU benchmark
    # -------------------------

    cpu_times = []

    for _ in range(ITERATIONS):
        start = time.perf_counter()

        C_cpu = A_cpu @ B_cpu

        end = time.perf_counter()

        cpu_times.append(end - start)

    # -------------------------
    # GPU warm-up
    # -------------------------

    for _ in range(5):
        C_gpu = A_gpu @ B_gpu

    torch.cuda.synchronize()

    # -------------------------
    # GPU benchmark
    # -------------------------

    gpu_times = []

    for _ in range(ITERATIONS):
        torch.cuda.synchronize()

        start = time.perf_counter()

        C_gpu = A_gpu @ B_gpu

        torch.cuda.synchronize()

        end = time.perf_counter()

        gpu_times.append(end - start)

    # -------------------------
    # Result validation
    # -------------------------

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

    # -------------------------
    # Calculate statistics
    # -------------------------

    cpu_average = statistics.mean(cpu_times)
    gpu_average = statistics.mean(gpu_times)

    cpu_median = statistics.median(cpu_times)
    gpu_median = statistics.median(gpu_times)

    cpu_min = min(cpu_times)
    gpu_min = min(gpu_times)

    # -------------------------
    # Calculate GFLOPS
    # -------------------------

    flops = 2 * (N ** 3)

    cpu_gflops = flops / cpu_average / 1e9
    gpu_gflops = flops / gpu_average / 1e9

    # -------------------------
    # Display results
    # -------------------------

    print("\nResults")
    print("-------")

    print(f"Results match: {results_match}")
    print(f"Max absolute error: {max_absolute_error:.8e}")

    print(f"CPU average: {cpu_average * 1000:.3f} ms")
    print(f"CPU median:  {cpu_median * 1000:.3f} ms")
    print(f"CPU minimum: {cpu_min * 1000:.3f} ms")

    print(f"GPU average: {gpu_average * 1000:.3f} ms")
    print(f"GPU median:  {gpu_median * 1000:.3f} ms")
    print(f"GPU minimum: {gpu_min * 1000:.3f} ms")

    print(f"CPU throughput: {cpu_gflops:.2f} GFLOPS")
    print(f"GPU throughput: {gpu_gflops:.2f} GFLOPS")

    # -------------------------
    # Return structured results
    # -------------------------

    return {
        "matrix_size": N,
        "cpu_average_ms": cpu_average * 1000,
        "cpu_median_ms": cpu_median * 1000,
        "cpu_minimum_ms": cpu_min * 1000,
        "gpu_average_ms": gpu_average * 1000,
        "gpu_median_ms": gpu_median * 1000,
        "gpu_minimum_ms": gpu_min * 1000,
        "cpu_gflops": cpu_gflops,
        "gpu_gflops": gpu_gflops,
        "results_match": results_match,
        "max_absolute_error": max_absolute_error,
    }


# -------------------------
# Main benchmark
# -------------------------

matrix_sizes = [1024, 2048, 4096, 8192]

results = []

for N in matrix_sizes:
    result = benchmark_matrix_size(N)
    results.append(result)


# -------------------------
# Save results to CSV
# -------------------------

csv_path = "results/benchmark.csv"

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
    "results_match",
    "max_absolute_error",
]

with open(csv_path, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to: {csv_path}")
