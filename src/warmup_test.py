import time
import torch

N = 4096

A = torch.randn(N, N, device="cuda")
B = torch.randn(N, N, device="cuda")

print("First few matrix multiplication timings:\n")

for i in range(10):
    torch.cuda.synchronize()

    start = time.perf_counter()

    C = A @ B

    torch.cuda.synchronize()

    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000

    print(f"Iteration {i + 1:2d}: {elapsed_ms:.3f} ms")
