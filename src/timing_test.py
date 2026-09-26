import time
import torch

N = 4096

A = torch.randn(N, N, device="cuda")
B = torch.randn(N, N, device="cuda")

# Warm up the GPU
C = A @ B
torch.cuda.synchronize()

# Timing without explicit synchronization
start = time.perf_counter()
C = A @ B
end = time.perf_counter()

print("Without synchronization:")
print(f"{(end - start) * 1000:.3f} ms")

# Timing with synchronization
torch.cuda.synchronize()

start = time.perf_counter()
C = A @ B
torch.cuda.synchronize()
end = time.perf_counter()

print("With synchronization:")
print(f"{(end - start) * 1000:.3f} ms")
