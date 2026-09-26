import torch

N = 4

A = torch.ones(N, N, device="cuda")
B = torch.ones(N, N, device="cuda")

C = A @ B

torch.cuda.synchronize()

print("A:")
print(A)

print("\nB:")
print(B)

print("\nC = A @ B:")
print(C)

print("\nC device:")
print(C.device)
