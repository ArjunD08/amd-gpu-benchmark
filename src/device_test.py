import torch

print("PyTorch version:", torch.__version__)
print("HIP version:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())

cpu_tensor = torch.ones(5, device="cpu")
gpu_tensor = torch.ones(5, device="cuda")

print("\nCPU tensor:")
print(cpu_tensor)
print("Device:", cpu_tensor.device)

print("\nGPU tensor:")
print(gpu_tensor)
print("Device:", gpu_tensor.device)

print("\nGPU name:")
print(torch.cuda.get_device_name(0))
