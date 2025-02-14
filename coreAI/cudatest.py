import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Should return True if CUDA is enabled

print(torch.cuda.device_count())  # Should show number of GPUs
print(torch.cuda.get_device_name(0))  # Should display name of GPU
