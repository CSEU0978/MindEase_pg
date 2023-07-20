import torch

print("cuda available ", torch.cuda.is_available())
print("cuda is_initialized? ", torch.cuda.is_initialized())
print("cuda compiled ", torch.cuda._is_compiled())
print("cuda gencode flags: ", torch.cuda.get_gencode_flags())
print("cuda device count", torch.cuda.device_count())
print("cuda device capability", torch.cuda.get_device_capability())
print("cuda current devices", torch.cuda.current_device())
print("cuda available", torch.cuda.is_available())
