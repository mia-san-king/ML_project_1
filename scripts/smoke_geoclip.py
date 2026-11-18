#!/usr/bin/env python3
"""Quick check after ml-p2 on DHLAB."""
import torch
from geoclip import GeoCLIP

print("torch", torch.__version__)
print("gpus", torch.cuda.device_count(), "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
GeoCLIP()
print("GeoCLIP OK")
