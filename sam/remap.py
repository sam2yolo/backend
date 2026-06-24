import torch
import sys

# Load your downloaded checkpoint
origin_ckpt = torch.load(sys.argv[1])
mapped_ckpt = {}

for k, v in origin_ckpt.items():
    if k.startswith("tracker.model."):
        mapped_ckpt[k.replace("tracker.model.", "")] = v
    elif k.startswith("detector."):
        mapped_ckpt[k.replace("detector.", "")] = v
    mapped_ckpt[k] = v

# Save the corrected weights
torch.save(mapped_ckpt, sys.argv[2])
print("Checkpoint remapped successfully!")