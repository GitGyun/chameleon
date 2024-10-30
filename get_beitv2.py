import timm
import os
import glob
import shutil
from pathlib import Path

beitv2_large = timm.create_model("hf_hub:timm/beitv2_large_patch16_224.in1k_ft_in22k", pretrained=True)
file_path = glob.glob('/root/.cache/huggingface/hub/models--timm--beitv2_large_patch16_224.in1k_ft_in22k/snapshots/**/pytorch_model.bin')[0]

Path('/root/.cache/torch/hub/checkpoints/').mkdir(parents=True, exist_ok=True)
os.system(f'ln -s {file_path} /root/.cache/torch/hub/checkpoints/beitv2_large_patch16_224_pt1k_ft21kto1k.pth')  
