import timm
import os
import glob
import shutil
from pathlib import Path
import os; import ipdb; ipdb.set_trace(context=15) if os.environ.get("LOCAL_RANK", '0') == '0' else None

beitv2_large = timm.create_model("hf_hub:timm/beitv2_large_patch16_224.in1k_ft_in22k", pretrained=True)
# import os; import ipdb; ipdb.set_trace(context=15) if os.environ.get("LOCAL_RANK", '0') == '0' else None
file_path = glob.glob('/root/.cache/huggingface/hub/models--timm--beitv2_large_patch16_224.in1k_ft_in22k/snapshots/**/pytorch_model.bin')[0]

Path('/root/.cache/torch/hub/checkpoints/').mkdir(parents=True, exist_ok=True)
os.system(f'ln -s {file_path} /root/.cache/torch/hub/checkpoints/beitv2_large_patch16_224_pt1k_ft21kto1k.pth')  
