import torch
import argparse
from tqdm import tqdm
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)
args = parser.parse_args()

ckpt_paths = glob(f'{args.root}/**/*.ckpt', recursive=True)
pbar = tqdm(total=len(ckpt_paths), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for ckpt_path in ckpt_paths:
    ckpt = torch.load(ckpt_path)
    ckpt['hyper_parameters']['config'].separate_alpha = False
    # torch.save(ckpt, ckpt_path)
    pbar.update(1)
    break