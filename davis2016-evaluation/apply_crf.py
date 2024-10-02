import argparse
import os
from crf import dense_crf
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Resize
from torchvision.utils import save_image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True)
parser.add_argument('--result_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
args = parser.parse_args()

upsample = Resize((480, 854))

obj_dirs = []
save_dirs = []
for obj in os.listdir(args.result_dir):
    obj_dir = os.path.join(args.result_dir, obj)
    if os.path.isdir(obj_dir):
        obj_dirs.append(obj_dir)
        save_dir = os.path.join(args.save_dir, obj)
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)

pbar = tqdm(total=len(obj_dirs), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for obj_dir, save_dir in zip(obj_dirs, save_dirs):
    obj = obj_dir.split('/')[-1]
    for file in os.listdir(obj_dir):
        if file.split('.')[-1] == 'pth':
            score_path = os.path.join(obj_dir, file)
            score = torch.load(score_path).float()

            img_path = os.path.join(args.image_dir, obj, file.replace('.pth', '.jpg'))
            img = np.array(Image.open(img_path))

            save_path = os.path.join(save_dir, file.replace('.pth', '.png'))

            prob = upsample(torch.cat((torch.zeros_like(score),
                                       score)).softmax(dim=0).cpu()).numpy()
            pred = torch.argmax(torch.from_numpy(dense_crf(img, prob)), dim=0).float()
            save_image(pred, save_path)
    pbar.update()
