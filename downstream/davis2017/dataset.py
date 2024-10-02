import os
from PIL import Image
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize

from dataset.utils import crop_arrays


class DAVIS2017(Dataset):
    '''
    base class for DAVIS2017 dataset
    '''
    CLASS_NAMES = ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows',
                   'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high',
                   'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick', 'motocross-jump', 'paragliding-launch',
                   'parkour', 'pigs', 'scooter-black', 'shooting', 'soapbox']
    NUM_INSTANCES = [2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 5, 2, 3, 2, 3, 5, 1, 3, 2, 2, 3, 1, 3, 2, 3, 3]


class DAVIS2017Dataset(DAVIS2017):
    '''
    DAVIS2017 dataset
    '''
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        super().__init__()
        assert config.class_name in self.CLASS_NAMES

        # configure paths
        self.base_size = base_size
        data_root = config.path_dict[config.dataset]
        data_dir = f'resized_{config.base_size[1]}'
        self.image_dir = os.path.join(data_root, data_dir, config.class_name, 'images')
        self.label_dir = os.path.join(data_root, data_dir, config.class_name, 'labels')

        self.img_size = crop_size
        self.eval_mode = eval_mode
        self.resize = resize
        self.shot = config.shot
        self.support_idx = config.support_idx
        self.n_channels = self.NUM_INSTANCES[self.CLASS_NAMES.index(config.class_name)] + 1
        self.precision = config.precision
        self.randomscale = config.randomscale

        self.toten = ToTensor()
        self.resizer = Resize(self.img_size)

        self.data_idxs = np.array(sorted(os.listdir(self.image_dir)))

        assert split in ['train', 'valid', 'test']
        if split == 'train':
            train_idxs = [0]
            for i in range(1, self.shot):
                train_idxs.append((len(self.data_idxs) // self.shot) * i)
            assert len(train_idxs) == self.shot, f'{len(train_idxs)} != {self.shot}'
            self.data_idxs = self.data_idxs[train_idxs]
        elif split == 'valid':
            n_vis = 10
            valid_idxs_reordered = []
            valid_idxs = torch.arange(len(self.data_idxs))
            vis_idxs = torch.linspace(min(valid_idxs), max(valid_idxs), n_vis).round().long().tolist()
            vis_idxs = [min(valid_idxs, key=lambda x:abs(x-vis_idx)) for vis_idx in vis_idxs]
            for i in vis_idxs:
                valid_idxs_reordered.append(i)
            for i in valid_idxs:
                if i not in vis_idxs:
                    valid_idxs_reordered.append(i)
            self.data_idxs = self.data_idxs[valid_idxs_reordered]

        if dset_size < 0:
            self.dset_size = len(self.data_idxs)
        elif not eval_mode:
            self.dset_size = dset_size
        else:
            self.dset_size = min(dset_size, len(self.data_idxs))
        
    def __len__(self):
        return self.dset_size

    def __getitem__(self, idx):
        img_path = self.data_idxs[idx % len(self.data_idxs)]
        image, label = self.load_data(img_path)
        
        return self.postprocess_data(image, label)
    
    def load_data(self, img_path):
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        
        lbl_path = img_path.replace('jpg', 'png')
        label = Image.open(os.path.join(self.label_dir, lbl_path))
        
        return image, label

    def postprocess_data(self, image, label):
        X = self.toten(image)
        Y = self.toten(label).squeeze(0)
            
        Y = (Y*255).long()
        Y = F.one_hot(Y, self.n_channels).permute(2, 0, 1).float()
        Y = Y[1:] # remove background channel

        if (not self.eval_mode):
            if self.randomscale and random.random() > 0.5:
                max_scale = 1.5
                scale_h = random.uniform(self.img_size[0] / self.base_size[0], max_scale)
                if random.random() > 0.5:
                    scale_w = scale_h
                else:
                    scale_w = random.uniform(self.img_size[1] / self.base_size[1], max_scale)
                target_size = (max(self.img_size[0], int(self.base_size[0] * scale_h)),
                               max(self.img_size[1], int(self.base_size[1] * scale_w)))
                X = F.interpolate(X.unsqueeze(0), target_size, mode='bilinear', align_corners=False).squeeze(0)
                Y = F.interpolate(Y.unsqueeze(0), target_size, mode='nearest').squeeze(0)

        if self.resize:
            X = self.resizer(X)
            Y = self.resizer(Y)
        else:
            X, Y = crop_arrays(X, Y,
                               base_size=X.size()[-2:],
                               crop_size=self.img_size,
                               random=(not self.eval_mode))
            
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
            
        M = torch.ones_like(Y)
            
        return X, Y, M