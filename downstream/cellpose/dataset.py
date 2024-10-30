import os
from PIL import Image
import numpy as np
from einops import repeat
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, ColorJitter, RandomRotation

from dataset.utils import crop_arrays


class CELLPOSEDataset(Dataset):
    '''
    cellpose dataset
    '''
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        super().__init__()
        self.base_size = base_size
        data_root = config.path_dict[config.dataset]
        self.eval_mode = eval_mode
        if split == 'valid':
            split = 'test'
        self.data_dir = os.path.join(data_root, split)

        self.img_size = crop_size
        self.eval_mode = eval_mode
        self.resize = resize
        self.shot = config.shot
        self.support_idx = config.support_idx
        self.precision = config.precision
        
        if split == 'train':
            idxs_path = os.path.join('dataset', 'meta_info', 'cellpose', 'train_idxs_perm.pth')
            if not os.path.exists(idxs_path):
                idxs = torch.randperm(540)
                torch.save(idxs, idxs_path)
            else:
                idxs = torch.load(idxs_path)
            self.data_idxs = idxs[self.support_idx*self.shot:(self.support_idx + 1)*self.shot]
        else:
            self.data_idxs = [i for i in range(0, 68)]
        
        self.toten = ToTensor()
        self.base_resizer = Resize(base_size)
        self.resizer = Resize(self.img_size)
        self.resize = resize
        self.eval_mode = eval_mode

        self.randomflip = config.randomflip
        self.randomjitter = config.randomjitter
        self.randomrotate = config.randomrotate
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.rotate = RandomRotation(30)
        self.max_h = 576
        self.max_w = 720

        if dset_size < 0:
            self.dset_size = len(self.data_idxs)
        elif not eval_mode:
            self.dset_size = dset_size
        else:
            self.dset_size = min(dset_size, len(self.data_idxs))   

    def __len__(self):
        return self.dset_size

    def load_data(self, img_path, mask_path):
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) 
        image = self.toten(image)
        image = torch.stack([image[1], image[0]]) # make order to (cytoplasm, nuclei)
        
        mask = self.toten(mask)
        flow = torch.from_numpy(np.load(mask_path.replace("masks.png", "flows.npy")))
        return image, flow, mask
    
    def postprocess_data(self, image, flow, mask):
        """
        image: 2 H W
        flow: 2 H W
        mask: 1 H W
        """
        
        # for evaluation
        aux = {}
        aux["full_mask"] = F.pad(mask, (0, self.max_w - mask.shape[-1], 0 ,self.max_h - mask.shape[-2]))
        aux["full_flow"] = F.pad(flow,  (0, self.max_w - mask.shape[-1], 0 ,self.max_h - mask.shape[-2]))
        aux["full_semmask"] = F.pad((mask > 0).float(), (0, self.max_w - mask.shape[-1], 0 ,self.max_h - mask.shape[-2]))
        aux["full_res"] = mask.shape[-2:]
        
        # normalize flow to [0, 1]
        flow = (flow + 1) / 2
        
        # repeat single-channel images to rgb
        X = repeat(image, "N H W -> (N C) H W", C=3) # 2 H W -> 6 H W

        # foreground mask
        Y_mask = (mask > 0).float()

        # append label channels
        Y = torch.cat([flow, Y_mask], dim=0) # 3 H W
        M = torch.ones_like(Y)

        # resize images to base_size or adaptive size
        if (X.shape[-1] < self.img_size[-1] or X.shape[-2] < self.img_size[-2]):
            # autocropping
            # adaptive_img_size = (int(self.img_size[-1] * self.img_size[-1] / 224), int(self.img_size[-2] * self.img_size[-2] / 224))
            adaptive_img_size = self.img_size
            
            min_ratio = max(adaptive_img_size[-1] / X.shape[-1] , adaptive_img_size[-2] / X.shape[-2])
            adaptive_resizer = Resize((1, 1))
            adaptive_resizer.size = (max(adaptive_img_size[-2], int(min_ratio * X.shape[-2])),
                                            max(adaptive_img_size[-1], int(min_ratio * X.shape[-1])) )
            X = adaptive_resizer(X)
            Y = adaptive_resizer(Y)
            M = adaptive_resizer(M)
        
        # image augmentation
        if not self.eval_mode:
            if self.randomflip:
                if random.random() > 0.5:
                    X = torch.flip(X, dims=[-1])
                    Y = torch.flip(Y, dims=[-1])
                    Y[1] = 1-Y[1]
                    M = torch.flip(M, dims=[-1])
                if random.random() > 0.5:
                    X = torch.flip(X, dims=[-2])
                    Y = torch.flip(Y, dims=[-2])
                    Y[0] = 1-Y[0]
                    M = torch.flip(M, dims=[-2])
            
            if self.randomjitter and random.random() > 0.5:
                X1, X2 = X.split(3, dim=0)
                X12 = torch.cat([X1, X2], dim=-1)
                X12 = self.jitter(X12)
                X1, X2 = X12.split(X1.size()[-1], dim=-1)
                X1 = repeat(X1[:1], '1 H W -> 3 H W')
                X2 = repeat(X2[:1], '1 H W -> 3 H W')
                X = torch.cat([X1, X2], dim=0)
        
        # resize or crop image to img_size
        if self.resize:
            X = self.resizer(X)
            Y = self.resizer(Y) 
            M = self.resizer(M)
        elif not self.eval_mode:
            X, Y, M = crop_arrays(X, Y, M,
                               base_size=X.size()[-2:],
                               crop_size=self.img_size,
                               random=(not self.eval_mode))
        
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
            M = M.to(torch.bfloat16)

        if self.eval_mode:
            return X, Y, M, aux 
        else:
            return X, Y, M

    def __getitem__(self, idx):
        cur_idx = self.data_idxs[idx % len(self.data_idxs)]
        cur_idx = str(cur_idx).zfill(3)
        img_path = os.path.join(self.data_dir, f'{cur_idx}_img.png')
        mask_path = os.path.join(self.data_dir, f'{cur_idx}_masks.png')
        
        image, flow, mask = self.load_data(img_path, mask_path)
        return self.postprocess_data(image, flow, mask)
