import os
from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, ColorJitter, RandomRotation, GaussianBlur

from dataset.utils import crop_arrays


class LINEMOD(Dataset):
    '''
    base class for LINEMOD dataset
    '''
    CLASS_NAMES = ['ape', 'benchviseblue', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']


class LINEMODDataset(LINEMOD):
    '''
    LINEMOD dataset for 6D pose estimation
    '''

    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        super().__init__()
        assert config.class_name in self.CLASS_NAMES

        # configure paths
        self.base_size = base_size
        data_root = config.path_dict[config.dataset]

        assert split in ['train', 'valid', 'test']
        self.split = split

        if config.coord_path is None or config.coord_path == 'none' or split != 'test' or not eval_mode:
            data_dir = f'resized_and_scaledcropped_uvw_{base_size[1]}'
            self.image_dir = os.path.join(data_root, data_dir, config.class_name, 'images')
            self.label_dir = os.path.join(data_root, data_dir, config.class_name, 'labels')
            self.pose_dir = os.path.join(data_root, 'pose_data', config.class_name)
            self.coord = None
        else:
            self.image_dir = os.path.join(data_root, 'color_data', config.class_name)
            self.label_dir = os.path.join(data_root, 'texture_data', config.class_name)
            self.pose_dir = os.path.join(data_root, 'pose_data', config.class_name)
            coord_path = config.coord_path
            if not os.path.exists(coord_path):
                coord_path = os.path.join('experiments', config.log_dir, coord_path)
            self.coord = np.load(coord_path)

        self.img_size = crop_size
        self.eval_mode = eval_mode
        self.resize = resize
        self.shot = config.shot
        self.support_idx = config.support_idx
        self.precision = config.precision

        self.toten = ToTensor()
        self.resizer = Resize(self.img_size)
        self.randomjitter = config.randomjitter
        self.randomblur = config.randomblur
        self.randomrotate = config.randomrotate

        if self.randomjitter:
            self.jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        if self.randomblur:
            self.blur = GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        if self.randomrotate:
            self.rotate = RandomRotation(30)

        if split == 'train':
            with open(os.path.join(data_root, 'meta_data', config.class_name, 'support.txt'), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            self.data_idxs = [f"{int(line.split('/')[-1].split('.')[0]):04d}.jpg" for line in lines]
            self.data_idxs = self.data_idxs[:self.shot]
        else:
            with open(os.path.join(data_root, 'meta_data', config.class_name, f'{split.replace("valid", "val")}.txt'), 'r') as f:
                lines = [line.strip() for line in f.readlines()]

            if self.coord is None:
                self.data_idxs = [f"{int(line.split('/')[-1].split('.')[0]):04d}.jpg" for line in lines]
            else:
                self.data_idxs = [f"color{int(line.split('/')[-1].split('.')[0])}.jpg" for line in lines]

            n_vis = 10
            if split == 'valid' and len(self.data_idxs) > n_vis:
                idxs = np.arange(len(self.data_idxs))
                if dset_size > 0:
                    idxs = idxs[::len(self.data_idxs)//dset_size][:dset_size]
                    assert len(idxs) == dset_size

                valid_idxs_reordered = []
                vis_idxs = torch.linspace(min(idxs), max(idxs), n_vis).round().long().tolist()
                vis_idxs = [min(idxs, key=lambda x:abs(x-vis_idx)) for vis_idx in vis_idxs]
                for i in vis_idxs:
                    valid_idxs_reordered.append(i)
                for i in idxs:
                    if i not in vis_idxs:
                        valid_idxs_reordered.append(i)
                self.data_idxs = [self.data_idxs[i] for i in valid_idxs_reordered]

        if dset_size < 0:
            self.dset_size = len(self.data_idxs)
        elif not eval_mode:
            self.dset_size = dset_size
        else:
            self.dset_size = min(dset_size, len(self.data_idxs))
    
    def get_rot_tra(self, rot_path, tra_path):
        rot_matrix = np.loadtxt(rot_path, skiprows=1)
        trans_matrix = np.loadtxt(tra_path, skiprows=1)
        trans_matrix = np.reshape(trans_matrix, (3, 1))
        rigid_transformation = np.append(rot_matrix, trans_matrix, axis=1)

        return rigid_transformation

    def load_data(self, img_path):
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        
        if self.coord is None:
            lbl_path = img_path.replace('.jpg', '.png')
            label = Image.open(os.path.join(self.label_dir, lbl_path))
        else:
            lbl_path = img_path.replace('.jpg', '.png').replace('color', 'corr')
            label = Image.open(os.path.join(self.label_dir, lbl_path))

        if self.eval_mode:
            img_id = int(img_path.split('.')[0].strip('color'))
            rot_path = f'rot{img_id}.rot'
            tra_path = f'tra{img_id}.tra'
            pose = self.get_rot_tra(os.path.join(self.pose_dir, rot_path),
                                    os.path.join(self.pose_dir, tra_path))
            
            image_full = image
            label_full = label
            if self.coord is None:
                coord = np.loadtxt(os.path.join(self.label_dir, lbl_path.replace('.png', '.bbox')))
            else:
                coord = self.coord[self.data_idxs.index(img_path)]
                image = image.crop((coord[0], coord[1], coord[2], coord[3])).resize(self.base_size)
                label = label.crop((coord[0], coord[1], coord[2], coord[3])).resize(self.base_size)
            
            return image, label, (pose, coord, image_full, label_full)
        else:
            return image, label

    def postprocess_data(self, image, label, meta=None):
        X = self.toten(image)
        Y = self.toten(label)
        if meta is not None:
            meta = meta[0], meta[1], self.toten(meta[2]), self.toten(meta[3])

        if self.resize:
            X = self.resizer(X)
            Y = self.resizer(Y)
        else:
            X, Y = crop_arrays(X, Y,
                               base_size=X.size()[-2:],
                               crop_size=self.img_size,
                               random=(not self.eval_mode))
            
        if not self.eval_mode:
            if self.randomjitter and random.random() > 0.5:
                X = self.jitter(X)
            if self.randomblur and random.random() > 0.5:
                X = self.blur(X)
            if self.randomrotate and random.random() > 0.5:
                XY = torch.cat((X, Y), dim=0)
                XY = self.rotate(XY)
                X, Y = XY[:3], XY[3:]

        Y = torch.cat((Y.sum(dim=0, keepdim=True).bool().to(Y.dtype), Y))

        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
            if meta is not None:
                meta = meta[0], meta[1], meta[2].to(torch.bfloat16), meta[3].to(torch.bfloat16)
        M = torch.ones_like(Y)
            
        if meta is not None:
            return X, Y, M, meta
        else:
            return X, Y, M

    def __getitem__(self, idx):
        img_path = self.data_idxs[idx % len(self.data_idxs)]
        if self.eval_mode:
            image, label, meta = self.load_data(img_path)
        else:
            image, label = self.load_data(img_path)
            meta = None
        
        return self.postprocess_data(image, label, meta)
        
    def __len__(self):
        return self.dset_size


class LINEMODMaskDataset(LINEMOD):
    '''
    LINEMOD dataset for semantic segmentation
    '''
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        super().__init__()
        assert config.class_name in self.CLASS_NAMES

        # configure paths
        self.base_size = base_size
        data_root = config.path_dict[config.dataset]
        data_dir = f'resized_mask_{base_size[1]}'
        self.image_dir = os.path.join(data_root, data_dir, config.class_name, 'images')
        self.label_dir = os.path.join(data_root, data_dir, config.class_name, 'labels')

        self.img_size = crop_size
        self.eval_mode = eval_mode
        self.resize = resize
        self.shot = config.shot
        self.support_idx = config.support_idx
        self.precision = config.precision

        assert split in ['train', 'valid', 'test']
        self.split = split

        self.toten = ToTensor()
        self.resizer = Resize(config.img_size)
        self.resize = resize
        self.randomflip = config.randomflip
        self.randomjitter = config.randomjitter
        self.randomblur = config.randomblur
        self.randomrotate = config.randomrotate

        if self.randomjitter:
            self.jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        if self.randomblur:
            self.blur = GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        if self.randomrotate:
            self.rotate = RandomRotation(30)

        if split == 'train':
            with open(os.path.join(data_root, 'meta_data', config.class_name, 'support.txt'), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            self.data_idxs = [f"{int(line.split('/')[-1].split('.')[0]):04d}.jpg" for line in lines]
            self.data_idxs = self.data_idxs[:self.shot]
        else:
            with open(os.path.join(data_root, 'meta_data', config.class_name, f'{split.replace("valid", "val")}.txt'), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            self.data_idxs = [f"{int(line.split('/')[-1].split('.')[0]):04d}.jpg" for line in lines]

            n_vis = 10
            if split == 'valid' and len(self.data_idxs) > n_vis:
                idxs = np.arange(len(self.data_idxs))
                if dset_size > 0:
                    idxs = idxs[::len(self.data_idxs)//dset_size][:dset_size]
                    assert len(idxs) == dset_size

                valid_idxs_reordered = []
                vis_idxs = torch.linspace(min(idxs), max(idxs), n_vis).round().long().tolist()
                vis_idxs = [min(idxs, key=lambda x:abs(x-vis_idx)) for vis_idx in vis_idxs]
                for i in vis_idxs:
                    valid_idxs_reordered.append(i)
                for i in idxs:
                    if i not in vis_idxs:
                        valid_idxs_reordered.append(i)
                self.data_idxs = [self.data_idxs[i] for i in valid_idxs_reordered]

        if dset_size < 0:
            self.dset_size = len(self.data_idxs)
        elif not eval_mode:
            self.dset_size = dset_size
        else:
            self.dset_size = min(dset_size, len(self.data_idxs))

    def load_data(self, img_path):
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        
        lbl_path = img_path.replace('.jpg', '.png')
        label = Image.open(os.path.join(self.label_dir, lbl_path))

        return image, label

    def postprocess_data(self, image, label):
        X = self.toten(image)
        Y = self.toten(label).bool().float()

        if self.resize:
            X = self.resizer(X)
            Y = self.resizer(Y)
        else:
            X, Y = crop_arrays(X, Y,
                               base_size=X.size()[-2:],
                               crop_size=self.img_size,
                               random=(not self.eval_mode))
            
        if not self.eval_mode:
            if self.randomjitter and random.random() > 0.5:
                X = self.jitter(X)
            if self.randomblur and random.random() > 0.5:
                X = self.blur(X)
            if self.randomrotate and random.random() > 0.5:
                XY = torch.cat((X, Y), dim=0)
                XY = self.rotate(XY)
                X, Y = XY[:3], XY[3:]

        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
        M = torch.ones_like(Y)
            
        return X, Y, M

    def __getitem__(self, idx):
        img_path = self.data_idxs[idx % len(self.data_idxs)]
        image, label = self.load_data(img_path)
        
        return self.postprocess_data(image, label)
        
    def __len__(self):
        return self.dset_size
