import json
import os
from PIL import Image
import numpy as np
from itertools import zip_longest
import random
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, ColorJitter

from dataset.utils import crop_arrays


class FSC147Dataset(Dataset):
    '''
    fsc147 dataset
    '''
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        super().__init__()
        if split == 'valid':
            split = 'val'
        assert split in ['train', 'val', 'test']

        self.base_size = base_size
        self.img_size = crop_size
        data_root = config.path_dict[config.dataset]
        self.eval_mode = eval_mode
        self.image_dir = os.path.join(data_root, f'images_384_VarV2')
        self.ann_dir = os.path.join(data_root, f'gt_density_map_adaptive_{self.img_size[0]}_VarV2')
        if eval_mode:
            self.ann_dir = os.path.join(data_root, f'gt_density_map_adaptive_384_VarV2')
        meta_file = os.path.join(data_root, f'annotation_FSC147_384.json')
        self.meta_file = json.load(open(meta_file, 'r'))
        split_file = os.path.join(data_root, f'Train_Test_Val_FSC_147.json')
        self.split_file = json.load(open(split_file, 'r'))
        self.image_files = self.split_file[split]

        self.cnt = 200
        self.resize = resize
        self.shot = config.shot
        self.support_idx = config.support_idx
        self.precision = config.precision

        self.data_root = data_root

        self.image2classname = { }
        self.classname2image = { }
        with open(os.path.join(data_root, f'ImageClasses_FSC147.txt'), 'r') as f:
            while True: 
                line = f.readline()
                if len(line) == 0:
                    break
                file_name, class_name = line.strip().split("\t")
                if file_name not in self.image_files:
                    continue
                
                self.image2classname[file_name] = class_name
                if class_name not in self.classname2image:
                    self.classname2image[class_name] = []
                self.classname2image[class_name].append(file_name)

        self.image_files_shuffle = []
        for bundle in zip_longest(*self.classname2image.values(), fillvalue="none"):
            self.image_files_shuffle += list(bundle) # this will cut-off the images, based on the species with least images.
        self.image_files_shuffle = [image_file for image_file in self.image_files_shuffle if image_file != "none"]

        self.toten = ToTensor()
        self.base_resizer = Resize(base_size)
        self.resizer = Resize(self.img_size)
        self.resize = resize
        
        self.randomflip = config.randomflip
        self.randomjitter = config.randomjitter
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

        if dset_size < 0:
            self.dset_size = len(self.image_files_shuffle)
        elif not eval_mode:
            self.dset_size = dset_size
        else:
            self.dset_size = min(dset_size, len(self.image_files_shuffle))   

        if split == "train":
            self.image_files_shuffle = self.image_files_shuffle[self.shot*self.support_idx:self.shot*(self.support_idx + 1)] 
        elif split == "val":
            self.image_files_shuffle = self.image_files_shuffle[:self.dset_size]

    def __len__(self):
        return self.dset_size
    
    def _load_data(self, image_path, ann_path):
        image = Image.open(image_path).convert('RGB')
        X = self.toten(image)
        if ann_path is None:
            label = None
            Y = None
        else:
            label = np.load(ann_path)[None, ...]
            # base resizer ? 
            Y = torch.from_numpy(label)
        
        _, H, W = X.shape
        
        X = self.base_resizer(X)
        if Y is not None:
            if not self.eval_mode:
                Y = self.base_resizer(Y)
                Y = Y * self.img_size[0] ** 2 / (Y.size(1) * Y.size(2))
            else:
                Y_sum = Y.sum((1, 2))
                Y = self.base_resizer(Y)
                Y = Y * Y_sum / Y.sum((1,2))

        return X, Y, (H, W) 

    def _flip(self, X, Y=None):
        X = torch.flip(X, dims=[-1])
        if Y is not None:
            Y = torch.flip(Y, dims=[-1])
        else:
            Y = None
        return X, Y
    
    def generate_cutmix_bbox(self, X, image_size, meta, aux):
        original_X = X.clone()
        bbox_coordinates = meta["box_examples_coordinates"]
        
        # normalize bbox coordinates to [0, 1]
        bbox_coordinates = torch.tensor([(x1, y1, x2, y2) for (x1, y1), _, (x2, y2), _ in bbox_coordinates]).float()
        bbox_coordinates /= torch.tensor([image_size[1], image_size[0], image_size[1], image_size[0]]).unsqueeze(0).float()
        if self.eval_mode:
            bbox_coordinates = bbox_coordinates[:3] # 3-shot setting
            
        aux["bbox_coords"] = bbox_coordinates.clone()[:3] # used in eval
        
        bbox_coordinates = (bbox_coordinates * self.base_size[0]).long() # denormalize to base_size
        
        # save exemplar objects and their sizes
        h, w = X.size(1), X.size(2) 
        objects = []
        object_sizes = [ ]
        cnt = 0
        for x1, y1, x2, y2 in bbox_coordinates:
            oh = y2 - y1
            ow = x2 - x1
            objects += [original_X[:, y1:y1+oh, x1:x1+ow]] 
            object_sizes += [(oh, ow)]
        
        # set object scale range for pasting
        if self.eval_mode:
            scale_min = scale_max = 1
        else:
            scale_min = 0.75
            scale_max = 1.25

        # adjust scale range for objects that are too big
        scale_dict = {}
        for i, object_size in enumerate(object_sizes):
            oh, ow = object_size
            if oh / h  > 0.95 or ow / w > 0.95:
                scale_dict[i] = (scale_min, 1)
            else:
                scale_dict[i] = (scale_min, min(scale_max, h/object_size[0].item(), w/object_size[1].item()))

        # set mask region (has value 1 if unavailable region)
        mask_region = torch.zeros(len(objects), h, w)
        for o, (oh, ow) in enumerate(object_sizes):
            mask_region[o, :-math.floor(oh.item() * scale_dict[o][1]),  :-math.floor(ow.item() * scale_dict[o][1])] = 1
            mask_region[o, 0, 0] = 1
        
        # make grid for sampling positions to be pasted
        grid = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'), dim=-1)

        # object-pasted image
        drawing = torch.zeros_like(original_X)
        
        # paste objects
        while mask_region.sum() > 0:
            for o, ((oh, ow), obj) in enumerate(zip(object_sizes, objects)):
                # if no more region to paste, skip
                if mask_region[o].sum() == 0:
                    continue
                
                # sample position (left corner) to paste
                if self.eval_mode:
                    idx = 0
                else:
                    idx = np.random.choice(len(grid[mask_region[o] == 1]))
                cur_y, cur_x = grid[mask_region[o] == 1][idx]

                # sample object scale
                scale_min, scale_max = scale_dict[o]
                if scale_min == scale_max:
                    scale = scale_min
                else:
                    scale = np.random.uniform(scale_min, scale_max)
                
                # scale object
                target_size = (target_size_h, target_size_w) = (math.floor(oh.item() * scale), math.floor(ow.item() * scale))
                target_size = (target_size_h, target_size_w) = min(target_size[0], h), min(target_size[1], w)
                obj = F.interpolate(obj[None], target_size, mode="bilinear", align_corners=False)

                # paste object
                drawing[:, cur_y: cur_y + target_size_h, cur_x: cur_x + target_size_w] = obj
                
                # update mask region
                for o_update, (oh_update, ow_update) in enumerate(object_sizes):
                    min_y = max(0, cur_y - math.floor(oh_update.item() * scale_max))
                    min_x = max(0, cur_x - math.floor(ow_update.item() * scale_max))
                    max_y = min(h, cur_y + target_size_h)
                    max_x = min(w, cur_x + target_size_w)
                    
                    if self.eval_mode:
                        margin_y = h // 50
                        margin_x = w // 50
                        min_y = max(0, min_y - margin_y)
                        min_x = max(0, min_x - margin_x)
                        max_y = min(h, max_y + margin_y)
                        max_x = min(w, max_x + margin_x)
                    
                    mask_region[o_update, min_y:max_y, min_x:max_x] = 0
                cnt += 1

            # if max number of objects is reached, stop
            if not self.eval_mode and cnt > self.cnt:
                break

        return drawing

    def postprocess_data(self, X, Y, image_size, meta=None):
        aux = {}
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
        if self.randomjitter and not self.eval_mode and random.random() > 0.5:
            X = self.jitter(X)
        if self.randomflip and not self.eval_mode and random.random() > 0.5:
            X, Y = self._flip(X, Y) 
        
        X2 = self.generate_cutmix_bbox(X, image_size, meta, aux)
        X = torch.cat([X, X2], dim=0)
 
        if self.resize:
            X = self.resizer(X)
            # adjust density to be summed to number of objects
            Y_sum = Y.sum((1, 2))
            Y = self.resizer(Y) 
            Y = Y * Y_sum / Y.sum((1, 2))
        else:
            X, Y = crop_arrays(X, Y,
                               base_size=X.size()[-2:],
                               crop_size=self.img_size,
                               random=(not self.eval_mode))
        
        aux["mode_value"] = Y.max()
        if Y.max() > 0:
            Y = Y / Y.max()
        
        M = torch.ones_like(Y)

        if self.eval_mode:
            return X, Y, M, (aux["mode_value"], aux["bbox_coords"])
        else:
            return X, Y, M

    def __getitem__(self, idx):
        image_file = self.image_files_shuffle[idx % len(self.image_files_shuffle)]
        ann_file = image_file.replace("jpg", "npy")
        
        image_path = os.path.join(self.image_dir, image_file)
        ann_path = os.path.join(self.ann_dir, ann_file)
        
        meta = self.meta_file[image_file]
        image, label, image_size = self._load_data(image_path, ann_path)
        
        return self.postprocess_data(image, label, image_size, meta)
    