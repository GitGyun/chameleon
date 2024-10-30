import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset
from .utils import crop_arrays, SobelEdgeDetector
from .augmentation import RandomHorizontalFlip, FILTERING_AUGMENTATIONS, RandomCompose, Mixup, CustomTrivialAugmentWide
from .unified_constants import TASKS_GROUP_DICT


class BaseDataset(Dataset):
    def __init__(self, root_dir, domains, tasks, component, base_size=(256, 256),
                 img_size=(224, 224), seed=None, precision='fp32', meta_dir='meta_info'):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        self.data_root = os.path.join(root_dir, f'{component}_{base_size[0]}_merged')
        self.domains = sorted(domains)
        
        self.subtasks = tasks
                
        self.base_size = base_size
        self.img_size = img_size
        self.precision = precision

        self.meta_info_path = os.path.join('dataset', meta_dir)
        self.edge_params = torch.load(os.path.join(self.meta_info_path, 'edge_params.pth'))
        self.sobel_detectors = [SobelEdgeDetector(kernel_size=k, sigma=s) for k, s in self.edge_params['params']]

    def load_image(self, img_path):
        raise NotImplementedError
    
    def load_task(self, task, img_path):
        raise NotImplementedError
    
    def preprocess_batch(self, task, imgs, labels, masks, channels=None, drop_background=True):
        raise NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self, root_dir, domains, tasks, shot, tasks_per_batch, domains_per_batch,
                 image_augmentation, unary_augmentation, binary_augmentation,
                 dset_size=-1, **kwargs):
        super().__init__(root_dir, domains, tasks, **kwargs)
        
        assert shot > 0
        self.shot = shot
        self.tasks_per_batch = tasks_per_batch
        self.domains_per_batch = min(len(domains)//2, domains_per_batch)
        self.dset_size = dset_size
        
        if image_augmentation:
            self.image_augmentation = RandomHorizontalFlip()
        else:
            self.image_augmentation = None
        
        if unary_augmentation:
            self.unary_augmentation = RandomCompose(
                [augmentation(**kwargs) for augmentation, kwargs in FILTERING_AUGMENTATIONS.values()],
                p=0.8,
            )
        else:
            self.unary_augmentation = None

        if binary_augmentation is not None:
            self.binary_augmentation = Mixup(order=True)
        else:
            self.binary_augmentation = None

    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.img_paths) // self.shot
        
    def __getitem__(self, idx):
        return self.sample(idx, self.tasks_per_batch)
    
    def sample(self, idx, n_channels):
        raise NotImplementedError
    

class ContinuousDataset(BaseDataset):
    def __init__(self, root_dir, domains, task, channel_idx=-1, dset_size=-1, **kwargs):
        super().__init__(root_dir, domains, [task], **kwargs)
        
        self.task = task
        self.channel_idx = channel_idx
        self.dset_size = dset_size
        self.n_channels = len(TASKS_GROUP_DICT[task])
    
    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx % len(self.img_paths)]

        # load image, label, and mask
        img, success = self.load_img(img_path)
        label, mask = self.load_task(self.task, img_path)
        if not success:
            mask = np.zeros_like(label)

        # preprocess labels
        imgs, labels, masks = self.preprocess_batch(self.task,
                                                    img[None],
                                                    None if label is None else label[None],
                                                    None if mask is None else mask[None],
                                                    channels=([self.channel_idx] if self.channel_idx >= 0 else None),
                                                    drop_background=False)
        
        
        X, Y, M = imgs[0], labels[0], masks[0]
        if self.image_augmentation is not None:
            X, Y, M = self.image_augmentation(X, Y, M)
        
        # crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              crop_size=self.img_size,
                              random=True)
        
        return X, Y, M

    
class SegmentationDataset(BaseDataset):
    def __init__(self, root_dir, domains, semseg_class=-1, dset_size=-1, **kwargs):
        super().__init__(root_dir, domains, ['segment_semantic'], **kwargs)

        self.semseg_class = semseg_class
        self.img_paths = torch.load(os.path.join(self.meta_info_path, 'img_files.pth')) # use global path dictionary
        self.n_channels = 1
        self.dset_size = dset_size

    def generate_class_idxs(self):
        raise NotImplementedError
    
    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.class_idxs)
            
    def __getitem__(self, idx):
        path_idx = self.class_idxs[idx % len(self.class_idxs)]
        img_path = self.img_paths[path_idx]

        # load image, label, and mask
        img, success = self.load_img(img_path)
        label, mask = self.load_task('segment_semantic', img_path)
        if not success:
            mask = np.zeros_like(mask)

        # preprocess labels
        imgs, labels, masks = self.preprocess_batch('segment_semantic',
                                                    img[None],
                                                    None if label is None else label[None],
                                                    None if mask is None else mask[None],
                                                    [self.semseg_class])
        
        X, Y, M = imgs[0], labels[0], masks[0]
        
        # crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              crop_size=self.img_size,
                              random=True)
        
        return X, Y, M
