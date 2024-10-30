import os
import random
import numpy as np
from PIL import Image
import math
from einops import rearrange, repeat

import torch
from torch.utils.data import Dataset
from dataset.augmentation import get_filtering_augmentation
from dataset.utils import crop_arrays


def open_float16(image_path):
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    return img


def trajectory_from_path(path):
    return path.split('_')[-2]


class MidAirStereo(Dataset):
    NAME = 'midair_stereo'

    TRAJECTORY_KITE_TRAIN = [f'{i}0{str(j).rjust(2, "0")}' for i in [0, 1, 2] for j in range(30)]
    TRAJECTORY_KITE_VAL = [f'{i}0{str(j).rjust(2, "0")}' for i in [3] for j in range(30)]
    TRAJECTORY_PLE_TRAIN =  [f'{i}0{str(j).rjust(2, "0")}' for i in [4, 5] for j in range(24)]
    TRAJECTORY_PLE_VAL =  [f'{i}0{str(j).rjust(2, "0")}' for i in [6] for j in range(24)]

    TRAJECTORY_KITE = TRAJECTORY_KITE_TRAIN + TRAJECTORY_KITE_VAL
    TRAJECTORY_PLE = TRAJECTORY_PLE_TRAIN + TRAJECTORY_PLE_VAL
    TRAJECTORY = TRAJECTORY_KITE + TRAJECTORY_PLE

    # Task Groups
    TASK_GROUPS_BASE = []
    TASK_GROUPS_CONTINUOUS = []
    TASK_GROUPS_CATEGORICAL = []

    # Continuous Tasks
    TASKS_NORMAL = [f'stereo_normal_{c}' for c in range(3)]
    TASK_GROUPS_CONTINUOUS.append(TASKS_NORMAL)

    TASKS_DEPTH = [f'stereo_depth_0']
    TASK_GROUPS_CONTINUOUS.append(TASKS_DEPTH)

    # Task Group Names
    TASK_GROUP_NAMES_BASE = ['_'.join(task_group[0].split('_')[:-1])
                             for task_group in TASK_GROUPS_BASE]
    TASK_GROUP_NAMES_CONTINUOUS = ['_'.join(task_group[0].split('_')[:-1])
                                   for task_group in TASK_GROUPS_CONTINUOUS]
    TASK_GROUP_NAMES_CATEGORICAL = ['_'.join(task_group[0].split('_')[:-1])
                                    for task_group in TASK_GROUPS_CATEGORICAL]
    TASK_GROUP_NAMES = TASK_GROUP_NAMES_BASE + TASK_GROUP_NAMES_CONTINUOUS + TASK_GROUP_NAMES_CATEGORICAL

    # All Task Groups, Task Group Dict, and Tasks
    TASK_GROUPS = TASK_GROUPS_BASE + TASK_GROUPS_CONTINUOUS + TASK_GROUPS_CATEGORICAL
    TASK_GROUP_DICT = {name: group for name, group in zip(TASK_GROUP_NAMES, TASK_GROUPS)}

    TASKS_BASE = [task for task_group in TASK_GROUPS_BASE for task in task_group]
    TASKS_CONTINUOUS = [task for task_group in TASK_GROUPS_CONTINUOUS for task in task_group]
    TASKS_CATEGORICAL = [task for task_group in TASK_GROUPS_CATEGORICAL for task in task_group]
    TASKS = TASKS_BASE + TASKS_CONTINUOUS + TASKS_CATEGORICAL

    # Channels Dictionary
    CHANNELS_DICT = {}
    for task_group_name, task_group in TASK_GROUP_DICT.items():
        CHANNELS_DICT[task_group_name] = len(task_group)

    def __init__(self, path_dict, split, component='training', base_size=(256, 256), crop_size=(224, 224), 
                 seed=None, precision='fp32', meta_dir='meta_info'):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        self.data_root = os.path.join(path_dict['midair'], f'{component}_{base_size[0]}_merged')
        if split == 'train':
            self.domains = self.TRAJECTORY_KITE_TRAIN + self.TRAJECTORY_PLE_TRAIN
        else:
            self.domains = self.TRAJECTORY_KITE_VAL + self.TRAJECTORY_PLE_VAL
        
        self.base_size = base_size
        self.crop_size = crop_size
        self.precision = precision
        self.meta_info_path = os.path.join('dataset', meta_dir, 'midair')
            
        # register euclidean depth and disparity statistics
        self.log_depth_range = torch.load(os.path.join(self.meta_info_path, 'midair_log_depth_range.pth'))
        self.log_disparity_range = torch.load(os.path.join(self.meta_info_path, 'midair_log_disparity_range.pth'))
        
    def _load_image(self, img_file):
        img_path1 = os.path.join(self.data_root, 'color_left', img_file)
        img_path2 = os.path.join(self.data_root, 'color_right', img_file)

        # open image files
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        img1 = np.asarray(img1)
        img2 = np.asarray(img2)
        
        # type conversion
        img1 = img1.astype('float32') / 255
        img2 = img2.astype('float32') / 255
        
        # shape conversion
        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))

        # concatenate images
        img = np.concatenate((img1, img2), axis=0)
        
        return img

    def _load_label(self, task_group, img_file):
        # get label path
        if task_group == 'stereo_depth':
            task_root = os.path.join(self.data_root, 'depth')
        elif task_group == 'stereo_normal':
            task_root = os.path.join(self.data_root, 'normals')
        elif task_group == 'stereo_disparity':
            task_root = os.path.join(self.data_root, 'stereo_disparity')
        elif task_group == 'stereo_occlusion':
            task_root = os.path.join(self.data_root, 'stereo_occlusion')
        else:
            raise NotImplementedError
        
        label_path = os.path.join(task_root, img_file.replace("JPEG", "PNG"))

        if task_group in ['stereo_depth', 'stereo_disparity']:
            # treat this specially because of float16 dtype
            label = open_float16(label_path)
        else:
            # open label file
            label = Image.open(label_path)
            label = np.asarray(label)
        
        # type conversion
        if label.dtype == 'uint8':
            label = label.astype('float32') / 255
        else:
            label = label.astype('float32')
            
        # shape conversion
        if label.ndim == 2:
            label = label[np.newaxis, ...]
        elif label.ndim == 3:
            label = np.transpose(label, (2, 0, 1))
        
        return label
        
    def _load_and_preprocess_label(self, task_group, img_file):
        # load label
        label = self._load_label(task_group, img_file)
            
        if task_group in self.TASK_GROUP_NAMES_CONTINUOUS:
            if task_group in ['stereo_depth', 'stereo_disparity']:
                np.clip(label, 1, 1250, label)
                label = np.log(label) / np.log(1250)
                label = label.astype('float32')
            # if task_group == 'stereo_depth':
            #         label = np.log((1 + label))
            #         log_min = self.log_depth_range[0].item()
            #         log_max = self.log_depth_range[1].item()
            #         label = (label - log_min) / (log_max - log_min)
            # elif task_group == 'stereo_disparity':
            #         label = np.log((1 + label))
            #         log_min = self.log_disparity_range[0].item()
            #         log_max = self.log_disparity_range[1].item()
            #         label = (label - log_min) / (log_max - log_min)

            # bound label values
            label = np.clip(label, 0, 1)
            mask = None
    
        # preprocess label
        elif task_group in self.TASK_GROUP_NAMES_CATEGORICAL:
            label = (255*label).astype("long")
            mask = None

        else:
            raise ValueError(f'Invalid task group: {task_group}')
            
        return label, mask
   
    def _postprocess_default(self, labels, masks, channels):
        labels = torch.from_numpy(labels).float()
        labels = labels[:, channels]

        if masks is not None:
            masks = torch.from_numpy(masks).float().expand_as(labels)
        else:
            masks = torch.ones_like(labels)
            
        return labels, masks

    def _postprocess(self, imgs, labels, masks, channels, task):
        # process all channels if not given
        if channels is None:
            if task == 'segment_semantic':
                channels = self.CLASS_IDXS
            else:
                channels = range(len(self.TASK_GROUP_DICT[task]))
            
        # task-specific preprocessing
        labels, masks = self._postprocess_default(labels, masks, channels)

        # ensure label values to be in [0, 1]
        labels = labels.clip(0, 1)

        return imgs, labels, masks

    def load_images(self, file_idxs):
        # load images
        imgs = []
        for file_idx in file_idxs:
            # index image path
            img_file = self.img_files[file_idx]

            # load image
            img = self._load_image(img_file)
            imgs.append(img)
        imgs = np.stack(imgs)

        # convert to torch tensor
        imgs = torch.from_numpy(imgs)

        return imgs
    
    def load_labels(self, imgs, task, channels, file_idxs):
        # load labels
        labels = []
        masks = []
        for file_idx in file_idxs:
            # index image path
            img_file = self.img_files[file_idx]

            # load label
            label, mask = self._load_and_preprocess_label(task, img_file)
            labels.append(label)
            masks.append(mask)
        labels = np.stack(labels) if labels[0] is not None else None
        masks = np.stack(masks) if masks[0] is not None else None

        # postprocess labels
        imgs, labels, masks = self._postprocess(imgs, labels, masks, channels, task)
        
        # precision conversion
        if self.precision == 'fp16':
            imgs = imgs.half()
            labels = labels.half()
            masks = masks.half()
        elif self.precision == 'bf16':
            imgs = imgs.bfloat16()
            labels = labels.bfloat16()
            masks = masks.bfloat16()

        return imgs, labels, masks

    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError


class MidAirStereoDataset(MidAirStereo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load all image files
        img_file_path = os.path.join(self.meta_info_path, 'img_files.pth')
        if not os.path.exists(img_file_path):
            img_files = sorted(os.listdir(os.path.join(self.data_root, 'color_left')))
            torch.save(img_files, img_file_path)
        else:
            img_files = torch.load(img_file_path)

        # permute image files
        idxs_perm_path = os.path.join(self.meta_info_path, 'idxs_perm_all.pth')
        if not os.path.exists(idxs_perm_path):
            idxs_perm = torch.randperm(len(img_files))
            torch.save(idxs_perm, idxs_perm_path)
        else:
            idxs_perm = torch.load(idxs_perm_path)
        img_files = [img_files[idx] for idx in idxs_perm]

        # register image files within the specified domains
        self.img_files = [img_file for img_file in img_files
                          if trajectory_from_path(img_file) in self.domains]
        self.domain_dict = {
            domain: [idx for idx, img_file in enumerate(self.img_files)
                     if trajectory_from_path(img_file) == domain]
            for domain in self.domains
        }
        
    def __len__(self):
        return len(self.img_files)


class MidAirStereoTrainDataset(MidAirStereoDataset):
    def __init__(self, label_augmentation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_augmentation = get_filtering_augmentation() if label_augmentation else None

    def __len__(self):
        return len(self.img_files)

    def sample_domains(self, n_domains):
        return np.random.choice(self.domains, n_domains, replace=False)
    
    def sample_files(self, domains, n_files):
        assert n_files % len(domains) == 0

        # sample image file indices for each domain
        file_idxs = np.array([], dtype=np.int64)
        for domain in domains:
            file_idxs = np.concatenate((file_idxs,
                                        np.random.choice(self.domain_dict[domain], 
                                                         n_files // len(domains), replace=False)))
        return file_idxs
    
    def sample_episode(self, n_files, n_channels=1, n_domains=1):
        # sample task group
        task_group_idx = np.random.choice(len(self.task_groups))
        task_group = self.task_groups[task_group_idx]
        task_group_name = '_'.join(task_group[0].split('_')[:-1])
        g_idx = torch.tensor(self.TASK_GROUP_NAMES.index(task_group_name))

        # decide on number of chunks and create channel mask
        max_channels = min(len(task_group), n_channels)
        chunk_size = random.randint(math.ceil(max_channels * 1.0 / 2), max_channels)
        chunks = torch.ones(n_channels).split(chunk_size, dim=0)
        channel_mask = torch.block_diag(*[chunk[:, None] @ chunk[None, :] for chunk in chunks]).bool()

        X = []
        Y = []
        M = []
        t_idx = []

        # sample images and labels for each chunk
        for chunk in chunks:
            # sample channels
            task_idxs = np.random.choice(len(task_group), len(chunk), replace=False)
            tasks = [task_group[t] for t in task_idxs]
            t_idx_ = torch.tensor([self.TASKS.index(task) for task in tasks])
            channels = [int(task.split('_')[-1]) for task in tasks]

            # sample domains
            domains = self.sample_domains(n_domains)

            # sample image paths
            file_idxs = self.sample_files(domains, n_files)

            # load images and labels
            X_ = self.load_images(file_idxs)
            X_, Y_, M_ = self.load_labels(X_, task_group_name, channels, file_idxs)

            # apply label task augmentation
            if self.label_augmentation:
                Y_, M_ = self.label_augmentation(Y_, M_)

            # expand channel dimension
            X_ = repeat(X_, 'N D H W -> C N D H W', C=len(chunk))
            Y_ = rearrange(Y_, 'N C H W -> C N 1 H W')
            M_ = rearrange(M_, 'N C H W -> C N 1 H W')

            X.append(X_)
            Y.append(Y_)
            M.append(M_)
            t_idx.append(t_idx_)
        
        # concatenate chunks
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        M = torch.cat(M, dim=0)
        t_idx = torch.cat(t_idx, dim=0)

        # random-crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              crop_size=self.crop_size,
                              random=True)
        
        return X, Y, M, t_idx, g_idx, channel_mask


class MidAirStereoTestDataset(MidAirStereoDataset):
    def __init__(self, task_group, dset_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_group = task_group
        self.dset_size = dset_size

    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.img_files)
        
    def __getitem__(self, idx):
        X = self.load_images([idx])
        X, Y, M = self.load_labels(X, self.task_group, None, [idx])

        return X[0], Y[0], M[0]


class MidAirStereoContinuousTrainDataset(MidAirStereoTrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = self.TASKS_CONTINUOUS
        self.task_groups = self.TASK_GROUPS_CONTINUOUS


class MidAirStereoCategoricalTrainDataset(MidAirStereoTrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = self.TASKS_CATEGORICAL
        self.task_groups = self.TASK_GROUPS_CATEGORICAL