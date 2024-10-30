import os
import random
import numpy as np
from PIL import Image
import skimage
import math
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataset.augmentation import get_filtering_augmentation
from dataset.utils import crop_arrays, SobelEdgeDetector


def open_float16(image_path):
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    return img


def trajectory_from_path(path):
    return path.split('_')[-2]


class MidAir(Dataset):
    NAME = 'midair'

    TRAJECTORY_KITE_TRAIN = [f'{i}0{str(j).rjust(2, "0")}' for i in [0, 1, 2] for j in range(30)]
    TRAJECTORY_KITE_VAL = [f'{i}0{str(j).rjust(2, "0")}' for i in [3] for j in range(30)]
    TRAJECTORY_PLE_TRAIN =  [f'{i}0{str(j).rjust(2, "0")}' for i in [4, 5] for j in range(24)]
    TRAJECTORY_PLE_VAL =  [f'{i}0{str(j).rjust(2, "0")}' for i in [6] for j in range(24)]

    TRAJECTORY_KITE = TRAJECTORY_KITE_TRAIN + TRAJECTORY_KITE_VAL
    TRAJECTORY_PLE = TRAJECTORY_PLE_TRAIN + TRAJECTORY_PLE_VAL
    TRAJECTORY = TRAJECTORY_KITE + TRAJECTORY_PLE

    # Class Splits for Semantic Segmentation
    # CLASS_IDXS = [2, 4, 5, 6, 8] # 7 is empty, 1 is too small, 3, 9, 10, 11, 12, 13 are not in KITE environment
    CLASS_IDXS = [2, 3, 4, 5, 6, 8, 10, 11]
    CLASS_RANGE = range(1, 14)
    CLASS_NAMES = ['Animals', 'Trees', 'Dirt ground', 'Ground vegetation', 'Rocky ground',
                   'Boulders', 'none', 'Water plane', 'Man-made construction', 'Road', 'Train track',
                   'Road sign', 'Other man-made objects']
    CLASS_IDX_TO_NAME = {c: name for c, name in zip(CLASS_RANGE, CLASS_NAMES)}
    CLASS_IDXS_VAL = CLASS_IDXS

    # Task Groups
    TASK_GROUPS_BASE = []
    TASK_GROUPS_CONTINUOUS = []
    TASK_GROUPS_CATEGORICAL = []

    # Base Tasks
    TASKS_AUTOENCODING = [f'autoencoding_{c}' for c in range(3)]
    TASK_GROUPS_BASE.append(TASKS_AUTOENCODING)

    TASKS_DENOISING = [f'denoising_{c}' for c in range(3)]
    TASK_GROUPS_BASE.append(TASKS_DENOISING)

    TASKS_EDGE2D = [f'edge_texture_{c}' for c in range(3)]
    TASK_GROUPS_BASE.append(TASKS_EDGE2D)

    # Continuous Tasks
    TASKS_NORMAL = [f'normal_{c}' for c in range(3)]
    TASK_GROUPS_CONTINUOUS.append(TASKS_NORMAL)

    TASKS_DEPTH = [f'depth_0']
    TASK_GROUPS_CONTINUOUS.append(TASKS_DEPTH)

    # Categorical Tasks
    TASKS_SEMSEG = [f'segment_semantic_{c}' for c in CLASS_IDXS]
    TASK_GROUPS_CATEGORICAL.append(TASKS_SEMSEG)

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
        if task_group_name == 'segment_semantic':
            CHANNELS_DICT[task_group_name] = 1
        else:
            CHANNELS_DICT[task_group_name] = len(task_group)

    def __init__(self, path_dict, split, component='training', base_size=(256, 256), crop_size=(224, 224), 
                 seed=None, precision='fp32', meta_dir='meta_info'):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        self.data_root = os.path.join(path_dict[self.NAME], f'{component}_{base_size[0]}_merged')
        if split == 'train':
            self.domains = self.TRAJECTORY_KITE_TRAIN + self.TRAJECTORY_PLE_TRAIN
        else:
            self.domains = self.TRAJECTORY_KITE_VAL + self.TRAJECTORY_PLE_VAL
        
        self.base_size = base_size
        self.crop_size = crop_size
        self.precision = precision
        self.meta_info_path = os.path.join('dataset', meta_dir, self.NAME)
            
        # register euclidean depth and occlusion edge statistics, sobel edge detectors, and class dictionary
        self.edge_params = torch.load(os.path.join(self.meta_info_path, 'edge_params.pth'))
        self.sobel_detectors = [SobelEdgeDetector(kernel_size=k, sigma=s) for k, s in self.edge_params['params']]
        self.log_depth_range = torch.load(os.path.join(self.meta_info_path, 'midair_log_depth_range.pth'))
        self.class_dict = torch.load(os.path.join(self.meta_info_path, 'midair_class_dict.pth'))
        
    def _load_image(self, img_file):
        img_path = os.path.join(self.data_root, 'color_left', img_file)

        # open image file
        img = Image.open(img_path)
        img = np.asarray(img)
        
        # type conversion
        img = img.astype('float32') / 255
        
        # shape conversion
        img = np.transpose(img, (2, 0, 1))
        
        return img

    def _load_label(self, task_group, img_file):
        # get label path
        if task_group == 'segment_semantic':
            task_root = os.path.join(self.data_root, 'segmentation')
        elif task_group == 'depth':
            task_root = os.path.join(self.data_root, 'depth')
        elif task_group == 'normal':
            task_root = os.path.join(self.data_root, 'normals')
        else:
            raise NotImplementedError
        
        label_path = os.path.join(task_root, img_file.replace("JPEG", "PNG"))

        if task_group == 'depth':
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
        if task_group in self.TASK_GROUP_NAMES_BASE:
            label = mask = None
        else:
            # load label
            label = self._load_label(task_group, img_file)
                
            if task_group in self.TASK_GROUP_NAMES_CONTINUOUS:
                if task_group == 'depth':
                    np.clip(label, 1, 1250, label)
                    label = np.log(label) / np.log(1250)
                    label = label.astype('float32')

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
    
    def _postprocess_autoencoding(self, imgs, channels):
        labels = imgs.clone()
        masks = torch.ones_like(labels)
            
        labels = labels[:, channels]
        masks = masks[:, channels]

        return labels, masks

    def _postprocess_denoising(self, imgs, channels):
        labels = imgs.clone()
        masks = torch.ones_like(labels)
        imgs = torch.from_numpy(skimage.util.random_noise(imgs, var=0.01))
            
        labels = labels[:, channels]
        masks = masks[:, channels]

        return imgs, labels, masks
    
    def _postprocess_edge_texture(self, imgs, channels):
        labels = []
        # detect sobel edge with different set of pre-defined parameters
        for c in channels:
            labels_ = self.sobel_detectors[c].detect(imgs)
            labels.append(labels_)
        labels = torch.cat(labels, 1)

        # thresholding and re-normalizing
        labels = torch.clip(labels, 0, self.edge_params['threshold'])
        labels = labels / self.edge_params['threshold']

        masks = torch.ones_like(labels)
        
        return labels, masks
    
    def _postprocess_depth(self, labels):
        labels = torch.from_numpy(labels).float()
        masks = torch.ones_like(labels)
        
        return labels, masks
    
    def _postprocess_segment_semantic(self, labels, channels):
        # regard non-support classes as background
        for c in self.CLASS_RANGE:
            if c not in channels:
                labels = np.where(labels == c,
                                  np.zeros_like(labels),
                                  labels)

        # re-label support classes
        for i, c in enumerate(sorted(channels)):
            labels = np.where(labels == c,
                              (i + 1)*np.ones_like(labels),
                              labels)

        # one-hot encoding
        labels = torch.from_numpy(labels).long().squeeze(1)
        try:
            labels = F.one_hot(labels, len(channels) + 1).permute(0, 3, 1, 2).float()
        except:
            labels = labels.bool().long()
            labels = F.one_hot(labels, len(channels) + 1).permute(0, 3, 1, 2).float()
        if len(channels) == 1:
            labels = labels[:, 1:]
        masks = torch.ones_like(labels)
        
        return labels, masks
   
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
        if task == 'autoencoding':
            labels, masks = self._postprocess_autoencoding(imgs, channels)
        elif task == 'denoising':
            imgs, labels, masks = self._postprocess_denoising(imgs, channels)
        elif task == 'edge_texture':
            labels, masks = self._postprocess_edge_texture(imgs, channels)
        elif task == 'depth':
            labels, masks = self._postprocess_depth(labels)
        elif task == 'segment_semantic':
            labels, masks = self._postprocess_segment_semantic(labels, channels)
        else:
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


class MidAirBaseDataset(MidAir):
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


class MidAirBaseTrainDataset(MidAirBaseDataset):
    def __init__(self, label_augmentation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_augmentation = get_filtering_augmentation() if label_augmentation else None
        self.tasks = self.TASKS_BASE
        self.task_groups = self.TASK_GROUPS_BASE

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


class MidAirContinuousTrainDataset(MidAirBaseTrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = self.TASKS_CONTINUOUS
        self.task_groups = self.TASK_GROUPS_CONTINUOUS


class MidAirBaseTestDataset(MidAirBaseDataset):
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
    

class MidAirCategoricalDataset(MidAir):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load all image files
        img_file_path = os.path.join(self.meta_info_path, 'img_files.pth')
        if not os.path.exists(img_file_path):
            img_files = sorted(os.listdir(os.path.join(self.data_root, 'color_left')))
            torch.save(img_files, img_file_path)
        else:
            img_files = torch.load(img_file_path)
        self.img_files = img_files
            
        # permute per-class files
        idxs_perm_path = os.path.join(self.meta_info_path, 'idxs_perm_classes.pth')
        if not os.path.exists(idxs_perm_path):
            idxs_perm = {}
            for c in self.CLASS_IDXS:
                idxs_perm[c] = torch.randperm(len(self.class_dict[c]))
            torch.save(idxs_perm, idxs_perm_path)
        else:
            idxs_perm = torch.load(idxs_perm_path)

        self.class_dict = {
            c: [self.class_dict[c][idx] for idx in idxs_perm[c]
                if trajectory_from_path(self.img_files[self.class_dict[c][idx]]) in self.domains]
            for c in self.CLASS_IDXS
        }
        self.n_imgs = sum([len(v) for v in self.class_dict.values()])

        # # register per-class image files within the specified domains
        np_img_files = np.array(self.img_files)
        np_class_dict = {k: np.array(v) for k, v in self.class_dict.items()} # {class: idxs}
        domain_info =  {c: np.array(list(map(trajectory_from_path, np_img_files[np_class_dict[c]]))) for c in self.CLASS_IDXS} # {class: domains}
        self.domain_dict = {c:dict() for c in self.CLASS_IDXS}
        for c in self.CLASS_IDXS:
            for domain in self.domains:
                domain_mask = (domain_info[c] == domain)
                self.domain_dict[c][domain] = list(np_class_dict[c][domain_mask])

        # Remove tasks if there is no image with the specified class label.
        self.valid_classes = []
        for c in self.CLASS_IDXS:
            empty = True
            for d in self.domains:
                if len(self.domain_dict[c][d]) != 0:
                    empty = False
                    break
            if not empty:
                self.valid_classes.append(c)
        assert len(self.valid_classes) > 0, 'No valid classes found.'

    def __len__(self):
        return self.n_imgs


class MidAirCategoricalTrainDataset(MidAirCategoricalDataset):
    def __init__(self, label_augmentation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_augmentation = get_filtering_augmentation() if label_augmentation else None
        self.tasks = self.TASKS_CATEGORICAL
        self.task_groups = self.TASK_GROUPS_CATEGORICAL

    def sample_domains(self, c, n_domains, n_files):
        domains = [d for d in self.domains if len(self.domain_dict[c][d]) > n_files // n_domains]
        return np.random.choice(domains, n_domains, replace=False)
    
    def sample_files(self, c, domains, n_files):
        assert n_files % len(domains) == 0
        
        # sample image file indices for each domain
        file_idxs = np.array([], dtype=np.int64)
        for domain in domains:
            file_idxs = np.concatenate((file_idxs,
                                        np.random.choice(self.domain_dict[c][domain], 
                                                         n_files // len(domains), replace=False)))
        return file_idxs
    
    def sample_episode(self, n_files, n_channels=1, n_domains=1):
        # sample task group
        task_group_idx = np.random.choice(len(self.task_groups))
        task_group = self.task_groups[task_group_idx]
        task_group_name = '_'.join(task_group[0].split('_')[:-1])
        g_idx = torch.tensor(self.TASK_GROUP_NAMES.index(task_group_name))

        # decide on number of chunks and create channel mask
        chunk_size = 1
        chunks = torch.ones(n_channels).split(chunk_size, dim=0)
        channel_mask = torch.block_diag(*[chunk[:, None] @ chunk[None, :] for chunk in chunks]).bool()

        X = []
        Y = []
        M = []
        t_idx = []
        c_idx = []

        # sample images and labels for each chunk
        for chunk in chunks:
            # sample channels
            task_idxs = np.random.choice(len(task_group), len(chunk), replace=False)
            tasks = [task_group[t] for t in task_idxs]
            t_idx_ = torch.tensor([self.TASKS.index(task) for task in tasks])
            channels = [int(task.split('_')[-1]) for task in tasks]

            # sample domains
            domains = self.sample_domains(channels[0], n_domains, n_files)

            # sample image paths
            file_idxs = self.sample_files(channels[0], domains, n_files)

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


class MidAirCategoricalTestDataset(MidAirCategoricalDataset):
    def __init__(self, task_group, class_id, dset_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_id = class_id
        self.dset_size = dset_size
        self.task_group = task_group

    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.class_dict[self.class_id])
        
    def __getitem__(self, idx):
        idx_ = self.class_dict[self.class_id][idx]
        X = self.load_images([idx_])
        X, Y, M = self.load_labels(X, self.task_group, [self.class_id], [idx_])

        return X[0], Y[0], M[0]
