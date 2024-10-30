import os
import random
import numpy as np
from PIL import Image
import skimage
import scipy
import scipy.ndimage
from einops import reduce, repeat, rearrange
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from dataset.augmentation import get_filtering_augmentation
from dataset.utils import SobelEdgeDetector, crop_arrays
from dataset.coco_api_wrapper import SingletonCOCOFactory


class DeepFashion(Dataset):
    NAME = 'deepfashion'

    KP_CLASSES = ["right_neck", "left_neck", "right_arm", "left_arm",
                  "right_waist", "left_waist", "right_leg", "left_leg"]
    KP_IDXS = list(range(1, 9)) # 1-8
    KP_IDX_TO_NAME = {c: name for c, name in zip(KP_IDXS, KP_CLASSES)}

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
    TASKS_KP_HUMAN = ['keypoints_human_0'] # Original single channel, Gaussian label.
    TASK_GROUPS_CONTINUOUS.append(TASKS_KP_HUMAN)

    # Categorical Tasks
    TASKS_KP_SEMANTIC = [f'keypoints_semantic_{c}' for c in KP_IDXS]
    TASK_GROUPS_CATEGORICAL.append(TASKS_KP_SEMANTIC)

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

    def __init__(self, path_dict, split, base_size=(256, 256), crop_size=(224, 224), 
                 seed=None, precision='fp32', meta_dir='meta_info'):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        self.base_size = base_size
        self.crop_size = crop_size
        self.precision = precision
        self.meta_info_path = os.path.join('dataset', meta_dir, self.NAME)
        
        if split == 'valid':
            split = 'val'
        assert split in ['train', 'val'], split  # TODO: add train+val
        self.img_root = os.path.join(path_dict[self.NAME], f'img_resized_{self.base_size[0]}')
        self.ann_root = os.path.join(path_dict[self.NAME], 'annotations')
        self.split = split
            
        # register sobel edge detectors
        self.edge_params = torch.load(os.path.join(self.meta_info_path, 'edge_params.pth'))
        self.sobel_detectors = [SobelEdgeDetector(kernel_size=k, sigma=s) for k, s in self.edge_params['params']]

        # transforms
        self.toten = ToTensor()

    def _resize(self, target, size=None, mode='bilinear'):
        '''
        Resize image or label to given size(default to self.base_size with bilinear).
        Dimension is assumed to be 3D (C, H, W)
        '''
        if size is None:
            size = self.base_size
        assert mode in ['bilinear', 'nearest']
        target = F.interpolate(target.unsqueeze(0), size=size, mode=mode).squeeze(0)
        return target

    def _load_image(self, imgId):
        '''
        get image idx and return tensor
        '''
        # get path
        file_name = self.coco.imgs[imgId]['file_name']
        img_path = file_name.replace('img/', f'{self.img_root}/')
        
        # open image file
        img = Image.open(img_path)
        img = self.toten(img)
        if len(img) == 1:
            img = img.repeat(3, 1, 1)
        
        return img

    def _load_label(self, task_group, imgId, class_id=None):
        '''
        return torch Tensor
        '''
        keypoints = self._load_keypoint(imgId)
        if class_id is None:
            class_id = list(range(len(self.KP_IDXS)))
        else:
            class_id = [c - 1 for c in class_id]

        label = torch.zeros(len(self.KP_IDXS), *self.base_size)
        for c, (w, h, v) in enumerate(keypoints):
            if v == 1:
                slice_h = slice(max(0, h-self.kp_radius), min(self.base_size[0], h+self.kp_radius+1))
                slice_w = slice(max(0, w-self.kp_radius), min(self.base_size[1], w+self.kp_radius+1))
                try:
                    label[c, slice_h, slice_w] = self.gaussian[max(0, self.kp_radius - h):max(0, self.kp_radius - h)+slice_h.stop-slice_h.start,
                                                                max(0, self.kp_radius - w):max(0, self.kp_radius - w)+slice_w.stop-slice_w.start]
                except:
                    print(imgId, c)

        if task_group == 'keypoints_human':
            label = reduce(label, 'c h w -> 1 h w', 'max')
            mask = None
        elif class_id is not None:
            label = label[class_id]
            mask = repeat((keypoints[class_id, 2] == 1).float(), 'c -> c h w', h=self.base_size[0], w=self.base_size[1])
        
        return label, mask
    
    def _load_keypoint(self, imgId):
        annId = self.coco.getAnnIds(imgId)[0]
        keypoints = torch.tensor(self.coco.anns[annId]['keypoints']).reshape(-1, 3)

        # move w, h coordinates, because of resizing
        height, width = self.coco.imgs[imgId]['height'], self.coco.imgs[imgId]['width']
        keypoints[..., 0] = keypoints[..., 0] * self.base_size[1] / width
        keypoints[..., 1] = keypoints[..., 1] * self.base_size[0] / height

        keypoints = keypoints.long()

        keypoints[:, 2] = torch.where(keypoints[:, :2].sum(dim=1) > 0,
                                      keypoints[:, 2],
                                      torch.zeros_like(keypoints[:, 2]))

        return keypoints
            
    def _load_and_preprocess_label(self, task_group, imgId, channels):
        # load label
        if task_group in self.TASK_GROUP_NAMES_BASE:
            label = mask = None
        else:
            if task_group in self.TASK_GROUP_NAMES_CONTINUOUS:
                # load label
                label, mask = self._load_label(task_group, imgId)

                # bound label values
                label = torch.clip(label, 0, 1)
        
            # preprocess label
            elif task_group in self.TASK_GROUP_NAMES_CATEGORICAL:
                # load label
                label, mask = self._load_label(task_group, imgId, class_id=channels)

            else:
                raise ValueError(f'Invalid task group: {task_group}')
            
        return label, mask
    
    def _postprocess_keypoints_human(self, labels, channels):
        '''
        labels: N C(=1) H W
        '''
        return labels, torch.ones_like(labels)

    def _postprocess_autoencoding(self, imgs, channels):
        labels = imgs.clone()
        labels = labels[:, channels]
        masks = torch.ones_like(labels)

        return labels, masks

    def _postprocess_denoising(self, imgs, channels):
        labels = imgs.clone()
        imgs = torch.from_numpy(skimage.util.random_noise(imgs, var=0.01))
            
        labels = labels[:, channels]
        masks = torch.ones_like(labels)

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

    def _postprocess(self, imgs, labels, masks, channels, task):
        # process all channels if not given
        if channels is None:
            assert task != 'segment_semantic'
            channels = range(len(self.TASK_GROUP_DICT[task]))
            
        # task-specific preprocessing
        if task == 'autoencoding':
            labels, masks = self._postprocess_autoencoding(imgs, channels)
        elif task == 'denoising':
            imgs, labels, masks = self._postprocess_denoising(imgs, channels)
        elif task == 'edge_texture':
            labels, masks = self._postprocess_edge_texture(imgs, channels)
        # keypoints_human, keypoints_multiclass, keypoints_semantic
        elif task == 'keypoints_human':
            labels, masks = self._postprocess_keypoints_human(labels, channels)
        else:
            pass

        # ensure label values to be in [0, 1]
        labels = labels.clip(0, 1)

        return imgs, labels, masks

    def load_images(self, imgIds):
        # load images
        imgs = []
        for imgId in imgIds:
            # load image
            img = self._load_image(imgId)
            imgs.append(img)
        imgs = torch.stack(imgs)

        return imgs
    
    def load_labels(self, imgs, task_group, channels, imgIds):
        # load labels
        labels = []
        masks = []
        for imgId in imgIds:
            # load label
            label, mask = self._load_and_preprocess_label(task_group, imgId, channels)
            labels.append(label)
            masks.append(mask)
        labels = torch.stack(labels) if labels[0] is not None else None
        masks = torch.stack(masks) if masks[0] is not None else None

        # postprocess labels
        imgs, labels, masks = self._postprocess(imgs, labels, masks, channels, task_group)
        
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


class DeepFashionBaseDataset(DeepFashion):
    def __init__(self, *args, coco=None, **kwargs):
        super().__init__(*args, **kwargs)
        if coco is not None:
            self.coco = coco
        else:
            self.coco = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'{self.split}.json'))

        self.imgIds = sorted(self.coco.getImgIds())
        perm_path = os.path.join('dataset', 'meta_info', self.NAME, f'idxs_perm_{self.split}.pth')
        if os.path.exists(perm_path):
            perm = torch.load(perm_path)
        else:
            perm = np.random.permutation(len(self.imgIds))
            torch.save(perm, perm_path)
        self.imgIds = [self.imgIds[i] for i in perm]
        ignore_Ids = [48170, 48938, 51480, 53748]
        self.imgIds = [imgId for imgId in self.imgIds if imgId not in ignore_Ids]

    def __len__(self):
        return len(self.imgIds)
    
    def initialize_keypoints(self):
        self.kp_imgIds = self.imgIds

        # Some attributes related to construct continuous keypoint labels
        self.kp_sigma = 3.0
        self.kp_max = 1.0 
        self.kp_truncate = 3.0
        self.kp_radius = round(self.kp_truncate * self.kp_sigma)
        base_array = np.zeros((2*self.kp_radius+1, 2*self.kp_radius+1))
        base_array[self.kp_radius, self.kp_radius] = 1
        self.gaussian = scipy.ndimage.gaussian_filter(base_array, sigma=self.kp_sigma, mode='constant', truncate=self.kp_truncate)
        self.gaussian = torch.tensor(self.gaussian / self.gaussian.max() * self.kp_max)


class DeepFashionBaseTrainDataset(DeepFashionBaseDataset):
    def __init__(self, label_augmentation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_augmentation = get_filtering_augmentation() if label_augmentation else None
        self.tasks = self.TASKS_BASE
        self.task_groups = self.TASK_GROUPS_BASE
        
    def sample_episode(self, n_files, n_channels=1, n_domains=None):
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

            # sample image paths
            file_idxs = np.random.choice(self.imgIds, n_files, replace=False)

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
    

class DeepFashionBaseTestDataset(DeepFashionBaseDataset):
    def __init__(self, task_group, dset_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_group = task_group
        self.dset_size = dset_size

    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.imgIds)
        
    def __getitem__(self, idx):
        imgIds = [self.imgIds[idx]]
        X = self.load_images(imgIds)
        X, Y, M = self.load_labels(X, self.task_group, None, imgIds)

        return X[0], Y[0], M[0]


class DeepFashionContinuousTrainDataset(DeepFashionBaseTrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_keypoints()
        self.tasks = self.TASKS_CONTINUOUS
        self.task_groups = self.TASK_GROUPS_CONTINUOUS


class DeepFashionContinuousTestDataset(DeepFashionBaseTestDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_keypoints()


class DeepFashionCategoricalDataset(DeepFashionBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_keypoints()
     

class DeepFashionCategoricalTrainDataset(DeepFashionCategoricalDataset):
    def __init__(self, label_augmentation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_augmentation = get_filtering_augmentation() if label_augmentation else None
        self.tasks = self.TASKS_CATEGORICAL
        self.task_groups = self.TASK_GROUPS_CATEGORICAL
        
    def sample_episode(self, n_files, n_channels=1, n_domains=None):
        # sample task group
        task_group_idx = np.random.choice(len(self.task_groups))
        task_group = self.task_groups[task_group_idx]
        task_group_name = '_'.join(task_group[0].split('_')[:-1])
        g_idx = torch.tensor(self.TASK_GROUP_NAMES.index(task_group_name))

        # decide on number of chunks and create channel mask
        max_channels = min(len(self.KP_IDXS), n_channels)
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

            # sample image paths
            file_idxs = np.random.choice(self.kp_imgIds, n_files, replace=False)

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


class DeepFashionCategoricalTestDataset(DeepFashionCategoricalDataset):
    def __init__(self, task_group, dset_size, *args, class_id=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_id = class_id
        self.dset_size = dset_size
        self.task_group = task_group

    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.kp_imgIds)
    
    def __getitem__(self, idx):
        imgId = self.kp_imgIds[idx]
        X = self.load_images([imgId])
        X, Y, M = self.load_labels(X, self.task_group, None, [imgId])
        Y_sparse = self._load_keypoint(imgId)

        return X[0], Y[0], M[0], (Y_sparse, imgId)
        
    def create_support_keypoint_index(self, shot, support_idx=0):
        print('Reading keypoints...')
        kp_dict = {kp: [] for kp in self.KP_IDXS}
        for imgId in self.imgIds:
            Y_sparse = self._load_keypoint(imgId)
            
            for kp in self.KP_IDXS:
                if Y_sparse[kp - 1, 2] == 1:
                    kp_dict[kp].append((imgId, (Y_sparse[:, 2] == 1).long()))

        for kp in self.KP_IDXS:
            kp_dict[kp] = sorted(kp_dict[kp], key=lambda x: x[1].sum(), reverse=True)

        print('Creating support keypoint index...')
        n_kps = torch.zeros(len(self.KP_IDXS))
        imgIds = []
        while len(imgIds) < shot*(support_idx+1):
            c = torch.argmin(n_kps).item()
            imgId, visibility = kp_dict[self.KP_IDXS[c]].pop(0)
            if imgId not in imgIds:
                imgIds.append(imgId)
                n_kps += visibility
        
        print('Done.')
        self.imgIds = self.kp_imgIds = imgIds[shot*support_idx:]
