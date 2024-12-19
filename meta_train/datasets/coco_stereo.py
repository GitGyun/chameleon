import os
import random
import numpy as np
from PIL import Image
from einops import repeat, rearrange

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from dataset.augmentation import get_filtering_augmentation
from dataset.utils import crop_arrays, draw_keypoints
from dataset.coco_api_wrapper import SingletonCOCOFactory


class COCOStereo(Dataset):
    NAME = 'coco_stereo'
    # 1. Multiclasss keypoint task is not used.
    # 2. Images with more than one instance are not used for keypoint task.
    # 3. Generating keypoint labels are done in a different way (parameter changed and use maximum).
    # 4. Spatial softmax loss is used.

    # Class Splits for Semantic Segmentation
    CLASS_IDXS = list(i for i in range(1, 92) if i not in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91])
    # 183 == other
    # 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91 are removed from COCO.
    CLASS_IDXS_VAL = [8, 15, 17, 27, 31, 51, 77, 81, 84, 85]
    # Manually set this because there are too many classes in COCO. These are top-10 by img count.
    # tree, wall-concrete, sky-other, building-other, grass | person, chair, car, cup, bottle
    CLASS_RANGE = range(92)
    CLASS_NAMES = \
        ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', # Things
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',  'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    CLASS_IDX_TO_NAME = {c: name for c, name in zip(CLASS_IDXS, CLASS_NAMES)}

    # Task Groups
    TASK_GROUPS_BASE = []
    TASK_GROUPS_CONTINUOUS = []
    TASK_GROUPS_CATEGORICAL = []

    # Categorical Tasks
    TASKS_INTERSEG = [f'segment_interactive_{c}' for c in CLASS_IDXS]
    TASK_GROUPS_CATEGORICAL.append(TASKS_INTERSEG)

    # Task Group Names
    TASK_GROUP_NAMES_BASE = ['_'.join(task_group[0].split('_')[:-1]) for task_group in TASK_GROUPS_BASE]
    TASK_GROUP_NAMES_CONTINUOUS = ['_'.join(task_group[0].split('_')[:-1]) for task_group in TASK_GROUPS_CONTINUOUS]
    TASK_GROUP_NAMES_CATEGORICAL = ['_'.join(task_group[0].split('_')[:-1]) for task_group in TASK_GROUPS_CATEGORICAL]
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
        CHANNELS_DICT[task_group_name] = 1

    def __init__(self, path_dict, split, base_size=(256, 256), crop_size=(224, 224), 
                 seed=None, precision='fp32', meta_dir='meta_info'):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        if split == 'valid':
            split = 'val'
        assert split in ['train', 'val'], split  # TODO: add train+val
        self.img_root = os.path.join(path_dict['coco'], 'images', f'{split}2017_{base_size[0]}')
        self.ann_root = os.path.join(path_dict['coco'], 'annotations')
        self.split = split
        
        self.base_size = base_size
        self.crop_size = crop_size
        self.precision = precision
        self.meta_info_path = os.path.join('dataset', meta_dir, 'coco')
            
        # register class dictionary
        self.class_dict = torch.load(os.path.join(self.meta_info_path, f'coco_{split}_class_dict.pth'))

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
        img_path = os.path.join(self.img_root, file_name)
        
        # open image file
        img = Image.open(img_path)
        img = self.toten(img)
        if img.ndim == 2:
            img = img[None]
        if len(img) == 1:
            img = img.repeat(3, 1, 1)
        
        return img

    def _load_label(self, task_group, imgId, class_id=None, eval_mode=False):
        '''
        return torch Tensor
        '''
        assert len(class_id) == 1 # only support binary segmentation
        class_id = class_id[0]
        label = np.zeros((self.coco.imgs[imgId]['height'],
                          self.coco.imgs[imgId]['width']), dtype='uint8')
        # things
        annIds = self.coco.getAnnIds(imgIds=[imgId], catIds=[class_id])
        anns = self.coco.loadAnns(annIds)

        # sample instances
        if eval_mode:
            anns = [anns[0]]
        else:
            k = random.randint(1, max(2, len(anns)//2))
            anns = random.choices(anns, k=k)

        keypoints = []

        # create label
        for ann in anns:
            segmap = self.coco.annToMask(ann)
            if eval_mode:
                keypoints += self.find_keypoints_inside(segmap, n_keypoints=3, is_random=False)
            else:
                n_keypoints = random.choice(list(range(1, 6)))
                keypoints += self.find_keypoints_inside(segmap, n_keypoints, is_random=True)
            
            label += segmap

        label = self.toten(label) * 255
        label = self._resize(label, mode='nearest')
        mask = None
        
        return label, keypoints, mask
    
    def find_keypoints_inside(self, segmentation_map, n_keypoints=-1, is_random=False):
        assert n_keypoints > 0
        H, W = segmentation_map.shape
        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1)
        possible = grid[segmentation_map == 1]
        possible = rearrange(possible, '... C -> (...) C')
        n_keypoints = min(n_keypoints, len(possible))
        if is_random:
            idxs = np.random.choice(list(range(len(possible))), n_keypoints, replace=False)
            keypoints = possible[idxs].tolist()
        else:
            if len(possible) < 3:
                keypoints = possible[:n_keypoints].tolist()
            else:
                chunk_size = len(possible) // (1 + n_keypoints)
                if chunk_size == 0:
                    keypoints = possible[:n_keypoints].tolist()
                else:
                    keypoints = possible[::chunk_size][1:-1].tolist()
        keypoints = [ [x / W, y / H] for y, x in keypoints]

        return keypoints
            
    def _load_and_preprocess_label(self, task_group, img_file, channels, eval_mode):
        # load label
        label, keypoints, mask = self._load_label(task_group, img_file, class_id=channels, eval_mode=eval_mode)
            
        return label, keypoints, mask
         
    def _postprocess_segment_interactive(self, imgs, labels, keypoints, masks):
        if self.stereo_method == "overlay":
            imgs2 = torch.stack([self._single_image_prompts(img, keypoint, self.radius, "blue", 0.5) for img, keypoint in zip(imgs, keypoints)])
        elif self.stereo_method == "pointonly":
            imgs2 = torch.stack([self._single_image_prompts(img, keypoint, self.radius, "blue", 0) for img, keypoint in zip(imgs, keypoints)])
            
        imgs = torch.cat([imgs, imgs2], dim=1)

        masks = torch.ones_like(labels)

        return imgs, labels, masks
    
    def _single_image_prompts(self, image, points, radius, color="white", alpha=0):
        """
        image: (3, H, W) image tensor
        points: normalized point tensor [0, 1] with shape of (N, 2)
        
        return (3, H, W) tensor 
        """
        
        if color == "black":
            point_map = torch.ones(3, image.size(1), image.size(2)).byte()
        else:
            point_map = torch.zeros(3, image.size(1), image.size(2)).byte()
            
        points = torch.tensor(points)[None]
        if points.ndim == 3:
            points[:, :, 0] *= image.size(2)
            points[:, :, 1] *= image.size(1)
            points = points.long()
            
            point_map = draw_keypoints(image=point_map, keypoints=points, colors=color, radius=radius, width=0)
            point_map = point_map.float() / 255
            if color == "black":
                point_map_mask = point_map == 0
            else:
                point_map_mask = point_map > 0
            
            point_map_mask = point_map_mask.to(image.dtype)
            point_map = point_map.to(image.dtype)
            
            background = image * (1 - point_map_mask)
            
            return alpha * background + point_map
        else:
            return image
    
    def _postprocess(self, imgs, labels, keypoints, masks):
        # task-specific preprocessing
        imgs, labels, masks = self._postprocess_segment_interactive(imgs, labels, keypoints, masks)

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
    
    def load_labels(self, imgs, task_group, channels, imgIds, eval_mode=False):
        # load labels
        labels = []
        keypoints = []
        masks = []
        for imgId in imgIds:
            # load label
            label, keypoint, mask = self._load_and_preprocess_label(task_group, imgId, channels, eval_mode=eval_mode)
            labels.append(label)
            keypoints.append(keypoint)
            masks.append(mask)
        labels = torch.stack(labels) if labels[0] is not None else None
        keypoints = keypoints if keypoints[0] is not None else None
        masks = torch.stack(masks) if masks[0] is not None else None

        # postprocess labels
        imgs, labels, masks = self._postprocess(imgs, labels, keypoints, masks)
        
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


class COCOStereoBaseDataset(COCOStereo):
    def __init__(self, *args, coco=None, **kwargs):
        super().__init__(*args, **kwargs)

        if coco is not None:
            self.coco = coco
        else:
            self.coco = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'instances_{self.split}2017.json'))
        
        self.imgIds = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.imgIds)
    

class COCOStereoCategoricalDataset(COCOStereoBaseDataset):
    '''
    In principle this should inherit from COCO, but it is really similar to COCOBaseDataset so we inherit from it.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        perm_path = os.path.join('dataset', 'meta_info', 'coco', f'idxs_perm_categorical_{self.split}.pth')
        if os.path.exists(perm_path):
            perm = torch.load(perm_path)
        else:
            perm = np.random.permutation(len(self.imgIds))
            torch.save(perm, perm_path)

        self.imgIds = [self.imgIds[i] for i in perm]

        self.stereo_method = "overlay"
        self.radius = 3
     

class COCOStereoCategoricalTrainDataset(COCOStereoCategoricalDataset):
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
        chunk_size = 1
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
            c = channels[0]
            file_idxs = np.random.choice(self.class_dict[c], n_files, replace=False)

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


class COCOStereoCategoricalTestDataset(COCOStereoCategoricalDataset):
    def __init__(self, task_group, dset_size, *args, class_id=-1, **kwargs):
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
        imgId = self.class_dict[self.class_id][idx]
        X = self.load_images([imgId])
        X, Y, M = self.load_labels(X, self.task_group, [self.class_id], [imgId], eval_mode=True)

        return X[0], Y[0], M[0]
    