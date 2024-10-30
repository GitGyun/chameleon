import os
from PIL import Image
import numpy as np
import scipy
from einops import repeat
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, ColorJitter, RandomRotation

from dataset.utils import crop_arrays
from dataset.coco_api_wrapper import SilentCOCO


class AP10K(Dataset):
    '''
    base class for AP10K dataset
    '''
    CLASS_NAMES = ['left_eye', 'right_eye', 'nose', 'neck', 'root_of_tail', 'left_shoulder',
                   'left_elbow', 'left_front_paw', 'right_shoulder', 'right_elbow', 'right_front_paw',
                   'left_hip', 'left_knee', 'left_back_paw', 'right_hip', 'right_knee', 'right_back_paw']
    SPECIES = ['Antelope', 'Argali Sheep', 'Bison', 'Buffalo', 'Cow', 'Sheep', 'Dog', 'Fox', 'Wolf', 'Beaver',
               'Alouatta', 'Monkey', 'Noisy Night Monkey', 'Spider Monkey', 'Uakari', 'Deer', 'Moose', 'Hamster', 'Elephant', 'Horse',
               'Zebra', 'Bobcat', 'Cat', 'Cheetah', 'Jaguar', 'King Cheetah', 'Leopard', 'Lion', 'Panther', 'Snow Leopard',
               'Tiger', 'Giraffe', 'Hippo', 'Chimpanzee', 'Gorilla', 'Rabbit', 'Skunk', 'Mouse', 'Rat', 'Otter',
               'Weasel', 'Raccoon', 'Rhino', 'Marmot', 'Squirrel', 'Pig', 'Black Bear', 'Brown Bear', 'Panda', 'Polar Bear']
    

class AP10KDataset(AP10K):
    '''
    AP10K dataset
    '''
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        assert config.class_name in self.SPECIES

        if split == 'valid':
            split = 'val'
        assert split in ['train', 'val', 'test']
        self.split = split

        # configure paths
        self.base_size = base_size
        data_root = config.path_dict[config.dataset]
        self.image_dir = os.path.join(data_root, f'images_cropped_with_extension_and_resized_{base_size[0]}')
        self.ann_dir = os.path.join(data_root, 'annotations', f'ap10k-{split}-split1.json')

        self.img_size = crop_size
        self.eval_mode = eval_mode
        self.resize = resize
        self.shot = config.shot
        self.support_idx = config.support_idx
        self.precision = config.precision
        self.eval_mode = eval_mode

        self.class_name = config.class_name
        self.coco = SilentCOCO(self.ann_dir)
        
        self.kp_classes = list(self.coco.cats.values())[0]['keypoints']
        self.species_ids = list(i for i in range(1, 55) if i not in [7, 37, 49, 54]) # 54 species, 4 species removed (no images)

        meta_dir = os.path.join('dataset', 'meta_info', 'ap10k')
        os.makedirs(meta_dir, exist_ok=True)
        class_dict_path = os.path.join(meta_dir, f'{self.split}_class_dict_skip_crowd.pth')
        
        if os.path.exists(class_dict_path):
            self.class_dict = torch.load(class_dict_path)
        else:
            self.generate_class_dict(class_dict_path, skip_crowd=True)

        assert self.class_name in self.SPECIES
        self.class_id = self.species_ids[self.SPECIES.index(self.class_name)]
        self.data_idxs = self.class_dict[self.class_id]

        # Some attributes related to construct continuous keypoint labels
        self.eval_mode = eval_mode
        self.gaussian = self._make_gaussian_kernel(3, 3, 1.)
        self.kp_radius = 9

        self.randomflip = config.randomflip
        self.randomjitter = config.randomjitter
        self.randomrotate = config.randomrotate

        self.toten = ToTensor()
        self.resizer = Resize(self.img_size)
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.rotate = RandomRotation(30)

        if dset_size < 0:
            self.dset_size = len(self.data_idxs)
        elif not eval_mode:
            self.dset_size = dset_size
        else:
            self.dset_size = min(dset_size, len(self.data_idxs))

        if split == 'train':
            self.create_support_keypoint_index(self.shot, self.support_idx)
        
    def _make_gaussian_kernel(self, sigma, truncate, max_value=1.0):
        radius = truncate * sigma
        base_array = np.zeros((2*radius+1, 2*radius+1))
        base_array[radius, radius] = 1
        gaussian = scipy.ndimage.gaussian_filter(base_array, sigma=sigma, mode='constant', truncate=float(truncate))
        gaussian = torch.tensor(gaussian / gaussian.max() * max_value)
        return gaussian 

    def generate_class_dict(self, class_dict_path, save=True, skip_crowd=True):
        self.class_dict = {j: [] for j in self.species_ids}

        if skip_crowd:
            skip_imgIds = set(imgId for imgId in self.coco.getImgIds() if len(self.coco.getAnnIds(imgIds=imgId)) != 1)
        else:
            skip_imgIds = set()

        for j in self.species_ids: # 54 species
            annIds = self.coco.getAnnIds(catIds=j)
            anns = self.coco.loadAnns(annIds)
            for ann in anns:
                imgId = ann['image_id']
                annId = ann['id']
                if imgId not in skip_imgIds:
                    self.class_dict[j].append((imgId, annId))

            self.class_dict[j] = np.array(self.class_dict[j])
            np.random.shuffle(self.class_dict[j])

        if save:
            torch.save(self.class_dict, class_dict_path)

    def __len__(self):
        return self.dset_size
    
    def _load_image(self, imgId, annId):
        # get path
        img_path = os.path.join(self.image_dir, f'{imgId}_{annId}.jpg')
        
        # open image file
        img = Image.open(img_path)
        img = self.toten(img)
        if len(img) == 1:
            img = img.repeat(3, 1, 1)
        
        return img
    
    def _load_keypoint(self, imgId, annId):
        keypoints = np.array(self.coco.anns[annId]['keypoints']).reshape(17, 3)
        W, H = self.coco.imgs[imgId]['width'], self.coco.imgs[imgId]['height']
        x1, y1, w1, h1 = self.coco.anns[annId]['bbox']
        x2 = x1 + w1
        y2 = y1 + h1

        x = round(max(0, (x1 + x2)/2 - w1/2*1.2))
        y = round(max(0, (y1 + y2)/2 - h1/2*1.2))
        w = round(min(W - max(0, (x1 + x2) / 2 - w1/2*1.2), w1*1.2))
        h = round(min(H - max(0, (y1 + y2) / 2 - h1/2*1.2), h1*1.2))

        # move w, h coordinates, because of resizing
        keypoints[..., 0] = (keypoints[..., 0] - x) / w * self.base_size[1]
        keypoints[..., 1] = (keypoints[..., 1] - y) / h * self.base_size[0]
        keypoints = keypoints.astype(int)

        keypoints = torch.from_numpy(keypoints)

        return keypoints

    def _load_label(self, imgId, annId):
        keypoints = self._load_keypoint(imgId, annId)
        label = self._sparse_to_dense(keypoints.numpy(), self.gaussian, self.kp_radius)
        mask = repeat((keypoints[..., 2] == 2).float(), 'c -> c h w', h=self.base_size[0], w=self.base_size[1])

        return label, mask, keypoints
    
    def _sparse_to_dense(self, keypoints, kernel, kp_radius):
        label = torch.zeros(17, *self.base_size)

        for c, (w, h, v) in enumerate(keypoints):
            if v == 2:
                slice_h = slice(max(0, h-kp_radius), min(self.base_size[0], h+kp_radius+1))
                slice_w = slice(max(0, w-kp_radius), min(self.base_size[1], w+kp_radius+1))
                label[c, slice_h, slice_w] = kernel[max(0, kp_radius - h):max(0, kp_radius - h)+slice_h.stop-slice_h.start,
                                                    max(0, kp_radius - w):max(0, kp_radius - w)+slice_w.stop-slice_w.start]
                
        return label

    def _load_data(self, imgId, annId):
        image = self._load_image(imgId, annId)
        label, mask, keypoints = self._load_label(imgId, annId)
                
        return image, label, mask, keypoints
    
    def _postprocess_data(self, image, label, mask, keypoints):
        X, Y, M = image, label, mask

        if self.randomjitter and not self.eval_mode and random.random() > 0.5:
            X = self.jitter(X)

        if self.randomrotate and not self.eval_mode and random.random() > 0.5:
            X, Y = self._rotate(X, Y)
        
        if (not self.eval_mode) and self.randomflip and random.random() > 0.5:
            X, Y, M, keypoints = self._flip(X, Y, M, keypoints)

        if self.resize:
            X = self.resizer(X)
            Y = self.resizer(Y)
            M = self.resizer(M)
            keypoints[..., 0] = keypoints[..., 0] / self.base_size[1] * self.img_size[1]
            keypoints[..., 1] = keypoints[..., 1] / self.base_size[0] * self.img_size[0]
        else:
            (X, Y, M), offsets = crop_arrays(X, Y, M,
                                                base_size=X.size()[-2:],
                                                crop_size=self.img_size,
                                                random=(not self.eval_mode),
                                                get_offsets=True,
                                                )
            keypoints[..., 0] = keypoints[..., 0] - offsets[1]
            keypoints[..., 1] = keypoints[..., 1] - offsets[0]

            keypoints[..., 2] = 2 * Y.sum((1, 2)).bool().float()
            M = M * (keypoints[..., 2][:, None, None] / 2)

            assert (Y.sum((1, 2)) + (1 - M).sum((1, 2))).bool().float().sum() == 17
        
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
            M = M.to(torch.bfloat16)
            
        return X, Y, M, keypoints

    def _flip(self, X, Y, M, keypoints=None):
        X = torch.flip(X, dims=[-1])
        Y = torch.flip(Y, dims=[-1])
        M = torch.flip(M, dims=[-1])
        flip_inds = [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]
        Y = Y[..., flip_inds, :, :]
        M = M[..., flip_inds, :, :]
        if keypoints is not None:
            keypoints = keypoints[flip_inds]
            return X, Y, M, keypoints
        else:
            return X, Y, M
        
    def _rotate(self, X, Y):
        XY = torch.cat([X, Y], dim=0)
        XY = self.rotate(XY)
        X, Y = torch.split(XY, [3, 17], dim=0)
        return X, Y

    def __getitem__(self, idx):
        imgId, annId = self.data_idxs[idx % len(self.data_idxs)]
        image, label, mask, keypoints = self._load_data(imgId, annId)
        X, Y, M, keypoints = self._postprocess_data(image, label, mask, keypoints)
        if self.eval_mode:
            return X, Y, M, (keypoints, (imgId, annId))
        else:
            return X, Y, M

    def create_support_keypoint_index(self, shot, support_idx=0):
        print('Reading keypoints...')
        # construct keypoint dict for each joint
        kp_dict = {kp: [] for kp in range(len(self.CLASS_NAMES))}
        for imgId in self.data_idxs:
            Y_sparse = self._load_keypoint(*imgId)
            for kp in range(len(self.CLASS_NAMES)):
                if Y_sparse[kp, 2] == 2:
                    kp_dict[kp].append((imgId, (Y_sparse[:, 2] == 2).long()))

        # sort keypoint dict by number of visible joints
        for kp in range(len(self.CLASS_NAMES)):
            kp_dict[kp] = sorted(kp_dict[kp], key=lambda x: x[1].sum(), reverse=True)

        print('Creating support keypoint index...')
        # add maximal keypoints for half of the shot
        n_kps = torch.zeros(len(self.CLASS_NAMES))
        imgIds = []
        top_k_imgs = min(shot*(support_idx+1)//2, len(self.data_idxs)//2)
        while len(imgIds) < top_k_imgs and n_kps.min() < float('inf'):
            # choose class with minimal number of keypoints in current support
            c = torch.argmin(n_kps).item()

            # if no more examples left for that class, skip
            if len(kp_dict[c]) == 0:
                n_kps[c] = float('inf')
                continue
            # add example with maximal number of keypoints for that class
            else:
                imgId, visibility = kp_dict[c].pop(0)

            # update number of included keypoints
            if (imgId[0], imgId[1]) not in imgIds:
                imgIds.append((imgId[0], imgId[1]))
                n_kps += visibility

        # support idx adjustment
        if len(self.data_idxs) - shot*support_idx >= shot:
            imgIds = imgIds[shot*support_idx:]
        else:
            imgIds = imgIds[shot*support_idx:] + imgIds[:shot*support_idx-len(self.data_idxs)]

        # add random keypoints for the rest half of the shot
        data_idxs = np.concatenate([self.data_idxs[shot*support_idx:], self.data_idxs[:shot*support_idx]])

        random.seed(0)
        remIds = sum([kp_dict[c] for c in kp_dict], []) # add all remaining images
        remIds = [(imgId[0], imgId[1]) for imgId, visibility in remIds if (imgId[0], imgId[1]) not in imgIds] # remove already added images
        remIds = [(imgId[0], imgId[1]) for imgId in data_idxs if (imgId[0], imgId[1]) in remIds][:min(shot, len(self.data_idxs)) - len(imgIds)] # add random images

        # join the two halves
        imgIds = np.array(imgIds + remIds)
        print(f'Done: registered {len(imgIds)} images out of {len(self.data_idxs)}')
        self.data_idxs = imgIds
        