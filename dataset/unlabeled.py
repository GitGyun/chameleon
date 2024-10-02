'''
Dataset for all the unlabeled images. Only base tasks and knowledge distillation tasks are available.
To use it, add images to the {data_root}/{split}_{img_size}_merged/{domain}/ after preprocessing.
I assume all the resizing and cropping is already done.
'''

import os
import random
import numpy as np
from PIL import Image
import skimage

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .augmentation import get_filtering_augmentation, Mixup
from .utils import SobelEdgeDetector, crop_arrays


class Unlabeled(Dataset):
    NAME = 'unlabeled'

    # Task Groups
    TASK_GROUPS_BASE = []
    TASK_GROUPS_CONTINUOUS = [] # This will not be used
    TASK_GROUPS_CATEGORICAL = [] # This will not be used

    # Base Tasks
    TASKS_AUTOENCODING = [f'autoencoding_{c}' for c in range(3)]
    TASK_GROUPS_BASE.append(TASKS_AUTOENCODING)

    TASKS_DENOISING = [f'denoising_{c}' for c in range(3)]
    TASK_GROUPS_BASE.append(TASKS_DENOISING)

    TASKS_EDGE2D = [f'edge_texture_{c}' for c in range(3)]
    TASK_GROUPS_BASE.append(TASKS_EDGE2D)

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

    # Available domains for using unlabeld images.
    DOMAINS = ('ph2', 'animals10', 'potsdam')
    

    def __init__(self, path_dict, split, base_size=(256, 256), crop_size=(224, 224), 
                 seed=None, precision='fp32', meta_dir='meta_info', domains=None):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.domains = self.DOMAINS
        # In future, we may set the domains to use
        # if 'all' in domains:
        #     self.domains = self.DOMAINS # use all
        # else:
        #     assert all([domain in self.DOMAINS for domain in domains]), f'Invalid domain: {domains}'
        #     self.domains = tuple(domains)

        self.meta_info_path = os.path.join('dataset', meta_dir, self.NAME)
        self.img_dir = os.path.join(path_dict[self.NAME], f'{split}_{base_size[0]}_merged')

        # load list of image files if avilable
        assert split in ['train', 'valid'], split
        img_file_path = os.path.join(self.meta_info_path, f'img_files_{split}_{"_".join(self.domains)}.pth')
        if os.path.exists(img_file_path):
            self.img_files = torch.load(img_file_path)
        else:
            self.img_files = {domain: sorted(os.listdir(os.path.join(self.img_dir, domain))) for domain in self.domains}
            # for domain in self.domains:
            #     domain_imgs = sorted(os.listdir(os.path.join(self.img_dir, domain)))
            #     self.img_files.extend(map(lambda x: os.path.join(domain, x), domain_imgs))
            torch.save(self.img_files, img_file_path)

        self.dset_size = sum(map(len, self.img_files.values()))
        self.base_size = base_size
        self.crop_size = crop_size
        self.precision = precision
            
        # register sobel edge detectors
        self.edge_params = torch.load(os.path.join(self.meta_info_path, 'edge_params.pth'))
        self.sobel_detectors = [SobelEdgeDetector(kernel_size=k, sigma=s) for k, s in self.edge_params['params']]

        # transforms
        self.toten = ToTensor()

    def __len__(self):
        return self.dset_size
        
    def _load_image(self, img_file, domain):
        img_path = os.path.join(self.img_dir, domain, img_file)

        # open image file
        img = Image.open(img_path)

        # resize if necessary
        if img.size != self.base_size:
            img = img.resize(self.base_size, Image.BILINEAR)

        # convert to torch tensor
        img = self.toten(img)

        # convert to 3 channels if necessary
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if img.shape[0] > 3:
            img = img[:3]
        
        return img
    
    def _load_label(self, task_group, img_file):
        raise NotImplementedError
        
    def _load_and_preprocess_label(self, task_group, img_file):
        # load label
        if task_group in self.TASK_GROUP_NAMES_BASE:
            label = mask = None
        else:
            raise NotImplementedError
            
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

    def _postprocess(self, imgs, labels, masks, channels, task):
        # process all channels if not given
        if channels is None:
            if task == 'segment_semantic':
                raise NotImplementedError
            else:
                channels = range(len(self.TASK_GROUP_DICT[task]))
            
        # task-specific preprocessing
        if task == 'autoencoding':
            labels, masks = self._postprocess_autoencoding(imgs, channels)
        elif task == 'denoising':
            imgs, labels, masks = self._postprocess_denoising(imgs, channels)
        elif task == 'edge_texture':
            labels, masks = self._postprocess_edge_texture(imgs, channels)
        else:
            raise NotImplementedError

        # ensure label values to be in [0, 1]
        labels = labels.clip(0, 1)

        return imgs, labels, masks
  
    def load_images(self, file_idxs, domains=None):
        if domains is None:
            domains = [self.domains[file_idx % len(self.domains)] for file_idx in file_idxs]
            file_idxs = [file_idx % len(self.img_files[domain]) for file_idx, domain in zip(file_idxs, domains)]
        # load images
        imgs = []
        for (domain, file_idx) in zip(domains, file_idxs):
            # index image path
            img_file = self.img_files[domain][file_idx]

            # load image
            img = self._load_image(img_file, domain=domain)
            imgs.append(img)
        imgs = torch.stack(imgs)

        return imgs
    
    def load_labels(self, imgs, task, channels, file_idxs, domains=None):
        if domains is None:
            domains = [self.domains[file_idx % len(self.domains)] for file_idx in file_idxs]
            file_idxs = [file_idx % len(self.img_files[domain]) for file_idx, domain in zip(file_idxs, domains)]

        # load labels
        labels = []
        masks = []
        for (domain, file_idx) in zip(domains, file_idxs):
            # index image path
            img_file = self.img_files[domain][file_idx]

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


class UnlabeledBaseDataset(Unlabeled):
    pass


class UnlabeledBaseTrainDataset(UnlabeledBaseDataset):
    def __init__(self, unary_augmentation, binary_augmentation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unary_augmentation = get_filtering_augmentation() if unary_augmentation else None
        self.binary_augmentation = Mixup() if binary_augmentation else None
        self.tasks = self.TASKS_BASE
        
    def sample_task(self, from_all=False):
        if from_all:
            task = np.random.choice(self.TASKS)
        else:
            task = np.random.choice(self.tasks)
        task_group, channel = '_'.join(task.split('_')[:-1]), int(task.split('_')[-1])
        t_idx = torch.tensor(self.TASKS.index(task))

        return task_group, channel, t_idx


    def sample_domains_with_files(self, n_domains, n_files):
        ret_domains = []
        ret_file_idxs = []
        files_per_domain = n_files // n_domains
        assert n_files % n_domains == 0, 'n_files must be divisible by n_domains, but {} % {} != 0'.format(n_files, n_domains)
        for domain in np.random.choice(self.domains, n_domains, replace=(n_domains > len(self.domains))):
            ret_file_idxs.extend(np.random.choice(len(self.img_files[domain]), files_per_domain))
            ret_domains.extend([domain] * files_per_domain)
        assert len(ret_file_idxs) == n_files

        return ret_domains, ret_file_idxs


    def sample_episode(self, n_files, n_domains=None):
        # sample tasks
        task_group, channel, t_idx = self.sample_task()
        if self.binary_augmentation is not None:
            task_group_aux, channel_aux, _ = self.sample_task(from_all=True)

        # sample domains and paths
        domains, file_idxs = self.sample_domains_with_files(n_domains, n_files)

        # load images
        imgs = self.load_images(file_idxs, domains=domains)

        # load labels
        X, Y, M = self.load_labels(imgs, task_group, [channel], file_idxs, domains)
        if self.binary_augmentation:
            _, Y_aux, M_aux = self.load_labels(imgs, task_group_aux, [channel_aux], file_idxs, domains)

        # apply unary task augmentation
        if self.unary_augmentation:
            Y, M = self.unary_augmentation(Y, M)
            if self.binary_augmentation:
                Y_aux, M_aux = self.unary_augmentation(Y_aux, M_aux)

        # apply binary taks augmentation
        if self.binary_augmentation:
            Y, M = self.binary_augmentation(Y, Y_aux, M, M_aux)
        
        # random-crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              crop_size=self.crop_size,
                              random=True)
        
        return X, Y, M, t_idx


class UnlabeledBaseTestDataset(UnlabeledBaseDataset):
    def __init__(self, task_group, dset_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_group = task_group
        if dset_size > 0:
            self.dset_size = dset_size

    def __len__(self):
        return self.dset_size
        
    def __getitem__(self, idx):
        X = self.load_images([idx])
        X, Y, M = self.load_labels(X, self.task_group, None, [idx])

        return X[0], Y[0], M[0]
    

class UnlabeledUnsupervisedTrainDataset(Unlabeled):
    def sample_domains_with_files(self, n_domains, n_files):
        ret_domains = []
        ret_file_idxs = []
        files_per_domain = n_files // n_domains
        assert n_files % n_domains == 0, 'n_files must be divisible by n_domains, but {} % {} != 0'.format(n_files, n_domains)
        for domain in np.random.choice(self.domains, n_domains, replace=(n_domains > len(self.domains))):
            ret_file_idxs.extend(np.random.choice(len(self.img_files[domain]), files_per_domain))
            ret_domains.extend([domain] * files_per_domain)
        assert len(ret_file_idxs) == n_files

        return ret_domains, ret_file_idxs


    def sample_episode(self, n_files, n_domains=None):
        # sample domains and paths
        domains, file_idxs = self.sample_domains_with_files(n_domains, n_files)

        # load images
        X = self.load_images(file_idxs, domains=domains)

        # crop arrays
        X = crop_arrays(X,
                        base_size=self.base_size,
                        crop_size=self.crop_size,
                        random=True)
        
        # create empty labels and masks
        Y = torch.empty(X.shape[0], 1, X.shape[2], X.shape[3])
        M = torch.empty(X.shape[0], 1, X.shape[2], X.shape[3])
        t_idx = torch.tensor(-1)
        
        # precision conversion
        if self.precision == 'fp16':
            X = X.half()
            Y = Y.half()
            M = M.half()
        elif self.precision == 'bf16':
            X = X.bfloat16()
            Y = Y.bfloat16()
            M = M.bfloat16()

        return X, Y, M, t_idx
    

class UnlabeledUnsupervisedTestDataset(Unlabeled):
    def __init__(self, dset_size=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if dset_size > 0:
            self.dset_size = dset_size

    def __len__(self):
        return self.dset_size
        
    def __getitem__(self, idx):
        domain = self.domains[idx]

        # load image
        X = self._load_image(self.img_files[domain][idx], domain=domain)
        
        # create empty labels and masks
        Y = torch.empty(1, X.shape[1], X.shape[2])
        M = torch.empty(1, X.shape[1], X.shape[2])
        
        # precision conversion
        if self.precision == 'fp16':
            X = X.half()
            Y = Y.half()
            M = M.half()
        elif self.precision == 'bf16':
            X = X.bfloat16()
            Y = Y.bfloat16()
            M = M.bfloat16()

        return X