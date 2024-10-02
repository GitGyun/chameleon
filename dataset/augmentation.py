import random
import math
import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur
from torchvision.transforms import ColorJitter
import cv2
from PIL import Image


def normalize(x):
    if x.max() == x.min():
        return x - x.min()
    else:
        return (x - x.min()) / (x.max() - x.min())


def linear_sample(p_range):
    if isinstance(p_range, float):
        return p_range
    else:
        return p_range[0] + random.random()*(p_range[1] - p_range[0])
    
    
def log_sample(p_range):
    if isinstance(p_range, float):
        return p_range
    else:
        return math.exp(math.log(p_range[0]) + random.random()*(math.log(p_range[1]) - math.log(p_range[0])))
    
    
def categorical_sample(p_range):
    if isinstance(p_range, (float, int)):
        return p_range
    else:
        return p_range[np.random.randint(len(p_range))]
    
    
def rand_bbox(size, lam):
    H, W = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Augmentation:
    pass


class RandomHorizontalFlip(Augmentation):
    def __init__(self):
        self.augmentation = lambda x: torch.flip(x, dims=[-1])
        
    def __str__(self):
        return 'RandomHorizontalFlip Augmentation'
        
    def __call__(self, *arrays, get_augs=False):
        if random.random() < 0.5:
            if len(arrays) == 1:
                if get_augs:
                    return self.augmentation(arrays[0]), self.augmentation
                else:
                    return self.augmentation(arrays[0])
            else:
                arrays_flipped = []
                for array in arrays:
                    arrays_flipped.append(self.augmentation(array))
                if get_augs:
                    return arrays_flipped, self.augmentation
                else:
                    return arrays_flipped
        else:
            if len(arrays) == 1:
                if get_augs:
                    return arrays[0], lambda x: x
                else:
                    return arrays[0]
            else:
                if get_augs:
                    return arrays, lambda x: x
                else:
                    return arrays
    
    
class RandomCompose(Augmentation):
    def __init__(self, augmentations, n_aug=2, p=0.5, verbose=False):
        assert len(augmentations) >= n_aug
        self.augmentations = augmentations
        self.n_aug = n_aug
        self.p = p
        self.verbose = verbose # for debugging
    
    def __call__(self, label, mask, get_augs=False):
        augmentations = [
            self.augmentations[i] 
            for i in np.random.choice(len(self.augmentations), size=self.n_aug, replace=False)
        ]
        
        for augmentation in augmentations:
            if random.random() < self.p:
                label, mask = augmentation(label, mask)
                if self.verbose:
                    print(augmentation)
            elif self.verbose:
                print('skipped')
            
        if get_augs:
            return label, mask, augmentations
        else:
            return label, mask
    

class RandomJitter(Augmentation):
    def __init__(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast
        
    def __str__(self):
        return f'RandomJitter Augmentation (brightness = {self.brightness}, contrast = {self.contrast})'
        
    def __call__(self, label, mask):
        brightness = linear_sample(self.brightness)
        contrast = linear_sample(self.contrast)
        
        alpha = 1 + contrast
        beta = brightness
        
        label = alpha * label + beta
        label = torch.clip(label, 0, 1)
        label = normalize(label)
        
        return label, mask
    
    
class RandomPolynomialTransform(Augmentation):
    def __init__(self, degree):
        self.degree = degree
        
    def __str__(self):
        return f'RandomPolynomialTransform Augmentation (degree = {self.degree})'
        
    def __call__(self, label, mask):
        degree = log_sample(self.degree)
        
        label = label.pow(degree)
        label = normalize(label)
        return label, mask


class RandomSigmoidTransform(Augmentation):
    def __init__(self, temperature):
        self.temperature = temperature
        
    def __str__(self):
        return f'RandomSigmoidTransform Augmentation (temperature = {self.temperature})'
    
    def __call__(self, label, mask):
        cast = False
        if label.dtype != torch.float32:
            dtype = label.dtype
            cast = True
            label = label.float()
        
        temperature = categorical_sample(self.temperature)
        
        label = torch.sigmoid(label / temperature)
        label = normalize(label)
        
        if cast:
            label = label.to(dtype)
        
        return label, mask


class RandomGaussianBlur(Augmentation):
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __str__(self):
        return f'RandomGaussianBlur Augmentation (kernel_size = {self.kernel_size}, sigma = {self.sigma})'
    
    def __call__(self, label, mask):
        cast = False
        if label.dtype != torch.float32:
            dtype = label.dtype
            cast = True
            label = label.float()
        
        kernel_size = [categorical_sample(self.kernel_size)]*2
        sigma = categorical_sample(self.sigma)
        
        label = gaussian_blur(label, kernel_size, sigma)
        label = normalize(label)
        
        if cast:
            label = label.to(dtype)
        
        return label, mask
    
    
class BinaryAugmentation(Augmentation):
    pass


class Mixup(BinaryAugmentation):
    def __init__(self, alpha=1.0, order=False):
        self.alpha = alpha
        self.order = order
        
    def __call__(self, label_1, label_2, mask_1, mask_2):
        lam = np.random.beta(self.alpha, self.alpha)
        if self.order:
            lam = max(lam, 1 - lam)
        label_mix = lam*label_1 + (1 - lam)*label_2
        mask_mix = torch.logical_and(mask_1, mask_2, out=mask_1.new(mask_1.size()))
        
        return label_mix, mask_mix


class Cutmix(BinaryAugmentation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, label_1, label_2, mask_1, mask_2):
        assert label_1.size() == label_2.size()
        
        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(label_1.size()[-2:], lam)
        
        label_mix = label_1.clone()
        label_mix[:, :, bbx1:bbx2, bby1:bby2] = label_2[:, :, bbx1:bbx2, bby1:bby2]
        mask_mix = mask_1.clone()
        mask_mix[:, :, bbx1:bbx2, bby1:bby2] = mask_2[:, :, bbx1:bbx2, bby1:bby2]

        return label_mix, mask_mix
    
     
FILTERING_AUGMENTATIONS = {
    'jitter': (RandomJitter, {"brightness": (-0.5, 0.5), 
                              "contrast": (-0.5, 0.5)}),
    'polynomial': (RandomPolynomialTransform, {"degree": (1.0/3, 3.0)}),
    'sigmoid': (RandomSigmoidTransform, {"temperature": [0.1, 0.2, 0.5, 2e5, 5e5, 1e6, 2e6]}),
    'gaussianblur': (RandomGaussianBlur, {"kernel_size": [9, 17, 33], 
                                          "sigma": [0.5, 1.0, 2.0, 5.0, 10.0]}),
    }


def get_filtering_augmentation():
    filtering_augmentation = RandomCompose(
        [augmentation(**kwargs) for augmentation, kwargs in FILTERING_AUGMENTATIONS.values()],
        p=0.8,
    )
    return filtering_augmentation


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(img1), dtype=np.uint8)
            img2 = np.array(self.photo_aug(img2), dtype=np.uint8)

        # symmetric
        else:
            img1 = np.asarray(img1)
            img2 = np.asarray(img2)
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)

        flow = flow.transpose(1, 2, 0)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)
        flow = flow.transpose(2, 0, 1)

        img1 = Image.fromarray(np.ascontiguousarray(img1))
        img2 = Image.fromarray(np.ascontiguousarray(img2))
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow