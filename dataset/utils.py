from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import math

from skimage.morphology import extrema
import itertools
import os
import plyfile
import scipy
import scipy.spatial

from PIL import Image, ImageDraw
from torchvision.utils import _log_api_usage_once


def inverse_sigmoid(x):
    return torch.log(x / (1-x))


################# Crop Tools ####################
def crop_arrays(*arrays, base_size=(256, 256), crop_size=(224, 224), random=True, get_offsets=False, offset_cuts=None):
    '''
    Crop arrays from base_size to img_size.
    Apply center crop if not random.
    '''
    if isinstance(base_size, int):
        base_size = (base_size, base_size)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    if base_size[0] == crop_size[0] and base_size[1] == crop_size[1]:
        if len(arrays) == 1:
            arrays = arrays[0]
        
        if get_offsets:
            return arrays, (0, 0)
        else:
            return arrays

    # TODO: check offset_cuts in LMFinetuneDataset and refactor here.
    if random and offset_cuts is not None and len(offset_cuts) == 2:
        off_H = np.random.randint(min(base_size[0] - crop_size[0] + 1,  offset_cuts[0] + 1))
        off_W = np.random.randint(min(base_size[1] - crop_size[1] + 1,  offset_cuts[1] + 1))

    elif random:
        min_H = 0
        min_W = 0
        max_H = base_size[0] - crop_size[0]
        max_W = base_size[1] - crop_size[1]
        if offset_cuts is not None:
            assert len(offset_cuts) == 4
            h_min, w_min, h_max, w_max = offset_cuts
            min_H = max(min_H, h_min - crop_size[0])
            min_W = max(min_W, w_min - crop_size[1])
            max_H = min(max_H, h_max)
            max_W = min(max_W, w_max)
            
        off_H = np.random.randint(max(1, max_H - min_H + 1)) + min_H
        off_W = np.random.randint(max(1, max_W - min_W + 1)) + min_W

    else:
        if offset_cuts is not None:
            assert len(offset_cuts) == 2
            off_H, off_W = offset_cuts
        else:
            off_H = (base_size[0] - crop_size[0]) // 2
            off_W = (base_size[1] - crop_size[1]) // 2

    slice_H = slice(off_H, off_H + crop_size[0])
    slice_W = slice(off_W, off_W + crop_size[1])

    arrays_cropped = []
    for array in arrays:
        if array is not None:
            assert array.ndim >= 2
            array_cropped = array[..., slice_H, slice_W]
            arrays_cropped.append(array_cropped)
        else:
            arrays_cropped.append(array)

    if len(arrays_cropped) == 1:
        arrays_cropped = arrays_cropped[0]

    if get_offsets:
        return arrays_cropped, (off_H, off_W)
    else:
        return arrays_cropped


class AutoCrop:
    def __init__(self, crop_size, min_overlap=0.5, rescale=1, sigma=None):
        self.crop_size = crop_size
        self.rescale = rescale
        self.stride = (round((1-min_overlap)*crop_size[0]), round((1-min_overlap)*crop_size[1]))
        if sigma is None:
            sigma = [crop_size[0] // 2, crop_size[1] // 2]
        # sigma = [crop_size[0] // 4, crop_size[1] // 4]
        base_array = np.zeros(crop_size)
        base_array[crop_size[0] // 2, crop_size[1] // 2] = 1
        gaussian = scipy.ndimage.gaussian_filter(base_array, sigma=sigma, mode='constant')
        self.confidence_map = torch.from_numpy(gaussian)

    def get_offset_cuts(self, base_size):
        if base_size[0] % self.stride[0] == 0:
            n_crops_H = base_size[0] // self.stride[0]
        else:
            n_crops_H = math.ceil(base_size[0] / self.stride[0]) - 1
        
        if base_size[1] % self.stride[1] == 0:
            n_crops_W = base_size[1] // self.stride[1] 
        else:
            n_crops_W = math.ceil(base_size[1] / self.stride[1]) - 1

        offset_cuts_list = []
        for i in range(n_crops_H):
            for j in range(n_crops_W):
                offset_cuts_list.append((min(i*self.stride[0], base_size[0] - self.crop_size[0]),
                                         min(j*self.stride[1], base_size[1] - self.crop_size[1])))
        
        return offset_cuts_list

    def crop(self, img):
        if self.rescale:
            img = F.interpolate(img.float(), scale_factor=self.rescale, mode='bilinear').to(img.dtype)

        base_size = img.shape[-2:]
        assert base_size[0] >= self.crop_size[0]
        assert base_size[1] >= self.crop_size[1]

        offset_cuts_list = self.get_offset_cuts(base_size)

        img_crops = []
        for offset_cuts in offset_cuts_list:
            img_cropped = crop_arrays(img, base_size=base_size, crop_size=self.crop_size, offset_cuts=offset_cuts, random=False)
            img_crops.append(img_cropped)

        return img_crops
    
    def tile(self, imgs, base_size):
        assert base_size[0] >= self.crop_size[0]
        assert base_size[1] >= self.crop_size[1]
        
        B, C, H, W = imgs[0].shape
        assert H == self.crop_size[0]
        assert W == self.crop_size[1]

        offset_cuts_list = self.get_offset_cuts(base_size)
        img = torch.zeros(B, C, *base_size, device=imgs[0].device, dtype=imgs[0].dtype)
        weight = torch.zeros(B, 1, *base_size, device=imgs[0].device, dtype=imgs[0].dtype)
        confidence_map = self.confidence_map.to(device=imgs[0].device, dtype=imgs[0].dtype)

        for i, offset_cuts in enumerate(offset_cuts_list):
            img[:, :, offset_cuts[0]:offset_cuts[0]+self.crop_size[0], offset_cuts[1]:offset_cuts[1]+self.crop_size[1]] += (confidence_map*imgs[i])
            weight[:, :, offset_cuts[0]:offset_cuts[0]+self.crop_size[0], offset_cuts[1]:offset_cuts[1]+self.crop_size[1]] += confidence_map

        weight = torch.where(weight > 0, weight, torch.ones_like(weight))
        img = img / weight
        return img


################# General Tools ####################
def to_device(data, device=None, dtype=None):
    '''
    Load data with arbitrary structure on device.
    '''
    def to_device_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        elif isinstance(data, tuple):
            return tuple(map(to_device_wrapper, data))
        elif isinstance(data, list):
            return list(map(to_device_wrapper, data))
        elif isinstance(data, dict):
            return {key: to_device_wrapper(data[key]) for key in data}
        else:
            return data
            
    return to_device_wrapper(data)


def get_data_iterator(data_loader, device=None, dtype=None):
    '''
    Iterator wrapper for dataloader
    '''
    def get_batch():
        while True:
            for batch in data_loader:
                yield to_device(batch, device, dtype)
    return get_batch()


################# Sobel Edge ####################
class SobelEdgeDetector:
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

        # compute gaussian kernel
        size = kernel_size // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

        self.gaussian_kernel = torch.from_numpy(g)[None, None, :, :].float()
        self.Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)[None, None, :, :]
        self.Ky = -torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float)[None, None, :, :]

    def detect(self, img, normalize=True):
        squeeze = False
        if len(img.shape) == 3:
            img = img[None, ...]
            squeeze = True

        img = pad_by_reflect(img, padding=self.kernel_size//2)
        img = F.conv2d(img, self.gaussian_kernel.repeat(1, img.size(1), 1, 1))

        img = pad_by_reflect(img, padding=1)
        Gx = F.conv2d(img, self.Kx)
        Gy = F.conv2d(img, self.Ky)

        G = (Gx.pow(2) + Gy.pow(2)).pow(0.5)
        if normalize:
            G = G / G.max()
        if squeeze:
            G = G[0]

        return G


def pad_by_reflect(x, padding=1):
    x = torch.cat((x[..., :padding], x, x[..., -padding:]), dim=-1)
    x = torch.cat((x[..., :padding, :], x, x[..., -padding:, :]), dim=-2)
    return x
    

################# Keypoint Related Utils ####################
def preprocess_kpmap(arrs:torch.Tensor, pre_threshold=0.1, post_threshold=0.4) -> torch.Tensor:
    # arr is B x 1 x H x W matrix or 1 x H x W or H x W matrix.
    # threshold, normalize to (0,1) and threshold again.
    unsqueezed = False
    if arrs.ndim == 2:
        arrs = arrs[None]
        unsqueezed = True

    if (arrs > pre_threshold).any():
        arrs = arrs.clip(pre_threshold, 1.0)
        minimums = arrs.min(dim=-2, keepdim=True)[0].min(dim=-1, keepdim=True)[0]
        arrs -= minimums
        maximums = arrs.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
        arrs = arrs / maximums
        arrs[arrs < post_threshold] = 0
    else:
        # if all values are below threshold, return all zeros except the max point (to ensure at leas one mode)
        arrs = torch.where(arrs==arrs.max(), max(1e-4, arrs.max()), 0) # 1e-4 for numerical stability

    if unsqueezed:
        arrs = arrs[0]

    return arrs


def preprocess_kpmap2(heatmap, threshold=0.1):
    assert heatmap.ndim == 2
    maxima = extrema.h_maxima((heatmap).numpy(), threshold)
    grid = torch.stack(torch.meshgrid(torch.arange(heatmap.shape[0]), torch.arange(heatmap.shape[1]), indexing='ij'), dim=2)
    modes = grid[maxima.astype(bool)]
    return modes


def get_modes(arrs:torch.Tensor, return_scores=True, top_one=True, threshold=0.1):
    '''
    arrs: 1 x H x W or H x W matrix
    returns a list of modes, each mode is a (x, y) tuple.
    '''
    assert arrs.ndim in [2, 3, 4]
    if arrs.ndim == 2:
        arrs = arrs[None]
    elif arrs.ndim == 4:
        modes = []
        for arr in arrs:
            kps, scores = get_modes(arr, True, True)
            modes.append(np.concatenate((np.array(kps)[:, 0], np.array(scores)), axis=1))
        return np.stack(modes, axis=0)

    coms_wh_list = []
    scores_list = []
    for arr_ in arrs:
        arr_ = preprocess_kpmap(arr_, threshold).float().cpu().numpy() # H x W

        # get local maximum points for each batch.
        labeled, n_feat = ndimage.label(arr_)
        coms_hw:List[Tuple[float, float]] = ndimage.center_of_mass(arr_, labeled, range(1, n_feat+1)) # in [[h, w]] format
        coms_hw = np.round(coms_hw).astype(int).tolist()
        if len(coms_hw) == 0:
            coms_hw = [(0, 0)]
        coms_wh = [(c[1], c[0]) for c in coms_hw] # in [[w, h]] format
        coms_wh.sort()
        scores:List[float] = []
        for w, h in coms_wh:
            try:
                scores.append(arr_[h, w])
            except:
                scores.append(-1.)
        if top_one:
            # get only one point with the largest score
            score, idx = torch.max(torch.tensor(scores), dim=0)
            coms_wh = [coms_wh[idx]]
            scores = [score]

        coms_wh_list.append(coms_wh)
        scores_list.append(scores)

    if return_scores:
        return coms_wh_list, scores_list
    return coms_wh_list


def modes_to_array(modes:List[List[Tuple]], scores:List[List[float]], max_detection=None):
    # assert len(modes) == 17 # 17 keypoints per instances => 17, num_inst(varying), (x, y)
    # assert len(scores) == 17 # 17 x num_modes(varying)
    # convert to array
    prod = list(itertools.product(*[[(*a, s) for a, s in zip(modes[i], scores[i])] for i in range(len(modes))]))
    if max_detection is not None:
        arr = np.zeros((len(modes), max_detection, 3), dtype=int)
        pred_score = np.zeros(max_detection)
    else:
        arr = np.zeros((len(modes), len(prod), 3), dtype=int)
        pred_score = np.zeros(len(prod))

    for d, kps in enumerate(prod):
        if max_detection is not None and d >= max_detection:
            break
        arr[:, d] = np.array(kps)
        pred_score[d] = arr[:, d, 2].mean()
        arr[:, d, 2] = 1

    return arr, pred_score


def ret_to_coco(all_ret, gt_coco, img_size=256):
    '''
    all_ret : dict {imgId: (17 x max_det x 3 ndarray, max_det score)}
    '''
    pred_dict = {}
    for imgId, (detection, scores) in all_ret.items():
        img = gt_coco.imgs[imgId]
        w = img['width']
        h = img['height']
        cat = gt_coco.anns[gt_coco.getAnnIds(imgIds=imgId)[0]]['category_id'] # assume one category for one img
        # convert size
        for d in range(detection.shape[1]):
            one_det = detection[:, d, :]  # 17 x 3
            # resize
            one_det[:, 0] = np.round(one_det[:, 0] * w / img_size)
            one_det[:, 1] = np.round(one_det[:, 1] * h / img_size)
            one_det = one_det.astype(int)
            res = {
                'image_id': imgId,
                'category_id': cat,
                'keypoints': one_det.reshape(-1).tolist(),
                'score': scores[d]
            }
            if imgId not in pred_dict:
                pred_dict[imgId] = []
            pred_dict[imgId].append(res)

    oks_thr = 0.9
    sigmas = np.array([
        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
        0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089
    ])
    valid_kpts = []
    for image_id in pred_dict.keys():
        img_kpts = pred_dict[image_id]
        for n_p in img_kpts:
            box_score = n_p['score']
            n_p['keypoints'] = np.array(n_p['keypoints']).reshape(-1, 3)
            kpt_score = 0
            valid_num = 0
            x_min = np.min(n_p['keypoints'][:, 0])
            x_max = np.max(n_p['keypoints'][:, 0])
            y_min = np.min(n_p['keypoints'][:, 1])
            y_max = np.max(n_p['keypoints'][:, 1])
            area = (x_max - x_min) * (y_max - y_min)
            n_p['area'] = int(area)
            valid_num = len(n_p['keypoints']) # assume all visible
            kpt_score = kpt_score / valid_num
            # rescoring
            n_p['score'] = float(kpt_score * box_score)
        keep = oks_nms(list(img_kpts), thr=oks_thr, sigmas=sigmas)
        valid_kpts.append([img_kpts[_keep] for _keep in keep])
    ret = []
    for each in valid_kpts:
        for det in each:
            det['keypoints'] = det['keypoints'].reshape(-1).astype(int).tolist()
            ret.append(det)

    return ret


def ret_to_coco_cropped(all_ret, gt_coco, img_size=256, extended=False):
    '''
    all_ret : dict {imgId: {annId: (17 x 3 ndarray, score)}}
    '''
    pred_dict = {imgId: {} for imgId in all_ret}
    for imgId in all_ret:
        for annId in all_ret[imgId]:
            if annId not in pred_dict[imgId]:
                pred_dict[imgId][annId] = []

            detection, scores = all_ret[imgId][annId]
            cat = gt_coco.anns[annId]['category_id']

            for d in range(detection.shape[1]):
                one_det = detection[:, d, :]  # 17 x 3

                # one_det[:, 2] = np.array(gt_coco.anns[annId]['keypoints']).reshape(-1, 3)[:, 2] # use gt visibility

                if extended:
                    W, H = gt_coco.imgs[imgId]['width'], gt_coco.imgs[imgId]['height']
                    x1, y1, w1, h1 = gt_coco.anns[annId]['bbox']
                    x2 = x1 + w1
                    y2 = y1 + h1

                    x = round(max(0, (x1 + x2)/2 - w1/2*1.2))
                    y = round(max(0, (y1 + y2)/2 - h1/2*1.2))
                    w = round(min(W - max(0, (x1 + x2) / 2 - w1/2*1.2), w1*1.2))
                    h = round(min(H - max(0, (y1 + y2) / 2 - h1/2*1.2), h1*1.2))
                else:
                    x, y, w, h = gt_coco.anns[annId]['bbox']

                # resize and shift
                one_det[:, 0] = np.round(one_det[:, 0] * w / img_size) + x
                one_det[:, 1] = np.round(one_det[:, 1] * h / img_size) + y
                
                one_det = one_det.astype(int)
                res = {
                    'id': annId,
                    'image_id': imgId,
                    'category_id': cat,
                    'keypoints': one_det.reshape(-1).tolist(),
                    'score': scores[d].item()
                }
                pred_dict[imgId][annId].append(res)

    oks_thr = 0.9
    sigmas = np.array([
        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
        0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089
    ])
    valid_kpts = []
    for image_id in pred_dict.keys():
        for ann_id in pred_dict[image_id]:
            img_kpts = pred_dict[image_id][ann_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                n_p['keypoints'] = np.array(n_p['keypoints']).reshape(-1, 3)
                kpt_score = 0
                valid_num = 0
                x_min = np.min(n_p['keypoints'][:, 0])
                x_max = np.max(n_p['keypoints'][:, 0])
                y_min = np.min(n_p['keypoints'][:, 1])
                y_max = np.max(n_p['keypoints'][:, 1])
                area = (x_max - x_min) * (y_max - y_min)
                n_p['area'] = int(area)
                valid_num = len(n_p['keypoints']) # assume all visible
                kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = float(kpt_score * box_score)
            keep = oks_nms(list(img_kpts), thr=oks_thr, sigmas=sigmas)
            valid_kpts.append([img_kpts[_keep] for _keep in keep])

    ret = []
    for each in valid_kpts:
        for det in each:
            det['keypoints'] = det['keypoints'].reshape(-1).astype(int).tolist()
            ret.append(det)

    return ret


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.
    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = [vg[i] > vis_thr and vd[i] > vis_thr for i in range(len(vg))]
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious


def oks_nms(kpts_db, thr=0.9, sigmas=None, vis_thr=None, score_per_joint=False):
    """OKS NMS implementations.
    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
        score_per_joint: the input scores (in kpts_db) are per joint scores
    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep


def dense_to_sparse(label):
    label_sparse = torch.zeros(len(label), 3, dtype=torch.float)
    for i in range(len(label)):
        label_sparse[i] = torch.FloatTensor((*get_modes(label[i], return_scores=False, top_one=True)[0][0], 1))
        
    return label_sparse


################# Keypoint Visualization Tools ####################
# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Keypoints(object):
    def __init__(self, keypoints, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)
        
        # TODO should I split them?
        # self.visibility = keypoints[..., 2]
        self.keypoints = keypoints# [..., :2]

        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        keypoints = type(self)(resized_data, size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                    "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = type(self).FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0

        keypoints = type(self)(flipped_data, self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def to(self, *args, **kwargs):
        keypoints = type(self)(self.keypoints.to(*args, **kwargs), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints

    def __getitem__(self, item):
        keypoints = type(self)(self.keypoints[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.keypoints))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


def _create_flip_indices(names, flip_map):
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


class PersonKeypoints(Keypoints):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }


class PersonMPIIKeypoints(Keypoints):
    NAMES = ['right_ankle', 'right_knee', 'right_hip', 'left_hip', 'left_knee', 'left_ankle',
             'pelvis', 'throax', 'upper_neck', 'head_top', 'right_wrist',
             'right_elbow', 'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist']
    FLIP_MAP = {
        'right_ankle': 'left_ankle',
        'right_knee': 'left_knee',
        'right_hip': 'left_hip',
        'right_ankle': 'left_ankle',
        'right_wrist': 'left_wrist',
        'right_elbow': 'left_elbow',
        'right_shoulder': 'left_shoulder',
    }


class FashionKeypoints(Keypoints):
    NAMES = ["right_neck", "left_neck", "right_arm", "left_arm",
             "right_waist", "left_waist", "right_leg", "left_leg"]
    FLIP_MAP = {
        'right_neck': 'left_neck',
        'right_arm': 'left_arm',
        'right_waist': 'left_waist',
        'right_leg': 'left_leg',
    }


class HandKeypoints(Keypoints):
    NAMES = ['wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4',
             'forefinger1', 'forefinger2', 'forefinger3', 'forefinger4',
             'middle_finger1', 'middle_finger2', 'middle_finger3', 'middle_finger4',
             'ring_finger1', 'ring_finger2', 'ring_finger3', 'ring_finger4',
             'pinky_finger1', 'pinky_finger2', 'pinky_finger3', 'pinky_finger4']
    FLIP_MAP = {}


class AnimalKeypoints(Keypoints):
    NAMES = [
        'left_eye',
        'right_eye',
        'nose',
        'neck',
        'root_of_tail',
        'left_shoulder',
        'left_elbow',
        'left_front_paw',
        'right_shoulder',
        'right_elbow',
        'right_front_paw',
        'left_hip',
        'left_knee',
        'left_back_paw',
        'right_hip',
        'right_knee',
        'right_back_paw',
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_front_paw': 'right_front_paw',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_back_paw': 'right_back_paw',
    }


PersonKeypoints.FLIP_INDS = _create_flip_indices(PersonKeypoints.NAMES, PersonKeypoints.FLIP_MAP)
PersonMPIIKeypoints.FLIP_INDS = _create_flip_indices(PersonMPIIKeypoints.NAMES, PersonMPIIKeypoints.FLIP_MAP)
FashionKeypoints.FLIP_INDS = _create_flip_indices(FashionKeypoints.NAMES, FashionKeypoints.FLIP_MAP)
HandKeypoints.FLIP_INDS = _create_flip_indices(HandKeypoints.NAMES, HandKeypoints.FLIP_MAP)
AnimalKeypoints.FLIP_INDS = _create_flip_indices(AnimalKeypoints.NAMES, AnimalKeypoints.FLIP_MAP)


def kp_connections_person(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


def kp_connections_person_mpii(keypoints):
    kp_lines = [
        [keypoints.index('right_ankle'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_wrist'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('upper_neck'), keypoints.index('head_top')],
        [keypoints.index('upper_neck'), keypoints.index('throax')],
        [keypoints.index('throax'), keypoints.index('pelvis')],
        [keypoints.index('throax'), keypoints.index('right_shoulder')],
        [keypoints.index('throax'), keypoints.index('left_shoulder')],
    ]
    return kp_lines


def kp_connections_fashion(keypoints):
    kp_lines = [
        [keypoints.index('right_neck'), keypoints.index('left_neck')],
        [keypoints.index('right_neck'), keypoints.index('right_arm')],
        [keypoints.index('right_neck'), keypoints.index('right_waist')],
        [keypoints.index('right_waist'), keypoints.index('right_leg')],
        [keypoints.index('left_neck'), keypoints.index('left_arm')],
        [keypoints.index('left_neck'), keypoints.index('left_waist')],
        [keypoints.index('left_waist'), keypoints.index('left_leg')],
        [keypoints.index('right_leg'), keypoints.index('left_leg')],
    ]
    return kp_lines


def kp_connections_hand(keypoints):
    kp_lines = [
        [keypoints.index('wrist'), keypoints.index('thumb1')],
        [keypoints.index('thumb1'), keypoints.index('thumb2')],
        [keypoints.index('thumb2'), keypoints.index('thumb3')],
        [keypoints.index('thumb3'), keypoints.index('thumb4')],
        [keypoints.index('wrist'), keypoints.index('forefinger1')],
        [keypoints.index('forefinger1'), keypoints.index('forefinger2')],
        [keypoints.index('forefinger2'), keypoints.index('forefinger3')],
        [keypoints.index('forefinger3'), keypoints.index('forefinger4')],
        [keypoints.index('wrist'), keypoints.index('middle_finger1')],
        [keypoints.index('middle_finger1'), keypoints.index('middle_finger2')],
        [keypoints.index('middle_finger2'), keypoints.index('middle_finger3')],
        [keypoints.index('middle_finger3'), keypoints.index('middle_finger4')],
        [keypoints.index('wrist'), keypoints.index('ring_finger1')],
        [keypoints.index('ring_finger1'), keypoints.index('ring_finger2')],
        [keypoints.index('ring_finger2'), keypoints.index('ring_finger3')],
        [keypoints.index('ring_finger3'), keypoints.index('ring_finger4')],
        [keypoints.index('wrist'), keypoints.index('pinky_finger1')],
        [keypoints.index('pinky_finger1'), keypoints.index('pinky_finger2')],
        [keypoints.index('pinky_finger2'), keypoints.index('pinky_finger3')],
        [keypoints.index('pinky_finger3'), keypoints.index('pinky_finger4')],
    ]
    return kp_lines


def kp_connections_animal(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('nose'), keypoints.index('neck')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_front_paw')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_front_paw')],
        [keypoints.index('neck'), keypoints.index('root_of_tail')],
        [keypoints.index('root_of_tail'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_back_paw')],
        [keypoints.index('root_of_tail'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_back_paw')],
    ]
    return kp_lines


PersonKeypoints.CONNECTIONS = kp_connections_person(PersonKeypoints.NAMES)
PersonMPIIKeypoints.CONNECTIONS = kp_connections_person_mpii(PersonMPIIKeypoints.NAMES)
FashionKeypoints.CONNECTIONS = kp_connections_fashion(FashionKeypoints.NAMES)
HandKeypoints.CONNECTIONS = kp_connections_hand(HandKeypoints.NAMES)
AnimalKeypoints.CONNECTIONS = kp_connections_animal(AnimalKeypoints.NAMES)


def vis_keypoints(img, kps, kp_thresh=0.5, alpha=0.7, lth=1, crad=1, object_type='person'):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (3, #keypoints) where 4 rows are (x, y, prob).
    """
    if object_type == 'person':
        dataset_keypoints = PersonKeypoints.NAMES
        kp_lines = PersonKeypoints.CONNECTIONS
    elif object_type == 'person_mpii':
        dataset_keypoints = PersonMPIIKeypoints.NAMES
        kp_lines = PersonMPIIKeypoints.CONNECTIONS
    elif object_type == 'fashion':
        dataset_keypoints = FashionKeypoints.NAMES
        kp_lines = FashionKeypoints.CONNECTIONS
    elif object_type == 'hand':
        dataset_keypoints = HandKeypoints.NAMES
        kp_lines = HandKeypoints.CONNECTIONS
    elif object_type == 'animal':
        dataset_keypoints = AnimalKeypoints.NAMES
        kp_lines = AnimalKeypoints.CONNECTIONS
    else:
        raise NotImplementedError

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    if object_type == 'person':
        mid_shoulder = (
            kps[:2, dataset_keypoints.index('right_shoulder')] +
            kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
        sc_mid_shoulder = np.minimum(
            kps[2, dataset_keypoints.index('right_shoulder')],
            kps[2, dataset_keypoints.index('left_shoulder')])
        mid_hip = (
            kps[:2, dataset_keypoints.index('right_hip')] +
            kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
        sc_mid_hip = np.minimum(
            kps[2, dataset_keypoints.index('right_hip')],
            kps[2, dataset_keypoints.index('left_hip')])
    
        mid_shoulder = mid_shoulder.astype(np.int64)
        mid_hip = mid_hip.astype(np.int64)
        
    kps = kps.astype(np.int64)
    
    if object_type == 'person':
        nose_idx = dataset_keypoints.index('nose')
        if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
                color=colors[len(kp_lines)], thickness=lth, lineType=cv2.LINE_AA)
        if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder), tuple(mid_hip),
                color=colors[len(kp_lines) + 1], thickness=lth, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=lth, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=crad, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=crad, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    vis = (1 - alpha) * img + kp_mask * alpha
    return vis


### custom fix of torchvision.utils.draw_keypoints
@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
    visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    """
    Draws Keypoints on given RGB image.
    The image values should be uint8 in [0, 255] or float in [0, 1].
    Keypoints can be drawn for multiple instances at a time.

    This method allows that keypoints and their connectivity are drawn based on the visibility of this keypoint.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8 or float.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoint locations for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where each tuple contains a pair of keypoints
            to be connected.
            If at least one of the two connected keypoints has a ``visibility`` of False,
            this specific connection is not drawn.
            Exclusions due to invisibility are computed per-instance.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.
        visibility (Tensor): Tensor of shape (num_instances, K) specifying the visibility of the K
            keypoints for each of the N instances.
            True means that the respective keypoint is visible and should be drawn.
            False means invisible, so neither the point nor possible connections containing it are drawn.
            The input tensor will be cast to bool.
            Default ``None`` means that all the keypoints are visible.
            For more details, see :ref:`draw_keypoints_with_visibility`.

    Returns:
        img (Tensor[C, H, W]): Image Tensor with keypoints drawn.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(draw_keypoints)
    # validate image
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    # validate keypoints
    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    # validate visibility
    if visibility is None:  # set default
        visibility = torch.ones(keypoints.shape[:-1], dtype=torch.bool)
    # If the last dimension is 1, e.g., after calling split([2, 1], dim=-1) on the output of a keypoint-prediction
    # model, make sure visibility has shape (num_instances, K).
    # Iff K = 1, this has unwanted behavior, but K=1 does not really make sense in the first place.
    if visibility.ndim != 2:
        raise ValueError(f"visibility must be of shape (num_instances, K). Got ndim={visibility.ndim}")
    if visibility.shape != keypoints.shape[:-1]:
        raise ValueError(
            "keypoints and visibility must have the same dimensionality for num_instances and K. "
            f"Got {visibility.shape = } and {keypoints.shape = }"
        )

    original_dtype = image.dtype
    if original_dtype.is_floating_point:
        from torchvision.transforms.v2.functional import to_dtype  # noqa

        image = to_dtype(image, dtype=torch.uint8, scale=True)

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()
    img_vis = visibility.cpu().bool().tolist()

    for kpt_inst, vis_inst in zip(img_kpts, img_vis):
        for kpt_coord, kp_vis in zip(kpt_inst, vis_inst):
            if not kp_vis:
                continue
            x1 = kpt_coord[0] - radius
            x2 = kpt_coord[0] + radius
            y1 = kpt_coord[1] - radius
            y2 = kpt_coord[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                if (not vis_inst[connection[0]]) or (not vis_inst[connection[1]]):
                    continue
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    out = torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)
    if original_dtype.is_floating_point:
        out = to_dtype(out, dtype=original_dtype, scale=True)
    return out

### DAVIS2017 Evaluation
def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


### LINEMOD 6D Pose
fx = 572.41140
px = 325.26110
fy = 573.57043
py = 242.04899
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


class MeshModel:
    def __init__(self, model, scale=10, transform=None):
        self.vertices = None
        self.indices = None
        self.colors = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.diameter = None
        self.scale = scale
        self.transform = transform
        self.frames = []
        self.load(model)

    def load(self, model):
        ply = plyfile.PlyData.read(model)
        self.vertices = np.zeros((ply['vertex'].count, 3))
        self.vertices[:, 0] = np.array(ply['vertex']['x']) / self.scale
        self.vertices[:, 1] = np.array(ply['vertex']['y']) / self.scale
        self.vertices[:, 2] = np.array(ply['vertex']['z']) / self.scale
        if self.transform is not None:
            self.vertices = np.dot(self.vertices, self.transform[:, :3].T) + self.transform[:, 3]

        self.bb = []
        self.minx, self.maxx = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        self.miny, self.maxy = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        self.minz, self.maxz = min(self.vertices[:, 2]), max(self.vertices[:, 2])
       
        self.bb.append([self.minx, self.miny, self.minz])
        self.bb.append([self.minx, self.maxy, self.minz])
        self.bb.append([self.minx, self.miny, self.maxz])
        self.bb.append([self.minx, self.maxy, self.maxz])
        self.bb.append([self.maxx, self.miny, self.minz])
        self.bb.append([self.maxx, self.maxy, self.minz])
        self.bb.append([self.maxx, self.miny, self.maxz])
        self.bb.append([self.maxx, self.maxy, self.maxz])
        self.bb = np.asarray(self.bb, dtype=np.float32)
        self.diameter = max(scipy.spatial.distance.pdist(self.bb, 'euclidean'))
        self.colors = np.zeros((ply['vertex'].count, 3))
        self.colors[:, 0] = normalize(self.vertices[:, 0]) * 255
        self.colors[:, 1] = normalize(self.vertices[:, 1]) * 255
        self.colors[:, 2] = normalize(self.vertices[:, 2]) * 255


def load_mesh_model(model_path):
    if os.path.exists(model_path):
        model = MeshModel(model_path)
    else:
        transform_path = model_path.replace('mesh.ply', 'transform.dat')
        model_path = model_path.replace('mesh.ply', 'OLDmesh.ply')
        transform = np.loadtxt(transform_path, skiprows=1)[:, 1]
        transform = np.reshape(transform, newshape=[3, 4])
        transform[:, 3] *= 100
        model = MeshModel(model_path, transform=transform)

    return model


def texture_to_6dpose(Y, model, coord=None):
    if Y.ndim == 4:
        if coord is not None:
            return np.stack([texture_to_6dpose(y, model, c) for y, c in zip(Y, coord)])
        else:
            return np.stack([texture_to_6dpose(y, model) for y in Y])

    Y = torch.where(Y[:1].bool(), Y[1:], torch.zeros_like(Y[1:])) 
    Y = Y.permute(1, 2, 0).cpu().numpy()
    pose = predict_pose_uvw(Y, model, coord)

    return pose


def predict_pose_uvw(uvw_region, model, coord=None, return_inliers=False):
    """
    Predict pose given UVW correspondences
    Args:
        cam: camera matrix
        uvw_region: UVW region
        model: object model
        return_inliers: bool to return inliers

    Returns: estimated pose

    """
    nonzero_mask = uvw_region[:, :, 0] > 0
    uvw_values = uvw_region[nonzero_mask]

    uvw_region_u = uvw_values[:, 0] * (model.maxx - model.minx) + model.minx
    uvw_region_v = uvw_values[:, 1] * (model.maxy - model.miny) + model.miny
    uvw_region_w = uvw_values[:, 2] * (model.maxz - model.minz) + model.minz
    points_3d = np.stack([uvw_region_u, uvw_region_v, uvw_region_w], axis=1)

    grid_row, grid_column = np.nonzero(nonzero_mask.astype(np.int64))

    image_points = np.empty((len(grid_row), 2))
    image_points[:, 0] = grid_row
    image_points[:, 1] = grid_column
    
    if coord is not None:
        image_points[:, 0] = (image_points[:, 0].astype(np.float64) / uvw_region.shape[0] * (coord[3] - coord[1]))
        image_points[:, 1] = (image_points[:, 1].astype(np.float64) / uvw_region.shape[1] * (coord[2] - coord[0]))
        image_points[:, 0] = (image_points[:, 0].astype(np.float64) + coord[1])
        image_points[:, 1] = (image_points[:, 1].astype(np.float64) + coord[0])

    object_points = points_3d

    if return_inliers:
        predicted_pose, n_inliers = solvePnP(intrinsic_matrix, image_points, object_points, return_inliers)
        predicted_pose = predicted_pose[:3]
        return predicted_pose, n_inliers
    else:
        predicted_pose = solvePnP(intrinsic_matrix, image_points, object_points, return_inliers)
        predicted_pose = predicted_pose[:3]
        return predicted_pose


def solvePnP(cam, image_points, object_points, return_inliers=False, ransac_iter=250):
    """
    Solve PnP problem using resulting correspondences
    Args:
        cam: Camera matrix
        image_points: Correspondence points on the image
        object_points: Correspondence points on the model
        return_inliers: Bool for inliers return
        ransac_iter: Number of RANSAC iterations

    Returns: Resulting object pose (+ number of inliers)

    """
    dist_coeffs = None  # Assuming no lens distortion
    if image_points.shape[0] < 4:
        pose = np.eye(4)
        inliers = []
    else:
        image_points[:, [0, 1]] = image_points[:, [1, 0]]
        object_points = np.expand_dims(object_points, 1)
        image_points = np.expand_dims(image_points, 1)

        try:
            success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points.astype(float), cam,
                                                                                       dist_coeffs, iterationsCount=ransac_iter,
                                                                                       reprojectionError=1.)
        except:
            success = False
            inliers = None

        # Get a rotation matrix
        pose = np.eye(4)
        if success:
            pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            pose[:3, 3] = np.squeeze(translation_vector)

        if inliers is None:
            inliers = []

    if return_inliers:
        return pose, len(inliers)
    else:
        return pose


def create_bounding_box(img, pose, pt_cld_data, color=(1, 0, 0), thickness=1, base_size=(480, 640), coord=None):
    "Create a bounding box around the object"
    img = img.copy()
    if (pose == 0).all():
        return img
    
    # 8 corner points of the ptcld data
    min_x, min_y, min_z = pt_cld_data.min(axis=0)
    max_x, max_y, max_z = pt_cld_data.max(axis=0)
    corners_3D = np.array([[max_x, min_y, min_z],
                           [max_x, min_y, max_z],
                           [min_x, min_y, max_z],
                           [min_x, min_y, min_z],
                           [max_x, max_y, min_z],
                           [max_x, max_y, max_z],
                           [min_x, max_y, max_z],
                           [min_x, max_y, min_z]])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    if coord is not None and coord.astype('bool').any():
        coord_2D[0] -= coord[0]
        coord_2D[1] -= coord[1]
        coord_2D[0] = coord_2D[0] * img.shape[1] / (coord[2] - coord[0])
        coord_2D[1] = coord_2D[1] * img.shape[0] / (coord[3] - coord[1])
    else:
        coord_2D[0] = coord_2D[0] * img.shape[1] / base_size[1]
        coord_2D[1] = coord_2D[1] * img.shape[0] / base_size[0]
    
    thres = 2e3
    coord_2D = coord_2D.clip(-thres, thres)
    coord_2D = ((np.floor(coord_2D)).T).astype(int)
    
    # pad images if the 2D coordinates are outside the image
    pad_w1 = -min(0, coord_2D[:, 0].min())
    pad_w2 = max(img.shape[1], coord_2D[:, 0].max()) - img.shape[1]
    pad_h1 = -min(0, coord_2D[:, 1].min())
    pad_h2 = max(img.shape[0], coord_2D[:, 1].max()) - img.shape[0]
    img = cv2.copyMakeBorder(img, pad_h1, pad_h2, pad_w1, pad_w2, cv2.BORDER_CONSTANT, value=0)
    coord_2D[:, 0] += pad_w1
    coord_2D[:, 1] += pad_h1

    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[4]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[2]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[5]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[3]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[6]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[3]), tuple(coord_2D[7]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[7]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[5]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[5]), tuple(coord_2D[6]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[6]), tuple(coord_2D[7]), color, thickness)

    # crop the image to remove the padding
    img = img[pad_h1:img.shape[0] - pad_h2, pad_w1:img.shape[1] - pad_w2]

    return img


def vis_pose(label, img, mesh_model, coord=None, thickness=1):
    if label.ndim == 3:
        assert img.ndim == 4
        if coord is not None:
            assert coord.ndim == 2
            return torch.stack([vis_pose(label_, img_, mesh_model, coord_, thickness=thickness) for label_, img_, coord_ in zip(label, img, coord)])
        else:
            return torch.stack([vis_pose(label_, img_, mesh_model, thickness=thickness) for label_, img_ in zip(label, img)])
    
    img = img.permute(1, 2, 0).numpy()
    vis = create_bounding_box(img, label, mesh_model.vertices, coord=coord, thickness=thickness)
    vis = torch.from_numpy(vis).permute(2, 0, 1).float()
    return vis


def ADD_score(pt_cld, true_pose, pred_pose, diameter):
    "Evaluation metric - ADD score"
    pred_pose[0:3, 0:3][np.isnan(pred_pose[0:3, 0:3])] = 1
    pred_pose[:, 3][np.isnan(pred_pose[:, 3])] = 0
    target = pt_cld @ true_pose[0:3, 0:3] + np.array(
        [true_pose[0, 3], true_pose[1, 3], true_pose[2, 3]])
    output = pt_cld @ pred_pose[0:3, 0:3] + np.array(
        [pred_pose[0, 3], pred_pose[1, 3], pred_pose[2, 3]])
    avg_distance = np.linalg.norm(output - target, axis=1).mean()
    threshold = diameter * 0.1
    if avg_distance <= threshold:
        return 1., avg_distance
    else:
        return 0., avg_distance
    
