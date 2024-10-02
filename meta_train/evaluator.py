import torch
import numpy as np
import json
import deepspeed
from dataset.coco_api_wrapper import SilentXTCOCO as COCO
from dataset.coco_api_wrapper import SilentXTCOCOeval as COCOeval
from dataset.utils import oks_nms


class KeypointEvaluator:
    def __init__(self, config, kp_json_path, ann_path, sigmas, base_size=(256, 256), local_rank=0, n_devices=1, trainer=None):
        self.config = config
        self.sigmas = sigmas
        self.base_size = base_size
        self.kp_json_path = kp_json_path
        self.ann_path = ann_path
        self.all_ret = {}
        self.local_rank = local_rank
        self.n_devices = n_devices
        self.trainer = trainer

    def reset(self):
        self.all_ret = {}

    def ret_to_coco(self, gt_coco):
        '''
        all_ret : dict {imgId: (17 x max_det x 3 ndarray, max_det score)}
        '''
        pred_dict = {}
        for imgId, (detection, scores) in self.all_ret.items():
            img = gt_coco.imgs[imgId]
            w = img['width']
            h = img['height']
            cat = gt_coco.anns[gt_coco.getAnnIds(imgIds=imgId)[0]]['category_id'] # assume one category for one img
            # convert size
            for d in range(detection.shape[1]):
                one_det = detection[:, d, :]  # 17 x 3
                # resize
                one_det[:, 0] = np.round(one_det[:, 0] * w / self.base_size[1])
                one_det[:, 1] = np.round(one_det[:, 1] * h / self.base_size[0])
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
        sigmas = np.array(self.sigmas)
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

    def ret_to_coco_cropped(self, gt_coco):
        '''
        self.all_ret : dict {imgId: {annId: (17 x 3 ndarray, score)}}
        '''
        pred_dict = {imgId: {} for imgId in self.all_ret}
        for imgId in self.all_ret:
            for annId in self.all_ret[imgId]:
                if annId not in pred_dict[imgId]:
                    pred_dict[imgId][annId] = []

                detection, scores = self.all_ret[imgId][annId]
                cat = gt_coco.anns[annId]['category_id']

                for d in range(detection.shape[1]):
                    one_det = detection[:, d, :]  # 17 x 3

                    W, H = gt_coco.imgs[imgId]['width'], gt_coco.imgs[imgId]['height']
                    x1, y1, w1, h1 = gt_coco.anns[annId]['bbox']
                    x2 = x1 + w1
                    y2 = y1 + h1

                    x = round(max(0, (x1 + x2)/2 - w1/2*1.2))
                    y = round(max(0, (y1 + y2)/2 - h1/2*1.2))
                    w = round(min(W - max(0, (x1 + x2) / 2 - w1/2*1.2), w1*1.2))
                    h = round(min(H - max(0, (y1 + y2) / 2 - h1/2*1.2), h1*1.2))

                    # resize and shift
                    one_det[:, 0] = np.round(one_det[:, 0] * w / self.base_size[1]) + x
                    one_det[:, 1] = np.round(one_det[:, 1] * h / self.base_size[0]) + y
                    
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
        sigmas = np.array(self.sigmas)
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

    def evaluate_keypoints(self):
        gt_coco = COCO(self.ann_path)
        ret_coco = self.ret_to_coco_cropped(gt_coco)

        # save coco result
        with open(self.kp_json_path.replace('.json', f'_{self.local_rank}.json'), 'w') as f:
            json.dump(ret_coco, f)

        # synchronize at this point
        if self.n_devices > 1:
            torch.distributed.barrier()

        if self.local_rank == 0:
            for local_rank in range(1, self.n_devices):
                with open(self.kp_json_path.replace('.json', f'_{local_rank}.json'), 'r') as f:
                    ret_coco += json.load(f)
            with open(self.kp_json_path, 'w') as f:
                json.dump(ret_coco, f)

            imgIds = [ret['image_id'] for ret in ret_coco]
            gt_coco.imgs = {k: v for k, v in gt_coco.imgs.items() if k in imgIds}
            annIds = [ret['id'] for ret in ret_coco]
            gt_coco.anns = {k: v for k, v in gt_coco.anns.items() if k in annIds}

            # evaluate
            coco_det = gt_coco.loadRes(self.kp_json_path)
            sigmas = np.array(self.sigmas)
            coco_eval = COCOeval(gt_coco, coco_det, 'keypoints', sigmas=sigmas)
            coco_eval.params.useSegm = None
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        else:
            coco_eval = None

        return coco_eval
    
    def evaluate_keypoints(self, cropped):
        gt_coco = COCO(self.ann_path)
        if cropped:
            ret_coco = self.ret_to_coco_cropped(gt_coco)
        else:
            ret_coco = self.ret_to_coco(gt_coco)

        # save coco result
        with open(self.kp_json_path.replace('.json', f'_{self.local_rank}.json'), 'w') as f:
            json.dump(ret_coco, f)

        # synchronize at this point
        if self.n_devices > 1:
            if self.config.strategy == 'deepspeed':
                deepspeed.comm.barrier()
            else:
                torch.distributed.barrier()

        if self.local_rank == 0:
            for local_rank in range(1, self.n_devices):
                with open(self.kp_json_path.replace('.json', f'_{local_rank}.json'), 'r') as f:
                    ret_coco += json.load(f)
            with open(self.kp_json_path, 'w') as f:
                json.dump(ret_coco, f)

            imgIds = [ret['image_id'] for ret in ret_coco]
            gt_coco.imgs = {k: v for k, v in gt_coco.imgs.items() if k in imgIds}
            if cropped:
                annIds = [ret['id'] for ret in ret_coco]
                gt_coco.anns = {k: v for k, v in gt_coco.anns.items() if k in annIds}
            else:
                gt_coco.anns = {k: v for k, v in gt_coco.anns.items() if v['image_id'] in imgIds}

            # evaluate
            coco_det = gt_coco.loadRes(self.kp_json_path)
            sigmas = np.array(self.sigmas)
            coco_eval = COCOeval(gt_coco, coco_det, 'keypoints', sigmas=sigmas)
            coco_eval.params.useSegm = None
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap = coco_eval.stats[0]
        else:
            ap = 0
            coco_eval = None

        if self.n_devices > 1:
            ap = self.trainer.all_gather(torch.tensor(ap, device=self.trainer.device))[0]

        return ap
