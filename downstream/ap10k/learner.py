from einops import rearrange, reduce
import numpy as np
import torch
import torch.nn.functional as F
import os

from ..base_learner import BaseLearner
from .dataset import AP10KDataset
from .utils import dense_to_sparse, vis_animal_keypoints, get_modes, modes_to_array
from .evaluator import AP10KEvaluator
from train.loss import spatial_softmax_loss


class AP10KLearner(BaseLearner):
    BaseDataset = AP10KDataset

    def register_evaluator(self):
        self.kp_classes = AP10KDataset.CLASS_NAMES
        self.kp_json_path = {}
        self.ann_path = {}

        if self.config.stage == 1:
            keys = ['mtest_train', 'mtest_valid']
        else:
            keys = ['mtest_test']
        self.evaluator = {key: None for key in keys}

        for key in keys:
            split = key.split('_')[1]
            kp_json_path = os.path.join(self.result_dir, f'{key}_temp.json')
            ann_path = os.path.join(self.config.path_dict[self.config.dataset], 'annotations',
                                    f'ap10k-{split.replace("valid", "val")}-split1.json')
            self.evaluator[key] = AP10KEvaluator(kp_json_path, ann_path, self.config.base_size, self.local_rank, self.n_devices)

    def reset_evaluator(self):
        for key in self.evaluator.keys():
            self.evaluator[key].reset()

    def compute_loss(self, Y_pred, Y, M):
        '''
        loss function that returns loss and a dictionary of its components
        '''
        loss = spatial_softmax_loss(Y_pred, Y, M, reduction='mean', scaled=self.config.scale_ssl)
        loss_values = {'loss': loss.detach(), 'loss_ssl': loss.detach()}

        return loss, loss_values

    def postprocess_logits(self, Y_pred_out):
        '''
        post-processing function for logits
        '''
        # spatial softmax
        H, W = Y_pred_out.shape[-2:]
        Y_pred_out = rearrange(Y_pred_out, '1 T N C H W -> 1 T N C (H W)')
        Y_pred_out = F.softmax(Y_pred_out, dim=-1)
        Y_pred_out = rearrange(Y_pred_out, '1 T N C (H W) -> 1 T N C H W', H=H, W=W)
        Y_pred_out = Y_pred_out / (1e-18 + reduce(Y_pred_out, '1 T N C H W -> 1 T N C 1 1', 'max'))
        return Y_pred_out
    
    def postprocess_vis(self, label, img=None, aux=None):
        '''
        post-processing function for visualization
        '''
        if label.ndim == 3:
            sparse = True
        else:
            sparse = False
        sparse_gt, _ = aux
        
        label_vis = []
        for i in range(len(label)):
            img_ = np.ascontiguousarray((img[i]*128).byte().permute(1, 2, 0).numpy())
            if sparse:
                kps = label[i].transpose(0, 1).numpy()
            else:
                kps = dense_to_sparse(label[i]).transpose(0, 1).numpy()
                if sparse_gt is not None:
                    kps[2] = sparse_gt[i, :, 2].float().cpu().numpy()
            vis = vis_animal_keypoints(img_, kps)
            vis = torch.from_numpy(vis).permute(2, 0, 1) / 255
            label_vis.append(vis)
        label = torch.stack(label_vis)

        return label

    def compute_metric(self, Y, Y_pred, M, aux, evaluator_key):
        '''
        compute evaluation metric
        '''
        metric = 0
        assert aux is not None
        assert evaluator_key is not None
        evaluator = self.evaluator[evaluator_key]

        _, (imgIds, annIds) = aux
        for i in range(len(Y_pred)):
            modes, scores = get_modes(Y_pred[i].cpu(), return_scores=True)
            arr, score = modes_to_array(modes, scores, max_detection=1)
            if imgIds[i].item() not in evaluator.all_ret:
                evaluator.all_ret[imgIds[i].item()] = {}
            evaluator.all_ret[imgIds[i].item()][annIds[i].item()] = (arr, score)
        return metric

    def log_metrics(self, loss_pred, log_dict, valid_tag):
        '''
        log evaluation metrics
        '''
        coco_eval = self.evaluator[valid_tag].evaluate_keypoints()
        ap = coco_eval.stats[0] if coco_eval is not None else 0
        if self.n_devices > 1:
            ap = self.trainer.all_gather(torch.tensor(ap, device=self.trainer.device))[0]
        log_dict[f'{valid_tag}/{self.vis_tag}_AP_inverted'] = 1 - ap

    def get_test_metrics(self, metrics_total):
        '''
        save test metrics
        '''
        coco_eval = self.evaluator['mtest_test'].evaluate_keypoints()
        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]
        metrics = list(zip(stats_names, coco_eval.stats))
        return metrics
    