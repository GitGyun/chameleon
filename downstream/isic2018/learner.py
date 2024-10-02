import torch
import torch.nn.functional as F
import os
from skimage import color
from einops import reduce
import numpy as np

from ..base_learner import BaseLearner
from .dataset import ISIC2018Dataset
from train.miou_fss import AverageMeter, Evaluator
from train.visualize import postprocess_semseg


class ISIC2018Learner(BaseLearner):
    BaseDataset = ISIC2018Dataset
    
    def register_evaluator(self):
        if self.config.stage == 1:
            keys = ['mtest_train', 'mtest_valid']
        else:
            keys = ['mtest_test']

        self.evaluator = {key: None for key in keys}
        for key in keys:
            self.evaluator[key] = AverageMeter(class_ids_interest=[0], semseg_classes=[0], device=torch.device(f'cuda:{self.local_rank}'))
        self.result_path = self.result_path.replace('.pth', f'_sid:{self.config.support_idx}.pth')

    def reset_evaluator(self):
        for key in self.evaluator:
            self.evaluator[key].reset()

    def compute_loss(self, Y_pred, Y, M):
        '''
        loss function that returns loss and a dictionary of its components
        '''
        loss = (M * F.binary_cross_entropy_with_logits(Y_pred, Y, reduction='none')).mean()
        loss_values = {'loss': loss.detach(), 'loss_bce': loss.detach()}
        return loss, loss_values
    
    def postprocess_final(self, Y_pred):
        Y_pred = (Y_pred.sigmoid() > 0.5).squeeze(1).to(Y_pred.dtype)
        return Y_pred
    
    def postprocess_vis(self, label, img=None, aux=None):
        '''
        post-processing function for visualization
        '''
        label = postprocess_semseg(label, img, aux)
        return label

    def compute_metric(self, Y, Y_pred, M, aux, evaluator_key=None):
        '''
        compute evaluation metric
        '''
        Y_full = aux
        evaluator = self.evaluator[evaluator_key]

        if Y_pred.ndim == 3:
            Y_pred = Y_pred.unsqueeze(1)
        assert Y_full.ndim == 4
        if Y_pred.shape[-2:] != (512, 512):
            Y_pred = F.interpolate(Y_pred.float(), (512, 512), mode='nearest')
        area_inter, area_union = Evaluator.classify_prediction(Y_pred.clone().float(), Y_full.float().round())
        class_id = torch.tensor([0]*len(Y_pred), device=Y_full.device) # use 0 for all classes
        area_inter = area_inter.to(Y_full.device)
        area_union = area_union.to(Y_full.device)
        evaluator.update(area_inter, area_union, class_id)
        metric = 0

        return metric

    def log_metrics(self, loss_pred, log_dict, valid_tag):
        '''
        log evaluation metrics
        '''
        evaluator = self.evaluator[valid_tag]
        if self.n_devices > 1:
            evaluator.intersection_buf = reduce(self.trainer.all_gather(evaluator.intersection_buf), 'G ... -> ...', 'sum')
            evaluator.union_buf = reduce(self.trainer.all_gather(evaluator.union_buf), 'G ... -> ...', 'sum')
        intersection = evaluator.intersection_buf.float()
        union = evaluator.union_buf.float()
        f1 = 2*intersection / torch.max(torch.stack([union + intersection, evaluator.ones]), dim=0)[0]
        log_dict[f'{valid_tag}/{self.vis_tag}_F1_inverted'] = 1 - f1[1, 0]

    def get_test_metrics(self, metrics_total):
        '''
        save test metrics
        '''
        evaluator = self.evaluator['mtest_test']
        if self.n_devices > 1:
            evaluator.intersection_buf = reduce(self.all_gather(evaluator.intersection_buf), 'G ... -> ...', 'sum')
            evaluator.union_buf = reduce(self.all_gather(evaluator.union_buf), 'G ... -> ...', 'sum')
        intersection = evaluator.intersection_buf.float()
        union = evaluator.union_buf.float()
        dsc = 2*intersection / torch.max(torch.stack([union + intersection, evaluator.ones]), dim=0)[0]
        dsc = dsc[1, 0].item()
        iou = evaluator.compute_iou()[0].cpu().item()
        metric = [dsc, iou]
        
        return metric