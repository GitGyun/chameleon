import torch
import torch.nn.functional as F

from skimage import color
import os

from ..base_learner import BaseLearner
from .dataset import DAVIS2017Dataset
from .utils import db_eval_iou, db_eval_boundary
from train.visualize import postprocess_semseg


class DAVIS2017Learner(BaseLearner):
    BaseDataset = DAVIS2017Dataset

    def compute_loss(self, Y_pred, Y, M):
        '''
        cross-entropy loss with implicit background class
        '''
        Y_bg = torch.zeros_like(Y[:, :1])
        Y = torch.argmax(torch.cat((Y_bg, Y), dim=1), dim=1)

        Y_pred_bg = torch.zeros_like(Y_pred[:, :1])
        Y_pred = torch.cat((Y_pred_bg, Y_pred), dim=1)

        loss = (M * F.cross_entropy(Y_pred, Y, reduction='none')).mean()
        loss_values = {'loss': loss.detach(), 'loss_ce': loss.detach()}

        return loss, loss_values

    def postprocess_final(self, Y_pred):
        '''
        post-processing function for final prediction
        '''
        if Y_pred.shape[1] == 1:
            Y_pred = (Y_pred.sigmoid() > 0.5).squeeze(1).to(Y_pred.dtype)
        else:
            Y_logits_bg = torch.zeros_like(Y_pred[:, :1])
            Y_logits = torch.cat((Y_logits_bg, Y_pred), dim=1)
            Y_pred = torch.argmax(Y_logits, dim=1).to(Y_pred.dtype)

        return Y_pred
    
    def postprocess_vis(self, label, img=None, aux=None):
        '''
        post-processing function for visualization
        '''
        label = postprocess_semseg(label, img, aux)
        return label

    def compute_metric(self, Y, Y_pred, M, aux, evaluator_key=None):
        '''
        J&F metric
        '''
        if Y.shape[1] > 1:
            Y = torch.argmax(torch.cat((0.5*torch.ones_like(Y[:, :1]), Y), dim=1), dim=1)
        else:
            Y = Y.squeeze(1).long()
        j_metric = db_eval_iou(Y.cpu().float().numpy(), Y_pred.cpu().float().numpy())
        f_metric = db_eval_boundary(Y.cpu().float().numpy(), Y_pred.cpu().float().numpy())
        metric = torch.tensor((j_metric + f_metric).mean() / 2).to(Y.device)

        return metric

    def log_metrics(self, loss_pred, log_dict, valid_tag):
        '''
        log inverted J&F metric
        '''
        log_dict[f'{valid_tag}/{self.vis_tag}_J&F_inverted'] = 1 - loss_pred

    def save_test_outputs(self, Y_pred, batch_idx):
        if self.config.class_name == 'bike-packing':
            target_size = (480, 910)
        elif self.config.class_name == 'shooting':
            target_size = (480, 1152)
        else:
            target_size = (480, 854)
        Y_pred = F.interpolate(Y_pred[:, None], target_size, mode='nearest')[:, 0]

        for i in range(len(Y_pred)):
            img = self.topil(Y_pred[i].cpu().byte())
            img.save(os.path.join(self.result_dir, f'{batch_idx*self.config.eval_batch_size+i:05d}.png'))
