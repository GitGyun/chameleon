import torch
import torch.nn.functional as F
from einops import rearrange
import flow_vis

from ..base_learner import BaseLearner
from .dataset import CELLPOSEDataset
from .utils import compute_masks, average_precision


class CELLPOSELearner(BaseLearner):
    BaseDataset = CELLPOSEDataset

    def compute_loss(self, Y_pred, Y, M):
        '''
        loss function that returns loss and a dictionary of its components
        '''
        loss = (M * F.binary_cross_entropy_with_logits(Y_pred, Y, reduction='none')).mean()
        loss_values = {'loss': loss.detach(), 'loss_bce': loss.detach()}
        return loss, loss_values
    
    def postprocess_logits(self, Y_pred_out):
        '''
        post-processing function for logits
        '''
        Y_pred_out = Y_pred_out.sigmoid()
        return Y_pred_out
    
    def postprocess_vis(self, label, img=None, aux=None):
        '''
        post-processing function for visualization
        '''
        vis = []
        label_vis = label.clone()
        label_vis[:, :2] = label_vis[:, :2] * 2 - 1
        label_vis[:, :2] = label_vis[:, :2].clip(-1, 1)
        for i in range(len(label)):
            vis_ = flow_vis.flow_to_color(rearrange(label_vis[i, :2] * (label_vis[i, 2:3] > 0.5).float(),  'C H W -> H W C').numpy())
            vis_ = torch.from_numpy(vis_/255)
            vis += [rearrange(vis_, 'H W C -> C H W')]
            
        label = torch.stack(vis)
        
        return label

    def compute_metric(self, Y, Y_pred, M, aux, evaluator_key=None):
        '''
        compute evaluation metric
        '''
        Y = Y.clone()
        Y_pred = Y_pred.clone()
        Y[:, :2] = Y[:, :2] * 2 - 1
        Y_pred[:, :2] = Y_pred[:, :2] * 2 - 1
        
        full_res = torch.stack(aux["full_res"]).permute(1, 0).long().cpu().numpy()
        full_mask = [mask[..., :h, :w] for mask, (h, w) in zip(aux["full_mask"], full_res)] 
        
        Y_full = [full_mask_[0].data.cpu().numpy() for full_mask_ in full_mask]
        flow = Y_pred[:, :2].cpu().numpy()
        prob_mask = Y_pred[:, -1].cpu().numpy()
        mask_preds = []
        for i in range(len(Y_full)):
            mask_pred, _ = compute_masks(5 * flow[i], prob_mask[i], cellprob_threshold=0.5,
                                         flow_threshold=0., resize=Y_full[i].shape[-2:], use_gpu=True)
            mask_preds += [mask_pred.astype('int32')]
        
        ap, tp, fp, fn = average_precision(Y_full, mask_preds, threshold=0.5)
        metric = torch.tensor(ap.mean()).to(Y.device)

        return metric

    def log_metrics(self, loss_pred, log_dict, valid_tag):
        '''
        log evaluation metrics
        '''
        log_dict[f'{valid_tag}/{self.vis_tag}_AP50_inverted'] = 1 - loss_pred
