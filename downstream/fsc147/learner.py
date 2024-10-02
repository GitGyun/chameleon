import torch
import torch.nn.functional as F
from einops import rearrange

from ..base_learner import BaseLearner
from .dataset import FSC147Dataset
from .utils import preprocess_kpmap, make_density_tensor, viridis


class FSC147Learner(BaseLearner):
    BaseDataset = FSC147Dataset

    def register_evaluator(self):
        if self.config.stage == 1:
            keys = ['mtest_train', 'mtest_valid']
        else:
            keys = ['mtest_test']
        self.evaluator = {key: None for key in keys}

        for key in keys:
            self.evaluator[key] = []

    def reset_evaluator(self):
        for key in self.evaluator.keys():
            self.evaluator[key] = []

    def compute_loss(self, Y_pred, Y, M):
        '''
        loss function that returns loss and a dictionary of its components
        '''
        loss = (M * F.mse_loss(Y_pred.sigmoid(), Y, reduction='none')).mean()
        loss_values = {'loss': loss.detach(), 'loss_con': loss.detach()}
        return loss, loss_values
    
    def postprocess_logits(self, Y_pred_out):
        Y_pred_out = Y_pred_out.sigmoid()
        return Y_pred_out
    
    def postprocess_vis(self, label, img=None, aux=None):
        '''
        post-processing function for visualization
        '''
        # process label by finding modes
        width = label.size(-1) // 64
        if width %2 == 1:
            width += 1

        label = label.clip(0, 1)
        label_vis = torch.stack([make_density_tensor(preprocess_kpmap(p[0], threshold=0.2), img_size=label.shape[-2:], width=width) for p in label]).float() 
        label_vis = viridis(label_vis)
        if img is not None:
            label_vis = label_vis * 0.5 + img[:len(label_vis), :3] * 0.5

        label = label_vis
        
        return label

    def compute_metric(self, Y, Y_pred, M, aux, evaluator_key=None):
        '''
        compute evaluation metric
        '''
        weights = aux[0]
        evaluator = self.evaluator[evaluator_key]
        
        assert len(Y) == 1
        if Y_pred.size(-1) == Y.size(-1) and Y_pred.size(-2) == Y.size(-2):
            ninecrop_sample = torch.tensor([False]).to(Y.device)
            denom = 1
        else:
            ninecrop_sample = torch.tensor([True]).to(Y.device)
            denom = Y_pred.size(-1) * Y_pred.size(-2) / (Y.size(-1) * Y.size(-2))
            # denom = 9
        
        Y = (Y.float() * rearrange(weights.float(), 'B -> B 1 1 1')).sum((1,2,3)).round().float()
        Y_pred_sum = ((Y_pred.float() * rearrange(weights.float(), 'B -> B 1 1 1')).sum((1,2,3)) / denom).round().float()
        Y_pred_mode = torch.tensor([len(preprocess_kpmap(p[0].data.cpu(), threshold=0.20)) for p in Y_pred]).float().to(Y.device)
        mask = (ninecrop_sample and Y_pred_sum > 3000).float() # apply mask to ninecrop samples whose sum is > 3000
        Y_pred = Y_pred_sum * mask + Y_pred_mode * (1 - mask)
        
        evaluator.append((Y, Y_pred))
        metric = (Y - Y_pred).abs().mean()

        return metric

    def log_metrics(self, loss_pred, log_dict, valid_tag):
        '''
        log evaluation metrics
        '''
        log_dict[f'{valid_tag}/{self.vis_tag}_MAE'] = loss_pred

    def get_test_metrics(self, metrics_total):
        Ys = torch.cat([y for y, _ in self.evaluator['mtest_test']])
        Y_preds = torch.cat([y_pred for _, y_pred in self.evaluator['mtest_test']])
        mae = (Ys - Y_preds).abs().mean()
        rmse = ((Ys - Y_preds)**2).mean().sqrt()
        metrics = [mae, rmse]
        return metrics