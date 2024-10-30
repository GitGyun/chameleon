import torch
import torch.nn.functional as F
import os
from skimage import color
from einops import reduce
import numpy as np

from ..base_learner import BaseLearner
from .dataset import LINEMODDataset, LINEMODMaskDataset
from .utils import texture_to_6dpose, ADD_score, load_mesh_model, vis_pose
from train.miou_fss import AverageMeter, Evaluator
from train.visualize import postprocess_semseg


class LINEMODLearner(BaseLearner):
    BaseDataset = LINEMODDataset
    
    def register_evaluator(self):
        mesh_path = os.path.join(self.config.path_dict[self.config.dataset], 'mesh_data', self.config.class_name, 'mesh.ply')
        self.mesh_model = load_mesh_model(mesh_path)

    def compute_loss(self, Y_pred, Y, M):
        '''
        loss function that returns loss and a dictionary of its components
        '''
        loss_bce = (M[:, :1] * F.binary_cross_entropy_with_logits(Y_pred[:, :1], Y[:, :1], reduction='none')).mean()
        loss_l1 = (Y[:, :1] * F.l1_loss(Y_pred[:, 1:].sigmoid(), Y[:, 1:], reduction='none')).mean()
        loss = loss_bce + loss_l1
        loss_values = {'loss': loss.detach(), 'loss_bce': loss_bce.detach(), 'loss_con': loss_l1.detach()}
        return loss, loss_values

    def postprocess_logits(self, Y_pred_out):
        '''
        post-processing function for logits
        '''
        Y_pred_out = Y_pred_out.sigmoid()
        return Y_pred_out
    
    def postprocess_final(self, Y_pred):
        ID_mask = (Y_pred[:, :1] > 0.5).to(Y_pred.dtype)
        UVW = Y_pred[:, 1:]
        Y_pred = torch.cat((ID_mask, UVW), dim=1)
        return Y_pred
    
    def postprocess_vis(self, label, img=None, aux=None):
        '''
        post-processing function for visualization
        '''
        _, coord, *_ = aux
        coord = coord.cpu().float().numpy()
        vis1 = torch.where(label[:, :1].bool(), label[:, 1:], torch.zeros_like(label[:, 1:])) 
        pose = texture_to_6dpose(label, self.mesh_model, coord)
        vis2 = vis_pose(pose, img, self.mesh_model, coord=coord, thickness=2)
        label = torch.cat((vis1, vis2), 0)

        return label

    def compute_metric(self, Y, Y_pred, M, aux, evaluator_key=None):
        '''
        compute evaluation metric
        '''
        gt_pose, coord, *_ = aux
        gt_pose = gt_pose.float().cpu().numpy()
        coord = coord.float().cpu().numpy()
        pred_pose = texture_to_6dpose(Y_pred, self.mesh_model, coord=coord)
        metric = torch.tensor([ADD_score(self.mesh_model.vertices, gt_pose[i], pred_pose[i], self.mesh_model.diameter)
                               for i in range(len(gt_pose))], device=Y.device)
        metric = metric.mean(dim=0)

        return metric

    def log_metrics(self, loss_pred, log_dict, valid_tag):
        '''
        log evaluation metrics
        '''
        log_dict[f'{valid_tag}/{self.config.dataset}_{self.config.task}_{self.config.class_name}_ADD0.1s_inverted'] = 1 - loss_pred[0]
        log_dict[f'{valid_tag}/{self.config.dataset}_{self.config.task}_{self.config.class_name}_AvgDistance'] = loss_pred[1]

    def save_test_outputs(self, Y_pred, batch_idx):
        '''
        save test outputs
        '''
        pass

    def get_test_metrics(self, metrics_total):
        '''
        save test metrics
        '''
        return metrics_total


class LINEMODMaskLearner(BaseLearner):
    BaseDataset = LINEMODMaskDataset
    
    def register_evaluator(self):
        if self.config.stage == 1:
            keys = ['mtest_train', 'mtest_valid']
            self.evaluator = {key: None for key in keys}
            for key in keys:
                self.evaluator[key] = AverageMeter(class_ids_interest=[0], semseg_classes=[0], device=torch.device(f'cuda:{self.local_rank}'))
        else:
            self.evaluator = {'mtest_test': []}
            self.result_path = os.path.join(self.config.result_dir, f'bbox_{self.config.class_name}.npy')

    def reset_evaluator(self):
        if self.config.stage == 1:
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
        metric = 0
        if self.config.stage == 1:
            assert self.evaluator is not None
            assert evaluator_key is not None
            evaluator = self.evaluator[evaluator_key]

            Y = Y.squeeze(1).long()
            area_inter, area_union = Evaluator.classify_prediction(Y_pred.clone().float(), Y.float())
            class_id = torch.tensor([evaluator.semseg_classes.index(0)]*len(Y_pred), device=Y.device)
            area_inter = area_inter.to(Y.device)
            area_union = area_union.to(Y.device)
            evaluator.update(area_inter, area_union, class_id)

        return metric

    def log_metrics(self, loss_pred, log_dict, valid_tag):
        '''
        log evaluation metrics
        '''
        evaluator = self.evaluator[valid_tag]
        if self.n_devices > 1:
            evaluator.intersection_buf = reduce(self.trainer.all_gather(evaluator.intersection_buf), 'G ... -> ...', 'sum')
            evaluator.union_buf = reduce(self.trainer.all_gather(evaluator.union_buf), 'G ... -> ...', 'sum')
        iou = evaluator.compute_iou()[0]
        log_dict[f'{valid_tag}/{self.vis_tag}_IoU_inverted'] = 1 - iou

    def save_test_outputs(self, Y_pred, batch_idx):
        '''
        save test outputs
        '''
        base_size = (480, 640)
        scale = 1.5

        Y_pred = F.interpolate(Y_pred[:, None], base_size, mode='nearest')[:, 0]
        for y in Y_pred:
            y = y.cpu().numpy()
            y_2d, x_2d = np.nonzero(y)
            if len(y_2d) == 0 or len(x_2d) == 0:
                coord = (0, 0, base_size[1], base_size[0])
            else:
                center_x = (x_2d.max() + x_2d.min()).astype(np.float32) / 2 
                center_y = (y_2d.max() + y_2d.min()).astype(np.float32) / 2

                w = round((x_2d.max() - x_2d.min()) * scale)
                h = round((y_2d.max() - y_2d.min()) * scale)

                corner_x = min(base_size[1] - w, max(0, round(center_x - w//2)))
                corner_y = min(base_size[0] - h, max(0, round(center_y - h//2)))

                coord = (corner_x, corner_y, corner_x + w, corner_y + h)
            self.evaluator['mtest_test'].append(coord)

    def get_test_metrics(self, metrics_total):
        '''
        save test metrics
        '''
        coord = np.array(self.evaluator['mtest_test'])
        return coord
