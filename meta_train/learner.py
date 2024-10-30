import os
from einops import rearrange, reduce
import torch
import torch.nn.functional as F
import habana_frameworks.torch.core as htcore

from torch.utils.data import DataLoader
from torchvision import transforms as T

from .unified import test_dataset_dict, base_class_dict, Unified, UnifiedTrainDataset, COCO, MPII, DeepFashion, FreiHand
from .utils import disassemble_batch, generate_task_mask, normalize_tensor
from dataset.utils import get_modes, modes_to_array
from .evaluator import KeypointEvaluator
from train.loss import spatial_softmax_loss
from train.miou_fss import AverageMeter, Evaluator
from train.visualize import (postprocess_semseg, postprocess_depth, postprocess_keypoints, postprocess_keypoints_mpii,
                             postprocess_keypoints_deepfashion, postprocess_keypoints_hand)


class MetaTrainLearner:
    SEMSEG_IDXS = [Unified.TASKS.index(task) for task in Unified.TASKS_CATEGORICAL if 'segment_semantic' in task[1] or 'segment_interactive' in task[1]]
    SEMKP_IDXS = [Unified.TASKS.index(task) for task in Unified.TASKS_CATEGORICAL if 'keypoints_semantic' in task[1]]
    CONTKP_IDXS = [Unified.TASKS.index(('coco', task)) for task in COCO.TASKS_KP_HUMAN] + \
                  [Unified.TASKS.index(('mpii', task)) for task in MPII.TASKS_KP_HUMAN] + \
                  [Unified.TASKS.index(('deepfashion', task)) for task in DeepFashion.TASKS_KP_HUMAN]
    
    def __init__(self, config, trainer):
        assert config.stage == 0, 'stage should be 0 (meta-training)'
        self.config = config
        self.trainer = trainer
        self.local_rank = self.trainer.local_rank
        self.n_devices = self.trainer.n_devices
        self.verbose = self.trainer.verbose

        self.topil = T.ToPILImage()
        result_dir = self.config.result_dir
        os.makedirs(result_dir, exist_ok=True)
        self.tag = ''

        # create dataset for episodic training
        if self.config.no_eval:
            dset_size = self.config.n_steps*self.config.global_batch_size # whole iterations in a single epoch
        else:
            dset_size = self.config.val_iter*self.config.global_batch_size # chunk iterations in validation steps

        self.train_data = UnifiedTrainDataset(
            split='train',
            dset_size=dset_size,
            config=self.config,
            verbose=self.verbose,
            label_augmentation=self.config.label_augmentation,
            path_dict=self.config.path_dict,
            base_size=self.config.base_size,
            crop_size=self.config.img_size,
            seed=None,
            precision=self.config.precision,
        )

    def register_evaluator(self):
        '''
        register evaluator
        '''
        self.evaluator = {}
        dset_names = set(dset_name for dset_name, _ in self.trainer.valid_tasks)
        for dset_name in dset_names:
            if 'categorical' in test_dataset_dict[dset_name] and dset_name not in ['mpii', 'deepfashion', 'freihand']:
                for key in self.trainer.valid_tags:
                    split = key.split('_')[1]
                    Dataset = base_class_dict[dset_name]
                    if dset_name == 'coco_stereo':
                        self.evaluator[(dset_name, 'segment_interactive', split)] = AverageMeter(class_ids_interest=range(len(Dataset.CLASS_IDXS_VAL)),
                                                                                             semseg_classes=Dataset.CLASS_IDXS_VAL,
                                                                                             device=torch.device(f'hpu:{self.local_rank}'))
                    else:
                        self.evaluator[(dset_name, 'segment_semantic', split)] = AverageMeter(class_ids_interest=range(len(Dataset.CLASS_IDXS_VAL)),
                                                                                          semseg_classes=Dataset.CLASS_IDXS_VAL,
                                                                                          device=torch.device(f'hpu:{self.local_rank}'))
                            
            if dset_name == 'coco' and (self.config.cat_task or (self.config.task_group == 'keypoints_semantic')):
                for key in self.trainer.valid_tags:
                    split = key.split('_')[1]
                    kp_json_path = os.path.join(self.config.result_dir, f'coco_{split}_temp.json')
                    ann_path = os.path.join(self.config.path_dict['coco'],
                                            'annotations', f'person_keypoints_{split.replace("valid", "val")}2017.json')
                    sigmas = [
                        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
                        0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089
                    ]
                    self.evaluator[('coco', 'keypoints_semantic', split)] = KeypointEvaluator(self.config, kp_json_path, ann_path, sigmas, local_rank=self.local_rank, n_devices=self.n_devices, trainer=self.trainer)

            if dset_name == 'mpii' and (self.config.cat_task or (self.config.task_group == 'keypoints_semantic')):
                for key in self.trainer.valid_tags:
                    split = key.split('_')[1]
                    kp_json_path = os.path.join(self.config.result_dir, f'mpii_{split}_temp.json')
                    ann_path = os.path.join(self.config.path_dict['mpii'], 'annotations', 'train.json')
                    sigmas = [
                        0.089, 0.083, 0.107, 0.107, 0.083, 0.089, 0.026, 0.026, 0.026, 0.026,
                        0.062, 0.072, 0.179, 0.179, 0.072, 0.062
                    ]
                    self.evaluator[('mpii', 'keypoints_semantic', split)] = KeypointEvaluator(self.config, kp_json_path, ann_path, sigmas, local_rank=self.local_rank, n_devices=self.n_devices, trainer=self.trainer)

            if dset_name == 'deepfashion' and (self.config.cat_task or (self.config.task_group == 'keypoints_semantic')):
                for key in self.trainer.valid_tags:
                    split = key.split('_')[1]
                    kp_json_path = os.path.join(self.config.result_dir, f'deepfashion_{split}_temp.json')
                    ann_path = os.path.join(self.config.path_dict['deepfashion'], 'annotations', f'{split.replace("valid", "val")}.json')
                    sigmas = [0.05]*len(DeepFashion.KP_CLASSES)
                    self.evaluator[('deepfashion', 'keypoints_semantic', split)] = KeypointEvaluator(self.config, kp_json_path, ann_path, sigmas, local_rank=self.local_rank, n_devices=self.n_devices, trainer=self.trainer)

            if dset_name == 'freihand' and (self.config.cat_task or (self.config.task_group == 'keypoints_semantic')):
                for key in self.trainer.valid_tags:
                    split = key.split('_')[1]
                    kp_json_path = os.path.join(self.config.result_dir, f'freihand_{split}_temp.json')
                    ann_path = os.path.join(self.config.path_dict['freihand'], 'annotations', f'freihand_{split.replace("valid", "val")}.json')
                    sigmas = [0.05]*len(FreiHand.KP_CLASSES)
                    self.evaluator[('freihand', 'keypoints_semantic', split)] = KeypointEvaluator(self.config, kp_json_path, ann_path, sigmas, local_rank=self.local_rank, n_devices=self.n_devices, trainer=self.trainer)

    def reset_evaluator(self):
        for key in self.evaluator:
            if self.evaluator[key] is not None:
                self.evaluator[key].reset()

    def get_train_loader(self):
        batch_size = self.config.global_batch_size  // self.n_devices
        train_loader = DataLoader(self.train_data, batch_size=batch_size,
                                  shuffle=(self.n_devices == 1), pin_memory=True, drop_last=True, num_workers=self.config.num_workers)
        
        return train_loader
    
    def get_support_data(self):
        data_path = f'support_data_{self.config.img_size[1]}_{self.config.dataset}.pth'
        if os.path.exists(data_path):
            support_data = torch.load(data_path)
        else:
            support_data = {}
        
        modified = False
        base_size = crop_size = self.config.base_size

        for g, (dataset_name, task_group) in enumerate(Unified.TASK_GROUP_NAMES):
            g_idx = torch.tensor([g])
            if self.config.task_group is not None and task_group != self.config.task_group:
                continue
            if getattr(self.config, dataset_name, False):
                if self.config.coco_cropped and dataset_name == 'coco' and task_group == 'keypoints_semantic':
                    task_group = 'keypoints_semantic_cropped'
                if task_group in ['segment_semantic', 'segment_interactive']:
                    class_indices = base_class_dict[dataset_name].CLASS_IDXS_VAL
                    for c in class_indices:
                        if (dataset_name, f'{task_group}_{c}') not in support_data:
                            TestDataset = test_dataset_dict[dataset_name]['categorical']
                            dset = TestDataset(
                                class_id=c,
                                task_group=task_group,
                                dset_size=self.config.eval_shot*(self.config.support_idx + 1),
                                path_dict=self.config.path_dict,
                                split='valid',
                                base_size=base_size,
                                crop_size=crop_size,
                                precision=self.config.precision,
                            )
                            dloader = DataLoader(dset, batch_size=self.config.eval_shot, shuffle=False, num_workers=0)
                            for idx, batch in enumerate(dloader):
                                if idx == self.config.support_idx:
                                    break
                            
                            t_idx = torch.tensor([[Unified.TASKS.index((dataset_name, f'{task_group}_{c}'))]])
                            batch = (*disassemble_batch(batch, self.config.img_size), t_idx, g_idx)
                            support_data[(dataset_name, f'{task_group}_{c}')] = batch
                            if self.verbose:
                                print(f'Generated {dataset_name} {task_group} {c} support data.')
                            modified = True
                else:
                    if (dataset_name, task_group) not in support_data:
                        kwargs = {}
                        if task_group == 'keypoints_semantic_cropped':
                            kwargs['cropped'] = True
                            task_type = Unified.TASK_GROUP_TYPE[(dataset_name, 'keypoints_semantic')]
                            task_group_ = 'keypoints_semantic'
                        else:
                            task_type = Unified.TASK_GROUP_TYPE[(dataset_name, task_group)]
                            task_group_ = task_group
                        TestDataset = test_dataset_dict[dataset_name][task_type]
                        dset = TestDataset(
                            task_group=task_group_,
                            dset_size=self.config.eval_shot*(self.config.support_idx + 1),
                            path_dict=self.config.path_dict,
                            split='valid',
                            base_size=base_size,
                            crop_size=crop_size,
                            precision=self.config.precision,
                            **kwargs,
                        )
                        if task_group.startswith('keypoints_semantic'):
                            dset.create_support_keypoint_index(self.config.eval_shot, self.config.support_idx)

                        dloader = DataLoader(dset, batch_size=self.config.eval_shot, shuffle=False, num_workers=0)
                        for idx, batch in enumerate(dloader):
                            if idx == self.config.support_idx or task_group.startswith('keypoints_semantic'):
                                break
                        
                        if task_group.startswith('keypoints_semantic'):
                            class_indices = test_dataset_dict[dataset_name]['categorical'].KP_IDXS
                            t_idx = torch.tensor([[Unified.TASKS.index((dataset_name, f'keypoints_semantic_{c}'))
                                                for c in class_indices]])
                        else:
                            t_idx = torch.tensor([[Unified.TASKS.index((dataset_name, f'{task_group}_{c}'))
                                                for c in range(len(TestDataset.TASK_GROUP_DICT[task_group]))]])

                        batch = (*disassemble_batch(batch, self.config.img_size), t_idx, g_idx)
                        support_data[(dataset_name, task_group)] = batch
                        if self.verbose:
                            print(f'Generated {dataset_name} {task_group} support data.')
                        modified = True

        if modified:
            torch.save(support_data, data_path)
        
        return support_data

    def get_val_loaders(self):
        valid_loaders_list = []
        loader_tags = ['mtrain_train', 'mtrain_valid']

        for loader_tag in loader_tags:
            split = loader_tag.split('_')[1]
            valid_loaders = {}

            # no crop for evaluation.
            crop_size = base_size = self.config.base_size

            for dataset_name, task_group in Unified.TASK_GROUP_NAMES:
                if self.config.task_group is not None and task_group != self.config.task_group:
                    continue
                if (not self.config.use_stereo_datasets) and 'stereo' in dataset_name:
                    continue
                if self.config.no_coco_kp and dataset_name == 'coco' and 'keypoints' in task_group:
                    continue
                if (not self.config.base_task) and (dataset_name, task_group) in Unified.TASKS_BASE:
                    continue
                if (not self.config.cont_task) and (dataset_name, task_group) in Unified.TASKS_CONTINUOUS:
                    continue
                if (not self.config.cat_task) and (dataset_name, task_group) in Unified.TASKS_CATEGORICAL:
                    continue

                if getattr(self.config, dataset_name, False):
                    if task_group in ['segment_semantic', 'segment_interactive']:
                        class_indices = base_class_dict[dataset_name].CLASS_IDXS_VAL
                        for c in class_indices:
                            TestDataset = test_dataset_dict[dataset_name]['categorical']
                            eval_data = TestDataset(
                                class_id=c,
                                task_group=task_group,
                                dset_size=self.config.eval_size,
                                path_dict=self.config.path_dict,
                                split=split,
                                base_size=base_size,
                                crop_size=crop_size,
                                precision=self.config.precision,
                            )
                            # create dataloader.
                            eval_loader = DataLoader(eval_data, batch_size=(self.config.eval_batch_size // self.n_devices),
                                                    shuffle=False, pin_memory=True, drop_last=False, num_workers=0)
                            valid_loaders[(dataset_name, f'{task_group}_{c}')] = eval_loader
                            if self.verbose:
                                print(f'Loaded {dataset_name} {task_group} {c} validation loader in {split} split.')
                    else:
                        task_type = Unified.TASK_GROUP_TYPE[(dataset_name, task_group)]
                        TestDataset = test_dataset_dict[dataset_name][task_type]
                        kwargs = {}
                        if dataset_name == 'coco' and task_group == 'keypoints_semantic':
                            kwargs['cropped'] = self.config.coco_cropped
                        eval_data = TestDataset(
                            task_group=task_group,
                            dset_size=self.config.eval_size,
                            path_dict=self.config.path_dict,
                            split=split,
                            base_size=base_size,
                            crop_size=crop_size,
                            precision=self.config.precision,
                            **kwargs,
                        )
                        # create dataloader.
                        eval_loader = DataLoader(eval_data, batch_size=(self.config.eval_batch_size // self.n_devices),
                                                shuffle=False, pin_memory=True, drop_last=False, num_workers=0)
                        valid_loaders[(dataset_name, task_group)] = eval_loader
                        if self.verbose:
                            print(f'Loaded {dataset_name} {task_group} validation loader in {split} split.')

            valid_loaders_list.append(valid_loaders)

        return valid_loaders_list, loader_tags

    def get_test_loader(self):
        raise NotImplementedError

    def compute_loss(self, Y_pred, Y, M, t_idx):
        '''
        Compute l1 loss for continuous tasks and bce loss for semantic segmentation.
        [loss_args]
            Y_src: unnormalized prediction of shape (B, T, N, 1, H, W)
            Y_tgt: normalized GT of shape (B, T, N, 1, H, W)
            M    : mask for loss computation of shape (B, T, N, 1, H, W)
            t_idx: task index of shape (B, T)
        '''
        Y_pred = rearrange(Y_pred, 'B T N C H W -> (B T) N C H W')
        Y = rearrange(Y, 'B T N C H W -> (B T) N C H W')
        M = rearrange(M, 'B T N C H W -> (B T) N C H W')
        t_idx = rearrange(t_idx, 'B T -> (B T)')

        # create loss masks
        bce_mask = generate_task_mask(t_idx, self.SEMSEG_IDXS + self.CONTKP_IDXS, 'cpu')
        ssl_mask = generate_task_mask(t_idx, self.SEMKP_IDXS, 'cpu')
        l1_mask = ~(bce_mask + ssl_mask)
        
        # prediction loss
        if bce_mask.sum() > 0:
            bce_mask = bce_mask.to(Y.device)
            loss_bce = (M[bce_mask] * F.binary_cross_entropy_with_logits(Y_pred[bce_mask], Y[bce_mask], reduction='none')).mean()
        else:
            loss_bce = 0
        if l1_mask.sum() > 0:
            l1_mask = l1_mask.to(Y.device)
            loss_l1 = (M[l1_mask] * F.l1_loss(Y_pred[l1_mask].sigmoid(), Y[l1_mask], reduction='none')).mean()
        else:
            loss_l1 = 0
        if ssl_mask.sum() > 0:
            ssl_mask = ssl_mask.to(Y.device)
            loss_ssl = spatial_softmax_loss(Y_pred[ssl_mask], Y[ssl_mask], M[ssl_mask], reduction='mean', scaled=self.config.scale_ssl)
        else:
            loss_ssl = 0

        loss = (self.config.bce_coef*loss_bce + self.config.ssl_coef*loss_ssl + self.config.con_coef*loss_l1).mean()
        
        # create log dictionary
        loss_values = {
            'loss': loss.detach(),
            'loss_bce': loss_bce.detach() if bce_mask.sum() > 0 else None,
            'loss_ssl': loss_ssl.detach() if ssl_mask.sum() > 0 else None,
            'loss_con': loss_l1.detach() if l1_mask.sum() > 0 else None,
        }

        return loss, loss_values

    def postprocess_logits(self, task, Y_pred_out):
        '''
        post-processing function for logits
        '''
        if 'segment_semantic' in task[1] or 'segment_interactive' in task[1]:
            pass # lazy post-processing for segmentation tasks
        elif 'keypoints_semantic' in task[1]:
            # spatial softmax
            H, W = Y_pred_out.shape[-2:]
            Y_pred_out = rearrange(Y_pred_out, '1 T N C H W -> 1 T N C (H W)')
            Y_pred_out = F.softmax(Y_pred_out, dim=-1)
            Y_pred_out = rearrange(Y_pred_out, '1 T N C (H W) -> 1 T N C H W', H=H, W=W)
            Y_pred_out = Y_pred_out / (1e-18 + reduce(Y_pred_out, '1 T N C H W -> 1 T N C 1 1', 'max'))
        else:
            Y_pred_out = Y_pred_out.sigmoid()

        return Y_pred_out

    def postprocess_final(self, task, Y_pred):
        '''
        post-processing function for final prediction
        '''
        if 'segment_semantic' in task[1] or 'segment_interactive' in task[1]:
            if Y_pred.shape[1] == 1:
                Y_pred = (Y_pred.sigmoid() > 0.5).squeeze(1).to(Y_pred.dtype)
            else:
                Y_logits_bg = torch.zeros_like(Y_pred[:, :1])
                Y_logits = torch.cat((Y_logits_bg, Y_pred), dim=1)
                Y_pred = torch.argmax(Y_logits, dim=1).to(Y_pred.dtype)

        return Y_pred
    
    def postprocess_vis(self, label, img=None, aux=None, task=None, dset_name=None):
        '''
        post-processing function for visualization
        '''
        assert task is not None
        # set task-specific post-processing function for visualization
        if 'segment_semantic' in task or 'segment_interactive' in task:
            postprocess_fn = postprocess_semseg
        elif 'depth_zbuffer' in task:
            postprocess_fn = postprocess_depth
        elif task == 'keypoints_semantic':
            assert dset_name is not None
            if dset_name == 'coco':
                postprocess_fn = postprocess_keypoints
            elif dset_name == 'mpii':
                postprocess_fn = postprocess_keypoints_mpii
            elif dset_name == 'deepfashion':
                postprocess_fn = postprocess_keypoints_deepfashion
            elif dset_name == 'freihand':
                postprocess_fn = postprocess_keypoints_hand
            else:
                raise NotImplementedError
        else:
            postprocess_fn = None

        if postprocess_fn is not None:
            label = postprocess_fn(label, img, aux)

        return label

    def compute_metric(self, task, Y, Y_pred, M, aux, evaluator_key=None):
        '''
        compute evaluation metric
        '''
        evaluator = self.evaluator[evaluator_key] if evaluator_key is not None else None

        # Mean Angle Error
        if task == 'normal':
            pred = normalize_tensor(Y_pred, dim=1)
            gt = normalize_tensor(Y, dim=1)
            deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
            metric = (M[:, 0] * deg_diff) / 90. # normalize to [0, 1]
            metric = metric.mean()
            
        # Mean IoU for binary segmentation
        elif 'segment_semantic' in task or 'segment_interactive' in task:
            Y = Y.squeeze(1).round().long()
            metric = 0
            assert evaluator is not None
            area_inter, area_union = Evaluator.classify_prediction(Y_pred.clone().float(), Y.float())
            if task == 'segment_semantic':
                semseg_class = 0
            else:
                semseg_class = int(task.split('_')[-1])
            class_id = torch.tensor([evaluator.semseg_classes.index(semseg_class)]*len(Y_pred), device=Y.device)
            area_inter = area_inter.to(Y.device)
            area_union = area_union.to(Y.device)
            evaluator.update(area_inter, area_union, class_id)

        elif task == 'keypoints_semantic':
            metric = 0
            assert evaluator is not None, task
            assert aux is not None
            aux = aux[1]
            for i in range(len(Y_pred)):
                modes, scores = get_modes(Y_pred[i].cpu(), return_scores=True)
                arr, score = modes_to_array(modes, scores, max_detection=1)
                # coco_cropped
                if aux.ndim == 2:
                    if aux[i, 0].item() not in evaluator.all_ret:
                        evaluator.all_ret[aux[i, 0].item()] = {}
                    evaluator.all_ret[aux[i, 0].item()][aux[i, 1].item()] = (arr, score)
                # ap10k and coco
                else:
                    evaluator.all_ret[aux[i].item()] = (arr, score)
        
        # Mean Squared Error
        else:
            metric = (M * F.mse_loss(Y, Y_pred, reduction='none').pow(0.5))
            metric = metric.mean()

        return metric

    def log_metrics(self, dset_name, task_name, split, loss_pred, log_dicts, valid_tag):
        '''
        log evaluation metrics
        '''
        log_dict, avg_loss, avg_loss_base, avg_loss_cont, avg_loss_seg, avg_loss_kp = log_dicts

        # semseg
        if 'segment_semantic' in task_name or 'segment_interactive' in task_name:
            task_group_name = '_'.join(task_name.split('_')[:-1])
            if 'segment_semantic' in task_name:
                semseg_classes = base_class_dict[dset_name].TASKS_SEMSEG
            else:
                semseg_classes = base_class_dict[dset_name].TASKS_INTERSEG
            if semseg_classes.index(task_name) == 0:
                evaluator = self.evaluator[(dset_name, task_group_name, split)]
                if self.n_devices > 1:
                    evaluator.intersection_buf = reduce(self.trainer.all_gather(evaluator.intersection_buf), 'G ... -> ...', 'sum')
                    evaluator.union_buf = reduce(self.trainer.all_gather(evaluator.union_buf), 'G ... -> ...', 'sum')
                iou = evaluator.compute_iou()[0]
                tag = f'{valid_tag}/{dset_name}_{task_group_name}_pred'
                log_dict[tag] = 1 - iou
                avg_loss[valid_tag].append(1 - iou)
                avg_loss_seg[valid_tag].append(1 - iou)

        elif task_name == 'keypoints_semantic':
            evaluator = self.evaluator[(dset_name, task_name, split)]
            cropped = (dset_name == 'coco' or dset_name == 'mpii')
            ap = evaluator.evaluate_keypoints(cropped=cropped)
            log_dict[f'{valid_tag}/{dset_name}_keypoints_semantic_pred'] = 1 - ap
            avg_loss[valid_tag].append(1 - ap)
            avg_loss_kp[valid_tag].append(1 - ap)

        # other
        else:
            log_dict[f'{valid_tag}/{dset_name}_{task_name}_pred'] = loss_pred
            avg_loss[valid_tag].append(loss_pred)
            if (task_name.startswith('autoencoding') or task_name.startswith('denoising') or task_name.startswith('edge_texture')):
                avg_loss_base[valid_tag].append(loss_pred)
            else:
                avg_loss_cont[valid_tag].append(loss_pred)
    
