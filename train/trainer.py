import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
import torch
import torch.distributed
import torchvision.transforms as T
from torchvision.utils import make_grid
from einops import rearrange, repeat
import numpy as np
import deepspeed
from tqdm import tqdm
import math

from meta_train.learner import MetaTrainLearner
from downstream.learner_factory import get_downstream_learner
from model.model_factory import get_model

from dataset.utils import to_device, AutoCrop, inverse_sigmoid
from meta_train.unified import base_class_dict

from .optim import get_optimizer
from .visualize import visualize_batch, visualize_alpha


class LightningTrainWrapper(pl.LightningModule):
    def __init__(self, config, verbose=True):
        '''
        Pytorch lightning wrapper for Visual Token Matching.
        '''
        super().__init__()
        # load model.
        self.model = get_model(config, verbose=verbose)
        self.config = config
        self.verbose = verbose
        self.n_devices = (torch.cuda.device_count() if not self.config.single_gpu else 1)

        # tools for validation.
        if self.config.autocrop:
            self.autocrop = AutoCrop(config.img_size, config.autocrop_minoverlap)
        else:
            self.upsample = T.Resize(config.base_size)
            self.downsample = T.Resize(config.img_size)
            self.tripleupsample = T.Resize((config.img_size[0] * 3, config.img_size[1] * 3))
            self.doubleupsample = T.Resize((config.img_size[0] * 2, config.img_size[1] * 2))
        
        self.vis_resizer = T.Resize(config.vis_size)

        if self.config.stage == 1:
            for attn in self.model.matching_module.matching:
                attn.attn_dropout.p = self.config.attn_dropout

        self.toten = T.ToTensor()

        # save hyper=parameters
        self.save_hyperparameters()
    
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # create learner class
        if self.config.stage == 0:
            self.learner = MetaTrainLearner(self.config, self)
        else:
            self.learner = get_downstream_learner(self.config, self)

        # create support data
        if not self.config.no_eval:
            self.support_data = self.prepare_support_data()

    def configure_optimizers(self):
        '''
        Prepare optimizer and lr scheduler.
        '''
        optimizer, self.lr_scheduler = get_optimizer(self.config, self.model)
        return optimizer
    
    def train_dataloader(self, verbose=True):
        '''
        Prepare training loader.
        '''
        if not self.config.no_train:
            train_loader = self.learner.get_train_loader()

            return train_loader
        
    def prepare_support_data(self):
        support_data = self.learner.get_support_data()
        if self.config.stage > 0:
            X, Y, M = support_data
            N = len(X)
            assert X.ndim == 4
            assert Y.ndim == 4

            X = repeat(X, '(B N) C H W -> B T N C H W', N=N, T=self.model.n_tasks)
            Y = rearrange(Y, '(B N) (T C) H W -> B T N C H W', N=N, T=self.model.n_tasks)
            M = rearrange(M, '(B N) (T C) H W -> B T N C H W', N=N, T=self.model.n_tasks)
            t_idx = repeat(torch.arange(self.model.n_tasks, device=X.device), 'T -> B T', B=len(X))
            g_idx = torch.tensor([0]*len(X))
            support_data = {self.config.task: (X, Y, M, t_idx, g_idx)}
        
        # convert to proper precision
        if self.config.precision == 'fp16':
            support_data = to_device(support_data, dtype=torch.half)
        elif self.config.precision == 'bf16':
            support_data = to_device(support_data, dtype=torch.bfloat16)
        
        return support_data
    
    def val_dataloader(self, verbose=True):
        '''
        Prepare validation loaders.
        '''
        if not self.config.no_eval:
            val_loaders_list, loader_tags = self.learner.get_val_loaders()
            self.valid_tags = loader_tags
            self.valid_tasks = list(val_loaders_list[0].keys())
            
            all_val_loaders = sum([list(val_loader.values()) for val_loader in val_loaders_list], [])
            self.validation_step_outputs = [[ ] for _ in range(len(all_val_loaders))]

            return all_val_loaders
    
    def test_dataloader(self):
        '''
        Prepare test loaders.
        '''
        test_loader = self.learner.get_test_loader()
        self.test_step_outputs = []

        return test_loader
        
    def forward(self, train_data):
        '''
        Forward data to model.
        '''
        if self.config.stage == 0:
            X, Y, M, t_idx, g_idx, _ = train_data
        else:
            X, Y, M = train_data

            assert X.ndim == 4
            assert Y.ndim == 4 or Y.ndim == 3
            T = Y.shape[1]

            # random shuffle
            perm = torch.randperm(len(X))
            X = X[perm]
            Y = Y[perm]
            M = M[perm]

            # reshape to 6d tensor
            X = repeat(X, '(B N) C H W -> B T N C H W', B=1, T=T)
            Y = rearrange(Y, '(B N) (T C) H W -> B T N C H W', B=1, T=T)
            M = rearrange(M, '(B N) (T C) H W -> B T N C H W', B=1, T=T)

            # create task index
            t_idx = repeat(torch.arange(self.model.n_tasks, device=X.device), 'T -> B T', B=len(X))

            # sample channels
            if self.config.channel_sampling > 0 and self.config.channel_sampling < Y.size(1):
                channels = torch.randperm(Y.size(1))[:self.config.channel_sampling]
                X = X[:, channels]
                Y = Y[:, channels]
                M = M[:, channels]
                t_idx = t_idx[:, channels]

            # create group index
            g_idx = torch.zeros(len(X), dtype=torch.long, device=X.device)

        # split query and support data
        X_S, X_Q = X.split(math.ceil(X.size(2) / 2), dim=2)
        Y_S, Y_Q = Y.split(math.ceil(Y.size(2) / 2), dim=2)
        M_S, M_Q = M.split(math.ceil(M.size(2) / 2), dim=2)

        # ignore masked region in support label
        Y_S_in = torch.where(M_S.bool(), Y_S, torch.ones_like(Y_S) * self.config.mask_value)

        # forward data
        Y_Q_pred = self.model(X_S, Y_S_in, X_Q, t_idx=t_idx, g_idx=g_idx)

        return Y_Q_pred, Y_Q, M_Q, t_idx

    def on_train_start(self, *args, **kwargs):
        super().on_train_start(*args, **kwargs)
        if self.n_devices > 1:
            self.trainer.train_dataloader.sampler.shuffle = False
            seed_everything(self.config.seed + self.local_rank, workers=True)
            # synchronize at this point
            if self.config.strategy == 'deepspeed':
                deepspeed.comm.barrier()
            else:
                torch.distributed.barrier()

    def training_step(self, batch, batch_idx):
        '''
        A single training iteration.
        '''
        if self.config.stage == 0 and self.config.use_stereo_datasets:
            batch_mono, batch_stereo = batch
            stereo_prob = len(self.learner.train_data.stereo_tasks) / (len(self.learner.train_data.stereo_tasks) + len(self.learner.train_data.tasks))
            if np.random.rand() < stereo_prob:
                batch = batch_stereo
            else:
                batch = batch_mono

        # forward model and compute loss.
        Y_Q_pred, Y_Q, M_Q, t_idx = self.forward(batch)
        if self.config.stage == 0:
            loss, loss_values = self.learner.compute_loss(Y_Q_pred, Y_Q, M_Q, t_idx)
        else:
            loss, loss_values = self.learner.compute_loss(Y_Q_pred, Y_Q, M_Q)

        # schedule learning rate.
        self.lr_scheduler.step(self.global_step + self.config.schedule_from)
        
        # log losses and learning rate.
        log_dict = {
            f'training/lr{self.learner.tag}': self.lr_scheduler.lr,
            'step': self.global_step,
        }
        for key, value in loss_values.items():
            if value is not None:
                log_dict[f'training/{key}{self.learner.tag}'] = value

        # log matching alpha
        for i, alpha in enumerate(self.model.matching_module.alpha):
            if self.config.stage == 0:
                for task_group in self.learner.train_data.task_group_names:
                    group_idx = self.learner.train_data.TASK_GROUP_NAMES.index(task_group)
                    log_dict[f'logging/matching_alpha_{i}_{i}_{"_".join(task_group)}{self.learner.tag}'] = \
                        (alpha.data[group_idx] / self.model.matching_module.alpha_temp).softmax(dim=0)[i].cpu().item()
            else:
                log_dict[f'logging/matching_alpha_{i}_{i}{self.learner.tag}'] = \
                    (alpha.data[0] / self.model.matching_module.alpha_temp).softmax(dim=0)[i].cpu().item()

        log_dict = {k: log_dict[k] for k in sorted(log_dict)}

        # tensorboard logging
        self.log_dict(
            log_dict,
            logger=True,
            on_step=True,
            rank_zero_only=True
        )
        
        return loss
    
    def encode_support(self, X, task, support_verbose=False):
        # support data
        X_S, Y_S, M_S, t_idx, g_idx, *_ = to_device(self.support_data[task], X.device)
        t_idx = t_idx.long()
        g_idx = g_idx.long()

        # predict labels on each crop
        Y_S_in = torch.where(M_S.bool(), Y_S, torch.ones_like(Y_S) * self.config.mask_value)

        # support encoding with chunking
        chunk_size = self.config.chunk_size if self.config.chunk_size > 0 else X_S.shape[1]
        if support_verbose:
            total = X_S.shape[2] // chunk_size + (1 if X_S.shape[2] % chunk_size > 0 else 0)
            pbar = tqdm(total=total, desc='Support Encoding')

        for X_S_, Y_S_in_ in zip(X_S.split(chunk_size, dim=2), Y_S_in.split(chunk_size, dim=2)):
            self.model.encode_support(X_S_, Y_S_in_, t_idx=t_idx, g_idx=g_idx)
            if support_verbose:
                pbar.update(chunk_size)

        if support_verbose:
            pbar.close()
                
    def preprocess_query(self, X, T):
        if X.shape[-2:] != self.config.img_size:
            if self.config.autocrop:
                X = self.autocrop.crop(X)
                n_crops = len(X)
                X = repeat(torch.stack(X), 'F B C H W -> 1 T (F B) C H W', T=T)
            else:
                X = repeat(self.downsample(X), 'B C H W -> 1 T B C H W', T=T)
                n_crops = 1
        else:
            X = repeat(X, 'B C H W -> 1 T B C H W', T=T)
            n_crops = 1
    
        return X, n_crops
    
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def inference(self, X, task, support_verbose=False, **kwargs):
        if not self.model.has_encoded_support:
            self.encode_support(X, task, support_verbose)

        T = self.model.t_idx.shape[1]

        # preprocess query 
        X_in, n_crops = self.preprocess_query(X, T)

        # determine instance chunk size
        chunk_size = self.config.chunk_size if self.config.chunk_size > 0 else X_in.shape[2]

        # determine channel chunk size
        if self.config.channel_chunk_size > 0:
            channel_chunk_size = min(self.config.channel_chunk_size, T)
        else:
            channel_chunk_size = T

        # iterate over channel chunks
        Y_pred_out = []
        for X_in_, channel_idxs in zip(X_in.split(channel_chunk_size, dim=1),
                                        torch.arange(T, device=X_in.device).split(channel_chunk_size)):
            Y_pred_out_ = []
            # iterate over instance chunks
            for X_in__ in X_in_.split(chunk_size, dim=2):
                # predict on query
                if getattr(self.config, "use_ninecrop", False):
                    is_stereo = X_in__.size(3) == 6
                    X_in__1 = self.tripleupsample(rearrange(X_in__[:, :, :, :3], '1 T B C H W -> (T B) C H W'))
                    X_in__1 = rearrange(X_in__1, '(T B) C H W -> 1 T B C H W', T=T)
                    
                    Y_pred_out__ = []
                    small_h = X_in__1.shape[-2] // 3
                    small_w = X_in__1.shape[-1] // 3
                    
                    X_in__2 = self.tripleupsample(rearrange(X_in__[:, :, :, 3:], '1 T B C H W -> (T B) C H W'))
                    X_in__2 = rearrange(X_in__2, '(T B) C H W -> 1 T B C H W', T=T)
                    X_in__2 = X_in__2[..., small_h: 2*small_h, small_w : 2 * small_w]
                        
                    for grid_y, grid_x in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
                        if is_stereo:
                            X_in___ = torch.cat([X_in__1[..., grid_y * small_h: (grid_y+1) * small_h,
                                                                grid_x * small_w : (grid_x+1)*small_w], X_in__2], dim=3)
                        else:
                            X_in___ = X_in__1[..., grid_y * small_h: (grid_y+1) * small_h, grid_x * small_w : (grid_x+1)*small_w]
                        Y_pred_out__.append(self.model.predict_query(X_in___, channel_idxs=channel_idxs, **kwargs))
                        
                    
                    Y_pred_out__ = torch.cat([torch.cat([Y_pred_out__[0], Y_pred_out__[1], Y_pred_out__[2]], dim=-1),
                                               torch.cat([Y_pred_out__[3], Y_pred_out__[4], Y_pred_out__[5]], dim=-1),
                                               torch.cat([Y_pred_out__[6], Y_pred_out__[7], Y_pred_out__[8]], dim=-1)], dim=-2)
                    
                    Y_pred_out_.append(Y_pred_out__)          

                elif self.config.task == "cellpose":
                    # DO Test-time augmentation if the task is equal to cellpsoe
                    Y_pred_1 = self.model.predict_query(X_in__, channel_idxs=channel_idxs, **kwargs).sigmoid()
                    Y_pred_2 = self.model.predict_query(X_in__.flip(-1), channel_idxs=channel_idxs, **kwargs).flip(-1).sigmoid()

                    Y_pred_3 = self.model.predict_query(X_in__.flip(-2), channel_idxs=channel_idxs, **kwargs).flip(-2).sigmoid()
                
                    Y_pred_4 = self.model.predict_query(X_in__.flip((-1, -2)), channel_idxs=channel_idxs, **kwargs).flip((-1, -2)).sigmoid()
                
                    
                    if channel_idxs.item() == 0:
                        Y_pred_out_.append(inverse_sigmoid((Y_pred_1 + Y_pred_2 + (1 - Y_pred_3) + (1 - Y_pred_4)) / 4))
                    elif channel_idxs.item() == 1:
                        Y_pred_out_.append(inverse_sigmoid((Y_pred_1 + (1-Y_pred_2) + Y_pred_3 + (1-Y_pred_4))/4))
                    elif channel_idxs.item() == 2:
                        Y_pred_out_.append(inverse_sigmoid((Y_pred_1 + Y_pred_2 + Y_pred_3 + Y_pred_4) / 4))
                    else:
                        assert False

                else:
                    Y_pred_out_.append(self.model.predict_query(X_in__, channel_idxs=channel_idxs, **kwargs))
    
            Y_pred_out.append(torch.cat(Y_pred_out_, dim=2))
        Y_pred_out = torch.cat(Y_pred_out, dim=1)

        if 'get_attn_map' in kwargs and kwargs['get_attn_map']:
            return Y_pred_out

        # post-process predictions
        if self.config.stage == 0:
            Y_pred_out = self.learner.postprocess_logits(task, Y_pred_out)
        else:
            Y_pred_out = self.learner.postprocess_logits(Y_pred_out)

        # remix the cropped predictions into a whole prediction
        if X.shape[-2:] != self.config.img_size:
            if self.config.autocrop:
                Y_pred_out = rearrange(Y_pred_out, '1 T (F B) C H W -> F B (T C) H W', F=n_crops)
                Y_pred = self.autocrop.tile(Y_pred_out, base_size=X.shape[-2:])
            else:
                Y_pred = rearrange(Y_pred_out, '1 T B C H W -> B (T C) H W')
                if getattr(self.config, "use_ninecrop", False):
                    Y_pred = self.tripleupsample(Y_pred)
                else:
                    Y_pred = self.upsample(Y_pred)
        else:
            Y_pred = rearrange(Y_pred_out, '1 T B C H W -> B (T C) H W')

        # post-process predictions for semantic segmentation
        if self.config.stage == 0:
            Y_pred = self.learner.postprocess_final(task, Y_pred)
        else:
            Y_pred = self.learner.postprocess_final(Y_pred)

        return Y_pred

    def on_validation_start(self):
        self.learner.register_evaluator()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Evaluate few-shot performance on validation dataset.
        '''
        if batch_idx == 0:
            self.model.reset_support()

        if self.config.stage == 0:
            task = self.valid_tasks[dataloader_idx % len(self.valid_tasks)]
            split = self.valid_tags[dataloader_idx // len(self.valid_tasks)].split('_')[1]
            dset_name, task_name = task
            if task_name == 'keypoints_semantic':
                evaluator_key = (dset_name, 'keypoints_semantic', split)
                if dset_name == 'coco' and self.config.coco_cropped:
                    task = (dset_name, 'keypoints_semantic_cropped')
            elif task_name.startswith('segment_semantic'):
                evaluator_key = (dset_name, 'segment_semantic', split)
            elif task_name.startswith('segment_interactive'):
                evaluator_key = (dset_name, 'segment_interactive', split)
            else:
                evaluator_key = None
        else:
            task = self.valid_tasks[0]
            dset_name = self.config.dataset
            task_name = task
            evaluator_key = self.valid_tags[dataloader_idx]
        
        # query data
        if len(batch) == 3:
            X, Y, M = batch
            aux = None
        elif len(batch) == 4:
            X, Y, M, aux = batch
        else:
            raise ValueError(f'Invalid batch size: {len(batch)}, {dset_name}, {task_name}')

        # inference
        Y_pred = self.inference(X, task) # B T H W
        if self.config.stage == 0:
            metric = self.learner.compute_metric(task_name, Y, Y_pred, M, aux, evaluator_key)
        else:
            metric = self.learner.compute_metric(Y, Y_pred, M, aux, evaluator_key)
        metric *= len(Y)
        N = torch.tensor(len(X), device=self.device)
        
        # visualize first batch
        if batch_idx == 0:
            if self.n_devices == 1:
                X_vis = self.vis_resizer(X)
                Y_vis = self.vis_resizer(Y)
                M_vis = self.vis_resizer(M)
                Y_pred_vis = self.vis_resizer(Y_pred)
                aux_vis = aux
            else:
                X_vis = rearrange(self.all_gather(self.vis_resizer(X)), 'G B ... -> (B G) ...')
                Y_vis = rearrange(self.all_gather(self.vis_resizer(Y)), 'G B ... -> (B G) ...')
                M_vis = rearrange(self.all_gather(self.vis_resizer(M)), 'G B ... -> (B G) ...')
                Y_pred_vis = rearrange(self.all_gather(self.vis_resizer(Y_pred)), 'G B ... -> (B G) ...')
                aux_vis = self.recursive_all_gather(aux) if aux is not None else None            
            
            vis_batch = (X_vis, Y_vis, M_vis, Y_pred_vis, aux_vis)
            vis_tag = self.valid_tags[dataloader_idx // len(self.valid_tasks)]
            self.vis_images(vis_batch, task_name, dset_name=dset_name, vis_tag=vis_tag)
        
        self.validation_step_outputs[dataloader_idx].append((metric, N))
        return metric, N
    
    def recursive_all_gather(self, x):
        if self.n_devices == 1:
            return x
        elif isinstance(x, (list, tuple)):
            return [self.recursive_all_gather(x_i) for x_i in x]
        elif isinstance(x, dict):
            return {k: self.recursive_all_gather(v) for k, v in x.items()}
        # elif isinstance(x, torch.IntTensor):
        elif x.dtype == torch.int16:
            return rearrange(self.all_gather(x.float()).int(), 'G B ... -> (B G) ...')           
        elif isinstance(x, torch.BoolTensor):
            return rearrange(self.all_gather(x.float()).bool(), 'G B ... -> (B G) ...')
        else:
            return rearrange(self.all_gather(x), 'G B ... -> (B G) ...')

    def on_validation_epoch_end(self):
        '''
        Aggregate losses of all validation datasets and log them into tensorboard.
        '''
        log_dict = {'step': self.global_step}
        if len(self.valid_tags) == 1:
            validation_step_outputs = [self.validation_step_outputs]
        else:
            validation_step_outputs = self.validation_step_outputs

        if self.config.stage == 0:
            avg_loss = {valid_tag: [] for valid_tag in self.valid_tags}
            avg_loss_base = {valid_tag: [] for valid_tag in self.valid_tags}
            avg_loss_cont = {valid_tag: [] for valid_tag in self.valid_tags}
            avg_loss_seg = {valid_tag: [] for valid_tag in self.valid_tags}
            avg_loss_kp = {valid_tag: [] for valid_tag in self.valid_tags}

            for split_idx in range(2):
                for task_idx in range(len(self.valid_tasks)):
                    valid_tag = self.valid_tags[split_idx]
                    task = self.valid_tasks[task_idx]
                    split = valid_tag.split('_')[1]
                    losses_batch = validation_step_outputs[split_idx*len(self.valid_tasks) + task_idx]

                    N_total = sum([losses[1] for losses in losses_batch])
                    loss_pred = sum([losses[0] for losses in losses_batch])
                    if self.n_devices > 1:
                        N_total = self.all_gather(N_total).sum(dim=0)
                        loss_pred = self.all_gather(loss_pred).sum(dim=0)

                    loss_pred = loss_pred / N_total

                    # log task-specific errors
                    dset_name, task_name = task
                    log_dicts = log_dict, avg_loss, avg_loss_base, avg_loss_cont, avg_loss_seg, avg_loss_kp
                    self.learner.log_metrics(dset_name, task_name, split, loss_pred, log_dicts, valid_tag)

            # log task-averaged error
            for valid_tag in self.valid_tags:
                avg_loss_total = sum(avg_loss[valid_tag]) / len(avg_loss[valid_tag])
                log_dict[f'summary/{valid_tag}_pred'] = avg_loss_total
                if len(avg_loss_base[valid_tag]) > 0:
                    avg_loss_base_total = sum(avg_loss_base[valid_tag]) / len(avg_loss_base[valid_tag])
                    log_dict[f'summary/{valid_tag}_base_pred'] = avg_loss_base_total
                if len(avg_loss_cont[valid_tag]) > 0:
                    avg_loss_cont_total = sum(avg_loss_cont[valid_tag]) / len(avg_loss_cont[valid_tag])
                    log_dict[f'summary/{valid_tag}_cont_pred'] = avg_loss_cont_total
                if len(avg_loss_seg[valid_tag]) > 0:
                    avg_loss_seg_total = sum(avg_loss_seg[valid_tag]) / len(avg_loss_seg[valid_tag])
                    log_dict[f'summary/{valid_tag}_seg_pred'] = avg_loss_seg_total
                if len(avg_loss_kp[valid_tag]) > 0:
                    avg_loss_kp_total = sum(avg_loss_kp[valid_tag]) / len(avg_loss_kp[valid_tag])
                    log_dict[f'summary/{valid_tag}_kp_pred'] = avg_loss_kp_total

        else:
            # keypoints and segmentation use offline evaluator
            for split_idx, valid_tag in enumerate(self.valid_tags):
                losses_batch = validation_step_outputs[split_idx]
                N_total = sum([losses[1] for losses in losses_batch])
                loss_pred = sum([losses[0] for losses in losses_batch])
                if self.n_devices > 1:
                    N_total = self.all_gather(N_total).sum(dim=0)
                    loss_pred = self.all_gather(loss_pred).sum(dim=0)
                loss_pred = loss_pred / N_total

                self.learner.log_metrics(loss_pred, log_dict, valid_tag)

        log_dict = {k: log_dict[k] for k in sorted(log_dict)}

        # tensorboard logging
        self.log_dict(
            log_dict,
            logger=True,
            rank_zero_only=True
        )

        if self.local_rank == 0:
            if self.config.stage == 0:
                for task_group in self.learner.train_data.task_group_names:
                    group_idx = self.learner.train_data.TASK_GROUP_NAMES.index(task_group)
                    vis = visualize_alpha(self.model.matching_module.alpha, group_idx, self.model.matching_module.alpha_temp)
                    # tensorboard logging
                    self.logger.experiment.add_image(f'model/matcing_alpha_{"_".join(task_group)}', self.toten(vis), self.global_step)
            else:
                vis = visualize_alpha(self.model.matching_module.alpha, temp=self.model.matching_module.alpha_temp)
                # tensorboard logging
                self.logger.experiment.add_image(f'model/matcing_alpha', self.toten(vis), self.global_step)
        # reset evaluator
        self.learner.reset_evaluator()

        # reset support
        self.model.reset_support()
        
        # reset validation_step_outputs
        self.validation_step_outputs.clear()
        if self.config.stage == 0:
            self.validation_step_outputs = [[ ] for _ in range(2 * len(self.valid_tasks))]
        else:
            self.validation_step_outputs = [[ ] for _ in range(len(self.valid_tags))]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Evaluate few-shot performance on test dataset.
        '''
        assert self.config.stage == 2, 'Test is only available in stage 2!'
        if batch_idx == 0:
            self.model.reset_support()

        task = self.config.task
        # query data
        if len(batch) == 3:
            X, Y, M = batch
            aux = None
        else:
            X, Y, M, aux = batch
        
        # support data
        Y_pred = self.inference(X, task) # B T H W
        metric = self.learner.compute_metric(Y, Y_pred, M, aux, 'mtest_test')
        metric *= len(Y)
        N = torch.tensor(len(X), device=self.device)

        self.learner.save_test_outputs(Y_pred, batch_idx)
        self.test_step_outputs.append((metric, N))
        return metric, N

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs

        N_total = sum([losses[1] for losses in test_step_outputs])
        metric_total = sum([losses[0] for losses in test_step_outputs])
        if self.n_devices > 1:
            N_total = self.all_gather(N_total).sum(dim=0)
            metric_total = self.all_gather(metric_total).sum(dim=0)
        metric_total = metric_total / N_total
        torch.save(metric_total.cpu(), self.learner.result_path)

        metric = self.learner.get_test_metrics(metric_total)
        if isinstance(metric, np.ndarray):
            np.save(self.learner.result_path, metric)
        else:
            torch.save(metric, self.learner.result_path)
            print(metric)
        
        self.test_step_outputs.clear() # free memory
        self.test_step_outputs = []

    @pl.utilities.rank_zero_only
    def vis_images(self, batch, task, dset_name=None, vis_tag='', **kwargs):
        '''
        Visualize query prediction into tensorboard.
        '''
        X, Y, M, Y_pred, aux = batch

        # handle stereo input
        if X.shape[1] == 6:
            X = torch.cat([X[:, :3], X[:, 3:]], dim=0)

        if self.config.stage == 0:
            postprocess_fn = lambda *args, **kwargs: self.learner.postprocess_vis(*args, **kwargs, task=task, dset_name=dset_name)
            if 'segment_semantic' in task or 'segment_interactive' in task:
                class_idx = int(task.split('_')[-1])
                class_name = base_class_dict[dset_name].CLASS_IDX_TO_NAME[class_idx]
                vis_tag = f'{vis_tag}/{dset_name}_segment_semantic_{class_name}'
            else:
                vis_tag = f'{vis_tag}/{dset_name}_{task}'
        else:
            postprocess_fn = self.learner.postprocess_vis
            vis_tag = f'{vis_tag}/{self.learner.vis_tag}'
        vis = visualize_batch(X, Y, M, Y_pred, aux, postprocess_fn=postprocess_fn, **kwargs)
        vis = make_grid(vis, nrow=len(Y), scale_each=True)

        # tensorboard logging
        self.logger.experiment.add_image(vis_tag, vis, self.global_step)
