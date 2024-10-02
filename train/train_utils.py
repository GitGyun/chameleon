import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

import os
import sys
import shutil
import random
import tqdm
import copy
from textwrap import dedent

import numpy as np
import torch

from .trainer import LightningTrainWrapper
from args import TRAIN_DATASETS
from meta_train.unified import Unified, base_class_dict
from model.transformers.helpers import resize_rel_pos_bias, resize_pos_embed

def configure_experiment(config, model, is_rank_zero=True):
    # set seeds
    # set_seeds(config.seed, config.debug_mode)
    
    # set directories
    log_dir, save_dir = set_directories(config,
                                        exp_name=config.exp_name,
                                        exp_subname=(config.exp_subname if config.stage >= 1 else ''),
                                        create_log_dir=(config.stage != 2 or config.result_dir is None or config.result_dir == 'none'),
                                        create_save_dir=(config.stage != 2),
                                        is_rank_zero=is_rank_zero,
                                        )

    # create lightning callbacks, logger, and checkpoint plugin
    if config.stage != 2:
        callbacks = set_callbacks(config, save_dir, config.monitor, ptf=config.save_postfix)
        logger = CustomTBLogger(log_dir, name='', version='', default_hp_metric=False)
    else:
        callbacks = set_callbacks(config, save_dir)
        logger = None
    
    # create profiler
    profiler = pl.profilers.PyTorchProfiler(log_dir) if config.profile_mode else None
        
    # parse precision
    precision = int(config.precision.strip('fp')) if config.precision in ['fp16', 'fp32'] else config.precision
    precision = f'{precision}-true'
    
    # choose accelerator
    strategy = set_strategy(config, precision)

    # choose plugins
    if config.stage == 1 and config.strategy == 'ddp' and not config.from_scratch:
        save_names = [f'model.{name}' for name in model.model.bias_parameter_names()]
        save_names += [f'model.matching_module.alpha.{name}' for name, _ in model.model.matching_module.alpha.named_parameters()]
        save_names += [f'model.matching_module.layernorm.{name}' for name, _ in model.model.matching_module.layernorm.named_parameters()]
        if config.head_tuning:
            save_names += [f'model.label_decoder.head.{name}' for name, _ in model.model.label_decoder.head.named_parameters()]
        if config.input_embed_tuning:
            save_names += [f'model.image_encoder.backbone.patch_embed.{name}'
                           for name, _ in model.model.image_encoder.backbone.patch_embed.named_parameters()]
        if config.output_embed_tuning:
            save_names += [f'model.label_encoder.backbone.patch_embed.{name}'
                           for name, _ in model.model.label_encoder.backbone.patch_embed.named_parameters()]
        if config.label_decoder_tuning:
            save_names += [f'model.label_decoder.{name}' for name, _ in model.model.label_decoder.named_parameters()]
        if config.relpos_tuning:
            save_names += [f'model.image_encoder.{name}' for name in model.model.image_encoder.relpos_parameter_names()]
            if getattr(model.model.image_encoder.backbone, "pos_embed", None) is not None:
                save_names += [f'model.image_encoder.backbone.pos_embed']
        plugins = [CustomCheckpointIO(save_names)]
    else:
        plugins = None
    
    return logger, log_dir, save_dir, callbacks, profiler, precision, strategy, plugins


def generate_general_print_info(config, model=None):
    if model is None:
        model_config = config
    else:
        model_config = model.config

    print_info = dedent(
        f'''\
        Running Stage {config.stage} with {config.strategy} Strategy:
        [General Info]
            > Exp Name: {config.exp_name}
            > Image Encoder: {model_config.image_encoder}
            > Label Encoder: {model_config.label_encoder}
            > Decoder Features: {model_config.decoder_features}
            > Num Attention Heads: {model_config.n_attn_heads}
            > Eval Batch Size: {config.eval_batch_size}
            > Num Workers: {config.num_workers}
            > Image Size: {config.img_size}
        '''
    )

    return print_info


def generate_mt_print_info(mt_config):
    print_info = dedent(
            f'''\
            [Meta-Train Info]
                > Dataset: {mt_config.dataset}
                    - Taskonomy: {mt_config.taskonomy}
                    - COCO: {mt_config.coco}
                    - MidAir: {mt_config.midair}
                    - MPII: {mt_config.mpii}
                    - DeepFashion: {mt_config.deepfashion}
                    - FreiHand: {mt_config.freihand}
                > Task: {mt_config.task}
                    - Base Tasks: {mt_config.base_task}
                    - Continuous Tasks: {mt_config.cont_task}
                    - Categorical Tasks: {mt_config.cat_task}
                > Image Size: {mt_config.img_size}
                > Base Size: {mt_config.base_size}
                > Global Batch Size: {mt_config.global_batch_size}
                > Loss Type: {mt_config.loss_type}
                > Num Steps: {mt_config.n_steps}
                > Optimizer: {mt_config.optimizer}
                > Weight Decay: {mt_config.weight_decay}
                > Learning Rate: {mt_config.lr}
                > Learning Rate for Pretrained Parameters: {mt_config.lr_pretrained}
                > Learning Rate Schedule: {mt_config.lr_schedule}
                > Learning Rate Warmup: {mt_config.lr_warmup}
            '''
        )

    return print_info


def generate_ft_print_info(ft_config):
    print_info = dedent(
            f'''\
            [Fine-Tune Info]
                > Dataset: {ft_config.dataset}
                > Task: {ft_config.task}
                > Class Name: {ft_config.class_name}
                > Shot: {ft_config.shot}
                > Support Idx: {ft_config.support_idx}
                > Eval Size: {ft_config.eval_size}
                > Learning Rate: {ft_config.lr}
                > Load Step: {ft_config.load_step}
                > Image Size: {ft_config.img_size}
                > Base Size: {ft_config.base_size}
                > Loss Type: {ft_config.loss_type}
                > Global Batch Size: {ft_config.global_batch_size}
                > Channel Sampling: {ft_config.channel_sampling}
                > Num Steps: {ft_config.n_steps}
                > Optimizer: {ft_config.optimizer}
                > Learning Rate: {ft_config.lr}
                > Learning Rate for Pretrained Parameters: {ft_config.lr_pretrained}
                > Learning Rate Schedule: {ft_config.lr_schedule}
                > Learning Rate Warmup: {ft_config.lr_warmup}
                > Early Stopping Patience: {ft_config.early_stopping_patience}
                > Auto Crop: {ft_config.autocrop}
            '''
        )
    
    return print_info


def generate_ts_print_info(ts_config):
    print_info = dedent(f'''\
            [Test Info]
                > Dataset: {ts_config.dataset}
                > Task: {ts_config.task}
                > Shot: {ts_config.shot}
                > Eval Shot: {ts_config.eval_shot}
                > Support Idx: {ts_config.support_idx}
                > Load Step: {ts_config.load_step}
                > Image Size: {ts_config.img_size}
                > Base Size: {ts_config.base_size}
                > Auto Crop: {ts_config.autocrop}
            '''
        )

    return print_info


def print_configs(config, model=None, mt_config=None, ft_config=None, ts_config=None):
    print_info = generate_general_print_info(config, model)
    if config.stage >= 0 and not config.from_scratch:
        print_info += generate_mt_print_info(mt_config if mt_config is not None else config)
    if config.stage >= 1:
        print_info += generate_ft_print_info(ft_config if ft_config is not None else config)
    if config.stage >= 2:
        print_info += generate_ts_print_info(ts_config if ts_config is not None else config)
    print(print_info)


def set_seeds(seed, debug_mode=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if debug_mode:
        torch.use_deterministic_algorithms(True)


def set_directories(config, root_dir='experiments', exp_name='', log_dir='logs', save_dir='checkpoints',
                    create_log_dir=True, create_save_dir=True, dir_postfix='', exp_subname='', is_rank_zero=True):
    # create the root directory
    os.makedirs(root_dir, exist_ok=True)

    # set logging directory
    if create_log_dir:
        os.makedirs(os.path.join(root_dir, config.log_dir), exist_ok=True)
        log_root = os.path.join(root_dir, config.log_dir, exp_name + dir_postfix)
        os.makedirs(log_root, exist_ok=True)
        if config.stage == 1:
            if exp_subname == '':
                exp_subname = f'task:{config.task}_shot:{config.shot}_is:{config.img_size[0]}_lr:{config.lr}_sid:{config.support_idx}{config.subname_postfix}'
                if config.class_wise:
                    exp_subname = exp_subname.replace(f'task:{config.task}', f'task:{config.task}_class:{config.class_name}')
        log_root = os.path.join(log_root, exp_subname)
        os.makedirs(log_root, exist_ok=True)
        log_dir = os.path.join(log_root, log_dir)

        # reset the logging directory if exists
        if is_rank_zero and os.path.exists(log_dir) and not config.continue_mode:
            try:
                shutil.rmtree(log_dir)
            except:
                pass
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = config.result_dir

    # set saving directory
    if create_save_dir:
        save_root = os.path.join(root_dir, config.save_dir, exp_name + dir_postfix)
        if config.stage == 1:
            if exp_subname == '':
                exp_subname = f'task:{config.task}_shot:{config.shot}_is:{config.img_size[0]}_lr:{config.lr}_support_idx:{config.support_idx}{config.subname_postfix}'
                if config.class_wise:
                    exp_subname = exp_subname.replace(f'task:{config.task}', f'task:{config.task}_class:{config.class_name}')
        save_root = os.path.join(save_root, exp_subname)
        os.makedirs(save_root, exist_ok=True)
        save_dir = os.path.join(save_root, save_dir)

        # create the saving directory if checkpoint doesn't exist or in skipping mode,
        # otherwise ask user to reset it
        if is_rank_zero and os.path.exists(save_dir):
            if config.continue_mode:
                print(f'resume from checkpoint ({save_dir})')
            elif config.skip_mode:
                print(f'skip the existing checkpoint ({save_dir})')
                sys.exit()
            elif config.debug_mode or config.reset_mode or config.benchmark_mode:
                print(f'remove existing checkpoint ({save_dir})')
                try:
                    shutil.rmtree(save_dir)
                except:
                    pass
            elif not config.slurm:
                while True:
                    print(f'redundant experiment name! ({save_dir}) remove existing checkpoints? (y/n)')
                    inp = input()
                    if inp == 'y':
                        try:
                            shutil.rmtree(save_dir)
                        except:
                            pass
                        break
                    elif inp == 'n':
                        print('quit')
                        sys.exit()
                    else:
                        print('invalid input')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    return log_dir, save_dir


def set_strategy(config, precision):
    if config.strategy == 'ddp' and not config.single_gpu:
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)
    elif config.strategy == 'deepspeed' and not config.single_gpu:
        strategy = pl.strategies.DeepSpeedStrategy(offload_optimizer=(config.optimizer == 'cpuadam'),
                                                   precision_plugin=pl.plugins.precision.DeepSpeedPrecisionPlugin(precision))
    else:
        strategy = pl.strategies.SingleDeviceStrategy(device='cuda:0') 
        
    return strategy


def set_callbacks(config, save_dir, monitor=None, ptf=''):
    callbacks = [
        CustomProgressBar(),
    ]
    if ((not config.no_eval) and
        monitor is not None and
        config.early_stopping_patience > 0):
        callbacks.append(CustomEarlyStopping(monitor=monitor, mode="min", patience=config.early_stopping_patience))

    if not config.no_save and save_dir is not None:
        # step checkpointing
        if config.stage == 0:
            if not config.temporary_checkpointing:
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=save_dir,
                    filename='step:{step:06d}' + ptf,
                    auto_insert_metric_name=False,
                    every_n_epochs=5,
                    save_top_k=-1,
                    save_last=False,
                )
            else:
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=save_dir,
                    filename='step:{step:06d}' + ptf,
                    auto_insert_metric_name=False,
                    every_n_epochs=5,
                    save_top_k=2,
                    save_last=False,
                    monitor='step',
                    mode='max'
                )
            checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
            callbacks.append(checkpoint_callback)

        # last checkpointing
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            filename=f'last{ptf}',
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_top_k=1,
            save_last=False,
            monitor='epoch',
            mode='max',
        )
        checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
        callbacks.append(checkpoint_callback)
        
        # best checkpointing
        if not (config.no_eval or monitor is None):
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename=f'best{ptf}',
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_top_k=1,
                save_last=False,
                monitor=monitor,
                mode='min',
            )
            checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
            callbacks.append(checkpoint_callback)
            
    return callbacks


def get_ckpt_path(load_dir, exp_name, load_step, exp_subname='', save_postfix='', reduced=False, load_path=None):
    if load_path is None or load_path == 'none':
        if load_step == 0:
            ckpt_name = f'best{save_postfix}.ckpt'
        elif load_step < 0:
            ckpt_name = f'last{save_postfix}.ckpt'
        else:
            ckpt_name = f'step:{load_step:06d}.ckpt'
        if reduced:
            ckpt_name = ckpt_name.replace('.ckpt', '.pth')
            
        load_path = os.path.join('experiments', load_dir, exp_name, exp_subname, 'checkpoints', ckpt_name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"checkpoint ({load_path}) does not exist!")
    
    return load_path


def copy_values(config_new, config_old):
    for key in config_new.__dir__():
        if key[:2] != '__':
            setattr(config_old, key, getattr(config_new, key))


def load_trained_ckpt(ckpt_path):
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, 'checkpoint', 'mp_rank_00_model_states.pt')
        ckpt = torch.load(ckpt_path)
        state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt['module'].items()}
        config = ckpt['hyper_parameters']['config']
    else:
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        config = ckpt['config']
        
    for key in list(state_dict.keys()):
        if 'teacher' in key:
            del state_dict[key]

    return state_dict, config


def load_adapted_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']

    return state_dict


def load_finetuned_ckpts(ckpt_paths):
    state_dicts = []
    configs = []
    if isinstance(ckpt_paths, str):
        ckpt_paths = [ckpt_paths]
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path)
        state_dicts.append(ckpt['state_dict'])
        configs.append(ckpt['hyper_parameters']['config'])

    return state_dicts, configs


def resized_load_state_dict(model, state_dict, config, verbose=True, mt_config=None):
    if mt_config is None or mt_config.n_input_images == 1:
        depth_src = 1
        depth_tar = config.n_input_images if not config.expand_input_embed else 1
    else:
        depth_src = depth_tar = mt_config.n_input_images

    # resize relative position bias table and pos embed
    for key in list(state_dict.keys()):
        if "relative_position_index" in key:
            state_dict[key] = model.state_dict()[key]

        if "relative_position_bias_table" in key:
            state_dict[key] = resize_rel_pos_bias(state_dict[key], model.state_dict()[key].size(0),
                                                  depth_src=(depth_src if 'image_encoder' in key else 1),
                                                  depth_tar=(depth_tar if 'image_encoder' in key else 1),
                                                  verbose=verbose, verbose_tag=key, dst_img_size=config.img_size)

        if 'pos_embed' in key:
            state_dict[key] = resize_pos_embed(state_dict[key], model.state_dict()[key].size(1),
                                               depth_src=(depth_src if 'image_encoder' in key else 1),
                                               depth_tar=(depth_tar if 'image_encoder' in key else 1),
                                               verbose=verbose, verbose_tag=key)

    print(model.load_state_dict(state_dict))


def select_task_specific_parameters(config, model, state_dict, t_idxs, g_idxs):
    # aggregate bias parameters
    bias_parameters = [f'model.{name}' for name in model.model.bias_parameter_names()]
    for key in state_dict.keys():
        if key in bias_parameters:
            state_dict[key] = state_dict[key][t_idxs].mean(0, keepdim=True).repeat(model.model.n_tasks, 1)

    # set matching alpha
    for i in range(len(model.model.matching_module.alpha)):
        n_alpha = model.model.n_tasks if config.separate_alpha else model.model.n_task_groups
        state_dict[f'model.matching_module.alpha.{i}'] = \
            state_dict[f'model.matching_module.alpha.{i}'][g_idxs].mean(0, keepdim=True).repeat(n_alpha, 1)


def remove_task_specific_parameters(config, model, state_dict):
    # handle bias parameters not fine-tuned
    bias_parameters = [f'model.{name}' for name in model.model.bias_parameter_names()]
    for key in state_dict.keys():
        if key[-4:] == 'bias' and state_dict[key].ndim == 2 and key not in bias_parameters:
            state_dict[key] = state_dict[key].mean(0)

    # remove mask token
    if 'model.image_encoder.backbone.mask_token' in state_dict and config.drop_patch_rate == 0:
        del state_dict['model.image_encoder.backbone.mask_token']


def extract_valid_task_idxs(config):
    if config.dataset in TRAIN_DATASETS:
        dset_names = [config.dataset]
    elif config.dataset == 'unified':
        dset_names = []
        for dset_name in TRAIN_DATASETS:
            if getattr(config, dset_name, False):
                dset_names.append(dset_name)
    else:
        raise NotImplementedError
    
    t_idxs = []
    g_idxs = []
    for dset_name in dset_names:
        for task_group_name in base_class_dict[dset_name].TASK_GROUP_NAMES:
            if config.task_group is None or task_group_name == config.task_group:
                g_idxs.append(Unified.TASK_GROUP_NAMES.index((dset_name, task_group_name)))
                for task in base_class_dict[dset_name].TASK_GROUP_DICT[task_group_name]:
                    t_idxs.append(Unified.TASKS.index((dset_name, task)))

    return t_idxs, g_idxs


def handle_unsaved_parameters(config, model, state_dict):
    if 'model.matching_module.layernorm.weight' not in state_dict:
        state_dict['model.matching_module.layernorm.weight'] = model.model.matching_module.layernorm.weight.data
    if 'model.matching_module.layernorm.bias' not in state_dict:
        state_dict['model.matching_module.layernorm.bias'] = model.model.matching_module.layernorm.bias.data
            

def load_model(config, verbose=True, reduced=False, load_explicitly=False):
    load_path = None
    ft_config = None
    ts_config = None

    # create trainer for episodic training
    if config.stage == 0:
        if not (config.continue_mode and load_explicitly):
            model = LightningTrainWrapper(config, verbose=verbose)
        if config.continue_mode:
            load_path = get_ckpt_path(config.load_dir, config.exp_name, config.load_step,
                                      save_postfix=config.save_postfix, reduced=reduced)
            if load_explicitly:
                state_dict, mt_config = load_trained_ckpt(load_path)
                if config.no_train:
                    mt_config.no_train = True
                if config.no_eval:
                    mt_config.no_eval = True
                config = mt_config
                model = LightningTrainWrapper(config=config, verbose=verbose)
                handle_unsaved_parameters(config, model, state_dict)
                print(model.load_state_dict(state_dict))
        mt_config = config

    # create trainer for fine-tuning or evaluation
    else:
        # load meta-trained checkpoint
        if config.stage == 1:
            config_new = ft_config = copy.deepcopy(config)
        else:
            config_new = ts_config = copy.deepcopy(config)

        ckpt_path = get_ckpt_path(config.load_dir, config.exp_name, config.load_step, reduced=reduced)
        state_dict, config = load_trained_ckpt(ckpt_path)
        if verbose:
            print(f'meta-trained checkpoint loaded from {ckpt_path}')
        mt_config = copy.deepcopy(config)

        mt_t_idxs, mt_g_idxs = extract_valid_task_idxs(mt_config)

        # load fine-tuned checkpoint
        if config_new.stage == 2:
            ft_ckpt_paths = get_ckpt_path(config_new.save_dir, config_new.exp_name, 0,
                                          config_new.exp_subname + config_new.subname_postfix, config_new.save_postfix,
                                          load_path=config_new.load_path)
            _, ft_configs = load_finetuned_ckpts(ft_ckpt_paths)
            ft_config = copy.deepcopy(ft_configs[0])

            # merge config
            copy_values(ft_config, config)

        # merge config
        copy_values(config_new, config)

        # image size adjustment
        if config.stage == 1 and config.img_size[0] != 224 or config.img_size[1] != 224:
            if config.img_size[0] == config.img_size[1]:
                config.image_encoder = config.image_encoder.replace('224', str(config.img_size[0]))
                config.label_encoder = config.label_encoder.replace('224', str(config.img_size[0]))
            else:
                config.image_encoder = config.image_encoder.replace('224', f'{str(config.img_size[0])}_{str(config.img_size[1])}')
                config.label_encoder = config.label_encoder.replace('224', f'{str(config.img_size[0])}_{str(config.img_size[1])}')

        # num input images adjustment
        if mt_config.n_input_images > 1 and config.n_input_images == 1:
            config.n_input_images = mt_config.n_input_images

        # create model
        model = LightningTrainWrapper(config=config, verbose=verbose)

        # remove task-specific parameters
        remove_task_specific_parameters(config, model, state_dict)
        handle_unsaved_parameters(config, model, state_dict)

        # select task-specific parameters for test task
        if config_new.stage == 1:
            select_task_specific_parameters(config, model, state_dict, mt_t_idxs, mt_g_idxs)

        # load fine-tuned checkpoint
        elif config_new.stage == 2:
            ft_ckpt_paths = get_ckpt_path(config.save_dir, config.exp_name, 0,
                                          config.exp_subname + config.subname_postfix, config.save_postfix,
                                          load_path=config.load_path)
            ft_state_dicts, _ = load_finetuned_ckpts(ft_ckpt_paths)
            if verbose:
                print('fine-tuned checkpoint loaded from')
                if isinstance(ft_ckpt_paths, str):
                    print(ft_ckpt_paths)
                else:
                    for path in ft_ckpt_paths:
                        print(path)

            for key in ft_state_dicts[0]:
                state_dict[key] = torch.cat([ft_state_dict[key] for ft_state_dict in ft_state_dicts], dim=0)

        resized_load_state_dict(model, state_dict, config, verbose=verbose, mt_config=mt_config)

    return model, config, load_path, mt_config, ft_config, ts_config

        
class CustomProgressBar(TQDMProgressBar):
    def __init__(self, rescale_validation_batches=1, initial_train_batch_idx=0):
        super().__init__()
        self.rescale_validation_batches = rescale_validation_batches
        self.initial_train_batch_idx = initial_train_batch_idx

    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        bar = tqdm.tqdm(
            desc="Training",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            initial=self.initial_train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self):
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = tqdm.tqdm(
            desc="Validation",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
    
    def init_test_tqdm(self):
        """Override this to customize the tqdm bar for testing."""
        bar = tqdm.tqdm(
            desc="Testing",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    
class CustomTBLogger(TensorBoardLogger):
    @pl.utilities.rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)

    
class CustomEarlyStopping(EarlyStopping):
    def _run_early_stopping_check(self, trainer):
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics
        if self.monitor not in logs:
            should_stop = False
            reason = None
        else:
            current = logs[self.monitor].squeeze()
            should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)


class CustomCheckpointIO(TorchCheckpointIO):
    def __init__(self, save_parameter_names):
        self.save_parameter_names = save_parameter_names
    
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        # store only task-specific parameters
        state_dict = checkpoint['state_dict']
        state_dict = {key: value for key, value in state_dict.items() if key in self.save_parameter_names}
        checkpoint['state_dict'] = state_dict
        
        super().save_checkpoint(checkpoint, path, storage_options)
