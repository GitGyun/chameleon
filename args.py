import argparse
import yaml
from easydict import EasyDict


TRAIN_DATASETS = ['taskonomy', 'coco', 'midair', 'mpii', 'deepfashion', 'freihand', 'midair_stereo', 'coco_stereo']
DOWNSTREAM_DATASETS = ['ap10k', 'davis2017', 'linemod', 'isic2018', 'fsc147', 'cellpose']
DOWNSTREAM_TASKS = ['animalkp', 'vos', 'pose_6d', 'segment_semantic', 'segment_medical', 'object_counting', 'cellpose']

base_sizes = {
    (224, 224): (256, 256),
    (384, 384): (448, 448),
    (416, 416): (480, 480),
    (448, 448): (518, 518),
    (512, 512): (592, 592),
}

def str2bool(v):
    if v == 'True' or v == 'true':
        return True
    elif v == 'False' or v == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

# argument parser
def parse_args(shell_script=None):
    parser = argparse.ArgumentParser()

    # necessary arguments
    parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')
    parser.add_argument('--continue_mode', '-cont', default=False, action='store_true')
    parser.add_argument('--skip_mode', '-skip', default=False, action='store_true')
    parser.add_argument('--no_train', '-nt', default=False, action='store_true')
    parser.add_argument('--no_eval', '-ne', default=False, action='store_true')
    parser.add_argument('--no_save', '-ns', default=False, action='store_true')
    parser.add_argument('--reset_mode', '-reset', default=False, action='store_true')
    parser.add_argument('--profile_mode', '-prof', default=False, action='store_true')
    parser.add_argument('--sanity_check', '-sc', default=False, action='store_true')
    parser.add_argument('--quick_mode', '-quick', default=False, action='store_true')
    parser.add_argument('--check_mode', '-check', default=False, action='store_true')
    parser.add_argument('--use_hpu', '-hpu', default=False, action='store_true')

    parser.add_argument('--large_mode', '-large', default=False, action='store_true')
    parser.add_argument('--dinov2_large_mode', '-dlarge', default=False, action='store_true')
    parser.add_argument('--dinov2_giant_mode', '-dgiant', default=False, action='store_true')
    parser.add_argument('--vit_large_mode', '-vlarge', default=False, action='store_true')
    parser.add_argument('--clip_large_mode', '-clarge', default=False, action='store_true')
    parser.add_argument('--mae_large_mode', '-mlarge', default=False, action='store_true')
    parser.add_argument('--beit_large_mode', '-blarge', default=False, action='store_true')
    parser.add_argument('--mae_huge_mode', '-mhuge', default=False, action='store_true')
    parser.add_argument('--reg4_dinov2_large_mode', '-rlarge', default=False, action='store_true')

    parser.add_argument('--temporary_checkpointing', '-tc', default=False, action='store_true')
    parser.add_argument('--development_mode', '-dev', default=False, action='store_true')
    parser.add_argument('--benchmark_mode', '-bm', default=False, action='store_true')
    parser.add_argument('--single_gpu', '-sgpu', default=False, action='store_true')
    parser.add_argument('--slurm', default=False, action='store_true')
    parser.add_argument('--no_coco_kp', default=False, action='store_true')
    
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--task', type=str, default=None, choices=DOWNSTREAM_TASKS)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--exp_subname', type=str, default='')
    parser.add_argument('--name_postfix', '-ptf', type=str, default='')
    parser.add_argument('--subname_postfix', '-snptf', type=str, default='')
    parser.add_argument('--save_postfix', '-sptf', type=str, default='')
    parser.add_argument('--result_postfix', '-rptf', type=str, default='')
    parser.add_argument('--result_dir', '-rdir', type=str, default=None)
    parser.add_argument('--num_nodes', '-n', type=int, default=None)
    parser.add_argument('--base_device_rank', '-bdr', type=int, default=0)
    parser.add_argument('--num_devices', '-nd', type=int, default=None)

    # optional arguments
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--strategy', '-str', type=str, default=None)
    parser.add_argument('--precision', '-prec', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None, choices=(['unified'] + TRAIN_DATASETS + DOWNSTREAM_DATASETS), nargs='+')
    for dataset in TRAIN_DATASETS:
        parser.add_argument(f'--{dataset}', type=str2bool, default=None)
    parser.add_argument('--base_task', type=str2bool, default=None)
    parser.add_argument('--cont_task', type=str2bool, default=None)
    parser.add_argument('--cat_task', type=str2bool, default=None)
    parser.add_argument('--task_group', type=str, default=None)
    parser.add_argument('--use_stereo_datasets', '-usd', type=str2bool, default=None)

    parser.add_argument('--num_workers', '-nw', type=int, default=None)
    parser.add_argument('--global_batch_size', '-gbs', type=int, default=None)
    parser.add_argument('--eval_batch_size', '-ebs', type=int, default=None)
    parser.add_argument('--eval_size', '-es', type=int, default=None)
    parser.add_argument('--shot', type=int, default=None)
    parser.add_argument('--eval_shot', '-eshot', type=int, default=None)
    parser.add_argument('--max_channels', '-mc', type=int, default=None)
    parser.add_argument('--support_idx', '-sid', type=int, default=None)
    parser.add_argument('--class_name', '-class', type=str, default=None)
    parser.add_argument('--autocrop', '-acrop', type=str2bool, default=None)
    parser.add_argument('--autocrop_minoverlap', '-acropmol', type=float, default=0.5)
    parser.add_argument('--chunk_size', '-cs', type=int, default=None)
    parser.add_argument('--channel_chunk_size', '-chs', type=int, default=None)
    parser.add_argument('--randomflip', '-rf', type=str2bool, default=None)
    parser.add_argument('--randomjitter', '-jit', type=str2bool, default=None)
    parser.add_argument('--randomrotate', '-rot', type=str2bool, default=None)
    parser.add_argument('--randomblur', '-blur', type=str2bool, default=None)
    parser.add_argument('--randomscale', '-scale', type=str2bool, default=None)

    parser.add_argument('--image_augmentation', '-ia', type=str2bool, default=None)
    parser.add_argument('--label_augmentation', '-la', type=str2bool, default=None)
    parser.add_argument('--image_encoder', '-ie', type=str, default=None)
    parser.add_argument('--label_encoder', '-le', type=str, default=None)
    parser.add_argument('--decoder_features', '-df', type=int, default=None)
    parser.add_argument('--image_encoder_drop_path_rate', '-iedpr', type=float, default=None)
    parser.add_argument('--label_encoder_drop_path_rate', '-ledpr', type=float, default=None)
    parser.add_argument('--n_attn_heads', '-nah', type=int, default=None)
    parser.add_argument('--loss_type', '-lt', type=str, default=None, help='Training loss type')
    parser.add_argument('--scale_ssl', '-sssl', type=str2bool, default=None)
    parser.add_argument('--matching_alpha_init', '-mai', type=float, default=None)
    parser.add_argument('--matching_alpha_temp', '-mat', type=float, default=None)
    parser.add_argument('--head_tuning', '-ht', type=str2bool, default=None)
    parser.add_argument('--label_decoder_tuning', '-ldt', type=str2bool, default=None)
    parser.add_argument('--input_embed_tuning', '-iet', type=str2bool, default=None)
    parser.add_argument('--output_embed_tuning', '-oet', type=str2bool, default=None)
    parser.add_argument('--relpos_tuning', '-rpt', type=str2bool, default=None)
    
    parser.add_argument('--separate_alpha', '-salpha', type=str2bool, default=None)

    parser.add_argument('--n_steps', '-nst', type=int, default=None)
    parser.add_argument('--n_schedule_steps', '-nscst', type=int, default=None)
    parser.add_argument('--optimizer', '-opt', type=str, default=None, choices=['sgd', 'adam', 'adamw', 'cpuadam'])
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr_pretrained', '-lrp', type=float, default=None)
    parser.add_argument('--lr_warmup', '-lrw', type=int, default=None)
    parser.add_argument('--lr_schedule', '-lrs', type=str, default=None, choices=['constant', 'sqroot', 'cos', 'poly'])
    parser.add_argument('--schedule_from', '-scf', type=int, default=None)
    parser.add_argument('--weight_decay', '-wd', type=float, default=None)
    parser.add_argument('--early_stopping_patience', '-esp', type=int, default=None)
    parser.add_argument('--from_scratch', '-fs', type=str2bool, default=None)
    parser.add_argument('--bce_coef', '-bcec', type=float, default=None)
    parser.add_argument('--con_coef', '-conc', type=float, default=None)
    parser.add_argument('--ssl_coef', '-sslc', type=float, default=None)
    parser.add_argument('--channel_sampling', '-chsa', type=int, default=None)

    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--val_iter', '-viter', type=int, default=None)
    parser.add_argument('--save_iter', '-siter', type=int, default=None)
    parser.add_argument('--load_step', '-ls', type=int, default=None)
    parser.add_argument('--load_path', '-lpath', type=str, default=None)
    parser.add_argument('--coord_path', '-cpath', type=str, default=None)
    parser.add_argument('--monitor', '-mt', type=str, default=None)

    parser.add_argument('--img_size', '-is', type=int, default=None, nargs='+')
    parser.add_argument('--base_size', '-bs', type=int, default=None, nargs='+')

    if shell_script is not None:
        args = parser.parse_args(args=shell_script.split(' '))
    else:
        args = parser.parse_args()

    # load config file
    if args.stage == 0:
        config_path = 'meta_train/train_config.yaml'
    elif args.stage == 1:
        assert len(args.dataset) == 1
        config_path = f'downstream/{args.dataset[0]}/configs/train_config.yaml'
    elif args.stage == 2:
        assert len(args.dataset) == 1
        config_path = f'downstream/{args.dataset[0]}/configs/test_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = EasyDict(config)

    # copy parsed arguments
    for key in args.__dir__():
        if key[:2] != '__' and getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))

    # retrieve data root
    with open('data_paths.yaml', 'r') as f:
        path_dict = yaml.safe_load(f)
        config.path_dict = path_dict

    ### large mode
    if args.large_mode:
        config.image_encoder = 'beitv2_large_patch16_224'
        config.label_encoder = 'vit_large_patch16_224'
        config.name_postfix = f'_LARGE{config.name_postfix}'
        config.n_attn_heads = 16
        config.decoder_features = 256

    # image size
    if isinstance(config.img_size, int):
        config.img_size = [config.img_size, config.img_size]
    elif len(config.img_size) == 1:
        config.img_size = [config.img_size[0], config.img_size[0]]
    else:
        assert len(config.img_size) == 2

    # base size
    if isinstance(config.base_size, int):
        if config.base_size < 0:
            config.base_size = base_sizes[tuple(config.img_size)]
        else:
            config.base_size = [config.base_size, config.base_size]
    elif len(config.base_size) == 1:
        config.base_size = [config.base_size[0], config.base_size[0]]
    else:
        assert len(config.base_size) == 2

    # vis size
    if isinstance(config.vis_size, int):
        if config.vis_size < 0:
            config.vis_size = config.base_size
        else:
            config.vis_size = [config.vis_size, config.vis_size]
    elif len(config.vis_size) == 1:
        if config.vis_size < 0:
            config.vis_size = config.base_size
        else:
            config.vis_size = [config.vis_size[0], config.vis_size[0]]
    else:
        assert len(config.vis_size) == 2

    if config.autocrop:
        config.chunk_size = 1

    # for debugging
    if config.debug_mode:
        if config.profile_mode:
            config.n_steps = 500
            config.log_iter = 10
            config.val_iter = 250
            config.save_iter = 250
        elif config.stage == 1:
            config.n_steps = 10
            config.log_iter = 1
            config.val_iter = 5
            config.save_iter = 5
            config.eval_size = 2*config.eval_batch_size
        else:
            config.n_steps = 10 if args.n_steps is None else args.n_steps
            config.log_iter = 1
            config.val_iter = 5 if args.val_iter is None else args.val_iter
            config.save_iter = 5
            config.eval_size = 2*config.eval_batch_size

        config.log_dir += '_debugging'
        if config.stage == 0:
            config.load_dir += '_debugging'
        config.save_dir += '_debugging'


    if isinstance(config.dataset, str):
        config.dataset = [config.dataset]

    if config.exp_name == '':
        config.exp_name = f'VTMv2{config.name_postfix}'

    # parse dataset name
    if len(config.dataset) == 1:
        config.dataset = config.dataset[0]
        if config.dataset in TRAIN_DATASETS:
            for dataset in TRAIN_DATASETS:
                if dataset == config.dataset:
                    setattr(config, dataset, True)
                else:
                    setattr(config, dataset, False)
            config.dataset = 'unified'
    else:
        assert config.stage == 0
        for dataset in TRAIN_DATASETS:
            if dataset in config.dataset:
                setattr(config, dataset, True)
            else:
                setattr(config, dataset, False)
        config.dataset = 'unified'

    if config.stage == 0:
        if config.n_schedule_steps < 0:
            config.n_schedule_steps = config.n_steps
    
        if config.task_group == 'None':
            config.task_group = None
        
        if config.use_stereo_datasets:
            config.n_input_images = 2

    elif config.stage == 1:
        if config.class_wise:
            config.monitor = f'mtest_valid/{config.dataset}_{config.task}_{config.class_name}_{config.monitor}'
        else:
            config.monitor = f'mtest_valid/{config.dataset}_{config.task}_{config.monitor}'

    elif config.stage == 2:
        config.single_gpu = True
    
    if config.num_devices is None:
        if config.single_gpu:
            config.num_devices = 1
        else:
            import habana_frameworks.torch.hpu as hthpu
            config.num_devices = hthpu.device_count()
    elif config.num_devices == 1:
        config.single_gpu = True

    return config
