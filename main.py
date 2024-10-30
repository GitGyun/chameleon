import os
# eager mode
os.environ['PT_HPU_LAZY_MODE'] = '0'
os.environ['PT_HPU_MAX_COMPOUND_OP_SIZE'] = '1'

# lazy mode
# os.environ['PT_HPU_LAZY_MODE'] = '1'
# os.environ['PT_HPU_LAZY_ACC_PAR_MODE'] = '1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import lightning.pytorch as pl
import torch
import warnings
import habana_frameworks.torch.gpu_migration

from args import parse_args
from train.train_utils import configure_experiment, load_model, print_configs
from lightning.pytorch import seed_everything

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)
    
    # parse args
    config = parse_args()
    seed_everything(config.seed, workers=True)

    if config.slurm:
        IS_RANK_ZERO = int(os.environ.get('SLURM_LOCALID', 0)) == 0
    else:
        IS_RANK_ZERO = int(os.environ.get('LOCAL_RANK', 0)) == 0

    if not config.check_mode:
        # load model
        model, config, ckpt_path, mt_config, ft_config, ts_config = load_model(config, verbose=IS_RANK_ZERO, reduced=(config.stage > 0))

        # environmental settings
        logger, log_dir, save_dir, callbacks, profiler, precision, strategy, plugins = configure_experiment(config, model, is_rank_zero=IS_RANK_ZERO)
        model.config.ckpt_dir = save_dir
        model.config.result_dir = log_dir

        # print configs
        if IS_RANK_ZERO:
            print_configs(config, model, mt_config, ft_config, ts_config)

        # set max epochs
        if (not config.no_eval) and config.stage <= 1:
            max_epochs = config.n_steps // config.val_iter
        else:
            max_epochs = 1

        # create pytorch lightning trainer.
        trainer = pl.Trainer(
            logger=logger,
            default_root_dir=save_dir,
            accelerator='hpu',
            max_epochs=max_epochs,
            log_every_n_steps=-1,
            num_sanity_val_steps=(2 if config.sanity_check else 0),
            callbacks=callbacks,
            benchmark=True,
            devices=config.num_devices,
            strategy=strategy,
            precision=precision,
            profiler=profiler,
            plugins=plugins,
            gradient_clip_val=config.gradient_clip_val,
            num_nodes=config.num_nodes,
        )

        # validation at start
        if config.stage == 1 or (config.stage == 0 and config.no_train):
            trainer.validate(model, verbose=False)
            
        # start evaluation
        if config.stage == 2:
            trainer.test(model)
        # start training or fine-tuning
        elif not config.no_train:
            trainer.fit(model, ckpt_path=ckpt_path)

