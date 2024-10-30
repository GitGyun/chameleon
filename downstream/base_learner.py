import os
from torch.utils.data import DataLoader
from torchvision import transforms as T


class BaseLearner:
    def __init__(self, config, trainer):
        assert config.stage > 0, 'stage should be 1 (fine-tuning) or 2 (testing)'
        self.config = config
        self.trainer = trainer
        self.local_rank = self.trainer.local_rank
        self.n_devices = self.trainer.n_devices
        self.topil = T.ToPILImage()

        result_dir = self.config.result_dir
        os.makedirs(result_dir, exist_ok=True)

        if config.class_wise:
            self.result_dir = os.path.join(result_dir, self.config.class_name)
            os.makedirs(self.result_dir, exist_ok=True)
            self.tag = f'_{self.config.dataset}_{self.config.task}_{self.config.class_name}'
            self.vis_tag = f'{self.config.dataset}_{self.config.task}_{self.config.class_name}'
            self.result_path = os.path.join(result_dir, f'result_class:{self.config.class_name}.pth')
        else:
            self.result_dir = result_dir
            self.tag = f'_{self.config.dataset}_{self.config.task}'
            self.vis_tag = f'{self.config.dataset}_{self.config.task}'
            self.result_path = os.path.join(result_dir, 'result.pth')

        self.register_evaluator()

    def register_evaluator(self):
        '''
        register evaluator
        '''
        self.evaluator = None

    def reset_evaluator(self):
        pass

    def get_train_loader(self):
        assert getattr(self, 'BaseDataset', None) is not None, 'BaseDataset should be specified!'

        if self.config.no_eval:
            dset_size = self.config.n_steps*self.config.global_batch_size # whole iterations in a single epoch
        else:
            dset_size = self.config.val_iter*self.config.global_batch_size # chunk iterations in validation steps
        base_size = self.config.base_size
        crop_size = self.config.img_size

        # create dataset for episodic training
        train_data = self.BaseDataset(
            self.config,
            split='train',
            base_size=base_size,
            crop_size=crop_size,
            eval_mode=False,
            resize=False,
            dset_size=dset_size,
        )

        batch_size = self.config.global_batch_size  // self.n_devices
        train_loader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=(self.n_devices == 1), pin_memory=True, drop_last=True, num_workers=self.config.num_workers)
        
        return train_loader
    
    def get_support_data(self):
        assert getattr(self, 'BaseDataset', None) is not None, 'BaseDataset should be specified!'

        dset_size = self.config.eval_shot if (self.config.eval_shot > 0 and not self.config.autocrop) else self.config.shot
        base_size = self.config.base_size
        crop_size = self.config.img_size

        # create dataset for episodic training
        train_data = self.BaseDataset(
            self.config,
            base_size=base_size,
            crop_size=crop_size,
            split='train',
            eval_mode=True,
            resize=True,
            dset_size=dset_size,
        )

        support_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False, drop_last=False, num_workers=0)
        for batch in support_loader:
            break
        X, Y, M, *_ = batch
        
        return X, Y, M

    def get_val_loaders(self):
        assert getattr(self, 'BaseDataset', None) is not None, 'BaseDataset should be specified!'
        
        valid_loaders_list = []
        for split in ['train', 'valid']:
            if split == 'train':
                dset_size = min(self.config.shot, self.config.eval_size)
            else:
                dset_size = self.config.eval_size
            base_size = crop_size = self.config.base_size
            
            eval_data = self.BaseDataset(
                self.config,
                base_size=base_size,
                crop_size=crop_size,
                split=split,
                eval_mode=True,
                resize=False,
                dset_size=dset_size,
            )

            batch_size = self.config.eval_batch_size // self.n_devices if not self.config.autocrop else 1

            valid_loaders = {
                self.config.task:
                DataLoader(eval_data, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=0)
            }
            valid_loaders_list.append(valid_loaders)

        loader_tags = ['mtest_train', 'mtest_valid']

        return valid_loaders_list, loader_tags

    def get_test_loader(self):
        assert getattr(self, 'BaseDataset', None) is not None, 'BaseDataset should be specified!'
        
        batch_size = self.config.eval_batch_size // self.n_devices
        dset_size = self.config.eval_size
        base_size = crop_size = self.config.base_size
        
        test_data = self.BaseDataset(
            self.config,
            base_size=base_size,
            crop_size=crop_size,
            split='test',
            eval_mode=True,
            resize=False,
            dset_size=dset_size,
        )
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=1)

        return test_loader

    def compute_loss(self, Y_pred, Y, M):
        '''
        loss function that returns loss and a dictionary of its components
        '''
        raise NotImplementedError

    def postprocess_logits(self, Y_pred_out):
        '''
        post-processing function for logits
        '''
        return Y_pred_out

    def postprocess_final(self, Y_pred):
        '''
        post-processing function for final prediction
        '''
        return Y_pred
    
    def postprocess_vis(self, label, img=None, aux=None):
        '''
        post-processing function for visualization
        '''
        return label

    def compute_metric(self, Y, Y_pred, M, aux, evaluator_key=None):
        '''
        compute evaluation metric
        '''
        raise NotImplementedError

    def log_metrics(self, loss_pred, log_dict, valid_tag):
        '''
        log evaluation metrics
        '''
        raise NotImplementedError

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
    
