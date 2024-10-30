import torch
from torch.utils.data import Dataset
from .datasets.taskonomy import *
from .datasets.coco import *
from .datasets.midair import *
from .datasets.mpii import *
from .datasets.deepfashion import *
from .datasets.freihand import *
from .datasets.midair_stereo import *
from .datasets.coco_stereo import *


base_class_dict = {
    'taskonomy': Taskonomy,
    'coco': COCO,
    'midair': MidAir,
    'mpii': MPII,
    'deepfashion': DeepFashion,
    'freihand': FreiHand,
    'midair_stereo': MidAirStereo,
    'coco_stereo': COCOStereo,
}


train_dataset_dict = {
    'taskonomy': {
        'base': TaskonomyBaseTrainDataset,
        'continuous': TaskonomyContinuousTrainDataset,
        'categorical': TaskonomyCategoricalTrainDataset,
    },
    'coco': {
        'base': COCOBaseTrainDataset,
        'continuous': COCOContinuousTrainDataset,
        'categorical': COCOCategoricalTrainDataset,
    },
    'midair': {
        'base': MidAirBaseTrainDataset,
        'continuous': MidAirContinuousTrainDataset,
        'categorical': MidAirCategoricalTrainDataset,
    },
    'mpii': {
        'base': MPIIBaseTrainDataset,
        'continuous': MPIIContinuousTrainDataset,
        'categorical': MPIICategoricalTrainDataset,
    },
    'deepfashion': {
        'base': DeepFashionBaseTrainDataset,
        'continuous': DeepFashionContinuousTrainDataset,
        'categorical': DeepFashionCategoricalTrainDataset,
    },
    'freihand': {
        'base': FreiHandBaseTrainDataset,
        'continuous': FreiHandContinuousTrainDataset,
        'categorical': FreiHandCategoricalTrainDataset,
    },
}

train_stereo_dataset_dict = {
    'midair_stereo': {
        'continuous': MidAirStereoContinuousTrainDataset,
    },
    'coco_stereo': {
        'categorical': COCOStereoCategoricalTrainDataset,
    },
}


test_dataset_dict = {
    'taskonomy': {
        'base': TaskonomyBaseTestDataset,
        'continuous': TaskonomyBaseTestDataset,
        'categorical': TaskonomyCategoricalTestDataset,
    },
    'coco': {
        'base': COCOBaseTestDataset,
        'continuous': COCOContinuousTestDataset,
        'categorical': COCOCategoricalTestDataset,
    },
    'midair': {
        'base': MidAirBaseTestDataset,
        'continuous': MidAirBaseTestDataset,
        'categorical': MidAirCategoricalTestDataset,
    },
    'mpii': {
        'base': MPIIBaseTestDataset,
        'continuous': MPIIContinuousTestDataset,
        'categorical': MPIICategoricalTestDataset,
    },
    'deepfashion': {
        'base': DeepFashionBaseTestDataset,
        'continuous': DeepFashionContinuousTestDataset,
        'categorical': DeepFashionCategoricalTestDataset,
    },
    'freihand': {
        'base': FreiHandBaseTestDataset,
        'continuous': FreiHandContinuousTestDataset,
        'categorical': FreiHandCategoricalTestDataset,
    },
    'midair_stereo': {
        'continuous': MidAirStereoTestDataset,
    },
    'coco_stereo': {
        'categorical': COCOStereoCategoricalTestDataset,
    },
}


class Unified(Dataset):
    '''
    base class for the unified dataset
    '''
    NAME = 'unified'

    # Tasks
    TASKS = []
    TASKS_BASE = []
    TASKS_CONTINUOUS = []
    TASKS_CATEGORICAL = []

    # Task Groups
    TASK_GROUP_NAMES = []
    TASK_GROUP_NAMES_BASE = []
    TASK_GROUP_NAMES_CONTINUOUS = []
    TASK_GROUP_NAMES_CATEGORICAL = []

    TASK_GROUP_TYPE = {}
    CHANNELS_DICT = {}

    for dataset_name, dataset in base_class_dict.items():
        for task_group_name in dataset.TASK_GROUP_NAMES_BASE:
            TASK_GROUP_NAMES.append((dataset_name, task_group_name))
            TASK_GROUP_NAMES_BASE.append((dataset_name, task_group_name))
            TASK_GROUP_TYPE[(dataset_name, task_group_name)] = 'base'
            for task in dataset.TASK_GROUP_DICT[task_group_name]:
                TASKS.append((dataset_name, task))
                TASKS_BASE.append((dataset_name, task))
            CHANNELS_DICT[(dataset_name, task_group_name)] = dataset.CHANNELS_DICT[task_group_name]

        for task_group_name in dataset.TASK_GROUP_NAMES_CONTINUOUS:
            TASK_GROUP_NAMES.append((dataset_name, task_group_name))
            TASK_GROUP_NAMES_CONTINUOUS.append((dataset_name, task_group_name))
            TASK_GROUP_TYPE[(dataset_name, task_group_name)] = 'continuous'
            for task in dataset.TASK_GROUP_DICT[task_group_name]:
                TASKS.append((dataset_name, task))
                TASKS_CONTINUOUS.append((dataset_name, task))
            CHANNELS_DICT[(dataset_name, task_group_name)] = dataset.CHANNELS_DICT[task_group_name]

        for task_group_name in dataset.TASK_GROUP_NAMES_CATEGORICAL:
            TASK_GROUP_NAMES.append((dataset_name, task_group_name))
            TASK_GROUP_NAMES_CATEGORICAL.append((dataset_name, task_group_name))
            TASK_GROUP_TYPE[(dataset_name, task_group_name)] = 'categorical'
            for task in dataset.TASK_GROUP_DICT[task_group_name]:
                TASKS.append((dataset_name, task))
                TASKS_CATEGORICAL.append((dataset_name, task))
            CHANNELS_DICT[(dataset_name, task_group_name)] = dataset.CHANNELS_DICT[task_group_name]


class UnifiedTrainDataset(Unified):
    '''
    unified dataset for meta-training
    '''
    def __init__(self, config, dset_size, verbose=True, **kwargs):
        self.shot = config.shot
        self.max_channels = config.max_channels
        self.dset_size = dset_size
        self.stereo = config.use_stereo_datasets

        self._register_datasets(config, verbose, **kwargs)
        self._register_samplers(config.task_sampling_weight)
        if self.stereo:
            self._register_stereo_datasets(config, verbose, **kwargs)
            self._register_stereo_samplers(config.task_sampling_weight)

    def _check_add_dataset(self, config, dataset_dict, task_type):
        assert task_type in ['base', 'continuous', 'categorical']
        alias_dict = {'base': 'base', 'continuous': 'cont', 'categorical': 'cat'}
        if getattr(config, f'{alias_dict[task_type]}_task') and task_type in dataset_dict:
            if config.task_group is not None:
                add = config.task_group in getattr(dataset_dict[task_type], f'TASK_GROUP_NAMES_{task_type.upper()}')
            else:
                add = True
        else:
            add = False

        return add
    
    def _add_dataset(self, datasets, tasks, task_group_names, task_type, config, dataset_name, dataset_dict, label_augmentation, **kwargs):
        assert task_type in ['base', 'continuous', 'categorical']
        datasets.append(
            dataset_dict[task_type](label_augmentation, **kwargs)
        )
    
        # register base tasks
        task_group_names_ = []
        task_groups = []
        for task in getattr(dataset_dict[task_type], f'TASKS_{task_type.upper()}'):
            task_group_name = '_'.join(task.split('_')[:-1])
            task_group = dataset_dict[task_type].TASK_GROUP_DICT[task_group_name]
            # if task_group is specified, only add tasks from the specified task group
            if config.task_group is not None:
                if task_group_name == config.task_group:
                    tasks.append((dataset_name, task))
                    if task_group_name not in task_group_names_:
                        task_group_names_.append(task_group_name)
                        task_groups.append(task_group)
            # otherwise, add all tasks
            else:
                tasks.append((dataset_name, task))
                if task_group_name not in task_group_names_:
                    task_group_names_.append(task_group_name)
                    task_groups.append(task_group)

        assert len(task_groups) > 0, (task_type, dataset_name)
        datasets[-1].task_groups = task_groups
        for task_group_name in task_group_names_:
            task_group_names.append((dataset_name, task_group_name))
        
    def _register_datasets(self, config, verbose, label_augmentation, **kwargs):
        if verbose:
            print('Registering datasets...')

        self.base_datasets = []
        self.continuous_datasets = []
        self.categorical_datasets = []
        self.tasks = []
        self.task_group_names = []
        for dataset_name, dataset_dict in train_dataset_dict.items():
            # iterate over datasets
            if getattr(config, dataset_name, False):
                # add base tasks
                if self._check_add_dataset(config, dataset_dict, 'base'):
                    self._add_dataset(self.base_datasets, self.tasks, self.task_group_names, 'base', config,
                                        dataset_name, dataset_dict, label_augmentation, **kwargs)
                    
                # add continuous tasks
                if self._check_add_dataset(config, dataset_dict, 'continuous'):
                    if not (config.no_coco_kp and dataset_name == 'coco'):
                        self._add_dataset(self.continuous_datasets, self.tasks, self.task_group_names, 'continuous', config,
                                            dataset_name, dataset_dict, label_augmentation, **kwargs)
                    
                # add categorical tasks
                if self._check_add_dataset(config, dataset_dict, 'categorical'):
                    if config.no_coco_kp and dataset_name == 'coco':
                        self._add_dataset(self.categorical_datasets, self.tasks, self.task_group_names, 'categorical', config,
                                            dataset_name, dataset_dict, label_augmentation, no_coco_kp=True, **kwargs)
                    else:
                        self._add_dataset(self.categorical_datasets, self.tasks, self.task_group_names, 'categorical', config,
                                            dataset_name, dataset_dict, label_augmentation, **kwargs)

                if verbose:
                    print(f'{dataset_name} dataset registered')
            
        self.base_size = sum([len(dset) for dset in self.base_datasets])
        self.continuous_size = sum([len(dset) for dset in self.continuous_datasets])
        self.categorical_size = sum([len(dset) for dset in self.categorical_datasets])
        if verbose:
            print(f'Total {self.base_size} base images, {self.continuous_size} continuous images, '
                  f'{self.categorical_size} categorical images with {len(self.tasks)} tasks of {len(self.task_group_names)} groups are registered.')
    
    def _register_stereo_datasets(self, config, verbose, label_augmentation, **kwargs):
        if verbose:
            print('Registering stereo datasets...')

        self.base_stereo_datasets = []
        self.continuous_stereo_datasets = []
        self.categorical_stereo_datasets = []
        self.stereo_tasks = []
        self.stereo_task_group_names = []
        for dataset_name, dataset_dict in train_stereo_dataset_dict.items():
            # iterate over datasets
            if getattr(config, dataset_name, False):
                # add base tasks
                if self._check_add_dataset(config, dataset_dict, 'base'):
                    self._add_dataset(self.base_stereo_datasets, self.stereo_tasks, self.stereo_task_group_names, 'base', config,
                                      dataset_name, dataset_dict, label_augmentation, **kwargs)
                    
                # add continuous tasks
                if self._check_add_dataset(config, dataset_dict, 'continuous'):
                    self._add_dataset(self.continuous_stereo_datasets, self.stereo_tasks, self.stereo_task_group_names, 'continuous', config,
                                      dataset_name, dataset_dict, label_augmentation, **kwargs)
                    
                # add categorical tasks
                if self._check_add_dataset(config, dataset_dict, 'categorical'):
                    self._add_dataset(self.categorical_stereo_datasets, self.stereo_tasks, self.stereo_task_group_names, 'categorical', config,
                                      dataset_name, dataset_dict, label_augmentation, **kwargs)

                if verbose:
                    print(f'{dataset_name} dataset registered')
            
        self.base_stereo_size = sum([len(dset) for dset in self.base_stereo_datasets])
        self.continuous_stereo_size = sum([len(dset) for dset in self.continuous_stereo_datasets])
        self.categorical_stereo_size = sum([len(dset) for dset in self.categorical_stereo_datasets])
        if verbose:
            print(f'Total {self.base_stereo_size} base stereo images, {self.continuous_stereo_size} continuous stereo images, '
                  f'{self.categorical_stereo_size} categorical stereo images with {len(self.stereo_tasks)} tasks of {len(self.stereo_task_group_names)} groups are registered.')

    def _register_samplers(self, task_sampling_weight=None):
        task_types = ['base', 'continuous', 'categorical']
        
        if task_sampling_weight is None:
            task_sampling_weight = [1. for _ in task_types]
        else:
            assert len(task_sampling_weight) == len(task_types), task_sampling_weight

        p_task_type = torch.tensor([task_sampling_weight[i] if len(getattr(self, f'{task_type}_datasets')) > 0 else 0.
                                    for i, task_type in enumerate(task_types)])
        assert p_task_type.sum() > 0
        p_task_type = p_task_type / p_task_type.sum()
        self.task_type_sampler = torch.distributions.Categorical(p_task_type)

        criterion_dict = {
            'base': lambda dset: len(dset),
            'continuous': lambda dset: len(dset.TASKS_CONTINUOUS),
            'categorical': lambda dset: len(dset.TASKS_CATEGORICAL),
        }

        for task_type in task_types:
            if len(getattr(self, f'{task_type}_datasets')) > 0:
                p_datasets = torch.tensor([criterion_dict[task_type](dset) for dset in getattr(self, f'{task_type}_datasets')])
                p_datasets = p_datasets / p_datasets.sum()
                setattr(self, f'{task_type}_dataset_sampler', torch.distributions.Categorical(p_datasets))

    def _register_stereo_samplers(self, ask_sampling_weight=None):
        task_types = ['base', 'continuous', 'categorical']
        
        if task_sampling_weight is None:
            task_sampling_weight = [1. for _ in task_types]
        else:
            assert len(task_sampling_weight) == len(task_types), task_sampling_weight

        p_task_type = torch.tensor([task_sampling_weight[i] if len(getattr(self, f'{task_type}_stereo_datasets')) > 0 else 0.
                                    for i, task_type in enumerate(task_types)])
        assert p_task_type.sum() > 0
        p_task_type = p_task_type / p_task_type.sum()
        self.stereo_task_type_sampler = torch.distributions.Categorical(p_task_type)

        criterion_dict = {
            'base': lambda dset: len(dset),
            'continuous': lambda dset: len(dset.TASKS_CONTINUOUS),
            'categorical': lambda dset: len(dset.TASKS_CATEGORICAL),
        }

        for task_type in task_types:
            if len(getattr(self, f'{task_type}_stereo_datasets')) > 0:
                p_datasets = torch.tensor([criterion_dict[task_type](dset) for dset in getattr(self, f'{task_type}_stereo_datasets')])
                p_datasets = p_datasets / p_datasets.sum()
                setattr(self, f'{task_type}_stereo_dataset_sampler', torch.distributions.Categorical(p_datasets))
        
    def __len__(self):
        return self.dset_size

    def sample_episode(self):
        # sample task type
        task_type = self.task_type_sampler.sample().item()

        # sample dataset
        if task_type == 0:
            dataset_idx = self.base_dataset_sampler.sample().item()
            dataset = self.base_datasets[dataset_idx]
        elif task_type == 1:
            dataset_idx = self.continuous_dataset_sampler.sample().item()
            dataset = self.continuous_datasets[dataset_idx]
        elif task_type == 2:
            dataset_idx = self.categorical_dataset_sampler.sample().item()
            dataset = self.categorical_datasets[dataset_idx]

        # sample episode
        X, Y, M, t_idx, g_idx, channel_mask = dataset.sample_episode(2*self.shot, n_channels=self.max_channels, n_domains=2)

        # reindex task
        t_idx_new = []
        for t_idx_ in t_idx:
            if t_idx_.item() >= 0:
                task = dataset.TASKS[t_idx_.item()]
                t_idx_new_ = torch.tensor(self.TASKS.index((dataset.NAME, task)))
            else:
                t_idx_new_ = t_idx_
            t_idx_new.append(t_idx_new_)
        t_idx = torch.stack(t_idx_new)

        # reindex group
        if g_idx.item() >= 0:
            task_group_name = dataset.TASK_GROUP_NAMES[g_idx.item()]
            g_idx = torch.tensor(self.TASK_GROUP_NAMES.index((dataset.NAME, task_group_name)))

        return X, Y, M, t_idx, g_idx, channel_mask

    def sample_stereo_episode(self):
        # sample task type
        task_type = self.stereo_task_type_sampler.sample().item()

        # sample dataset
        if task_type == 0:
            dataset_idx = self.base_stereo_dataset_sampler.sample().item()
            dataset = self.base_stereo_datasets[dataset_idx]
        elif task_type == 1:
            dataset_idx = self.continuous_stereo_dataset_sampler.sample().item()
            dataset = self.continuous_stereo_datasets[dataset_idx]
        elif task_type == 2:
            dataset_idx = self.categorical_stereo_dataset_sampler.sample().item()
            dataset = self.categorical_stereo_datasets[dataset_idx]

        # sample episode
        X, Y, M, t_idx, g_idx, channel_mask = dataset.sample_episode(2*self.shot, n_channels=max(1, self.max_channels//2), n_domains=2)

        # reindex task
        t_idx_new = []
        for t_idx_ in t_idx:
            if t_idx_.item() >= 0:
                task = dataset.TASKS[t_idx_.item()]
                t_idx_new_ = torch.tensor(self.TASKS.index((dataset.NAME, task)))
            else:
                t_idx_new_ = t_idx_
            t_idx_new.append(t_idx_new_)
        t_idx = torch.stack(t_idx_new)

        # reindex group
        if g_idx.item() >= 0:
            task_group_name = dataset.TASK_GROUP_NAMES[g_idx.item()]
            g_idx = torch.tensor(self.TASK_GROUP_NAMES.index((dataset.NAME, task_group_name)))

        return X, Y, M, t_idx, g_idx, channel_mask
    
    def __getitem__(self, idx):
        batch = self.sample_episode()
        if self.stereo:
            stereo_batch = self.sample_stereo_episode()
            return batch, stereo_batch
        else:
            return batch