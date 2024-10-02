from .davis2017.learner import DAVIS2017Learner
from .ap10k.learner import AP10KLearner
from .linemod.learner import LINEMODLearner, LINEMODMaskLearner
from .isic2018.learner import ISIC2018Learner
from .cellpose.learner import CELLPOSELearner
from .fsc147.learner import FSC147Learner


def get_downstream_learner(config, trainer):
    '''
    add custom learner here
    '''
    if config.dataset == 'davis2017':
        return DAVIS2017Learner(config, trainer)
    elif config.dataset == 'ap10k':
        return AP10KLearner(config, trainer)
    elif config.dataset == 'linemod':
        if config.task == 'pose_6d':
            return LINEMODLearner(config, trainer)
        elif config.task == 'segment_semantic':
            return LINEMODMaskLearner(config, trainer)
        else:
            raise NotImplementedError
    elif config.dataset == 'isic2018':
        return ISIC2018Learner(config, trainer)
    elif config.dataset == 'cellpose':
        return CELLPOSELearner(config, trainer)
    elif config.dataset == 'fsc147':
        return FSC147Learner(config, trainer)
    else:
        raise NotImplementedError