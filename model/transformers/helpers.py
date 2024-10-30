""" Model creation / weight loading / state_dict helpers
Hacked together by / Copyright 2020 Ross Wightman
"""
import collections.abc
import logging
import math
import os
import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch.utils.checkpoint import checkpoint

from timm.models.features import FeatureListNet, FeatureDictNet, FeatureHookNet
from timm.models.fx_features import FeatureGraphNet
from timm.models.hub import has_hf_hub, download_cached_file, load_state_dict_from_hf
from timm.models.layers import Conv2dSame, Linear, BatchNormAct2d
from .registry import get_pretrained_cfg
from einops import repeat, rearrange
import numpy as np
from scipy import interpolate


_logger = logging.getLogger(__name__)


# Global variables for rarely used pretrained checkpoint download progress and hash check.
# Use set_pretrained_download_progress / set_pretrained_check_hash functions to toggle.
_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_state_dict(checkpoint_path, use_ema=True):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=True, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            state_dict = clean_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def _resolve_pretrained_source(pretrained_cfg):
    cfg_source = pretrained_cfg.get('source', '')
    pretrained_url = pretrained_cfg.get('url', None)
    pretrained_file = pretrained_cfg.get('file', None)
    hf_hub_id = pretrained_cfg.get('hf_hub_id', None)
    # resolve where to load pretrained weights from
    load_from = ''
    pretrained_loc = ''
    if cfg_source == 'hf-hub' and has_hf_hub(necessary=True):
        # hf-hub specified as source via model identifier
        load_from = 'hf-hub'
        assert hf_hub_id
        pretrained_loc = hf_hub_id
    else:
        # default source == timm or unspecified
        if pretrained_file:
            load_from = 'file'
            pretrained_loc = pretrained_file
        elif pretrained_url:
            load_from = 'url'
            pretrained_loc = pretrained_url
        elif hf_hub_id and has_hf_hub(necessary=True):
            # hf-hub available as alternate weight source in default_cfg
            load_from = 'hf-hub'
            pretrained_loc = hf_hub_id
    if load_from == 'hf-hub' and 'hf_hub_filename' in pretrained_cfg:
        # if a filename override is set, return tuple for location w/ (hub_id, filename)
        pretrained_loc = pretrained_loc, pretrained_cfg['hf_hub_filename']
    return load_from, pretrained_loc


def set_pretrained_download_progress(enable=True):
    """ Set download progress for pretrained weights on/off (globally). """
    global _DOWNLOAD_PROGRESS
    _DOWNLOAD_PROGRESS = enable


def set_pretrained_check_hash(enable=True):
    """ Set hash checking for pretrained weights on/off (globally). """
    global _CHECK_HASH
    _CHECK_HASH = enable


def load_custom_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        load_fn: Optional[Callable] = None,
):
    r"""Loads a custom (read non .pth) weight file
    Downloads checkpoint file into cache-dir like torch.hub based loaders, but calls
    a passed in custom load fun, or the `load_pretrained` model member fn.
    If the object is already present in `model_dir`, it's deserialized and returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.
    Args:
        model: The instantiated model to load weights into
        pretrained_cfg (dict): Default pretrained model cfg
        load_fn: An external stand alone fn that loads weights into provided model, otherwise a fn named
            'laod_pretrained' on the model will be called if it exists
    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None) or {}
    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if not load_from:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    if load_from == 'hf-hub':  # FIXME
        _logger.warning("Hugging Face hub not currently supported for custom load pretrained models.")
    elif load_from == 'url':
        pretrained_loc = download_cached_file(pretrained_loc, check_hash=_CHECK_HASH, progress=_DOWNLOAD_PROGRESS)

    if load_fn is not None:
        load_fn(model, pretrained_loc)
    elif hasattr(model, 'load_pretrained'):
        model.load_pretrained(pretrained_loc)
    else:
        _logger.warning("Valid function to load pretrained weights is not available, using random initialization.")


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def load_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        num_classes: int = 1000,
        in_chans: int = 3,
        filter_fn: Optional[Callable] = None,
        strict: bool = True,
):
    """ Load pretrained checkpoint
    Args:
        model (nn.Module) : PyTorch model module
        pretrained_cfg (Optional[Dict]): configuration for pretrained weights / target dataset
        num_classes (int): num_classes for model
        in_chans (int): in_chans for model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint
    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None) or {}
    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if load_from == 'file':
        _logger.info(f'Loading pretrained weights from file ({pretrained_loc})')
        state_dict = load_state_dict(pretrained_loc)
    elif load_from == 'url':
        _logger.info(f'Loading pretrained weights from url ({pretrained_loc})')
        state_dict = load_state_dict_from_url(
            pretrained_loc, map_location='cpu', progress=_DOWNLOAD_PROGRESS, check_hash=_CHECK_HASH)
    elif load_from == 'hf-hub':
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        if isinstance(pretrained_loc, (list, tuple)):
            state_dict = load_state_dict_from_hf(*pretrained_loc)
        else:
            state_dict = load_state_dict_from_hf(pretrained_loc)
    else:
        _logger.warning("No pretrained weights exist or were found for this model. Using random initialization.")
        return

    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    input_convs = pretrained_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = pretrained_cfg.get('classifier', None)
    label_offset = pretrained_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                state_dict.pop(classifier_name + '.weight', None)
                state_dict.pop(classifier_name + '.bias', None)
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    custom_load_state_dict(model, state_dict, strict=strict)
    # model.load_state_dict(state_dict, strict=strict)


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_string(parent_module, model_string):
    separator = '***'
    state_dict = {}
    lst_shape = model_string.split(separator)
    for k in lst_shape:
        k = k.split(':')
        key = k[0]
        shape = k[1][1:-1].split(',')
        if shape[0] != '':
            state_dict[key] = [int(i) for i in shape]

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if isinstance(old_module, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        elif isinstance(old_module, BatchNormAct2d):
            new_bn = BatchNormAct2d(
                state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            new_bn.drop = old_module.drop
            new_bn.act = old_module.act
            set_layer(new_module, n, new_bn)
        elif isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)
        elif isinstance(old_module, nn.Linear):
            # FIXME extra checks to ensure this is actually the FC classifier layer and not a diff Linear layer?
            num_features = state_dict[n + '.weight'][1]
            new_fc = Linear(
                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                new_module.num_features = num_features
    new_module.eval()
    parent_module.eval()

    return new_module


def adapt_model_from_file(parent_module, model_variant):
    adapt_file = os.path.join(os.path.dirname(__file__), 'pruned', model_variant + '.txt')
    with open(adapt_file, 'r') as f:
        return adapt_model_from_string(parent_module, f.read().strip())


def pretrained_cfg_for_features(pretrained_cfg):
    pretrained_cfg = deepcopy(pretrained_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'crop_pct', 'classifier', 'global_pool')  # add default final pool size?
    for tr in to_remove:
        pretrained_cfg.pop(tr, None)
    return pretrained_cfg


def set_default_kwargs(kwargs, names, pretrained_cfg):
    for n in names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # pretrained_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])


def filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def update_pretrained_cfg_and_kwargs(pretrained_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model
    Args:
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)
    set_default_kwargs(kwargs, names=default_kwarg_names, pretrained_cfg=pretrained_cfg)
    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    filter_kwargs(kwargs, names=kwargs_filter)


def resolve_pretrained_cfg(variant: str, pretrained_cfg=None):
    if pretrained_cfg and isinstance(pretrained_cfg, dict):
        # highest priority, pretrained_cfg available and passed as arg
        return deepcopy(pretrained_cfg)
    # fallback to looking up pretrained cfg in model registry by variant identifier
    pretrained_cfg = get_pretrained_cfg(variant)
    if not pretrained_cfg:
        _logger.warning(
            f"No pretrained configuration specified for {variant} model. Using a default."
            f" Please add a config to the model pretrained_cfg registry or pass explicitly.")
        pretrained_cfg = dict(
            url='',
            num_classes=1000,
            input_size=(3, 224, 224),
            pool_size=None,
            crop_pct=.9,
            interpolation='bicubic',
            first_conv='',
            classifier='',
        )
    return pretrained_cfg


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        pretrained_cfg: Optional[Dict] = None,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[Dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        pretrained_custom_load: bool = False,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs):
    """ Build model with specified default_cfg and optional model_cfg
    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation
    Args:
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        pretrained_cfg (dict): model's pretrained weight/task config
        model_cfg (Optional[Dict]): model's architecture config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        pretrained_strict (bool): load pretrained weights strictly
        pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
        pretrained_custom_load (bool): use custom load fn, to load numpy or other non PyTorch weights
        kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=pretrained_cfg)
    update_pretrained_cfg_and_kwargs(pretrained_cfg, kwargs, kwargs_filter)
    pretrained_cfg.setdefault('architecture', variant)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    # Build the model
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat
    
    if pruned:
        model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        if pretrained_custom_load:
            # FIXME improve custom load trigger
            load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
        else:
            load_pretrained(
                model,
                pretrained_cfg=pretrained_cfg,
                num_classes=num_classes_pretrained,
                in_chans=kwargs.get('in_chans', 3),
                filter_fn=pretrained_filter_fn,
                strict=pretrained_strict)

    # Wrap the model in a feature extraction module if enabled
    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                elif feature_cls == 'fx':
                    feature_cls = FeatureGraphNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)  # add back default_cfg
        model.default_cfg = model.pretrained_cfg  # alias for backwards compat
    
    return model


def model_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params based on ordering
        return [p for p in model.parameters()][:-2]
    else:
        return model.parameters()


def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def named_modules(module: nn.Module, name='', depth_first=True, include_root=False):
    if not depth_first and include_root:
        yield name, module
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        yield from named_modules(
            module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        yield name, module


def named_modules_with_params(module: nn.Module, name='', depth_first=True, include_root=False):
    if module._parameters and not depth_first and include_root:
        yield name, module
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        yield from named_modules_with_params(
            module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if module._parameters and depth_first and include_root:
        yield name, module


MATCH_PREV_GROUP = (99999,)


def group_with_matcher(
        named_objects,
        group_matcher: Union[Dict, Callable],
        output_values: bool = False,
        reverse: bool = False
):
    if isinstance(group_matcher, dict):
        # dictionary matcher contains a dict of raw-string regex expr that must be compiled
        compiled = []
        for group_ordinal, (group_name, mspec) in enumerate(group_matcher.items()):
            if mspec is None:
                continue
            # map all matching specifications into 3-tuple (compiled re, prefix, suffix)
            if isinstance(mspec, (tuple, list)):
                # multi-entry match specifications require each sub-spec to be a 2-tuple (re, suffix)
                for sspec in mspec:
                    compiled += [(re.compile(sspec[0]), (group_ordinal,), sspec[1])]
            else:
                compiled += [(re.compile(mspec), (group_ordinal,), None)]
        group_matcher = compiled

    def _get_grouping(name):
        if isinstance(group_matcher, (list, tuple)):
            for match_fn, prefix, suffix in group_matcher:
                r = match_fn.match(name)
                if r:
                    parts = (prefix, r.groups(), suffix)
                    # map all tuple elem to int for numeric sort, filter out None entries
                    return tuple(map(float, chain.from_iterable(filter(None, parts))))
            return float('inf'),  # un-matched layers (neck, head) mapped to largest ordinal
        else:
            ord = group_matcher(name)
            if not isinstance(ord, collections.abc.Iterable):
                return ord,
            return tuple(ord)

    # map layers into groups via ordinals (ints or tuples of ints) from matcher
    grouping = defaultdict(list)
    for k, v in named_objects:
        grouping[_get_grouping(k)].append(v if output_values else k)

    # remap to integers
    layer_id_to_param = defaultdict(list)
    lid = -1
    for k in sorted(filter(lambda x: x is not None, grouping.keys())):
        if lid < 0 or k[-1] != MATCH_PREV_GROUP[0]:
            lid += 1
        layer_id_to_param[lid].extend(grouping[k])

    if reverse:
        assert not output_values, "reverse mapping only sensible for name output"
        # output reverse mapping
        param_to_layer_id = {}
        for lid, lm in layer_id_to_param.items():
            for n in lm:
                param_to_layer_id[n] = lid
        return param_to_layer_id

    return layer_id_to_param


def group_parameters(
        module: nn.Module,
        group_matcher,
        output_values=False,
        reverse=False,
):
    return group_with_matcher(
        module.named_parameters(), group_matcher, output_values=output_values, reverse=reverse)


def group_modules(
        module: nn.Module,
        group_matcher,
        output_values=False,
        reverse=False,
):
    return group_with_matcher(
        named_modules_with_params(module), group_matcher, output_values=output_values, reverse=reverse)


def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.
    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.
    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


def flatten_modules(named_modules, depth=1, prefix='', module_types='sequential'):
    prefix_is_tuple = isinstance(prefix, tuple)
    if isinstance(module_types, str):
        if module_types == 'container':
            module_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        else:
            module_types = (nn.Sequential,)
    for name, module in named_modules:
        if depth and isinstance(module, module_types):
            yield from flatten_modules(
                module.named_children(),
                depth - 1,
                prefix=(name,) if prefix_is_tuple else name,
                module_types=module_types,
            )
        else:
            if prefix_is_tuple:
                name = prefix + (name,)
                yield name, module
            else:
                if prefix:
                    name = '.'.join([prefix, name])
                yield name, module


def gen_relative_position_index(window_size, latent_size: int = 0) -> torch.Tensor:
    if len(window_size) == 2: # D x H x W
        window_size = (1, window_size[0], window_size[1])

    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1) + 3

    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1] * window_size[2]
    coords = torch.stack(torch.meshgrid(
        [torch.arange(window_size[0]),
         torch.arange(window_size[1]),
         torch.arange(window_size[2])], indexing='ij'))  # 3, Wd, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 2] += window_size[2] - 1
    relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
    relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
    relative_position_index = torch.zeros(size=(window_area + 1 + latent_size,) * 2, dtype=relative_coords.dtype)

    # token to token
    relative_position_index[1+latent_size:, 1+latent_size:] = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

    # token to cls & cls to token & cls to cls
    relative_position_index[latent_size, latent_size:] = num_relative_distance - 3
    relative_position_index[latent_size:, latent_size] = num_relative_distance - 2
    relative_position_index[latent_size, latent_size] = num_relative_distance - 1

    # latent to latent
    relative_position_index[:latent_size, :latent_size] = num_relative_distance

    # latent to token & token to latent
    relative_position_index[:latent_size, 1+latent_size:] = num_relative_distance + 1
    relative_position_index[1+latent_size:, :latent_size] = num_relative_distance + 2

    # latent to cls & cls to latent
    relative_position_index[:latent_size, latent_size] = num_relative_distance + 3
    relative_position_index[latent_size, :latent_size] = num_relative_distance + 4

    return relative_position_index


def resize_rel_pos_bias(rel_pos_bias, dst_num_pos, depth_src=1, depth_tar=1, verbose=True, verbose_tag='', dst_img_size=None, patch_size=16):
    src_num_pos, num_attn_heads = rel_pos_bias.size()
    num_extra_pos = 3

    src_size = (int(((src_num_pos - num_extra_pos) // (2*depth_src - 1)) ** 0.5),
                int(((src_num_pos - num_extra_pos) // (2*depth_src - 1)) ** 0.5))
    if dst_img_size is None:
        dst_size = (int(((dst_num_pos - num_extra_pos) // (2*depth_tar - 1)) ** 0.5),
                    int(((dst_num_pos - num_extra_pos) // (2*depth_tar - 1)) ** 0.5))
    else:
        dst_size = ((dst_img_size[0] // patch_size)*2 - 1, (dst_img_size[1] // patch_size)*2 - 1)
        assert dst_size[0]*dst_size[1]*(2*depth_tar - 1) + num_extra_pos == dst_num_pos, f'{dst_size} != {(dst_num_pos - num_extra_pos) // (2*depth_tar - 1)}'

    # spatial interpolation
    if src_size != tuple(dst_size):
        if verbose:
            print("Position interpolate for %s from %dx%d to %dx%d" % (
            verbose_tag, src_size[0], src_size[1], dst_size[0], dst_size[1]))
        extra_tokens = rel_pos_bias[-num_extra_pos:, :]
        rel_pos_bias = rel_pos_bias[:-num_extra_pos, :]

        def geometric_progression(a, r, n):
            return a * (1.0 - r ** n) / (1.0 - r)

        xy = []
        for src_size_, dst_size_ in zip(src_size, dst_size):
            left, right = 1.01, 1.5
            while right - left > 1e-6:
                q = (left + right) / 2.0
                gp = geometric_progression(1, q, src_size_ // 2)
                if gp > dst_size_ // 2:
                    right = q
                else:
                    left = q

            dis = []
            cur = 1
            for i in range(src_size_ // 2):
                dis.append(cur)
                cur += q ** (i + 1)

            r_ids = [-_ for _ in reversed(dis)]

            d = r_ids + [0] + dis
            xy.append(d)

        x, y = xy

        tx = dst_size[0] // 2.0
        ty = dst_size[1] // 2.0
        dx = np.arange(-tx, tx + 0.1, 1.0)
        dy = np.arange(-ty, ty + 0.1, 1.0)

        all_rel_pos_bias = []

        for i in range(num_attn_heads):
            z = rel_pos_bias[:, i].view((2*depth_src - 1, *src_size)).cpu().detach().float().numpy()
            rel_pos_bias_interp = []
            for d in range(len(z)):
                f = interpolate.interp2d(x, y, z[d], kind='cubic')
                rel_pos_bias_interp.append(torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))
            rel_pos_bias_interp = torch.cat(rel_pos_bias_interp, dim=0)
            all_rel_pos_bias.append(rel_pos_bias_interp)

        rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

        new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
    else:
        new_rel_pos_bias = rel_pos_bias

    src_size = dst_size
    # depth extrapolation
    if depth_tar != depth_src:
        assert depth_src == 1
        rel_pos_bias = new_rel_pos_bias
        window_size_2d = ((src_size[0] + 1) // 2, (src_size[1] + 1) // 2)
        window_size_3d = (depth_tar, *window_size_2d)
        if verbose:
            print("Position extrapolate for %s from %dx%dx%d to %dx%dx%d" % (
            verbose_tag, src_size[0], src_size[1], 1, dst_size[0], dst_size[1], depth_tar))

        new_rel_pos_bias = torch.zeros((dst_num_pos, num_attn_heads), device=rel_pos_bias.device, dtype=rel_pos_bias.dtype)

        relative_position_index_2d = gen_relative_position_index(window_size_2d)
        relative_position_index_3d = gen_relative_position_index(window_size_3d)

        # repeat spatial position bias
        pos_index_2d = torch.arange(relative_position_index_2d.max() + 1 - num_extra_pos, device=rel_pos_bias.device)
        pos_index_3d = pos_index_2d + src_size[0]*src_size[1]

        new_rel_pos_bias[pos_index_3d] = rel_pos_bias[pos_index_2d]

        # repeat cls position bias
        pos_index_cls2cls_2d = relative_position_index_2d[0, 0].reshape(-1)
        pos_index_cls2all_2d = relative_position_index_2d[0, 1:].reshape(-1)
        pos_index_all2cls_2d = relative_position_index_2d[1:, 0].reshape(-1)

        pos_index_cls2cls_3d = relative_position_index_3d[0, 0].reshape(-1)
        pos_index_cls2all_3d = relative_position_index_3d[0, 1:].reshape(-1)
        pos_index_all2cls_3d = relative_position_index_3d[1:, 0].reshape(-1)

        new_rel_pos_bias[pos_index_cls2cls_3d] = rel_pos_bias[pos_index_cls2cls_2d]
        new_rel_pos_bias[pos_index_cls2all_3d] = rel_pos_bias[pos_index_cls2all_2d].repeat(depth_tar, 1)
        new_rel_pos_bias[pos_index_all2cls_3d] = rel_pos_bias[pos_index_all2cls_2d].repeat(depth_tar, 1)

    return new_rel_pos_bias


def resize_pos_embed(posemb, ntok_new, num_prefix_tokens=1, gs_new=(), depth_src=1, depth_tar=1, verbose=True, verbose_tag=''):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    if posemb.size(1) == ntok_new:
        return posemb
    
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]

    gs_old = int(math.sqrt(len(posemb_grid) / depth_src))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new / depth_tar))] * 2
    assert len(gs_new) >= 2
    if verbose:
        print('Position embedding %s grid-size from %s to %s' % (verbose_tag, [gs_old, gs_old], gs_new))
    posemb_grid = posemb_grid.reshape(depth_src, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid.float(), size=gs_new, mode='bicubic', align_corners=False).to(posemb_grid.dtype)

    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(depth_src, gs_new[0] * gs_new[1], -1)
    if depth_src != depth_tar:
        assert depth_src == 1
        posemb_grid = posemb_grid.repeat(depth_tar, 1, 1)
    posemb_grid = rearrange(posemb_grid, 'd n e -> 1 (n d) e')

    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def custom_load_state_dict(model, state_dict, strict=True):
    if hasattr(model, 'blocks'):
        # print("Expand the shared relative position embedding to each transformer block. ")
        if 'rel_pos_bias.relative_position_bias_table' in state_dict:
            rel_pos_bias = state_dict['rel_pos_bias.relative_position_bias_table']

        for i, block in enumerate(model.blocks):
            if hasattr(block, 'attn') and getattr(block.attn, 'relative_position_bias_table', None) is not None:
                if f'blocks.{i}.attn.relative_position_bias_table' in state_dict:
                    rel_pos_bias = state_dict[f'blocks.{i}.attn.relative_position_bias_table']

                rel_pos_bias_resized = resize_rel_pos_bias(rel_pos_bias.clone(), block.attn.relative_position_bias_table.size(0),
                                                           depth_tar=model.n_input_images, verbose=False)

                if len(block.attn.relative_position_bias_table) == len(rel_pos_bias_resized) + 5:
                    rel_pos_bias_resized = torch.cat((rel_pos_bias_resized, rel_pos_bias_resized[-1:].repeat(5, 1)), dim=0)
                
                state_dict[f'blocks.{i}.attn.relative_position_bias_table'] = rel_pos_bias_resized

                if model.n_input_images > 1 and f'blocks.{i}.attn.relative_position_index' in state_dict:
                    state_dict[f'blocks.{i}.attn.relative_position_index'] = model.state_dict()[f'blocks.{i}.attn.relative_position_index']

    if hasattr(model, 'pos_embed') and 'pos_embed' in state_dict:
        # print("Expand the shared position embedding to each transformer block. ")
        img_size = model.img_size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        pos_embed_resized = resize_pos_embed(state_dict['pos_embed'], model.state_dict()['pos_embed'].size(1),
                                             num_prefix_tokens=model.num_prefix_tokens,
                                             gs_new=[imsize // model.patch_size for imsize in img_size],
                                             depth_tar=model.n_input_images, verbose=False)
        state_dict['pos_embed'] = pos_embed_resized

    ignore_keys = ['mask_token', 'norm.weight', 'norm.bias']
    for k in ignore_keys:
        if k in state_dict and not hasattr(model, k):
            del state_dict[k]

    for name, p in model.named_parameters():
        if (name[-4:] == 'bias' and
            name in state_dict and
            p.ndim == 2 and
            state_dict[name].ndim == 1):
            assert state_dict[name].shape[0] == p.shape[1]
            state_dict[name] = repeat(state_dict[name], 'd -> T d', T=p.shape[0])

    model.load_state_dict(state_dict, strict=strict)
