""" BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
Model from official source: https://github.com/microsoft/unilm/tree/master/beit
and
https://github.com/microsoft/unilm/tree/master/beit2
@inproceedings{beit,
title={{BEiT}: {BERT} Pre-Training of Image Transformers},
author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=p-BhZSz59o4}
}
@article{beitv2,
title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
year={2022},
eprint={2208.06366},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
At this point only the 1k fine-tuned classification weights and model configs have been added,
see original source above for pre-training models and procedure.
Modifications by / Copyright 2021 Ross Wightman, original copyrights below
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, gen_relative_position_index
from timm.models.layers import DropPath, trunc_normal_
from .registry import register_model
from .vision_transformer import checkpoint_filter_fn
from .custom_layers import Identity, Linear, LayerNorm, Mlp, PatchEmbed
from einops import rearrange, repeat
import random


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'beit_base_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth'),
    'beit_base_patch16_384': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_384_pt22k_ft22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_base_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth',
        num_classes=21841,
    ),
    'beit_base_patch16_384_in22k': _cfg(
        input_size=(3, 384, 384),
    ),
    'beit_large_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22kto1k.pth'),
    'beit_large_patch16_384': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_384_pt22k_ft22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_large_patch16_512': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_512_pt22k_ft22kto1k.pth',
        input_size=(3, 512, 512), crop_pct=1.0,
    ),
    'beit_large_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k.pth',
        num_classes=21841,
    ),

    'beitv2_base_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21kto1k.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_base_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth',
        num_classes=21841,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_base_patch16_224_in22k_ft21k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth',
        num_classes=21841,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21kto1k.pth',
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth',
        num_classes=21841,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_384': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384)
    ),
    'beitv2_large_patch16_416': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 416, 416)
    ),
    'beitv2_large_patch16_512': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 512, 512)
    ),
    'beitv2_large_patch16_224_512': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 224, 512)
    ),
    'beitv2_large_patch16_160_384': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 160, 384)
    ),
    'beitv2_large_patch16_224_384': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 224, 384)
    ),
    'beitv2_large_patch16_128_224': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 128, 224)
    ),
    'beitv2_large_patch16_320': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 320, 320)
    ),
}


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=None, attn_head_dim=None,
            n_bias_sets=0, additional_positions=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.additional_positions = additional_positions

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            if n_bias_sets > 0:
                self.q_bias = nn.Parameter(torch.zeros(n_bias_sets, all_head_dim))
                self.register_buffer('k_bias', torch.zeros(n_bias_sets, all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(n_bias_sets, all_head_dim))
            else:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        if window_size:
            if len(window_size) == 2:
                window_size = (1, window_size[0], window_size[1])
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1) + 3 + (5 if additional_positions > 0 else 0)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wd - 1 * 2*Wh-1 * 2*Ww-1 + 3, nH
            self.register_buffer("relative_position_index", gen_relative_position_index(window_size, additional_positions))
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(n_bias_sets, all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self, N):
        pos_index = self.relative_position_index[:N, :N].reshape(-1)
        relative_position_bias = self.relative_position_bias_table[pos_index].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, shared_rel_pos_bias=None, b_idx=None):
        # parse shape
        B, N, C = x.shape

        # create qkv bias
        if b_idx is None:
            q_bias = repeat(self.q_bias, 'd -> B d', B=B) if self.q_bias is not None else None
            k_bias = repeat(self.k_bias, 'd -> B d', B=B) if self.k_bias is not None else None
            v_bias = repeat(self.v_bias, 'd -> B d', B=B) if self.v_bias is not None else None
        else:
            q_bias = self.q_bias[b_idx]
            k_bias = self.k_bias[b_idx]
            v_bias = self.v_bias[b_idx]
        qkv_bias = torch.cat((q_bias[:, None], k_bias[:, None], v_bias[:, None]), 2) if q_bias is not None else None

        # qkv projection
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=None) + qkv_bias
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # compute attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # add relative position bias
        if self.relative_position_bias_table is not None:
            attn = attn + self._get_rel_pos_bias(N)
        if shared_rel_pos_bias is not None:
            attn = attn + shared_rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        # projection
        x = self.proj(x, b_idx=b_idx)
        x = self.proj_drop(x)

        return x
    

class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            window_size=None, attn_head_dim=None, n_bias_sets=0, additional_positions=0, post_block=None):
        super().__init__()

        self.norm1 = norm_layer(n_bias_sets, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            window_size=window_size, attn_head_dim=attn_head_dim, n_bias_sets=n_bias_sets, additional_positions=additional_positions)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(n_bias_sets, dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, n_bias_sets=n_bias_sets)

        if init_values:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.post_block = post_block
        
    def forward(self, x, shared_rel_pos_bias=None, b_idx=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x, b_idx), shared_rel_pos_bias=shared_rel_pos_bias, b_idx=b_idx))
            x = x + self.drop_path(self.mlp(self.norm2(x, b_idx), b_idx=b_idx))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x, b_idx), shared_rel_pos_bias=shared_rel_pos_bias, b_idx=b_idx))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x, b_idx), b_idx=b_idx))
        
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads, additional_positions=0):
        super().__init__()
        if len(window_size) == 2:
            window_size = (1, window_size[0], window_size[1])

        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1] * window_size[2]
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size - 1) + 3 + (5 if additional_positions > 0 else 0)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.register_buffer("relative_position_index", gen_relative_position_index(window_size, additional_positions))
        self.additional_positions = additional_positions

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area + 1 + self.additional_positions, self.window_area + 1 + self.additional_positions, -1)  # Wh*Ww*Wd,Wh*Ww*Wd,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww


class Beit(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
            self, config, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
            attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(LayerNorm, eps=1e-6),
            init_values=None, use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
            head_init_scale=0.001, n_bias_sets=0, n_input_images=1):
        super().__init__()
        # original beit attributes
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.grad_checkpointing = False

        # register patch embed
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # register CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # register positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # determine window size for relative position bias
        self.n_input_images = n_input_images
        if self.n_input_images > 1 and not config.expand_input_embed:
            window_size = (self.n_input_images, *self.patch_embed.grid_size)
        else:
            window_size = self.patch_embed.grid_size
        self.num_patch_tokens = self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1]

        # register relative position bias
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=window_size, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # register beit blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=window_size if use_rel_pos_bias else None, n_bias_sets=n_bias_sets)
            for i in range(depth)])
        
        # register norm and head
        use_fc_norm = self.global_pool == 'avg'
        self.norm = Identity() if use_fc_norm else norm_layer(0, embed_dim)
        self.fc_norm = norm_layer(0, False, embed_dim) if use_fc_norm else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed|rel_pos_bias',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, feature_idxs=None, b_idx=None):
        # tokenize input patches
        x = self.patch_embed(x)

        # append cls token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # add positional embeddings
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # get relative position bias
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        # forward blocks
        level = 0
        features, attn_maps = [], []
        if feature_idxs is None:
            feature_idxs = range(len(self.blocks))
        for i, blk in enumerate(self.blocks):
            # forward block
            x = blk(x, shared_rel_pos_bias=rel_pos_bias, b_idx=b_idx)

            # pass features at selected levels
            if i in feature_idxs:
                if x.size(1) >= 2*self.num_patch_tokens:
                    assert self.n_input_images > 1
                    features.append(x[:, -self.n_input_images*self.num_patch_tokens:-(self.n_input_images - 1)*self.num_patch_tokens])
                else:
                    features.append(x[:, -self.num_patch_tokens:]) # remove cls token

                if i == max(feature_idxs):
                    break

                level += 1

        return features

    def forward_head(self, x, pre_logits: bool = False):
        if self.fc_norm is not None:
            x = x[:, 1:].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x, b_idx=None):
        x = self.forward_features(x, b_idx=b_idx)
        x = self.forward_head(x)
        return x


def _beit_checkpoint_filter_fn(state_dict, model):
    if 'module' in state_dict:
        # beit v2 didn't strip module
        state_dict = state_dict['module']
    return checkpoint_filter_fn(state_dict, model)


def _create_beit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Beit models.')

    model = build_model_with_cfg(
        Beit, variant, pretrained,
        # FIXME an updated filter fn needed to interpolate rel pos emb if fine tuning to diff model sizes
        pretrained_filter_fn=_beit_checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def beit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_base_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_base_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_base_patch16_384_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_384_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_512', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beit_large_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_base_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_base_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_base_patch16_224_in22k_ft21k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_base_patch16_224_in22k_ft21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_416(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_416', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_512', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_224_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_224_512', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_160_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_160_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_224_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_224_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_128_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_128_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_320(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_320', pretrained=pretrained, **model_kwargs)
    return model
