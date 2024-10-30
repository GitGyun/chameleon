import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple
from einops import repeat, rearrange
from timm.models.layers.trace_utils import _assert
    

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        return args[0]


class Linear(nn.Linear):
    """
    Bias-Switching Linear layer
    """
    def __init__(self, n_bias_sets=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is None:
            n_bias_sets = 0

        self.n_bias_sets = n_bias_sets
        if self.n_bias_sets > 0:
            assert self.bias is not None
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_bias_sets).contiguous())

    def forward(self, input, b_idx=None):
        if self.n_bias_sets > 0:
            assert b_idx is not None
            output = F.linear(input, self.weight, None)
            bias = self.bias[b_idx][:, None]
            return output + bias
        else:
            return F.linear(input, self.weight, self.bias)


class LayerNorm(nn.LayerNorm):
    """
    Bias-Switching LayerNorm
    """
    def __init__(self, n_bias_sets=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is None:
            n_bias_sets = 0
        
        self.n_bias_sets = n_bias_sets
        if self.n_bias_sets > 0:
            assert self.elementwise_affine
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_bias_sets).contiguous())

    def forward(self, input, b_idx=None):
        if self.n_bias_sets > 0:
            assert b_idx is not None
            output = F.layer_norm(input, self.normalized_shape, self.weight, None, self.eps)
            if b_idx.ndim == 1:
                bias = self.bias[b_idx]
                for _ in range(output.ndim - 2):
                    bias = bias[:, None]
            else:
                assert False
                bias_mh = torch.stack(self.bias.split(self.bias.shape[1] // b_idx.shape[1], dim=1), 0)
                bias = torch.einsum('bhn,hnd->bhd', b_idx, bias_mh)
                bias = rearrange(bias, 'B h d -> B 1 (h d)')
            return output + bias
        else:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)


class Conv2d(nn.Conv2d):
    """
    Bias-Switching Conv2d layer
    """
    def __init__(self, n_bias_sets=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is None:
            n_bias_sets = 0

        self.n_bias_sets = n_bias_sets
        if self.n_bias_sets > 0:
            assert self.bias is not None
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_bias_sets).contiguous())

    def forward(self, input, b_idx=None):
        if self.n_bias_sets > 0:
            assert b_idx is not None
            output = F.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
            if b_idx.ndim == 1:
                bias = self.bias[b_idx][:, :, None, None]
            else:
                raise NotImplementedError

            return output + bias
        else:
            return F.conv2d(input, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)


class Mlp(nn.Module):
    """
    Bias-Switching MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True,
                 drop=0., n_bias_sets=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = Linear(n_bias_sets, in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = Linear(n_bias_sets, hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, b_idx=None):
        x = self.fc1(x, b_idx=b_idx)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x, b_idx=b_idx)
        x = self.drop2(x)
        return x


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            input = module(*inputs)
        return input


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.proj_switching = False
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        if C != self.proj.in_channels:
            assert C % self.proj.in_channels == 0
            x = rearrange(x, 'B (N C) H W -> (B N) C H W', C=self.proj.in_channels)
        x = self.proj(x)
        if C != self.proj.in_channels:
            x = rearrange(x, '(B N) C H W -> B C (N H) W', N=(C // self.proj.in_channels))
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
