import torch.nn as nn
from einops import rearrange, repeat

from .transformers.factory import create_model
from .transformers.custom_layers import Identity

        
class ViTEncoder(nn.Module):
    '''
    Vision Transformer encoder wrapper
    '''
    def __init__(self, config, backbone, pretrained, in_chans, **kwargs):
        super().__init__()
        self.backbone = create_model(
            backbone,
            config=config,
            pretrained=pretrained,
            in_chans=in_chans,
            global_pool='',
            num_classes=0,
            **kwargs
        )
        self.grid_size = self.backbone.patch_embed.grid_size
        self.backbone.norm = Identity()
        self.feature_idxs = [level * (len(self.backbone.blocks) // 4) - 1
                             for level in range(1, 5)]
    
    def bias_parameters(self):
        for name, p in self.backbone.named_parameters():
            if name.endswith('bias') and p.ndim == 2:
                yield p

    def bias_parameter_names(self):
        names = []
        for name, p in self.backbone.named_parameters():
            if name.endswith('bias') and p.ndim == 2:
                names.append(f'backbone.{name}')
        return names
    
    def relpos_parameters(self):
        for name, p in self.backbone.named_parameters():
            if name.endswith('relative_position_bias_table'):
                yield p

    def relpos_parameter_names(self):
        names = []
        for name, p in self.backbone.named_parameters():
            if name.endswith('relative_position_bias_table'):
                names.append(f'backbone.{name}')
        return names
        
    def forward(self, x, t_idx=None):
        '''
        [input]
            x: (B, T, N, C, H, W)
            t_idx: None or (B, T)
        [output]
            features: dict of (B, T, N, hw+1, d)
        '''
        B, T, N = x.shape[:3]

        # flatten tensors
        x = rearrange(x, 'B T N C H W -> (B T N) C H W').contiguous()

        # repeat task index for shots
        if t_idx is not None:
            t_idx = repeat(t_idx, 'B T -> (B T N)', N=N)

        features = self.backbone.forward_features(x, feature_idxs=self.feature_idxs, b_idx=t_idx)
        
        features = [rearrange(feat, '(B T N) n d -> B T N n d', B=B, T=T, N=N) for feat in features]

        return features
