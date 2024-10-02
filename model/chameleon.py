import torch
import torch.nn as nn

from .encoder import ViTEncoder
from .decoder import DPTDecoder
from .matching import MatchingModule


class Chameleon(nn.Module):
    def __init__(self, config, n_tasks, n_task_groups):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_task_groups = n_task_groups
        self.separate_alpha = getattr(config, 'separate_alpha', False)

        self.image_encoder = ViTEncoder(config, config.image_encoder, pretrained=(config.stage == 0 and not config.continue_mode),
                                        in_chans=3, drop_path_rate=config.image_encoder_drop_path_rate,
                                        n_bias_sets=self.n_tasks, n_input_images=config.n_input_images,)
        self.label_encoder = ViTEncoder(config, config.label_encoder, pretrained=False,
                                        in_chans=1, drop_path_rate=config.label_encoder_drop_path_rate,
                                        n_bias_sets=0, n_input_images=1,)
        self.matching_module = MatchingModule(self.image_encoder.backbone.embed_dim, self.label_encoder.backbone.embed_dim, config.n_attn_heads,
                                              alpha_init=config.matching_alpha_init, alpha_temp=config.matching_alpha_temp,
                                              n_alphas=(self.n_tasks if self.separate_alpha else self.n_task_groups))
        self.label_decoder = DPTDecoder(self.label_encoder.grid_size, self.label_encoder.backbone.embed_dim,
                                        hidden_features=[min(config.decoder_features*(2**i), 1024) for i in range(4)],
                                        out_chans=1, img_size=config.img_size)

        self.reset_support()
    
    def reset_support(self):
        # for support encoding
        self.has_encoded_support = False
        self.W_Ss = self.Z_Ss = None

    def bias_parameters(self):
        # bias parameters for similarity adaptation
        for p in self.image_encoder.bias_parameters():
            yield p

    def bias_parameter_names(self):
        names = [f'image_encoder.{name}' for name in self.image_encoder.bias_parameter_names()]

        return names

    def pretrained_parameters(self):
        for p in self.image_encoder.parameters():
            yield p
    
    def scratch_parameters(self):
        modules = [self.label_encoder, self.matching_module, self.label_decoder]
        for module in modules:
            for p in module.parameters():
                yield p

    def forward(self, X_S, Y_S, X_Q, t_idx=None, g_idx=None):
        # encode query input, support input and output
        W_Qs = self.image_encoder(X_Q, t_idx=t_idx)
        W_Ss = self.image_encoder(X_S, t_idx=t_idx)
        Z_Ss = self.label_encoder(Y_S)

        # mix support output by matching
        a_idx = t_idx if self.separate_alpha else g_idx
        Z_Q_preds = self.matching_module(W_Qs, W_Ss, Z_Ss, a_idx=a_idx)
        
        # decode support output
        Y_Q_pred = self.label_decoder(Z_Q_preds)
        
        return Y_Q_pred
    
    @torch.no_grad()
    def encode_support(self, X_S, Y_S, t_idx=None, g_idx=None):
        self.t_idx = t_idx
        self.g_idx = g_idx

        # encode query input, support input and output
        W_Ss = self.image_encoder(X_S, t_idx=t_idx)
        Z_Ss = self.label_encoder(Y_S)
        self.has_encoded_support = True

        # append suppot data
        if self.W_Ss is None:
            self.W_Ss = W_Ss
            self.Z_Ss = Z_Ss
        else:
            self.W_Ss = {level: torch.cat([self.W_Ss[level], W_Ss[level]], dim=2) for level in range(len(W_Ss))}
            self.Z_Ss = {level: torch.cat([self.Z_Ss[level], Z_Ss[level]], dim=2) for level in range(len(Z_Ss))}

    def predict_query(self, X_Q, channel_idxs=None, get_attn_map=False):
        assert self.has_encoded_support
        if channel_idxs is not None:
            W_Ss = {level: self.W_Ss[level][:, channel_idxs] for level in range(len(self.W_Ss))}
            Z_Ss = {level: self.Z_Ss[level][:, channel_idxs] for level in range(len(self.Z_Ss))}
            t_idx = self.t_idx[:, channel_idxs]
        else:
            W_Ss = self.W_Ss
            Z_Ss = self.Z_Ss
            t_idx = self.t_idx

        W_Qs = self.image_encoder(X_Q, t_idx=t_idx)

        a_idx = t_idx if self.separate_alpha else self.g_idx

        if get_attn_map:
            Z_Q_preds, As = self.matching_module(W_Qs, W_Ss, Z_Ss, a_idx=a_idx, get_attn_map=True)
            return torch.stack(As, dim=-2)
        else:
            Z_Q_preds = self.matching_module(W_Qs, W_Ss, Z_Ss, a_idx=a_idx, get_attn_map=False)
        
        Y_Q_pred = self.label_decoder(Z_Q_preds)

        return Y_Q_pred
