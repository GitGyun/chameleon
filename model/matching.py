import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class CrossAttention(nn.Module):
    '''
    Multi-Head Cross-Attention layer for Matching
    '''
    def __init__(self, dim_q, dim_v, dim_o, num_heads=16, temperature=-1, dr=0.1):
        super().__init__()
        
        self.dim_q = dim_q
        
        # heads and temperature
        self.num_heads = num_heads
        self.dim_split_q = dim_q // num_heads
        self.dim_split_v = dim_o // num_heads
        if temperature > 0:
            self.temperature = temperature
        else:
            self.temperature = math.sqrt(dim_o)
        
        # projection layers
        self.fc_q = nn.Linear(dim_q, dim_q, bias=False)
        self.fc_k = nn.Linear(dim_q, dim_q, bias=False)
        self.fc_v = nn.Linear(dim_v, dim_o, bias=False)
        self.fc_o = nn.Linear(dim_o, dim_o, bias=False)
        
        # nonlinear activation and dropout
        self.activation = nn.GELU()
        self.attn_dropout = nn.Dropout(dr)
        
        # layernorm layers
        self.pre_ln_q = self.pre_ln_k = nn.LayerNorm(dim_q)
        self.ln = nn.LayerNorm(dim_o)

    def forward(self, Q, K, V, mask=None, get_attn_map=False):
        # pre-layer normalization
        Q = self.pre_ln_q(Q)
        K = self.pre_ln_k(K)
        
        # lienar projection
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        # split into multiple heads
        Q_ = torch.cat(Q.split(self.dim_split_q, 2), 0)
        K_ = torch.cat(K.split(self.dim_split_q, 2), 0)
        V_ = torch.cat(V.split(self.dim_split_v, 2), 0)
        
        # scaled dot-product attention with mask and dropout
        L = Q_.bmm(K_.transpose(1, 2)) / self.temperature
        L = L.clip(-1e4, 1e4)
        
        # mask
        if mask is not None:
            L = L.masked_fill(~mask, -float('inf'))
            
        A = L.softmax(dim=2)
        if mask is not None:
            A.masked_fill(~mask, 0)
        A = self.attn_dropout(A)
        
        # apply attention to values
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        
        # layer normalization
        O = self.ln(O)
        
        # residual connection with non-linearity
        O = O + self.activation(self.fc_o(O))
            
        if get_attn_map:
            return O, A
        else:
            return O
        
        

class MatchingModule(nn.Module):
    '''
    Matching Module of VTMv2
    '''
    def __init__(self, dim_w, dim_z, n_heads=16, alpha_init=0, alpha_temp=0.05, n_alphas=1):
        super().__init__()
        self.matching = nn.ModuleList([CrossAttention(dim_w, dim_z, dim_z, num_heads=n_heads)
                                       for _ in range(4)])

        self.alpha = nn.ParameterList([nn.Parameter(alpha_init*F.one_hot(torch.tensor([level]*n_alphas), 4).float()) for level in range(4)])
        self.alpha_temp = alpha_temp
        self.layernorm = nn.LayerNorm(dim_w)
            
    def forward(self, W_Qs, W_Ss, Z_Ss, a_idx, get_attn_map=False):
        B, T, N = W_Qs[0].shape[:3]

        W_Qs = torch.stack([self.layernorm(W_Qs[level]) for level in range(4)])
        W_Ss = torch.stack([self.layernorm(W_Ss[level]) for level in range(4)])
        W_Qs_mix = []
        W_Ss_mix = []
        for level in range(4):
            alpha = (self.alpha[level][a_idx] / self.alpha_temp).softmax(dim=-1)
            if a_idx.ndim == 2:
                alpha = rearrange(alpha, 'B T L-> L B T 1 1 1')
            else:
                alpha = rearrange(alpha, 'B L-> L B 1 1 1 1')
            W_Qs_mix.append((alpha * W_Qs).sum(dim=0))
            W_Ss_mix.append((alpha * W_Ss).sum(dim=0))

        Z_Qs = []
        if get_attn_map:
            As = []
        for level in range(4):
            # drop the cls token
            Q = rearrange(W_Qs_mix[level], 'B T N n d -> (B T) (N n) d')
            K = rearrange(W_Ss_mix[level], 'B T N n d -> (B T) (N n) d')
            V = rearrange(Z_Ss[level], 'B T N n d -> (B T) (N n) d')

            O = self.matching[level](Q, K, V, get_attn_map=get_attn_map)
            if get_attn_map:
                O, A = O
                A = rearrange(A, '(nh B T) ... -> B T ... nh', B=B, T=T)
                As.append(A)
                
            Z_Q = rearrange(O, '(B T) (N n) d -> B T N n d', B=B, T=T, N=N)
            Z_Qs.append(Z_Q)

        if get_attn_map:
            return Z_Qs, As
        else:
            return Z_Qs