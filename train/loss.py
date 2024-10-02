import torch
import torch.nn.functional as F
from einops import rearrange


def spatial_softmax_loss(Y_pred, Y, M, reduction='mean', scaled=False):
    '''
    Compute spatial softmax loss for AnimalKP.
    '''
    if Y_pred.ndim == 6:
        M = rearrange(M, 'B T N C H W -> B (H W) T N C')
        Y_pred = rearrange(Y_pred, 'B T N C H W -> B (H W) T N C')
        Y = rearrange(Y, 'B T N C H W -> B (H W) T N C')
    else:
        M = rearrange(M, 'B N C H W -> B (H W) N C')
        Y_pred = rearrange(Y_pred, 'B N C H W -> B (H W) N C')
        Y = rearrange(Y, 'B N C H W -> B (H W) N C')
    loss = F.cross_entropy(Y_pred*M, Y*M, reduction='none')
        
    if reduction == 'mean':
        loss = loss.mean()
    if scaled:
        loss = loss / max(1, Y.sum())

    return loss


def spatio_channel_softmax_loss(Y_pred, Y, M, reduction='mean'):
    '''
    Compute spatial softmax loss for AnimalKP.
    '''
    assert Y_pred.ndim == 6

    # normalize over channels
    Y_pred = Y_pred - torch.logsumexp(Y_pred, dim=1, keepdim=True)
    
    # normalize over spatial dimensions
    Y_pred = rearrange(Y_pred, 'B T N C H W -> B (H W) T N C')
    M = rearrange(M, 'B T N C H W -> B (H W) T N C')
    Y = rearrange(Y, 'B T N C H W -> B (H W) T N C')
    loss = F.cross_entropy(Y_pred*M, Y*M, reduction='none')

    if reduction == 'mean':
        loss = loss.mean()
    return loss
