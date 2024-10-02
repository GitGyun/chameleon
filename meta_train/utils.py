from einops import rearrange, repeat
import torch
import torch.nn.functional as F


def disassemble_batch(batch, resize=None):
    X, Y, M, *_ = batch
    if resize is not None:
        X = F.interpolate(X, size=resize, mode='bilinear', align_corners=False)
        Y = F.interpolate(Y, size=resize, mode='nearest')
        M = F.interpolate(M, size=resize, mode='nearest')

    T = Y.size(1)
    X = repeat(X, 'N C H W -> 1 T N C H W', T=T)
    Y = rearrange(Y, 'N T H W -> 1 T N 1 H W')
    M = rearrange(M, 'N T H W -> 1 T N 1 H W')
    
    return X, Y, M


def generate_task_mask(t_idx, task_idxs):
    '''
    Generate binary mask whether the task is semantic segmentation (1) or not (0).
    '''
    task_mask = torch.zeros_like(t_idx, dtype=bool)
    for task_idx in task_idxs:
        task_mask = torch.logical_or(task_mask, t_idx == task_idx)

    return task_mask


def normalize_tensor(input_tensor, dim):
    '''
    Normalize Euclidean vector.
    '''
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out