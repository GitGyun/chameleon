from skimage import color
import numpy as np
import torch
from dataset.utils import dense_to_sparse, vis_keypoints
import matplotlib.pyplot as plt
import io
from PIL import Image


def visualize_batch(X=None, Y=None, M=None, Y_preds=None, aux=None, postprocess_fn=None):
    '''
    Visualize a global batch consists of N-shot images and labels for T channels.
    It is assumed that images are shared by all channels, thus convert channels into RGB and visualize at once.
    '''
    
    vis = []
    
    # shape check
    assert X is not None or Y is not None or Y_preds is not None
    
    # visualize image
    if X is not None:
        img = X.cpu().float()
        vis.append(img)
    else:
        img = None
        
    # flatten labels and masks
    Ys = []
    Ms = []
    if Y is not None:
        Ys.append(Y)
        Ms.append(M)
    
    if Y_preds is not None:
        if isinstance(Y_preds, torch.Tensor):
            Ys.append(Y_preds)
            Ms.append(None)
        elif isinstance(Y_preds, (tuple, list)):
            for Y_pred in Y_preds:
                Ys.append(Y_pred)
                Ms.append(None)
        else:
            ValueError(f'unsupported predictions type: {type(Y_preds)}')

    # visualize labels
    if len(Ys) > 0:
        for Y in Ys:
            label = Y.cpu().float()

            # fill masked region with random noise
            if M is not None and Y.ndim == 4:
                assert Y.ndim == M.ndim, (Y.shape, M.shape)
                if M.shape[1] == 1:
                    M = M.repeat(1, Y.shape[1], 1, 1)

                M = M.cpu().float()
                label = torch.where(M.bool(), label, torch.zeros_like(label))

            if postprocess_fn is not None:
                label = postprocess_fn(label, img, aux)
                # try:
                #     label = postprocess_fn(label, img, aux)
                # except:
                #     print('visualization failed')
                #     label = torch.zeros_like(img)

            label = visualize_label_as_rgb(label)
            vis.append(label)

    vis = torch.cat(vis)
    vis = vis.float().clip(0, 1)
    
    return vis


def postprocess_depth(label, img=None, aux=None):
    label = 0.6*label + 0.4
    label = torch.exp(label * np.log(2.0**16.0)) - 1.0
    label = torch.log(label) / 11.09
    label = (label - 0.64) / 0.18
    label = (label + 1.) / 2
    label = (label*255).byte().float() / 255.
    return label


def postprocess_semseg(label, img=None, aux=None, fixed_colors=True, colors=None):
    if colors is not None:
        COLORS = colors
    else:
        COLORS = ('red', 'blue', 'yellow', 'magenta', 
                  'green', 'indigo', 'darkorange', 'cyan', 'pink', 
                  'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
                  'purple', 'darkviolet')

    if label.ndim == 4:
        if label.shape[1] > 1:
            label = torch.argmax(torch.cat((0.5*torch.ones_like(label[:, :1]), label), dim=1), dim=1)
        else:
            label = label.squeeze(1).long()
    elif label.dtype != torch.long:
        label = label.float().round().long()
    
    label_vis = []
    if img is not None:
        for img_, label_ in zip(img, label):
            if fixed_colors:
                colors = [COLORS[(c - 1) % len(COLORS)] for c in label_.unique() if c != 0]

            label_vis.append(torch.from_numpy(color.label2rgb(label_.numpy(),
                                                              image=img_.permute(1, 2, 0).numpy(),
                                                              colors=colors,
                                                              kind='overlay')).permute(2, 0, 1))
    else:
        for label_ in label:
            if fixed_colors:
                colors = [COLORS[(c - 1) % len(COLORS)] for c in label_.unique() if c != 0]

            label_vis.append(torch.from_numpy(color.label2rgb(label_.numpy(),
                                                              colors=colors,
                                                              kind='overlay')).permute(2, 0, 1))
    
    label = torch.stack(label_vis)
    
    return label


def postprocess_keypoints(label, img, aux, lth=1, crad=2):
    if label.ndim == 3:
        sparse = True
    else:
        sparse = False
    
    sparse_gt = aux[0]
    label_vis = []
    for i in range(len(label)):
        img_ = np.ascontiguousarray((img[i]*128).byte().permute(1, 2, 0).numpy())
        if sparse:
            kps = label[i].transpose(0, 1).numpy()
        else:
            kps = dense_to_sparse(label[i]).transpose(0, 1).numpy()
            kps[2] = sparse_gt[i, :, 2].float().cpu().numpy()
        kps[2] = kps[2] / 2
        vis = vis_keypoints(img_, kps, lth=lth, crad=crad, object_type='person')
        vis = torch.from_numpy(vis).permute(2, 0, 1) / 255
        label_vis.append(vis)
    label = torch.stack(label_vis)

    return label


def postprocess_keypoints_mpii(label, img, aux, lth=1, crad=2):
    if label.ndim == 3:
        sparse = True
    else:
        sparse = False
    
    sparse_gt = aux[0]
    label_vis = []
    for i in range(len(label)):
        img_ = np.ascontiguousarray((img[i]*128).byte().permute(1, 2, 0).numpy())
        if sparse:
            kps = label[i].transpose(0, 1).numpy()
        else:
            kps = dense_to_sparse(label[i]).transpose(0, 1).numpy()
            kps[2] = sparse_gt[i, :, 2].float().cpu().numpy()
        vis = vis_keypoints(img_, kps, lth=lth, crad=crad, object_type='person_mpii')
        vis = torch.from_numpy(vis).permute(2, 0, 1) / 255
        label_vis.append(vis)
    label = torch.stack(label_vis)

    return label


def postprocess_keypoints_deepfashion(label, img, aux, lth=1, crad=2):
    if label.ndim == 3:
        sparse = True
    else:
        sparse = False
    
    sparse_gt = aux[0]
    label_vis = []
    for i in range(len(label)):
        img_ = np.ascontiguousarray((img[i]*128).byte().permute(1, 2, 0).numpy())
        if sparse:
            kps = label[i].transpose(0, 1).numpy()
        else:
            kps = dense_to_sparse(label[i]).transpose(0, 1).numpy()
            kps[2] = sparse_gt[i, :, 2].float().cpu().numpy()
        vis = vis_keypoints(img_, kps, lth=lth, crad=crad, object_type='fashion')
        vis = torch.from_numpy(vis).permute(2, 0, 1) / 255
        label_vis.append(vis)
    label = torch.stack(label_vis)

    return label


def postprocess_keypoints_hand(label, img, aux, lth=1, crad=2):
    if label.ndim == 3:
        sparse = True
    else:
        sparse = False
    
    sparse_gt = aux[0]
    label_vis = []
    for i in range(len(label)):
        img_ = np.ascontiguousarray((img[i]*128).byte().permute(1, 2, 0).numpy())
        if sparse:
            kps = label[i].transpose(0, 1).numpy()
        else:
            kps = dense_to_sparse(label[i]).transpose(0, 1).numpy()
            kps[2] = sparse_gt[i, :, 2].float().cpu().numpy()
        vis = vis_keypoints(img_, kps, lth=lth, crad=crad, object_type='hand')
        vis = torch.from_numpy(vis).permute(2, 0, 1) / 255
        label_vis.append(vis)
    label = torch.stack(label_vis)

    return label


def visualize_label_as_rgb(label):
    if label.size(1) == 1:
        label = label.repeat(1, 3, 1, 1)
    elif label.size(1) == 2:
        label = torch.cat((label, torch.zeros_like(label[:, :1])), 1)
    elif label.size(1) == 5:
        label = torch.stack((
            label[:, :2].mean(1),
            label[:, 2:4].mean(1),
            label[:, 4]
        ), 1)
    elif label.size(1) != 3:
        assert NotImplementedError
        
    return label


def visualize_alpha(alpha, idx=0, temp=1):
    n_matching_levels = len(alpha)
    n_image_levels = alpha[0].shape[1]
    levels = [f'Matching Level {i+1}' for i in range(n_matching_levels)]
    values = {
        f'Image Level {i+1}': torch.stack([(alpha[level].detach().cpu().data[idx] / temp).softmax(dim=0)[i]
                                           for level in range(n_matching_levels)]).float()
        for i in range(n_image_levels)
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    bottom = torch.zeros(4)

    for x, y in values.items():
        p = ax.bar(levels, y, width=0.5, label=x, bottom=bottom)
        bottom += y
        
    ax.set_yticks(torch.linspace(0, 1, 11))
    ax.set_title("Contribution of Image Levels to Matching Levels", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc="upper right", fontsize=14)
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img
