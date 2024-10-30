import torch
from skimage.morphology import extrema


def preprocess_kpmap(heatmap, threshold=0.1):
    assert heatmap.ndim == 2
    maxima = extrema.h_maxima((heatmap).numpy(), threshold)
    grid = torch.stack(torch.meshgrid(torch.arange(heatmap.shape[0]), torch.arange(heatmap.shape[1]), indexing='ij'), dim=2)
    modes = grid[maxima.astype(bool)]
    return modes


def make_density_tensor(kpmaps, img_size=(256,256), width=2):
    # N, 2
    assert width % 2 == 0
    density = torch.zeros(1, *img_size)
    for x, y in kpmaps:
        density[0, x - width // 2 : x + width // 2, y - width // 2 : y + width // 2] = 1

    return density


def viridis(x):
    c1 = torch.tensor([68., 1., 84.]) / 255
    c2 = torch.tensor([33., 145., 140.]) / 255
    c3 = torch.tensor([253., 231., 37.]) / 255
    
    for i in range(x.ndim - 1):
        c1 = c1[:, None]
        c2 = c2[:, None]
        c3 = c3[:, None]
    if x.ndim == 4:
        c1 = c1.transpose(0, 1)
        c2 = c2.transpose(0, 1)
        c3 = c3.transpose(0, 1)
    
    x = torch.where(
        x < 0.5,
        c1 * (0.5 - x) * 2 + c2 * x * 2,
        c2 * (1 - x) * 2 + c3 * (x - 0.5) * 2
    )
    return x