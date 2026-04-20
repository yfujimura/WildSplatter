from functools import cache

import torch
from einops import reduce
from lpips import LPIPS
from skimage.metrics import structural_similarity
#from torch import Tensor


@torch.no_grad()
def compute_psnr(ground_truth, predicted):
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()



@cache
def get_lpips(device):
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(ground_truth, predicted):
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(ground_truth, predicted):
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)