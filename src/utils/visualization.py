import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import torchvision.transforms.functional as tf
from einops import rearrange, repeat


def compare_image(ctx, tgt, pred, conf=None, mask=None):
    """
    ctx, tgt, pred: (V, 3, H, W)
    conf: (V, 1, H, W) or None
    """
    ctx = rearrange(ctx, "v c h w -> c (v h) w")
    tgt = rearrange(tgt, "v c h w -> c (v h) w")
    pred = rearrange(pred, "v c h w -> c (v h) w")

    if tgt.shape[1] < ctx.shape[1]:
        pad = torch.ones(
            tgt.shape[0],
            ctx.shape[1] - tgt.shape[1],
            tgt.shape[2],
            device=tgt.device,
        )
        tgt = torch.cat([tgt, pad], 1)
        pred = torch.cat([pred, pad], 1)

    images = [ctx, tgt, pred]

    if conf is not None:
        conf_rgb = conf_to_rgb(conf)  # (3, H, V*W)
        images.append(conf_rgb.to(ctx.device))

    if mask is not None:
        mask = repeat(mask, "v h w -> v c h w", c=3)
        mask = rearrange(mask, "v c h w -> c (v h) w")
        if mask.shape[1] < ctx.shape[1]:
            pad = torch.ones(
                mask.shape[0],
                ctx.shape[1] - mask.shape[1],
                mask.shape[2],
                device=mask.device,
            )
            mask = torch.cat([mask, pad], 1)
        images.append(mask)

    comparison = torch.cat(images, dim=-1)
    comparison = tf.to_pil_image(comparison)

    caption = "context / target / pred"
    if conf is not None:
        caption += " / confidence"
    if mask is not None:
        caption += " / mask"

    return wandb.Image(comparison, caption=caption)

#def compare_image(ctx, tgt, pred):
#    ctx = rearrange(ctx, "v c h w -> c (v h) w")
#    tgt = rearrange(tgt, "v c h w -> c (v h) w")
#    pred = rearrange(pred, "v c h w -> c (v h) w")
#
#    if tgt.shape[1] < ctx.shape[1]:
#        pad = torch.ones(tgt.shape[0], ctx.shape[1] - tgt.shape[1], tgt.shape[2], device=tgt.device)
#        tgt = torch.cat([tgt, pad], 1)
#        pred = torch.cat([pred, pad], 1)
#
#    comparison = torch.cat([ctx, tgt, pred], -1)
#    comparison = tf.to_pil_image(comparison)
#    return wandb.Image(comparison, caption="context/target/pred")

def conf_to_rgb(
    conf: torch.Tensor,        # (V, 1, H, W) or (V, H, W)
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Returns:
        conf_rgb: torch.Tensor (3, V*H, W), float in [0,1]
    """
    if conf.dim() == 4:
        conf = conf[:, 0]  # (V, H, W)

    # (V, H, W) -> (V*H, W)  ※ctx/tgt/pred と同じ高さにする
    conf_img = rearrange(conf, "v h w -> (v h) w").detach().float().cpu()
    conf_np = conf_img.numpy()

    # robust min/max
    if vmin is None:
        vmin = np.percentile(conf_np, 1)
    if vmax is None:
        vmax = np.percentile(conf_np, 99)

    if vmax <= vmin:
        vmax = vmin * 1.1

    conf_np = (conf_np - vmin) / (vmax - vmin + 1e-8)
    conf_np = np.clip(conf_np, 0, 1)

    cmap_fn = plt.get_cmap(cmap)
    conf_rgb = cmap_fn(conf_np)[..., :3]          # (V*H, W, 3)
    conf_rgb = torch.from_numpy(conf_rgb).permute(2, 0, 1)  # (3, V*H, W)

    return conf_rgb