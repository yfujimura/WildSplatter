from torch import nn
import torch.nn.functional as F

class LossAlign(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = "align"
        self.w_o = cfg.align.weight.origin
        self.w_d = cfg.align.weight.dir
        self.eps = 1e-8

    def forward(self, pred, target, global_step, batch_masks=None):
        ray_o = pred["ray_o"]
        ray_d = pred["ray_d"]
        ray_o_gt = target["ray_o"]
        ray_d_gt = target["ray_d"]

        lo = ((ray_o - ray_o_gt) ** 2).mean()

        d1 = F.normalize(ray_d, dim=-1, eps=self.eps)
        d2 = F.normalize(ray_d_gt, dim=-1, eps=self.eps)
        cos = (d1 * d2).sum(dim=-1)  # (B,S,H,W)
        ld = (1.0 - cos).mean()
        
        return self.w_o * lo + self.w_d * ld
        