import torch
from torch import nn

class LossConfidence(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = "conf"
        self.weight = cfg.conf.weight  # lambda

    def forward(self, pred, target):
        if "conf_s" not in pred:
            return torch.tensor(0.0, device=pred["color"].device)

        s = pred["conf_s"]   # (B,V,1,H,W)
        return -self.weight * s.mean()
