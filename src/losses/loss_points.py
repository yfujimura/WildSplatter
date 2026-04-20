from torch import nn

class LossPoints(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = "points"
        self.weight = cfg.points.weight
        self.apply_gs_means = cfg.points.apply_gs_means
        self.gs_end_step = cfg.points.gs_end_step

    def forward(self, pred, target, global_step, batch_masks=None):
        delta2 = target["weight"] * ((pred["points"] - target["points"]) ** 2).sum(dim=-1)  # (B, VHW)

        if batch_masks is not None:
            loss = (delta2*batch_masks[:,None]).mean()
        else:
            loss = delta2.mean()
        return self.weight * loss