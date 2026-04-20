from .loss_mse import LossMse
from .loss_lpips import LossLpips
from .loss_align import LossAlign
from .loss_confidence import LossConfidence
from .loss_points import LossPoints

LOSSES = {"mse": LossMse, "lpips": LossLpips, "align": LossAlign, "conf": LossConfidence, "points": LossPoints}

def get_losses(cfg):
    losses = []
    for loss in cfg.keys():
        losses.append(LOSSES[loss](cfg))
    return losses