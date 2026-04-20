from torch import nn

class LossMse(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = "mse"
        self.weight = cfg.mse.weight

    def forward(self, pred, target, global_step, batch_masks=None):
        pred_img = pred["color"]      # (B,V,3,H,W)
        tgt_img  = target["color"]

        if "occ_mask" in target:
            pred_img = pred_img * target["occ_mask"][:,:,None]
            tgt_img = tgt_img * target["occ_mask"][:,:,None]
        
        delta2 = (pred_img - tgt_img) ** 2  # (B,V,3,H,W)

        #if "conf" in pred:
        #    conf = pred["conf"]                           # (B,V,1,H,W)
        #    delta2 = delta2 * conf

        if batch_masks is not None:
            loss = (delta2*batch_masks[:,None,None,None,None]).mean()
        else:
            loss = delta2.mean()
        return self.weight * loss
        