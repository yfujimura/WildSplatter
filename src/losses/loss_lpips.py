from lpips import LPIPS
from torch import nn
from einops import rearrange

def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)

class LossLpips(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = "lpips"

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

        self.weight = cfg.lpips.weight

        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(self, pred, target, global_step, batch_masks=None):
        pred_img = pred["color"]      # (B,V,3,H,W)
        tgt_img  = target["color"]

        if "occ_mask" in target:
            pred_img = pred_img * target["occ_mask"][:,:,None]
            tgt_img = tgt_img * target["occ_mask"][:,:,None]

        #if "conf" in pred:
        #    conf = pred["conf"]       # (B,V,1,H,W)
        #    pred_img = pred_img * conf
        #    tgt_img  = tgt_img  * conf

        loss = self.lpips(
            rearrange(pred_img, "b v c h w -> (b v) c h w"),
            rearrange(tgt_img,  "b v c h w -> (b v) c h w"),
            normalize=True,
        )

        if batch_masks is not None:
            loss = loss * batch_masks[:,None,None,None]
        
        return self.weight * loss.mean()
        
