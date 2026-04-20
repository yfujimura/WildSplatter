import math

import torch
from torch import nn
from einops import rearrange, repeat
from minlora import get_lora_params

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from src.losses import get_losses
from src.utils.gs_utils import repeat_gaussians
from src.utils.visualization import compare_image

from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, map_pdf_to_opacity

class ModelWrapper(LightningModule):

    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.optimizer_cfg = cfg.optimizer
        self.n_views = cfg.dataset.n_views
        self.model = model
        self.losses = nn.ModuleList(get_losses(cfg.loss))
        

    def on_fit_start(self):
        self.model.da3.train()

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        intrs = batch["intrinsics"]
        extrs = batch["extrinsics"]
        depths = batch["depth"]
        skys = batch["sky"]
        masks = batch["mask"]
    
        B, H, W = images.shape[0], images.shape[-2], images.shape[-1]
        V = images.shape[1]
        V_ctx = 2
        V_tgt = V - V_ctx

        occ_masks = (masks > 0) + (skys > 0.3)
        occ_masks = occ_masks[:,V_ctx:]
    
        all_images = images.clone()
        ctx_images, tgt_images = all_images.split([V_ctx, V_tgt], dim=1)

        ctx_depths, tgt_depths = depths.split([V_ctx, V_tgt], dim=1)
        ctx_skys, tgt_skys = skys.split([V_ctx, V_tgt], dim=1)
        
        x = self.model._imagenet_normalize_bnxchw(all_images)
        
        extrs_norm = self.model._normalize_extrinsics(as_homogeneous(extrs))
        ctx_extrs, tgt_extrs = extrs_norm.split([V_ctx, V_tgt], dim=1)
    
        ctx_intrs, tgt_intrs = intrs.split([V_ctx, V_tgt], dim=1)
    
        if self.model.da3_backbone_train_mode == "frozen":
            with torch.no_grad():
                feats, aux_feats = self.model.da3.backbone(
                    x, cam_token=None, export_feat_layers=[self.model.da3.backbone.alt_start-1], n_ctx=V_ctx, ref_view_strategy="first"
                )
        else:
            feats, aux_feats = self.model.da3.backbone(
                x, cam_token=None, export_feat_layers=[self.model.da3.backbone.alt_start-1], n_ctx=V_ctx, ref_view_strategy="first"
            )

        aux_feat = aux_feats[0][:,-V_tgt:]
        app_embed = self.model._apply_app_encoder(aux_feat, H, W)
    
        with torch.autocast(device_type=x.device.type, enabled=False):
            # ===== Estimate Gaussians =====
            output = self.model.da3.head(feats, H, W, patch_start_idx=0)
            gs_outs = self.model.da3.gs_head(
                feats=feats,
                H=H,
                W=W,
                patch_start_idx=0,
                images=x[:,:V_ctx],
            )
            raw_gaussians = gs_outs.raw_gs
            densities = gs_outs.raw_gs_conf

            last_aux = output["last_aux"]
            feat = gs_outs["fused_feat"]

            # ===== Get ray origins and directions =====
            ray_o, ray_d = self.model._get_ray_origin_and_dir(output.ray, H, W)

            offset_depth = self.model._apply_offset_decoder(feat)[...,-1]

            # ===== Adapt raw Gaussians to be ready for rendering =====
            gaussians = self.model._adapt_gaussians(
                raw_gaussians=raw_gaussians,
                depths=output.depth,
                opacities=map_pdf_to_opacity(densities),
                extrinsics=ctx_extrs,
                intrinsics=ctx_intrs,
                H=H,
                W=W,
                ray_o=ray_o,
                ray_d=ray_d,
                offset_depth=offset_depth,
            )

            ## ===== Rescale Gaussians =====
            points_da3 = ray_o[:,:,None,None,:] + ray_d * output.depth[...,None]
            points_da3 = rearrange(points_da3, "b v h w c -> b (v h w) c")
            scale_bias, points, weights, batch_masks = self.model._compute_scale_bias(
                points_da3,
                ctx_depths.float(),
                ctx_intrs.float(),
                ctx_extrs.float(),
                ctx_skys.float(),
                return_mask=True,
            )
            gaussians.means = gaussians.means * scale_bias[:,None,:1] + scale_bias[:,None,1:]
            gaussians.scales = gaussians.scales * scale_bias[:,None,:1]
            points_da3 = points_da3 * scale_bias[:,None,:1] + scale_bias[:,None,1:]

            # ===== Gaussians with appearance =====
            gaussians = repeat_gaussians(gaussians, V_tgt)
            harmonics_app = torch.zeros_like(gaussians.harmonics)
            for i in range(V_tgt):
                raw_gaussians_app = self.model._apply_gs_decoder(feat, app_embed[:,i:i+1])[...,:-1]
                gaussians_app = self.model._adapt_gaussians(
                    raw_gaussians=raw_gaussians_app,
                    depths=output.depth,
                    opacities=map_pdf_to_opacity(densities),
                    extrinsics=ctx_extrs,
                    intrinsics=ctx_intrs,
                    H=H,
                    W=W,
                    ray_o=ray_o,
                    ray_d=ray_d,
                )
                harmonics_app[:,i] = gaussians_app.harmonics
            gaussians.harmonics = harmonics_app

            gaussians.means = rearrange(gaussians.means, "b v ... -> (b v) ...")
            gaussians.scales = rearrange(gaussians.scales, "b v ... -> (b v) ...")
            gaussians.rotations = rearrange(gaussians.rotations, "b v ... -> (b v) ...")
            gaussians.harmonics = rearrange(gaussians.harmonics, "b v ... -> (b v) ...")
            gaussians.opacities = rearrange(gaussians.opacities, "b v ... -> (b v) ...")
            tgt_extrs = rearrange(tgt_extrs, "b v ... -> (b v) 1 ...")
            tgt_intrs = rearrange(tgt_intrs, "b v ... -> (b v) 1 ...")
            tgt_images = rearrange(tgt_images, "b v ... -> (b v) 1 ...")
            occ_masks = rearrange(occ_masks, "b v ... -> (b v) 1 ...")

            # ===== Render =====
            all_colors, _ = self.model._render_3dgs(
                image_shape=(H,W),
                tgt_extrs=tgt_extrs,
                tgt_intrs=tgt_intrs,
                gaussians=gaussians,
            )


        target = {"color": tgt_images, "points": points, "weight": weights, "occ_mask": occ_masks}
        pred = {"color": all_colors, "points": points_da3, "gaussians": gaussians, "ray_o": ray_o}

        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(pred, target, self.global_step, batch_masks)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
            

        if (
            self.global_rank == 0
            and self.global_step % self.cfg.train.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)  
            
        return total_loss

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch["image"]
        intrs = batch["intrinsics"]
        extrs = batch["extrinsics"]
        depths = batch["depth"]
        skys = batch["sky"]
        masks = batch["mask"]
    
        B, H, W = images.shape[0], images.shape[-2], images.shape[-1]
        V = images.shape[1]
        V_ctx = 2
        V_tgt = V - V_ctx

        occ_masks = (masks > 0) + (skys > 0.3)
        occ_masks = occ_masks[:,V_ctx:]
    
        all_images = images.clone()
        ctx_images, tgt_images = all_images.split([V_ctx, V_tgt], dim=1)

        ctx_depths, tgt_depths = depths.split([V_ctx, V_tgt], dim=1)
        ctx_skys, tgt_skys = skys.split([V_ctx, V_tgt], dim=1)
        
        x = self.model._imagenet_normalize_bnxchw(all_images)
        
        extrs_norm = self.model._normalize_extrinsics(as_homogeneous(extrs))
        ctx_extrs, tgt_extrs = extrs_norm.split([V_ctx, V_tgt], dim=1)
    
        ctx_intrs, tgt_intrs = intrs.split([V_ctx, V_tgt], dim=1)
    
        feats, aux_feats = self.model.da3.backbone(
            x, cam_token=None, export_feat_layers=[self.model.da3.backbone.alt_start-1], n_ctx=V_ctx, ref_view_strategy="first"
        )

        aux_feat = aux_feats[0][:,-V_tgt:]
        app_embed = self.model._apply_app_encoder(aux_feat, H, W)
    
        with torch.autocast(device_type=x.device.type, enabled=False):
            # ===== Estimate Gaussians =====
            output = self.model.da3.head(feats, H, W, patch_start_idx=0)
            gs_outs = self.model.da3.gs_head(
                feats=feats,
                H=H,
                W=W,
                patch_start_idx=0,
                images=x[:,:V_ctx],
            )
            raw_gaussians = gs_outs.raw_gs
            densities = gs_outs.raw_gs_conf

            last_aux = output["last_aux"]
            feat = gs_outs["fused_feat"]

            # ===== Get ray origins and directions =====
            ray_o, ray_d = self.model._get_ray_origin_and_dir(output.ray, H, W)

            offset_depth = self.model._apply_offset_decoder(feat)[...,-1]

            # ===== Adapt raw Gaussians to be ready for rendering =====
            gaussians = self.model._adapt_gaussians(
                raw_gaussians=raw_gaussians,
                depths=output.depth,
                opacities=map_pdf_to_opacity(densities),
                extrinsics=ctx_extrs,
                intrinsics=ctx_intrs,
                H=H,
                W=W,
                ray_o=ray_o,
                ray_d=ray_d,
                offset_depth=offset_depth,
            )

            ## ===== Rescale Gaussians =====
            points_da3 = ray_o[:,:,None,None,:] + ray_d * output.depth[...,None]
            scale_bias, _, _, _ = self.model._compute_scale_bias(
                rearrange(points_da3, "b v h w c -> b (v h w) c"),
                ctx_depths.float(),
                ctx_intrs.float(),
                ctx_extrs.float(),
                ctx_skys.float(),
            )
            gaussians.means = gaussians.means * scale_bias[:,None,:1] + scale_bias[:,None,1:]
            gaussians.scales = gaussians.scales * scale_bias[:,None,:1]

            # ===== Gaussians with appearance =====
            gaussians = repeat_gaussians(gaussians, V_tgt)
            harmonics_app = torch.zeros_like(gaussians.harmonics)
            for i in range(V_tgt):
                raw_gaussians_app = self.model._apply_gs_decoder(feat, app_embed[:,i:i+1])[...,:-1]
                gaussians_app = self.model._adapt_gaussians(
                    raw_gaussians=raw_gaussians_app,
                    depths=output.depth,
                    opacities=map_pdf_to_opacity(densities),
                    extrinsics=ctx_extrs,
                    intrinsics=ctx_intrs,
                    H=H,
                    W=W,
                    ray_o=ray_o,
                    ray_d=ray_d,
                )
                harmonics_app[:,i] = gaussians_app.harmonics
            gaussians.harmonics = harmonics_app

            gaussians.means = rearrange(gaussians.means, "b v ... -> (b v) ...")
            gaussians.scales = rearrange(gaussians.scales, "b v ... -> (b v) ...")
            gaussians.rotations = rearrange(gaussians.rotations, "b v ... -> (b v) ...")
            gaussians.harmonics = rearrange(gaussians.harmonics, "b v ... -> (b v) ...")
            gaussians.opacities = rearrange(gaussians.opacities, "b v ... -> (b v) ...")
            tgt_extrs = rearrange(tgt_extrs, "b v ... -> (b v) 1 ...")
            tgt_intrs = rearrange(tgt_intrs, "b v ... -> (b v) 1 ...")
            tgt_images = rearrange(tgt_images, "b v ... -> (b v) 1 ...")
            occ_masks = rearrange(occ_masks, "b v ... -> (b v) 1 ...")

            # ===== Render =====
            all_colors, _ = self.model._render_3dgs(
                image_shape=(H,W),
                tgt_extrs=tgt_extrs,
                tgt_intrs=tgt_intrs,
                gaussians=gaussians,
            )

        comparison_image = compare_image(ctx_images[0], tgt_images[0], rearrange(all_colors, "(b v) ... -> b v ...", v=V_tgt)[0,:,0], mask=occ_masks[0])
        self.logger.experiment.log({
            "comparison_image": comparison_image,
            "global_step": self.global_step,
        })


    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        cfg = self.optimizer_cfg

        # -------------------------
        # helper
        # -------------------------
        def base_params(module):
            params = []
            for n, p in module.named_parameters():
                if p.requires_grad and "lora_" not in n:
                    params.append(p)
            return params
    
        param_groups = []
    
        # depth head
        head_params = base_params(self.model.da3.head)
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": cfg.lr.da3.head,
            })
    
        # gs head
        gs_head_params = base_params(self.model.da3.gs_head)
        if gs_head_params:
            param_groups.append({
                "params": gs_head_params,
                "lr": cfg.lr.da3.gs_head,
            })
    
        # appearance encoder
        app_params = base_params(self.model.app_encoder)
        if app_params:
            param_groups.append({
                "params": app_params,
                "lr": cfg.lr.app_encoder,
            })
    
        # gs decoder
        gs_decoder_params = base_params(self.model.gs_decoder)
        if gs_decoder_params:
            param_groups.append({
                "params": gs_decoder_params,
                "lr": cfg.lr.gs_decoder,
            })

        # offset decoder
        offset_decoder_params = base_params(self.model.offset_decoder)
        if offset_decoder_params:
            param_groups.append({
                "params": offset_decoder_params,
                "lr": cfg.lr.offset_decoder,
            })
    
        # -------------------------
        # LoRA params
        # -------------------------
    
        backbone_lora = []
        head_lora = []
    
        for n, p in self.model.named_parameters():
            if p.requires_grad and "lora_" in n:
    
                if "da3.backbone" in n:
                    backbone_lora.append(p)
    
                elif "da3.head" in n:
                    head_lora.append(p)
    
        if backbone_lora:
            param_groups.append({
                "params": backbone_lora,
                "lr": cfg.lr.lora_backbone,
                "weight_decay": 0.0,
            })
    
        if head_lora:
            param_groups.append({
                "params": head_lora,
                "lr": cfg.lr.lora_head,
                "weight_decay": 0.0,
            })
    
        # -------------------------
        # optimizer
        # -------------------------
    
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
    
        # -------------------------
        # scheduler
        # -------------------------
    
        warmup_steps = cfg.warm_up_steps
        max_steps = self.cfg.trainer.max_steps
    
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
    
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda] * len(param_groups),
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    

            


    
        

    