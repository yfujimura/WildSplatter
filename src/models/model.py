import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
#from minlora import add_lora, LoRAParametrization

from src.models.appearance_encoder import AppearanceEncoder
from src.models.utils import _copy_output_conv2, _copy_output_conv2_with_zero_init, _copy_output_conv2_with_app_in, _copy_output_conv2_aux_for_conf
from src.utils.training_utils import compute_weight_two_views, depth2points, linear_transform

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.model.utils.transform import cam_quat_xyzw_to_world_quat_wxyz
from depth_anything_3.utils.sh_helpers import rotate_sh
from depth_anything_3.specs import Gaussians
from depth_anything_3.model.utils.gs_renderer import render_3dgs

class WildSplatterModel(nn.Module):

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.da3 = DepthAnything3.from_pretrained(cfg.da3.pretrained_model).model
        
        self.da3_backbone_train_mode = cfg.da3.backbone.train_mode
        vit = self.da3.backbone.pretrained 
        vit.patch_embed.requires_grad_(False)
        
        alt_start = self.da3.backbone.alt_start
        if self.da3_backbone_train_mode == "frozen":
            for blk in vit.blocks:
                blk.requires_grad_(False)
        elif self.da3_backbone_train_mode == "lora":
            lora_config = {
                nn.Linear: {
                    "weight": partial(LoRAParametrization.from_linear, rank=cfg.da3.backbone.lora_rank)
                }
            }
            for blk in vit.blocks:
                blk.requires_grad_(False)
            for blk in vit.blocks[alt_start:]:
                 add_lora(blk.attn.qkv, lora_config=lora_config)
        else:
            raise ValueError(f"Unknown backbone train_mode: {self.da3_backbone_train_mode}")

        
        self.da3_head_train_mode = cfg.da3.head.train_mode
        depth_head = self.da3.head

        if self.da3_head_train_mode == "frozen":
            for p in depth_head.parameters():
                p.requires_grad_(False)
        elif self.da3_head_train_mode == "lora":
            lora_rank = cfg.da3.head.lora_rank
            lora_config = {
                nn.Conv2d: {
                    "weight": partial(
                        LoRAParametrization.from_conv2d,
                        rank=lora_rank
                    )
                },
                nn.Linear: {
                    "weight": partial(
                        LoRAParametrization.from_linear,
                        rank=lora_rank
                    )
                },
            }
            for p in depth_head.parameters():
                p.requires_grad_(False)
            for module in depth_head.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    add_lora(module, lora_config=lora_config)
        elif self.da3_head_train_mode == "full":
            for p in depth_head.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"Unknown head train_mode: {self.da3_head_train_mode}")
        
        self.app_encoder = AppearanceEncoder(cfg.wild_splatter.appearance_encoder)
        self.gs_decoder = _copy_output_conv2_with_app_in(
            self.da3.gs_head.scratch.output_conv2,
            app_dim=cfg.wild_splatter.appearance_encoder.out_dim,
        )
        self.offset_decoder = _copy_output_conv2(self.da3.gs_head.scratch.output_conv2)
        
        self.eps = 1e-8

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        """Normalize extrinsics"""
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        return ex_t_norm

    def _imagenet_normalize_bnxchw(
        self,
        images: torch.Tensor,
        inplace: bool = False,
    ) -> torch.Tensor:
        """
        ImageNet normalize for images with shape (B, N, C, H, W)
    
        Args:
            images: torch.Tensor, shape (B,N,3,H,W), range [0,1]
            inplace: if True, normalize in-place
    
        Returns:
            normalized images with same shape
        """
        assert images.ndim == 5, f"expected 5D tensor, got {images.shape}"
        assert images.size(2) == 3, "C must be 3 (RGB)"
    
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std  = images.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    
        if inplace:
            images.sub_(mean).div_(std)
            return images
        else:
            return (images - mean) / std

    def _viewwise_normalize(
        self,
        images: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        View-wise normalization for multi-view images (no denorm, keep RGB)
    
        Args:
            images: Tensor [B, V, 3, H, W] (ImageNet normalized)
            eps: numerical stability
    
        Returns:
            Tensor [B, V, 3, H, W]
        """
    
        assert images.dim() == 5 and images.size(2) == 3, \
            "Expected shape [B, V, 3, H, W]"
    
        # Bごとに全viewで統計を共有
        mean = images.mean(dim=[1, 3, 4], keepdim=True)   # [B,1,3,1,1]
        std  = images.std(dim=[1, 3, 4], keepdim=True) + eps
    
        images = (images - mean) / std
    
        return images

    def _denorm_grayscale_viewnorm_renorm(
        self,
        images: torch.Tensor,
        grayscale: bool = True,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Multi-view images preprocessing for pretrained models:
        - ImageNet denormalize
        - (optional) grayscale conversion
        - view-wise normalization (across V)
        - re-apply ImageNet normalization
    
        Args:
            images: Tensor [B, V, 3, H, W] (ImageNet normalized)
            grayscale: whether to convert to grayscale
            eps: numerical stability
    
        Returns:
            Tensor [B, V, 3, H, W] (ImageNet normalized)
        """
    
        assert images.dim() == 5 and images.size(2) == 3, \
            "Expected shape [B, V, 3, H, W]"
    
        device = images.device
    
        # --- ImageNet stats (local) ---
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,1,3,1,1)
        imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,1,3,1,1)
    
        # --- 1. denormalize ---
        x = images * imagenet_std + imagenet_mean
    
        # --- 2. grayscale (optional) ---
        if grayscale:
            x = (
                0.2989 * x[:, :, 0] +
                0.5870 * x[:, :, 1] +
                0.1140 * x[:, :, 2]
            ).unsqueeze(2)  # [B, V, 1, H, W]
    
            x = x.repeat(1, 1, 3, 1, 1)
    
        # --- 3. view-wise normalization ---
        mean = x.mean(dim=[1, 3, 4], keepdim=True)
        std  = x.std(dim=[1, 3, 4], keepdim=True) + eps
        x = (x - mean) / std
    
        # --- 4. re-apply ImageNet normalization ---
        x = (x - imagenet_mean) / imagenet_std
    
        return x

    def _upsample_ray_d(self, ray_d: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        ray_d: (B, N, h0, w0, 3)
        return: (B, N, H, W, 3)
        """
        B, N, h0, w0, _ = ray_d.shape
        x = ray_d.permute(0, 1, 4, 2, 3).reshape(B * N, 3, h0, w0)  # (B*N,3,h0,w0)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = x.reshape(B, N, 3, H, W).permute(0, 1, 3, 4, 2)          # (B,N,H,W,3)
        return x

    def _cam_centers_from_w2c(self, extr: torch.Tensor) -> torch.Tensor:
        """
        extr: (B,V,4,4) world2cam
        return: (B,V,3) camera centers in world
        """
        R = extr[..., :3, :3]
        t = extr[..., :3, 3]
        C = -(R.transpose(-1, -2) @ t[..., None])[..., 0]
        return C

    def _get_ray_origin_and_dir(self, ray_map, H, W):
        ray_d, ray_o = ray_map.split([3,3], dim=-1)
        ray_o = rearrange(ray_o, "b v h w d -> b v (h w) d").mean(2) # b n 3
        ray_d = self._upsample_ray_d(ray_d, H, W)
        return ray_o, ray_d
    
    @torch.no_grad()
    def _scale_tgt_translation_to_ctx_centers(
        self,
        ctx_centers_world: torch.Tensor,   # (B,V,3) e.g : ray origin (world)
        tgt_extr_w2c: torch.Tensor,        # (B,V,4,4) world2cam
        method: str = "mean",
        eps: float = 1e-8,
    ):
        """
        rotations are assumed aligned already.
        Align only translation scale of tgt_extr to match ctx centers' baseline scale.
    
        Returns:
          tgt_scaled: (B,V,4,4)  (t scaled)
          s: (B,) scale factors applied to tgt translation (t *= s)
        """
        assert ctx_centers_world.ndim == 3 and ctx_centers_world.shape[-1] == 3
        assert tgt_extr_w2c.ndim == 4 and tgt_extr_w2c.shape[-2:] == (4,4)
        assert ctx_centers_world.shape[0] == tgt_extr_w2c.shape[0]
        assert ctx_centers_world.shape[1] == tgt_extr_w2c.shape[1]
    
        Ctgt = self._cam_centers_from_w2c(tgt_extr_w2c)   # (B,V,3)
        Cctx = ctx_centers_world.to(Ctgt.device, Ctgt.dtype)
    
        Dc = torch.cdist(Cctx.float(), Cctx.float())  # (B,V,V)
        Dt = torch.cdist(Ctgt.float(), Ctgt.float())
    
        B, V, _ = Dc.shape
        triu = torch.triu(torch.ones((V, V), device=Dc.device, dtype=torch.bool), diagonal=1)
    
        dc = Dc[:, triu].clamp_min(eps)  # (B, V*(V-1)/2)
        dt = Dt[:, triu].clamp_min(eps)
    
        ratio = dc / dt
    
        if method == "median":
            s = ratio.median(dim=1).values
        elif method == "mean":
            s = ratio.mean(dim=1)
        else:
            raise ValueError("method must be 'median' or 'mean'")
    
        tgt_scaled = tgt_extr_w2c.clone()
        tgt_scaled[..., :3, 3] = tgt_scaled[..., :3, 3] * s[:, None, None]
        return tgt_scaled, s

    def _adapt_gaussians(
        self,
        raw_gaussians,
        depths,
        opacities,
        extrinsics,
        intrinsics,
        H, 
        W, 
        ray_o,
        ray_d,
        offset_depth=None,
    ):
        b, v = raw_gaussians.shape[:2]

        # 1. compute 3DGS means
        if offset_depth is not None:
            gs_depths = depths + offset_depth
        else:
            gs_depths = depths

        cam2worlds = affine_inverse(extrinsics)
        intr_normed = intrinsics.clone().detach()
        intr_normed[..., 0, :] /= W
        intr_normed[..., 1, :] /= H

        gs_means_world = ray_o[:,:,None,None,:] + ray_d * gs_depths[...,None]
        gs_means_world = rearrange(gs_means_world, "b v h w d -> b (v h w) d")

        # 2. compute other GS attributes
        # rau_gaussians[...,:2]: learned pixel offset
        # rau_gaussians[...,-1]: learned depth offset
        scales, rotations, sh = raw_gaussians[...,2:-1].split((3, 4, 3 * self.da3.gs_adapter.d_sh), dim=-1)

        scale_min = self.da3.gs_adapter.gaussian_scale_min
        scale_max = self.da3.gs_adapter.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        pixel_size = 1 / torch.tensor((W, H), dtype=raw_gaussians.dtype, device=intrinsics.device)
        multiplier = self.da3.gs_adapter.get_scale_multiplier(intr_normed, pixel_size)
        gs_scales = scales * gs_depths[..., None] * multiplier[..., None, None, None]
        gs_scales = rearrange(gs_scales, "b v h w d -> b (v h w) d")

        # 2.2) 3DGS quaternion (world space)
        # due to historical issue, assume quaternion in order xyzw, not wxyz
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + self.eps)
        # rotate them to world space
        cam_quat_xyzw = rearrange(rotations, "b v h w c -> b (v h w) c")
        c2w_mat = repeat(
            cam2worlds,
            "b v i j -> b (v h w) i j",
            h=H,
            w=W,
        )
        cam_quat_xyzw = cam_quat_xyzw.to(dtype=c2w_mat.dtype) 
        world_quat_wxyz = cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w_mat)
        gs_rotations_world = world_quat_wxyz  # b (v h w) c

        # 2.3) 3DGS color / SH coefficient (world space)
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        if not self.da3.gs_adapter.pred_color:
            sh = sh * self.da3.gs_adapter.sh_mask
        
        if self.da3.gs_adapter.pred_color or self.da3.gs_adapter.sh_degree == 0:
            # predict pre-computed color or predict only DC band, no need to transform
            gs_sh_world = sh
        else:
            gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
        gs_sh_world = rearrange(gs_sh_world, "b v h w xyz d_sh -> b (v h w) xyz d_sh")

        # 2.4) 3DGS opacity
        gs_opacities = rearrange(opacities, "b v h w ... -> b (v h w) ...")
        
        gs_world = Gaussians(
            means=gs_means_world.float(),
            harmonics=gs_sh_world.float(),
            opacities=gs_opacities.float(),
            scales=gs_scales.float(),
            rotations=gs_rotations_world.float(),
        )
        return gs_world

    def _render_3dgs(
        self,
        image_shape,
        tgt_extrs,
        tgt_intrs,
        gaussians,
        chunk_size=None,
    ):
        v = tgt_extrs.shape[1]
        tgt_c2w = affine_inverse(tgt_extrs)

        in_h, in_w = image_shape
        
        intr_normed = tgt_intrs.clone().detach()
        intr_normed[..., 0, :] /= in_w
        intr_normed[..., 1, :] /= in_h
        tgt_intrs = intr_normed

        if chunk_size is None:
            chunk_size = v
        chunk_size = min(v, chunk_size)
        all_colors = []
        all_depths = []

        for chunk_idx in range(math.ceil(v / chunk_size)):
            s = int(chunk_idx * chunk_size)
            e = int((chunk_idx + 1) * chunk_size)
            cur_n_view = tgt_extrs[:, s:e].shape[1]
            color, depth = render_3dgs(
                extrinsics=rearrange(tgt_extrs[:, s:e], "b v ... -> (b v) ...").float(),  # w2c
                intrinsics=rearrange(tgt_intrs[:, s:e], "b v ... -> (b v) ...").float(),  # normed
                image_shape=image_shape,
                gaussian=gaussians,
                num_view=cur_n_view,
                use_sh=True
            )
            all_colors.append(rearrange(color, "(b v) ... -> b v ...", v=cur_n_view))
            all_depths.append(rearrange(depth, "(b v) ... -> b v ...", v=cur_n_view))
        all_colors = torch.cat(all_colors, 1)

        all_colors.clamp(0,1) # b v c h w
        return all_colors, all_depths

    def _apply_gs_decoder(self, feat, app_embed, v_max=2.):
        """
        feat:      (B, V, C, H, W)
        app_embed: (B, 1, D)
        """
        B, V, C, H, W = feat.shape
    
        feat = rearrange(feat, "b v c h w -> (b v) c h w")
        app = repeat(app_embed, "b 1 d -> (b v) d h w", v=V, h=H, w=W)
    
        x = torch.cat([feat, app], dim=1)
        raw_gaussians = self.gs_decoder(x)
    
        raw_gaussians = rearrange(
            raw_gaussians, "(b v) c h w -> b v h w c", b=B, v=V
        )

        # limit range to (vmin, vmax)
        raw_gaussians = v_max * raw_gaussians.tanh()
    
        return raw_gaussians

    def _apply_offset_decoder(self, feat, activation="sigmoid"):
        """
        feat:      (B, V, C, H, W)
        """
        B, V, C, H, W = feat.shape
    
        x = rearrange(feat, "b v c h w -> (b v) c h w")
        raw_gaussians = self.offset_decoder(x)
    
        raw_gaussians = rearrange(
            raw_gaussians, "(b v) c h w -> b v h w c", b=B, v=V
        )

        if activation == "sigmoid":
            raw_gaussians = torch.sigmoid(raw_gaussians)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
        return raw_gaussians

    def _apply_app_encoder(
        self,
        feat, # (B, V, N, D)
        H,
        W,
    ):
        B, V, N, D = feat.shape
        feat = rearrange(feat, "b v n d -> (b v) n d")
        app_embed = self.app_encoder(feat, H, W)
        app_embed = rearrange(app_embed, "(b v) d -> b v d", b=B)
        return app_embed

    def _apply_conf_head(
        self,
        feat: torch.Tensor,          # (B, V, C, Hf, Wf)
        image_size: tuple[int, int], # (H_img, W_img)
        return_logit: bool = True,
    ):
        """
        Args:
            feat:        (B, V, C, Hf, Wf) 
            image_size:  (H_img, W_img) 
            return_logit: s も返すか
    
        Returns:
            conf: (B, V, 1, H_img, W_img)
            s:    (B, V, 1, H_img, W_img) (optional)
        """
        assert feat.dim() == 5, f"feat must be (B,V,C,H,W) but got {feat.shape}"
        B, V, C, Hf, Wf = feat.shape
        H_img, W_img = image_size
    
        # (B,V,C,Hf,Wf) -> (B*V,C,Hf,Wf)
        feat = rearrange(feat, "b v c h w -> (b v) c h w")
    
        # s: (B*V,1,Hf,Wf)
        s = self.conf_head(feat)
        s = s.clamp(-10, 5)
    
        # upsample to image resolution
        if (Hf, Wf) != (H_img, W_img):
            s = F.interpolate(
                s,
                size=(H_img, W_img),
                mode="bilinear",
                align_corners=False,
            )
    
        # (B*V,1,H,W) -> (B,V,1,H,W)
        s = rearrange(s, "(b v) c h w -> b v c h w", b=B, v=V)
    
        # conf = exp(s)
        conf = torch.exp(s)
    
        if return_logit:
            return conf, s
        return conf
        

    @torch.no_grad()
    def compute_rays_w2c(
        self,
        extrinsics: torch.Tensor,  # (B,S,4,4) world-to-camera
        intrinsics: torch.Tensor,  # (B,S,3,3)
        H: int,
        W: int,
    ):
        """
        Returns:
            ray_o: (B,S,3)
            ray_d: (B,S,H,W,3)
        """
        device = extrinsics.device
        dtype = extrinsics.dtype
        B, S = extrinsics.shape[:2]
    
        # ---- w2c -> c2w ----
        R = extrinsics[..., :3, :3]          # (B,S,3,3)
        t = extrinsics[..., :3, 3]           # (B,S,3)
    
        R_inv = R.transpose(-1, -2)           # (B,S,3,3)
        cam_center = -torch.einsum(
            "bsij,bsj->bsi", R_inv, t
        )                                     # (B,S,3)
    
        ray_o = cam_center
    
        # ---- pixel grid ----
        u = torch.arange(W, device=device, dtype=dtype) + 0.5
        v = torch.arange(H, device=device, dtype=dtype) + 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # (H,W)
    
        pix = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)  # (H,W,3)
        pix = pix.view(1, 1, H, W, 3).expand(B, S, H, W, 3)
    
        # ---- camera ray directions ----
        Kinv = torch.linalg.inv(intrinsics)    # (B,S,3,3)
        d_cam = torch.einsum(
            "bsij,bshwj->bshwi", Kinv, pix
        )                                      # (B,S,H,W,3)
    
        # ---- world ray directions ----
        ray_d = torch.einsum(
            "bsij,bshwj->bshwi", R_inv, d_cam
        )                                      # (B,S,H,W,3)
    
        ray_d = F.normalize(ray_d, dim=-1)
    
        return ray_o, ray_d

    @torch.no_grad()
    def _compute_scale_bias(
        self,
        gaussian_means,
        depths,
        intrs,
        extrs,
        skys,
        return_mask=False,
        mask_thr=0.5,
    ):
        B = gaussian_means.shape[0]
        
        weights = compute_weight_two_views(depths, intrs, extrs, skys) # (B, V, H, W)
        weights = rearrange(weights, "b v h w -> b (v h w)")
        points = depth2points(
            rearrange(depths, "b v ... -> (b v) ..."),
            rearrange(intrs, "b v ... -> (b v) ..."),
            rearrange(extrs[:,:,:3], "b v ... -> (b v) ..."),
        ) # (BV, H, W, 3)
        points = rearrange(points, "(b v) h w c -> b (v h w) c", b=B) 
        
        scale_bias = linear_transform(
            gaussian_means,
            points,
            weights,
        )

        if return_mask:
            scaled_points = gaussian_means * scale_bias[:,None,:1] + scale_bias[:,None,1:]
            pts_errors = (weights * torch.sum((scaled_points - points)**2, dim=2)).mean(dim=1) # B
            weights_sum = weights.sum(dim=1) # B
            masks = (pts_errors < mask_thr) * (weights_sum > 0)
            return scale_bias, points, weights, masks
        else:
            return scale_bias, points, weights, None
            
            
            
            
            
    
    
    
            