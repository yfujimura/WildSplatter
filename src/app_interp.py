import sys
sys.path.append("src/Depth-Anything-3/src")

import os
import glob
import hydra
from omegaconf import OmegaConf

import cv2
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, map_pdf_to_opacity

from src.models.model import WildSplatterModel
from src.models.model_wrapper import ModelWrapper

INPUT_DIR = "./assets"

def so3_log(R: torch.Tensor) -> torch.Tensor:
    """SO(3)->so(3): returns axis-angle vector w (shape: (3,))"""
    cos_theta = (torch.trace(R) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta < 1e-6:
        return torch.zeros(3, device=R.device, dtype=R.dtype)

    w = torch.stack([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2 * torch.sin(theta))

    return theta * w


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """so(3)->SO(3): input w (shape: (3,))"""
    theta = torch.norm(w)
    I = torch.eye(3, device=w.device, dtype=w.dtype)

    if theta < 1e-6:
        return I

    k = w / theta
    K = torch.tensor([
        [0,     -k[2],  k[1]],
        [k[2],   0,    -k[0]],
        [-k[1],  k[0],  0   ],
    ], device=w.device, dtype=w.dtype)

    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def interpolate_extrinsics_w2c(extrinsics: torch.Tensor, n: int, interp_center: bool = True) -> torch.Tensor:
    """
    extrinsics: (2, 4, 4) w2c (world->camera)
    n: total cameras INCLUDING endpoints (so middle cameras are n-2)
    interp_center:
      True  -> interpolate camera centers C in world coords (recommended)
      False -> interpolate t directly in w2c
    returns: (n, 4, 4) w2c
    """
    assert extrinsics.shape == (2, 4, 4)
    assert n >= 2

    E0, E1 = extrinsics[0], extrinsics[1]
    R0, t0 = E0[:3, :3], E0[:3, 3]
    R1, t1 = E1[:3, :3], E1[:3, 3]

    # Rotation SLERP on SO(3): R(alpha) = Exp(alpha * log(R1 R0^T)) R0
    R_rel = R1 @ R0.T
    w = so3_log(R_rel)

    # For w2c: camera center in world is C = -R^T t
    if interp_center:
        C0 = -R0.T @ t0
        C1 = -R1.T @ t1

    outs = []
    for i in range(n):
        alpha = i / (n - 1)

        R = so3_exp(alpha * w) @ R0

        if interp_center:
            C = (1 - alpha) * C0 + alpha * C1
            t = -R @ C  # back to w2c translation
        else:
            t = (1 - alpha) * t0 + alpha * t1

        E = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype)
        E[:3, :3] = R
        E[:3, 3] = t
        outs.append(E)

    return torch.stack(outs, dim=0)

def save_video_opencv(images: torch.Tensor, path: str, fps: int = 30):
    """
    images: (N, 3, H, W), float32, range [0,1], RGB
    """
    images = (images * 255).clamp(0, 255).byte()
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # (N,H,W,3)

    H, W = images.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))

    for frame in images:
        frame_bgr = frame[:, :, ::-1]  # RGB -> BGR
        writer.write(frame_bgr)

    writer.release()


def load_resize_crop(path, size, resample=Image.BICUBIC):
    """
    画像を読み込み、アスペクト比を保ってリサイズし、
    中央をクロップして (size, size) にする

    Args:
        path (str): 画像ファイルのパス
        size (int): 出力画像の一辺のサイズ
        resample: リサイズ時の補間方法

    Returns:
        PIL.Image.Image: 処理後の画像
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # アスペクト比を保ったまま、短辺が size 以上になるようにリサイズ
    scale = size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), resample=resample)

    # 中央クロップ
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    right = left + size
    bottom = top + size
    img = img.crop((left, top, right, bottom))

    return img, [scale, left, top]

def pil_list_to_tensor(images):
    """
    Args:
        images (List[PIL.Image.Image]): PIL Image のリスト

    Returns:
        torch.Tensor: (B, C, H, W)
    """
    to_tensor = transforms.ToTensor()  # (C, H, W), [0,1]

    tensors = [to_tensor(img) for img in images]
    batch = torch.stack(tensors, dim=0)  # (B, C, H, W)
    return batch
    

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main",
)
def main(cfg):
    torch.set_float32_matmul_precision("medium")

    output_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    wild_splatter_model = WildSplatterModel(cfg.model)
    model_wrapper = ModelWrapper.load_from_checkpoint(
        cfg.ckpt_path,
        cfg=cfg,
        model=wild_splatter_model,
    )

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    image_paths.sort()
    
    images = []
    for image_path in image_paths:
        image, _ = load_resize_crop(image_path, 504)
        images.append(image)
    images = pil_list_to_tensor(images)

    N = 60
    batch = {}
    batch["image"] = images.unsqueeze(0).cuda()
    render_imgs = []
    with torch.no_grad():
        images = batch["image"].clone()
        images = torch.cat([images, images], dim=1)
    
        B, H, W = images.shape[0], images.shape[-2], images.shape[-1]
        V = images.shape[1]
        V_ctx = 2
        V_tgt = V - V_ctx
        
        all_images = images.clone()
        ctx_images, tgt_images = all_images.split([V_ctx, V_tgt], dim=1)

        x = model_wrapper.model._imagenet_normalize_bnxchw(all_images)
    
        feats, aux_feats = model_wrapper.model.da3.backbone(
            x, cam_token=None, export_feat_layers=[model_wrapper.model.da3.backbone.alt_start-1], n_ctx=V_ctx, ref_view_strategy="first"
        )

        aux_feat = aux_feats[0][:,-V_tgt:]
        app_embed_org = model_wrapper.model._apply_app_encoder(aux_feat, H, W)
        for n in range(N):
            ratio = (N - n) / N
            app_embed = app_embed_org[:,0] * ratio + app_embed_org[:,1] * (1-ratio)
            app_embed = app_embed[:,None].expand(app_embed_org.shape[0], 2, -1)

            with torch.autocast(device_type=x.device.type, enabled=False):
                # ===== Estimate Gaussians =====
                output = model_wrapper.model.da3.head(feats, H, W, patch_start_idx=0)
                gs_outs = model_wrapper.model.da3.gs_head(
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
                ray_o, ray_d = model_wrapper.model._get_ray_origin_and_dir(output.ray, H, W)
        
                # ===== The scale of target extrinsics is adjusted to the DA3 scale =====
                output = model_wrapper.model.da3._process_ray_pose_estimation(output, H, W)
                ctx_intrs = tgt_intrs = output["intrinsics"]
                ctx_extrs = tgt_extrs = as_homogeneous(output["extrinsics"])
    
                interpolated_extrs = interpolate_extrinsics_w2c(tgt_extrs[0], N)
                interpolated_intrs = tgt_intrs[0,0][None].expand(N,3,3)

                offset_depth = model_wrapper.model._apply_offset_decoder(feat)[...,-1]
        
                # ===== Adapt raw Gaussians to be ready for rendering =====
                output.gaussians = model_wrapper.model._adapt_gaussians(
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
        
                # ===== Gaussians with appearance =====
                gaussians_app = []
                for i in range(1):
                    raw_gaussians_app = model_wrapper.model._apply_gs_decoder(feat, app_embed[:,i:i+1])[...,:-1]
                    gaussians_app.append(
                        model_wrapper.model._adapt_gaussians(
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
                    )
                output.gaussians.harmonics = gaussians_app[0].harmonics
        
    
            # ===== Render =====
            all_colors, _ = model_wrapper.model._render_3dgs(
                image_shape=(H,W),
                tgt_extrs=interpolated_extrs[None, n:n+1],
                tgt_intrs=interpolated_intrs[None, n:n+1],
                gaussians=output.gaussians,
            )
            all_colors = all_colors.clamp(0,1)
            render_imgs.append(all_colors[0])
    
    save_video_opencv(torch.cat(render_imgs,0), os.path.join(output_dir, "output_video.mp4"), fps=30)
    print("save" + os.path.join(output_dir, "output_video.mp4"))
    
   

if __name__ == "__main__":
    main()