#!/usr/bin/env python3
# ============================================================
# Precompute DA3 depth for MegaScenes (all images)
#   - preprocess: min-side resize -> 504x504 center crop
#   - DA3 metric depth (CUDA)
#   - render COLMAP sparse depth from *ONLY points observed in that image* (CUDA)
#   - affine align: sparse ≈ a*da3 + b   (RANSAC robust)
#   - save to: <depth_root>/{i0:03}/{i1:03}/<stem>_img<image_id>.npz
#
# Single-GPU
#   python precompute_depths_da3.py \
#       --root_dir /path/to/MegaScenes \
#       --depth_root /path/to/depths \
#       --i0_start 0 --i0_end 0 --batch_size 256
#
# Multi-GPU:
#   torchrun --nproc_per_node=4 precompute_depths_da3.py \
#       --root_dir /path/to/MegaScenes \
#       --depth_root /path/to/depths \
#       --i0_start 0 --i0_end 458 --batch_size 256
# ============================================================

import sys
sys.path.append("../Depth-Anything-3/src")

import os, glob, json, argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from tqdm import tqdm

from read_write_model import read_model, qvec2rotmat
from depth_anything_3.api import DepthAnything3

# safer: skip truncated rather than crash
ImageFile.LOAD_TRUNCATED_IMAGES = False


# ============================================================
# Distributed helpers
# ============================================================
def dist_is_available_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank_world():
    if dist_is_available_and_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1

def dist_init_from_env():
    # torchrun sets these
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


# ============================================================
# Image preprocess (same policy as your viewset maker)
# ============================================================
def preprocess_min_side_resize_center_crop(pil: Image.Image, out_size: int):
    """
    scale so min(w,h) == out_size, then center crop out_size x out_size.
    return: pil_crop, scale, left, top, (orig_w,h), (resized_w,h)
    """
    w, h = pil.size
    scale = out_size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pil_rs = pil.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - out_size) // 2
    top  = (new_h - out_size) // 2
    pil_cr = pil_rs.crop((left, top, left + out_size, top + out_size))
    return pil_cr, float(scale), int(left), int(top), (int(w), int(h)), (int(new_w), int(new_h))


# ============================================================
# COLMAP camera -> fx,fy,cx,cy and adjust for resize/crop
# ============================================================
def colmap_cam_to_fx_fy_cx_cy(cam):
    model = cam.model.upper()
    p = np.array(cam.params, dtype=np.float64)

    if model == "SIMPLE_PINHOLE":       # f, cx, cy
        f, cx, cy = p
        fx, fy = f, f
    elif model == "PINHOLE":            # fx, fy, cx, cy
        fx, fy, cx, cy = p
    elif model in ["SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"]:
        f, cx, cy, _k = p
        fx, fy = f, f
    elif model in ["RADIAL", "RADIAL_FISHEYE"]:
        f, cx, cy, _k1, _k2 = p
        fx, fy = f, f
    elif model in ["OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"]:
        fx, fy, cx, cy = p[:4]
    elif model == "FOV":
        f, cx, cy, _omega = p
        fx, fy = f, f
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {cam.model}")

    return float(fx), float(fy), float(cx), float(cy), cam.model

def adjust_intrinsics_resize_crop(fx, fy, cx, cy, scale, left, top):
    fx2 = fx * scale
    fy2 = fy * scale
    cx2 = cx * scale - left
    cy2 = cy * scale - top
    return fx2, fy2, cx2, cy2

def make_K_torch(fx, fy, cx, cy, device, dtype=torch.float32):
    return torch.tensor([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], device=device, dtype=dtype)

def w2c_from_qt(qvec, tvec):
    R = qvec2rotmat(qvec).astype(np.float64)  # world->cam
    t = np.array(tvec, dtype=np.float64).reshape(3, 1)
    Rt = np.concatenate([R, t], axis=1)       # 3x4
    return Rt


# ============================================================
# Sparse depth rendering (CUDA)
#   IMPORTANT: points_xyz_world must be only points observed in the image
# ============================================================
@torch.no_grad()
def render_sparse_depth(points_xyz_world: torch.Tensor,
                        w2c_3x4: torch.Tensor,
                        K: torch.Tensor,
                        H: int, W: int,
                        stride: int = 2):
    """
    points_xyz_world: (P,3) float32 CUDA
    w2c_3x4: (3,4) float32 CUDA
    K: (3,3) float32 CUDA
    returns:
      depth_sparse (H,W) float32 (0 where no point)
      mask_sparse  (H,W) bool
    """
    device = points_xyz_world.device
    P = int(points_xyz_world.shape[0])
    if P <= 0:
        depth = torch.zeros((H, W), device=device, dtype=torch.float32)
        mask = torch.zeros((H, W), device=device, dtype=torch.bool)
        return depth, mask

    ones = torch.ones((P, 1), device=device, dtype=points_xyz_world.dtype)
    Xw_h = torch.cat([points_xyz_world, ones], dim=1)  # (P,4)

    Xc = (w2c_3x4 @ Xw_h.T).T  # (P,3)
    z = Xc[:, 2]
    valid = z > 1e-6
    Xc = Xc[valid]
    z = z[valid]
    if Xc.numel() == 0:
        depth = torch.zeros((H, W), device=device, dtype=torch.float32)
        mask = torch.zeros((H, W), device=device, dtype=torch.bool)
        return depth, mask

    uvw = (K @ Xc.T).T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]

    ui = torch.round(u).long()
    vi = torch.round(v).long()
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    ui = ui[inb]
    vi = vi[inb]
    z  = z[inb]
    if ui.numel() == 0:
        depth = torch.zeros((H, W), device=device, dtype=torch.float32)
        mask = torch.zeros((H, W), device=device, dtype=torch.bool)
        return depth, mask

    if stride > 1:
        keep = ((ui % stride) == 0) & ((vi % stride) == 0)
        ui = ui[keep]; vi = vi[keep]; z = z[keep]
        if ui.numel() == 0:
            depth = torch.zeros((H, W), device=device, dtype=torch.float32)
            mask = torch.zeros((H, W), device=device, dtype=torch.bool)
            return depth, mask

    depth = torch.full((H, W), float("inf"), device=device, dtype=torch.float32)
    flat_idx = (vi * W + ui).to(torch.int64)
    depth_flat = depth.view(-1)

    try:
        depth_flat.scatter_reduce_(0, flat_idx, z.float(), reduce="amin", include_self=True)
    except Exception:
        depth[vi, ui] = torch.minimum(depth[vi, ui], z.float())

    depth = depth_flat.view(H, W)
    mask = torch.isfinite(depth)
    depth = torch.where(mask, depth, torch.zeros_like(depth))
    return depth, mask


# ============================================================
# RANSAC affine fit: sparse ≈ a*da + b
#   - returns a,b,n_inliers,n_fit,thresh_used,min_inliers_used,ransac_ok
# ============================================================
@torch.no_grad()
def fit_affine_ransac_da_to_sparse(
    da_depth: torch.Tensor,
    sparse_depth: torch.Tensor,
    sparse_mask: torch.Tensor,
    min_pts: int = 30,
    iters: int = 400,
    thresh: float = -1.0,          # if <0: auto
    min_inliers: int = -1,         # if <0: auto (max(min_pts, 0.3*N))
    eps: float = 1e-6,
):
    device = da_depth.device
    m = sparse_mask & (da_depth > eps) & (sparse_depth > eps)
    x = da_depth[m].float()
    y = sparse_depth[m].float()
    N = int(x.numel())

    if N < min_pts:
        a = torch.tensor(1.0, device=device, dtype=torch.float32)
        b = torch.tensor(0.0, device=device, dtype=torch.float32)
        return a, b, 0, N, float("nan"), 0, False

    if thresh < 0:
        med = float(torch.median(y).item())
        thresh = max(0.05 * med, 1e-3)

    if min_inliers < 0:
        min_inliers = max(min_pts, int(0.3 * N))

    idx = torch.arange(N, device=device)
    best_cnt = -1
    best_inl = None

    for _ in range(iters):
        ids = idx[torch.randint(0, N, (2,), device=device)]
        x1, x2 = x[ids[0]], x[ids[1]]
        y1, y2 = y[ids[0]], y[ids[1]]
        if torch.abs(x2 - x1) < 1e-6:
            continue
        a_h = (y2 - y1) / (x2 - x1)
        b_h = y1 - a_h * x1

        err = torch.abs(a_h * x + b_h - y)
        inl = err < thresh
        cnt = int(inl.sum().item())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl

    if best_inl is None or best_cnt < min_inliers:
        # fallback to plain LS (avoid crash)
        A = torch.stack([x, torch.ones_like(x)], dim=1)
        p = torch.linalg.lstsq(A, y).solution
        a = p[0].float()
        b = p[1].float()
        return a, b, 0, N, float(thresh), int(min_inliers), False

    # refine with least squares on inliers
    xi = x[best_inl]
    yi = y[best_inl]
    A = torch.stack([xi, torch.ones_like(xi)], dim=1)
    p = torch.linalg.lstsq(A, yi).solution
    a = p[0].float()
    b = p[1].float()
    return a, b, int(best_cnt), N, float(thresh), int(min_inliers), True


@torch.no_grad()
def apply_affine(depth: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    return torch.clamp(a * depth + b, min=1e-6)


# ============================================================
# DA3 metric depth inference (batched on N dimension)
# ============================================================
@torch.no_grad()
def da3_metric_depth_batch(model: DepthAnything3,
                           imgs_n3hw: torch.Tensor,
                           imagenet_mean: torch.Tensor,
                           imagenet_std: torch.Tensor):
    """
    imgs_n3hw: (N,3,H,W) float in [0,1] CUDA
    returns: depth_nhw (N,H,W) float32 CUDA
    """
    x = imgs_n3hw.unsqueeze(0)  # (1,N,3,H,W)
    x = (x - imagenet_mean) / imagenet_std
    out = model.model.da3_metric(x)
    depth = out.depth[0].float()  # (N,H,W)
    sky = out.sky[0].float()
    return depth, sky


# ============================================================
# Scene enumeration and basename mapping
# ============================================================
def find_scene_colmap_dir(root_dir: str, i0: int, i1: int) -> Optional[str]:
    cands = [
        os.path.join(root_dir, f"reconstruct/{i0:03}/{i1:03}/colmap/0"),
        os.path.join(root_dir, f"reconstruct/{i0:03}/{i1:03}/colmap/sparse/0"),
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return None

def build_by_base_for_scene_images(root_dir: str, i0: int, i1: int) -> Dict[str, str]:
    img_root = os.path.join(root_dir, f"images/{i0:03}/{i1:03}")
    all_imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        all_imgs += glob.glob(os.path.join(img_root, "**", ext), recursive=True)
    by_base = {}
    for p in all_imgs:
        b = os.path.basename(p)
        if b not in by_base:
            by_base[b] = p
    return by_base

def enumerate_scenes(root_dir: str, i0_start: int, i0_end: int, i1_start: int, i1_end: int) -> List[Tuple[int,int]]:
    scenes = []
    for i0 in range(i0_start, i0_end + 1):
        r0 = os.path.join(root_dir, f"reconstruct/{i0:03}")
        if not os.path.isdir(r0):
            continue
        for i1 in range(i1_start, i1_end + 1):
            r = os.path.join(root_dir, f"reconstruct/{i0:03}/{i1:03}")
            if os.path.isdir(r):
                scenes.append((i0, i1))
    return scenes


# ============================================================
# Output paths
# ============================================================
def out_npz_path(depth_root: str, i0: int, i1: int, image_id: int, basename: str) -> str:
    """
    Save as <stem>_img<image_id>.npz to avoid basename collisions in the same scene.
    """
    stem = os.path.splitext(basename)[0]
    return os.path.join(depth_root, f"{i0:03}", f"{i1:03}", f"{stem}_img{int(image_id):08d}.npz")


# ============================================================
# gather only points observed in an image
# ============================================================
def observed_points_xyz_for_image(points3D, image) -> Optional[np.ndarray]:
    try:
        p3d_ids = np.array(image.point3D_ids, dtype=np.int64)
    except Exception:
        return None

    p3d_ids = p3d_ids[p3d_ids >= 0]
    if p3d_ids.size == 0:
        return None
    p3d_ids = np.unique(p3d_ids)

    xyz = []
    for pid in p3d_ids:
        p = points3D.get(int(pid), None)
        if p is None:
            continue
        xyz.append(p.xyz)
    if len(xyz) == 0:
        return None
    return np.stack(xyz, axis=0).astype(np.float32)


# ============================================================
# Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, default="/path/to/MegaScenes")
    ap.add_argument("--depth_root", type=str, default="", help="default: <root_dir>/depths")

    ap.add_argument("--i0_start", type=int, required=True)
    ap.add_argument("--i0_end", type=int, required=True)
    ap.add_argument("--i1_start", type=int, default=0)
    ap.add_argument("--i1_end", type=int, default=999)

    ap.add_argument("--out_size", type=int, default=504)

    ap.add_argument("--batch_size", type=int, default=8, help="per-GPU batch over images (N)")
    ap.add_argument("--stride_sparse", type=int, default=2, help="projection stride for sparse depth rendering")
    ap.add_argument("--min_sparse_pts", type=int, default=10, help="min sparse samples to fit affine")
    ap.add_argument("--save_depth_dtype", type=str, default="float16", choices=["float16","float32"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log_jsonl", type=str, default="", help="default: <depth_root>/_precompute_log_rank*.jsonl")
    ap.add_argument("--model_id", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")
    ap.add_argument("--no_tqdm", action="store_true")

    ap.add_argument("--ransac_iters", type=int, default=400)
    ap.add_argument("--ransac_thresh", type=float, default=-1.0, help="if <0: auto threshold")

    return ap.parse_args()

def main():
    dist_init_from_env()
    rank, world = get_rank_world()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "This script expects CUDA GPUs."

    args = parse_args()

    root_dir = args.root_dir
    depth_root = args.depth_root if args.depth_root else os.path.join(root_dir, "depths")
    os.makedirs(depth_root, exist_ok=True)

    if args.log_jsonl:
        log_path = args.log_jsonl
    else:
        log_path = os.path.join(depth_root, f"_precompute_log_rank{rank:02d}.jsonl")

    model = DepthAnything3.from_pretrained(args.model_id)
    model = model.cuda()
    model.eval()

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,1,3,1,1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,1,3,1,1)

    scenes = enumerate_scenes(root_dir, args.i0_start, args.i0_end, args.i1_start, args.i1_end)
    if rank == 0:
        print(f"[INFO] scenes found in range: {len(scenes)}  (world_size={world})")
        print(f"[INFO] depth_root(abs): {os.path.abspath(depth_root)}")
        print(f"[INFO] root_dir(abs): {os.path.abspath(root_dir)}")
        print(f"[INFO] run command example:")
        print(f"  torchrun --nproc_per_node={world} precompute_depths_da3.py --root_dir {root_dir} --i0_start {args.i0_start} --i0_end {args.i0_end}")

    scenes_rank = [s for idx, s in enumerate(scenes) if (idx % world) == rank]
    pbar = tqdm(scenes_rank, disable=args.no_tqdm, desc=f"rank{rank}", ncols=110)

    stats = {
        "scenes_total": len(scenes_rank),
        "scenes_ok": 0,
        "images_total": 0,
        "images_saved": 0,
        "images_skipped_exists": 0,
        "images_skipped_missing_file": 0,
        "images_failed_load": 0,
        "images_failed_affine": 0,
        "images_failed_sparse_pts": 0,
        "scenes_skip_no_colmap": 0,
        "scenes_skip_read_model": 0,
        "scenes_skip_no_points": 0,
    }

    def log_line(obj: dict):
        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    for (i0, i1) in pbar:
        colmap_dir = find_scene_colmap_dir(root_dir, i0, i1)
        if colmap_dir is None:
            stats["scenes_skip_no_colmap"] += 1
            continue

        try:
            cameras, images, points3D = read_model(colmap_dir, ext=".bin")
        except Exception as e:
            stats["scenes_skip_read_model"] += 1
            log_line({"type": "scene_error", "i0": i0, "i1": i1, "colmap_dir": colmap_dir, "error": f"read_model:{repr(e)}"})
            continue

        if len(points3D) == 0:
            stats["scenes_skip_no_points"] += 1
            continue

        by_base = build_by_base_for_scene_images(root_dir, i0, i1)
        if len(by_base) == 0:
            continue

        usable = []
        for image_id, im in images.items():
            base = os.path.basename(im.name)
            if base in by_base:
                usable.append(image_id)
        usable = sorted(usable)
        if len(usable) == 0:
            continue

        stats["scenes_ok"] += 1

        batch_imgs = []
        batch_meta = []  # (image_id, base, path, K, w2c, pmeta)
        H = W = args.out_size

        def flush_batch():
            nonlocal batch_imgs, batch_meta
            if len(batch_imgs) == 0:
                return

            imgs_n3hw = torch.stack(batch_imgs, dim=0).to(device=device, non_blocking=True)

            try:
                da_depths, da_skys = da3_metric_depth_batch(model, imgs_n3hw, imagenet_mean, imagenet_std)
            except Exception as e:
                for meta in batch_meta:
                    (image_id, base, path, _, _, _pmeta) = meta
                    log_line({"type": "image_error", "i0": i0, "i1": i1, "image_id": int(image_id),
                              "basename": base, "path": path, "stage": "da3", "error": repr(e)})
                batch_imgs = []
                batch_meta = []
                return

            for bi, meta in enumerate(batch_meta):
                (image_id, base, path, K, w2c, pmeta) = meta
                out_path = out_npz_path(depth_root, i0, i1, image_id, base)  # CHANGED

                if (not args.overwrite) and os.path.exists(out_path):
                    stats["images_skipped_exists"] += 1
                    continue

                da = da_depths[bi]
                sky = da_skys[bi]

                try:
                    xyz = observed_points_xyz_for_image(points3D, images[image_id])
                    if xyz is None or xyz.shape[0] < args.min_sparse_pts:
                        stats["images_failed_sparse_pts"] += 1
                        log_line({"type": "image_error", "i0": i0, "i1": i1, "image_id": int(image_id),
                                  "basename": base, "path": path, "stage": "sparse_points",
                                  "error": f"not_enough_observed_points: {0 if xyz is None else int(xyz.shape[0])}"})
                        continue
                    pts_t = torch.from_numpy(xyz).to(device=device, non_blocking=True)
                except Exception as e:
                    stats["images_failed_sparse_pts"] += 1
                    log_line({"type": "image_error", "i0": i0, "i1": i1, "image_id": int(image_id),
                              "basename": base, "path": path, "stage": "sparse_points", "error": repr(e)})
                    continue

                try:
                    sparse, sparse_mask = render_sparse_depth(pts_t, w2c, K, H, W, stride=args.stride_sparse)

                    a, b, n_inl, n_fit, thr_used, min_inl_used, ransac_ok = fit_affine_ransac_da_to_sparse(
                        da, sparse, sparse_mask,
                        min_pts=args.min_sparse_pts,
                        iters=args.ransac_iters,
                        thresh=args.ransac_thresh,
                    )
                    depth_aligned = apply_affine(da, a, b)
                except Exception as e:
                    stats["images_failed_affine"] += 1
                    log_line({"type": "image_error", "i0": i0, "i1": i1, "image_id": int(image_id),
                              "basename": base, "path": path, "stage": "sparse/ransac_affine", "error": repr(e)})
                    continue

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if args.save_depth_dtype == "float16":
                    depth_to_save = depth_aligned.detach().cpu().to(torch.float16).numpy()
                    sky_to_save = sky.detach().cpu().to(torch.float16).numpy()
                else:
                    depth_to_save = depth_aligned.detach().cpu().to(torch.float32).numpy()
                    sky_to_save = sky.detach().cpu().to(torch.float32).numpy()

                K_cpu = K.detach().cpu().numpy().astype(np.float32)
                w2c_cpu = w2c.detach().cpu().numpy().astype(np.float32)

                np.savez_compressed(
                    out_path,
                    depth=depth_to_save,
                    sky=sky_to_save,
                    a=np.float32(a.detach().cpu().item()),
                    b=np.float32(b.detach().cpu().item()),
                    n_sparse=np.int32(n_fit),
                    n_inliers=np.int32(n_inl),
                    inlier_ratio=np.float32(float(n_inl) / max(1, int(n_fit))),
                    ransac_thresh=np.float32(thr_used) if np.isfinite(thr_used) else np.float32(-1.0),
                    min_inliers=np.int32(min_inl_used),
                    ransac_ok=np.bool_(bool(ransac_ok)),
                    K=K_cpu,
                    w2c=w2c_cpu,
                    image_id=np.int32(image_id),
                    basename=base,
                    src_path=path,
                    preprocess=json.dumps(pmeta),
                    colmap_dir=colmap_dir,
                )
                stats["images_saved"] += 1

            batch_imgs = []
            batch_meta = []

        for image_id in usable:
            im = images[image_id]
            base = os.path.basename(im.name)
            path = by_base.get(base, None)
            if path is None:
                stats["images_skipped_missing_file"] += 1
                continue

            out_path = out_npz_path(depth_root, i0, i1, image_id, base)  # CHANGED
            if (not args.overwrite) and os.path.exists(out_path):
                stats["images_skipped_exists"] += 1
                continue

            cam = cameras[im.camera_id]

            try:
                with Image.open(path) as pil:
                    pil = pil.convert("RGB")
                    pil_cr, scale, left, top, orig_sz, resized_sz = preprocess_min_side_resize_center_crop(pil, args.out_size)
                img_t = TF.to_tensor(pil_cr).to(dtype=torch.float32)
            except Exception as e:
                stats["images_failed_load"] += 1
                log_line({"type": "image_error", "i0": i0, "i1": i1, "image_id": int(image_id),
                          "basename": base, "path": path, "stage": "load/preprocess", "error": repr(e)})
                continue

            try:
                fx, fy, cx, cy, cam_model = colmap_cam_to_fx_fy_cx_cy(cam)
                fx2, fy2, cx2, cy2 = adjust_intrinsics_resize_crop(fx, fy, cx, cy, scale, left, top)
                K = make_K_torch(fx2, fy2, cx2, cy2, device=device, dtype=torch.float32)
            except Exception as e:
                stats["images_failed_affine"] += 1
                log_line({"type": "image_error", "i0": i0, "i1": i1, "image_id": int(image_id),
                          "basename": base, "path": path, "stage": "intrinsics", "error": repr(e)})
                continue

            Rt = w2c_from_qt(np.array(im.qvec), np.array(im.tvec)).astype(np.float32)
            w2c = torch.from_numpy(Rt).to(device=device)

            pmeta = {
                "orig_sz": orig_sz,
                "resized_sz": resized_sz,
                "scale": scale,
                "left": left,
                "top": top,
                "cam_model": cam_model,
                "fxfy_cxcy_out": [float(fx2), float(fy2), float(cx2), float(cy2)],
                "out_size": args.out_size,
            }

            batch_imgs.append(img_t)
            batch_meta.append((image_id, base, path, K, w2c, pmeta))
            stats["images_total"] += 1

            if len(batch_imgs) >= args.batch_size:
                flush_batch()

        flush_batch()

        if not args.no_tqdm:
            pbar.set_postfix_str(
                f"saved={stats['images_saved']} exist={stats['images_skipped_exists']} "
                f"loadfail={stats['images_failed_load']} sparsefail={stats['images_failed_sparse_pts']}"
            )

    summary = {"type": "rank_summary", "rank": rank, "world": world, "stats": stats, "log_path": log_path}
    log_line(summary)

    if rank == 0:
        print("\n==== DONE (per-rank logs) ====")
        print("depth_root:", depth_root)
        print("log files like:", os.path.join(depth_root, "_precompute_log_rank*.jsonl"))
        print("Tip: combine summaries by grepping type=rank_summary")

    if dist_is_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()