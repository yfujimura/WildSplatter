#!/usr/bin/env python3
# ============================================================
# Viewset maker
# View selection strategy:
#  1) Select two context views as endpoints:
#      - Choose one seed view (same as before: maximum degree)
#      - Filter candidates using coverage (seed_mincov_batch)
#        with a minimum coverage threshold and top_m selection
#      - Select v2 as the view with the largest viewing angle
#        from the seed (same as the first step of the previous
#        angle-only FPS)
#  2) Select one target view lying between the two context views:
#      - Let v1 be the seed and v2 the farthest view in terms of
#        viewing angle
#      - Require:
#          d13 < d12 and d23 < d12
#          ang13 < ang12 and ang23 < ang12
#      - Among the valid views, choose v3 as the one with the
#        highest angle-only score
#      - Additionally, restrict v3 candidates to those satisfying
#          coverage(v1, v3) >= coverage(v1, v2)
#
# The output always consists of three views:
# v1 and v2 are context views, and v3 is the target view.
# ============================================================

import os, glob, json, argparse
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm

from read_write_model import read_model, qvec2rotmat

ImageFile.LOAD_TRUNCATED_IMAGES = False


# ============================================================
# Geometry helpers
# ============================================================
def cam_center_world(qvec, tvec):
    R = qvec2rotmat(qvec)
    t = np.array(tvec, dtype=np.float64).reshape(3)
    return -R.T @ t

def viewing_dir_world(qvec):
    R = qvec2rotmat(qvec)
    z = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)  # camera +Z in world
    return z / (np.linalg.norm(z) + 1e-12)

def angle_deg(u, v):
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# ============================================================
# Image preprocess: resize so min-side==out then center crop
# ============================================================
def resize_and_center_crop_with_params(img: Image.Image, out_size: int):
    w, h = img.size
    scale = out_size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_rs = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - out_size) // 2
    top  = (new_h - out_size) // 2
    cropped = img_rs.crop((left, top, left + out_size, top + out_size))
    return cropped, scale, left, top, new_w, new_h, w, h


# ============================================================
# COLMAP camera -> fx,fy,cx,cy
# ============================================================
def colmap_cam_to_intrinsics(cam):
    model = cam.model.upper()
    p = np.array(cam.params, dtype=np.float64)

    if model == "SIMPLE_PINHOLE":       # f, cx, cy
        f, cx, cy = p
        fx, fy = f, f
    elif model == "PINHOLE":            # fx, fy, cx, cy
        fx, fy, cx, cy = p
    elif model in ["SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"]:  # f, cx, cy, k
        f, cx, cy, _k = p
        fx, fy = f, f
    elif model in ["RADIAL", "RADIAL_FISHEYE"]:                # f, cx, cy, k1, k2
        f, cx, cy, _k1, _k2 = p
        fx, fy = f, f
    elif model in ["OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"]:
        fx, fy, cx, cy = p[:4]
    elif model == "FOV":
        f, cx, cy, _omega = p
        fx, fy = f, f
    else:
        raise ValueError(f"Unsupported camera model: {cam.model}")

    return float(fx), float(fy), float(cx), float(cy), cam.model

def adjust_intrinsics_for_resize_crop(fx, fy, cx, cy, scale, left, top):
    fx2 = fx * scale
    fy2 = fy * scale
    cx2 = cx * scale - left
    cy2 = cy * scale - top
    return fx2, fy2, cx2, cy2


# ============================================================
# Depth npz paths
# ============================================================
def depth_npz_preferred_path(depth_root: str, i0: int, i1: int, stem: str, image_id: int) -> str:
    fn = f"{stem}__i0{i0:03}_i1{i1:03}_id{int(image_id):08d}.npz"
    return os.path.join(depth_root, f"{i0:03}", f"{i1:03}", fn)

def resolve_depth_npz_path(depth_root: str, i0: int, i1: int, basename: str, image_id: int) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(basename))[0]
    pref = depth_npz_preferred_path(depth_root, i0, i1, stem, image_id)
    if os.path.exists(pref):
        return pref

    scene_dir = os.path.join(depth_root, f"{i0:03}", f"{i1:03}")
    if not os.path.isdir(scene_dir):
        return None

    pats = [
        os.path.join(scene_dir, f"*{stem}*id{int(image_id)}*.npz"),
        os.path.join(scene_dir, f"*id{int(image_id)}*{stem}*.npz"),
        os.path.join(scene_dir, f"*{stem}*{int(image_id)}*.npz"),
        os.path.join(scene_dir, f"*{int(image_id)}*{stem}*.npz"),
    ]
    for pat in pats:
        hits = glob.glob(pat)
        if len(hits) > 0:
            hits.sort()
            return hits[0]
    return None


# ============================================================
# Build usable ids
# ============================================================
def build_usable_views_with_depth(
    root_dir: str,
    depth_root: str,
    i0: int,
    i1: int,
    images_dict,
    eps_dup_baseline: float = 1e-3,
):
    scene_img_root = os.path.join(root_dir, f"images/{i0:03}/{i1:03}")
    all_imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        all_imgs += glob.glob(os.path.join(scene_img_root, "**", ext), recursive=True)

    by_base = {}
    for p in all_imgs:
        b = os.path.basename(p)
        if b not in by_base:
            by_base[b] = p

    cand_ids = []
    cand_paths = []
    cand_depths = []
    cand_centers = []
    cand_dirs = []

    missing_img = 0
    missing_depth = 0
    ransac_bad = 0

    for image_id, im in images_dict.items():
        base = os.path.basename(im.name)
        p = by_base.get(base, None)
        if p is None:
            missing_img += 1
            continue

        dpath = resolve_depth_npz_path(depth_root, i0, i1, base, int(image_id))
        if dpath is None or (not os.path.exists(dpath)):
            missing_depth += 1
            continue

        try:
            data = np.load(dpath, allow_pickle=True)
            if "ransac_ok" in data and (not bool(data["ransac_ok"])):
                ransac_bad += 1
                continue
        except Exception:
            missing_depth += 1
            continue

        cand_ids.append(image_id)
        cand_paths.append(p)
        cand_depths.append(dpath)
        cand_centers.append(cam_center_world(im.qvec, im.tvec))
        cand_dirs.append(viewing_dir_world(im.qvec))

    if len(cand_ids) == 0:
        return (
            [], {}, {}, {}, {},
            missing_img, missing_depth, ransac_bad, 0
        )

    C = np.asarray(cand_centers, dtype=np.float32)
    N = C.shape[0]

    sq = np.sum(C * C, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (C @ C.T)
    dist2 = np.maximum(dist2, 0.0)

    eps2 = float(eps_dup_baseline * eps_dup_baseline)

    keep = np.ones(N, dtype=bool)
    skipped_duplicate_pose = 0

    for i in range(N):
        if not keep[i]:
            continue
        dup = dist2[i, i+1:] < eps2
        if np.any(dup):
            skipped_duplicate_pose += int(np.sum(keep[i+1:][dup]))
            keep[i+1:][dup] = False

    usable_ids = []
    id_to_path = {}
    id_to_depth_path = {}
    centers = {}
    dirs = {}

    for i, ok in enumerate(keep):
        if not ok:
            continue
        image_id = cand_ids[i]
        usable_ids.append(image_id)
        id_to_path[image_id] = cand_paths[i]
        id_to_depth_path[image_id] = cand_depths[i]
        centers[image_id] = C[i]
        dirs[image_id] = cand_dirs[i]

    return (
        usable_ids,
        id_to_path,
        id_to_depth_path,
        dirs,
        centers,
        missing_img,
        missing_depth,
        ransac_bad,
        skipped_duplicate_pose,
    )


@torch.no_grad()
def backward_warp_valid_occl_mask_batch_cuda(
    tgt_depth_bhw: torch.Tensor,      # (B,H,W)
    src_depth_bhw: torch.Tensor,      # (B,H,W)
    src_K_b33: torch.Tensor,          # (B,3,3)
    src_w2c_b34: torch.Tensor,        # (B,3,4)
    tgt_K_b33: torch.Tensor,          # (B,3,3)
    tgt_w2c_b34: torch.Tensor,        # (B,3,4)
    tgt_sky_bhw: torch.Tensor,        # (B,H,W)  sky prob (large=empty)
    LOG_THR: float,
    SKY_THR: float,
):
    device = tgt_depth_bhw.device
    B, H, W = tgt_depth_bhw.shape

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).T  # (3, HW)

    Kt_inv = torch.inverse(tgt_K_b33)
    rays = torch.bmm(Kt_inv, pix.unsqueeze(0).expand(B, -1, -1)).transpose(1, 2)

    z = tgt_depth_bhw.view(B, -1, 1)
    valid_z = (z[..., 0] > 1e-6)

    Xc_t = rays * z

    R_t = tgt_w2c_b34[:, :, :3]
    t_t = tgt_w2c_b34[:, :, 3:4]
    Xw = torch.bmm(R_t.transpose(1, 2), (Xc_t.transpose(1, 2) - t_t)).transpose(1, 2)

    R_s = src_w2c_b34[:, :, :3]
    t_s = src_w2c_b34[:, :, 3:4]
    Xc_s = (torch.bmm(R_s, Xw.transpose(1, 2)) + t_s).transpose(1, 2)

    zs = Xc_s[..., 2]
    valid_front = (zs > 1e-6)

    uvw = torch.bmm(src_K_b33, Xc_s.transpose(1, 2)).transpose(1, 2)
    u = uvw[..., 0] / (uvw[..., 2] + 1e-12)
    v = uvw[..., 1] / (uvw[..., 2] + 1e-12)

    inb = (u >= 0.0) & (u <= (W - 1.0)) & (v >= 0.0) & (v <= (H - 1.0))

    gx = (u / max(1.0, (W - 1.0))) * 2.0 - 1.0
    gy = (v / max(1.0, (H - 1.0))) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(B, H, W, 2)

    src_depth_1 = src_depth_bhw.unsqueeze(1)
    zsrc = F.grid_sample(
        src_depth_1,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )[:, 0]

    zsrc_flat = zsrc.view(B, -1)
    zs_flat = zs

    # ---- depth consistency (single criterion) ----
    # Use log-depth difference as a single, scale-invariant criterion.
    # LOG_THR corresponds to multiplicative tolerance: exp(LOG_THR).
    z_eps = 1e-6
    zs_pos   = zs_flat > z_eps
    zsrc_pos = zsrc_flat > z_eps
    
    log_err = (torch.log(zs_flat.clamp_min(z_eps)) - torch.log(zsrc_flat.clamp_min(z_eps))).abs()
    
    ok_depth = zs_pos & zsrc_pos & (log_err <= LOG_THR)

    # ---- sky mask (tgt canvas) ----
    nosky_mask = (tgt_sky_bhw.view(B, -1) < SKY_THR)

    valid = valid_z & valid_front & inb & ok_depth & nosky_mask
    return valid.view(B, H, W)


@torch.no_grad()
def canvas_coverage(
    mask_bhw: torch.Tensor,
    nosky_mask_bhw: torch.Tensor,
    stride: int = 1
) -> torch.Tensor:

    if stride > 1:
        mask = mask_bhw[:, ::stride, ::stride]
        nosky = nosky_mask_bhw[:, ::stride, ::stride]
    else:
        mask = mask_bhw
        nosky = nosky_mask_bhw

    valid = mask & nosky

    num = valid.float().sum(dim=(1, 2))
    denom = nosky.float().sum(dim=(1, 2)).clamp_min(1.0)

    return num / denom


# ============================================================
# Per-scene processing
# ============================================================
def process_scene_worker(args_dict: dict, i0: int, i1: int):
    root_dir = args_dict["root_dir"]
    depth_root = args_dict["depth_root"]
    out_root = args_dict["out_root"]

    out_size_save = int(args_dict["out_size_save"])
    out_size_warp = int(args_dict["out_size_warp"])
    png_compress_level = int(args_dict["png_compress_level"])

    # Output is fixed to three views (2 context + 1 target).
    k_set = int(args_dict["k_set"])
    if k_set != 3:
        k_set = 3

    min_view_angle = float(args_dict["min_view_angle"])
    max_view_angle = float(args_dict["max_view_angle"])

    min_baseline_ratio = float(args_dict["min_baseline_ratio"])
    max_baseline_ratio = float(args_dict["max_baseline_ratio"])

    mincov_thr = float(args_dict["mincov"])
    cov_stride = int(args_dict["cov_stride"])
    max_candidates_per_seed = int(args_dict["max_candidates_per_seed"])
    max_sets_per_scene = int(args_dict["max_sets_per_scene"])

    MIN_UNION_COV = float(args_dict["min_union_cov"])

    top_m = int(args_dict["top_m"])
    pair_batch = int(args_dict["pair_batch"])

    eps_dup_baseline = float(args_dict["eps_dup_baseline"])

    LOG_THR = float(args_dict["log_depth_thr"])

    SKY_THR = float(args_dict["sky_thr"])
    MIN_NOSKY_RATIO = float(args_dict["min_nosky_ratio"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        return {"status": "skip:no_cuda", "i0": i0, "i1": i1, "saved_sets": 0, "set_dirs": [], "bad_images": []}

    # ------------------------------------------------------------
    # COLMAP
    # ------------------------------------------------------------
    cand_colmap_dirs = [
        os.path.join(root_dir, f"reconstruct/{i0:03}/{i1:03}/colmap/0"),
        os.path.join(root_dir, f"reconstruct/{i0:03}/{i1:03}/colmap/sparse/0"),
    ]
    colmap_dir = next((p for p in cand_colmap_dirs if os.path.isdir(p)), None)
    if colmap_dir is None:
        return {"status": "skip:no_colmap_dir", "i0": i0, "i1": i1, "saved_sets": 0, "set_dirs": [], "bad_images": []}

    try:
        cameras, images, _ = read_model(colmap_dir, ext=".bin")
    except Exception as e:
        return {"status": f"skip:read_model_failed:{type(e).__name__}", "i0": i0, "i1": i1, "saved_sets": 0, "set_dirs": [], "bad_images": []}

    (
        usable_ids, id_to_path, id_to_depth_path, _dirs, centers,
        _missing_img, _missing_depth, _ransac_bad, _skipped_dup
    ) = build_usable_views_with_depth(
        root_dir, depth_root, i0, i1, images,
        eps_dup_baseline=eps_dup_baseline,
    )

    if len(usable_ids) < 3:
        return {"status": f"skip:too_few_usable:{len(usable_ids)}", "i0": i0, "i1": i1, "saved_sets": 0, "set_dirs": [], "bad_images": []}

    # ------------------------------------------------------------
    # NumPy precomputation
    # ------------------------------------------------------------
    uid_list = usable_ids
    uid_to_idx = {u: i for i, u in enumerate(uid_list)}
    idx_to_uid = {i: u for u, i in uid_to_idx.items()}
    N = len(uid_list)

    C = np.stack([centers[u] for u in uid_list], axis=0).astype(np.float32)

    # dist
    sq = np.sum(C * C, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (C @ C.T)
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2)  # (N,N)

    scale = float(np.median(dist[dist > 0]))
    if not np.isfinite(scale) or scale <= 0:
        return {"status": "skip:bad_scale", "i0": i0, "i1": i1, "saved_sets": 0, "set_dirs": [], "bad_images": []}

    min_base = min_baseline_ratio * scale
    max_base = max_baseline_ratio * scale

    # ang (viewing direction)
    D = np.stack([_dirs[u] for u in uid_list], axis=0).astype(np.float32)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    cosang = np.clip(Dn @ Dn.T, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))  # (N,N)

    # Common seed/candidate constraints used to filter coverage candidates.
    adj_full = (
        (ang >= min_view_angle) &
        (ang <= max_view_angle) &
        (dist >= min_base) &
        (dist <= max_base)
    )
    np.fill_diagonal(adj_full, False)

    remaining_mask = np.ones(N, dtype=bool)
    degree = adj_full.sum(axis=1).astype(np.int32)

    def prefilter_candidates_np(seed_idx: int, remaining_mask_: np.ndarray):
        m = remaining_mask_ & adj_full[seed_idx]
        m[seed_idx] = False
        idx = np.where(m)[0]
        idx = idx[np.argsort(dist[seed_idx, idx])]
        return idx[:max_candidates_per_seed]

    # ------------------------------------------------------------
    # depth cache
    # ------------------------------------------------------------
    depth_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def _scale_K(K_33: torch.Tensor, scale_xy: float) -> torch.Tensor:
        K2 = K_33.clone()
        K2[0, 0] *= scale_xy
        K2[1, 1] *= scale_xy
        K2[0, 2] *= scale_xy
        K2[1, 2] *= scale_xy
        return K2

    def load_depth_pack(iid: int):
        if iid in depth_cache:
            return depth_cache[iid], None
    
        dpath = id_to_depth_path.get(iid, None)
        if dpath is None:
            return None, {"stage": "resolve_depth_path", "iid": int(iid)}
    
        try:
            data = np.load(dpath, allow_pickle=True)
            depth_504 = torch.from_numpy(data["depth"]).to(device, torch.float32)
            sky_504 = torch.from_numpy(data["sky"]).to(device, torch.float32)
            K_504 = torch.from_numpy(data["K"]).to(device, torch.float32)
            w2c_t = torch.from_numpy(data["w2c"]).to(device, torch.float32)
        except Exception as e:
            return None, {"stage": "load_depth_npz", "iid": int(iid), "error": repr(e), "dpath": str(dpath)}
    
        depth_252 = F.interpolate(
            depth_504[None, None],
            size=(out_size_warp, out_size_warp),
            mode="bilinear",
            align_corners=False
        )[0, 0]
    
        sky_252 = F.interpolate(
            sky_504[None, None],
            size=(out_size_warp, out_size_warp),
            mode="nearest"
        )[0, 0]
    
        s = float(out_size_warp) / float(out_size_save)
        K_252 = _scale_K(K_504, s)

        nosky_ratio = float((sky_252 < SKY_THR).float().mean())
    
        pack = {
            "depth252": depth_252,
            "depth504": depth_504,
            "sky252": sky_252,
            "sky504": sky_504,
            "K252": K_252,
            "K504": K_504,
            "w2c": w2c_t,
            "dpath": dpath,
            "nosky_ratio": nosky_ratio,
        }
    
        depth_cache[iid] = pack
        return pack, None

    # ------------------------------------------------------------
    # seed-only mincov: returns list[(cov, uid)]
    # ------------------------------------------------------------
    @torch.no_grad()
    def seed_mincov_batch(seed_iid: int, cand_list: List[int]) -> List[Tuple[float, int]]:
        ps, _ = load_depth_pack(seed_iid)
        if ps is None:
            return []

        seed_depth = ps["depth252"]
        seed_sky   = ps["sky252"]
        seed_K = ps["K252"]
        seed_w2c = ps["w2c"]

        out: List[Tuple[float, int]] = []
        for st in range(0, len(cand_list), pair_batch):
            chunk = cand_list[st: st + pair_batch]
            packs, keep_ids = [], []
            for j in chunk:
                pj, _ = load_depth_pack(j)
                if pj is None:
                    continue
                packs.append(pj)
                keep_ids.append(j)
            if not keep_ids:
                continue

            B = len(keep_ids)
            cand_depth = torch.stack([p["depth252"] for p in packs])
            cand_sky   = torch.stack([p["sky252"] for p in packs])
            cand_K = torch.stack([p["K252"] for p in packs])
            cand_w2c = torch.stack([p["w2c"] for p in packs])

            # seed -> cand (tgt=cand, src=seed)
            src_depth_rep = seed_depth.unsqueeze(0).expand(B, -1, -1)
            src_K_rep = seed_K.unsqueeze(0).expand(B, -1, -1)
            src_w2c_rep = seed_w2c.unsqueeze(0).expand(B, -1, -1)

            m_s2c = backward_warp_valid_occl_mask_batch_cuda(
                cand_depth, src_depth_rep, src_K_rep, src_w2c_rep,
                cand_K, cand_w2c,
                cand_sky,
                LOG_THR, SKY_THR,
            )
            nosky_cand = cand_sky < SKY_THR
            cov_s2c = canvas_coverage(m_s2c, nosky_cand, stride=cov_stride)

            # cand -> seed (tgt=seed, src=cand)
            tgt_depth_rep = seed_depth.unsqueeze(0).expand(B, -1, -1)
            tgt_K_rep = seed_K.unsqueeze(0).expand(B, -1, -1)
            tgt_w2c_rep = seed_w2c.unsqueeze(0).expand(B, -1, -1)

            m_c2s = backward_warp_valid_occl_mask_batch_cuda(
                tgt_depth_rep, cand_depth, cand_K, cand_w2c,
                tgt_K_rep, tgt_w2c_rep,
                seed_sky.unsqueeze(0).expand(B, -1, -1),
                LOG_THR, SKY_THR,
            )
            nosky_seed = seed_sky.unsqueeze(0).expand(B, -1, -1) < SKY_THR
            cov_c2s = canvas_coverage(m_c2s, nosky_seed, stride=cov_stride)

            cov_min = torch.minimum(cov_s2c, cov_c2s)

            cov_min_np = cov_min.detach().cpu().numpy()
            for ii, j in enumerate(keep_ids):
                out.append((float(cov_min_np[ii]), int(j)))
        return out

    # ------------------------------------------------------------
    # selection helpers
    # ------------------------------------------------------------
    def pick_v2_by_angle_only(seed_idx: int, pool_uids: List[int]) -> Optional[int]:
        """
        v2 is the candidate with the largest viewing angle from the seed.
        """
        if not pool_uids:
            return None
        best_uid = None
        best_ang = -1.0
        for uid in pool_uids:
            pj, _ = load_depth_pack(uid)
            if pj is None or pj["nosky_ratio"] < MIN_NOSKY_RATIO:
                continue
            
            j = uid_to_idx[uid]
            a = float(ang[seed_idx, j])
            if a > best_ang:
                best_ang = a
                best_uid = uid
        return best_uid

    def pick_v3_by_visible_union(
        v1_uid: int,
        v2_uid: int,
        v1_idx: int,
        v2_idx: int,
        pool_uids: List[int],
        forbid: set,
        min_union_cov: float,   # ★ 追加
    ) -> Optional[int]:
        """
        Select a v3 candidate from pool_uids subject to interpolation constraints.
        
        For each candidate j:
          - Consider only candidates satisfying
              (d13 < d12 and d23 < d12) and
              (a13 < a12 and a23 < a12)
          - Compute valid(v3 -> v1) and valid(v3 -> v2)
          - Compute their union visibility mask
          - Discard candidates whose union coverage is below min_union_cov
          - Score the remaining candidates using
              union_cov * balance
        
        Select the candidate with the highest score.
        """
    
        p1, _ = load_depth_pack(v1_uid)
        p2, _ = load_depth_pack(v2_uid)
        if p1 is None or p2 is None:
            return None
    
        # interpolation thresholds (in idx space)
        d12 = float(dist[v1_idx, v2_idx])
        a12 = float(ang[v1_idx, v2_idx])
        if d12 <= 1e-12 or a12 <= 1e-12:
            return None
    
        best_uid = None
        best_score = -1.0
    
        for uid in pool_uids:
            if uid in forbid:
                continue
    
            j_idx = uid_to_idx[uid]
            if not remaining_mask[j_idx]:
                continue
    
            # ---- interpolation constraints ----
            d13 = float(dist[v1_idx, j_idx])
            d23 = float(dist[v2_idx, j_idx])
            a13 = float(ang[v1_idx, j_idx])
            a23 = float(ang[v2_idx, j_idx])
    
            if not (d13 < d12 and d23 < d12 and a13 < a12 and a23 < a12):
                continue
    
            pj, _ = load_depth_pack(uid)
            if pj is None:
                continue

            if pj["nosky_ratio"] < MIN_NOSKY_RATIO:
                continue
    
            depth_j = pj["depth252"].unsqueeze(0)
            sky_j   = pj["sky252"].unsqueeze(0)
            
            m_j2v1 = backward_warp_valid_occl_mask_batch_cuda(
                tgt_depth_bhw=depth_j,
                src_depth_bhw=p1["depth252"].unsqueeze(0),
                src_K_b33=p1["K252"].unsqueeze(0),
                src_w2c_b34=p1["w2c"].unsqueeze(0),
                tgt_K_b33=pj["K252"].unsqueeze(0),
                tgt_w2c_b34=pj["w2c"].unsqueeze(0),
                tgt_sky_bhw=sky_j,
                LOG_THR=LOG_THR,
                SKY_THR=SKY_THR,
            )
            
            m_j2v2 = backward_warp_valid_occl_mask_batch_cuda(
                tgt_depth_bhw=depth_j,
                src_depth_bhw=p2["depth252"].unsqueeze(0),
                src_K_b33=p2["K252"].unsqueeze(0),
                src_w2c_b34=p2["w2c"].unsqueeze(0),
                tgt_K_b33=pj["K252"].unsqueeze(0),
                tgt_w2c_b34=pj["w2c"].unsqueeze(0),
                tgt_sky_bhw=sky_j,
                LOG_THR=LOG_THR,
                SKY_THR=SKY_THR,
            )
    
            m_or = m_j2v1 | m_j2v2
            nosky_j = sky_j < SKY_THR
            #score = float(canvas_coverage(m_or, nosky_j, stride=cov_stride)[0])
            union_cov = float(canvas_coverage(m_or, nosky_j, stride=cov_stride)[0])
            # Apply the minimum union coverage threshold.
            if union_cov < min_union_cov:
                continue

            b13 = d13
            b23 = d23
            balance = 1.0 - abs(b13 - b23) / (d12 + 1e-6)
            
            score = union_cov * balance
    
            if score > best_score:
                best_score = score
                best_uid = uid
    
        return best_uid

    # ------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------
    saved = 0
    set_dirs = []
    bad_images_all = []

    while remaining_mask.sum() >= 3 and saved < max_sets_per_scene:
        tried_seed = np.zeros(N, dtype=bool)
        made_one = False

        while True:
            masked_degree = np.where(remaining_mask & (~tried_seed), degree, -1)
            seed_idx = int(np.argmax(masked_degree))
            best_deg = int(masked_degree[seed_idx])

            if best_deg < 1:
                break

            v1_uid = idx_to_uid[seed_idx]

            p_seed, _ = load_depth_pack(v1_uid)
            if p_seed is None or p_seed["nosky_ratio"] < MIN_NOSKY_RATIO:
                tried_seed[seed_idx] = True
                continue

            # ---- Build v2 candidates using coverage ----
            cand_idx = prefilter_candidates_np(seed_idx, remaining_mask)
            if len(cand_idx) < 1:
                tried_seed[seed_idx] = True
                continue

            cand_uids = [idx_to_uid[i] for i in cand_idx]

            scored = seed_mincov_batch(v1_uid, cand_uids)
            scored = [(m, j) for (m, j) in scored if m >= mincov_thr]
            if len(scored) < 1:
                tried_seed[seed_idx] = True
                continue

            scored.sort(key=lambda x: x[0], reverse=True)

            # Store the coverage between v1 and each candidate.
            # This value is later used as the reference coverage for v2.
            cov_dict = {j: m for (m, j) in scored}

            pool_uids = [j for (_m, j) in scored[: min(top_m, len(scored))]]

            # ---- Select v2 using only the viewing angle
            v2_uid = pick_v2_by_angle_only(seed_idx, pool_uids)
            if v2_uid is None:
                tried_seed[seed_idx] = True
                continue
            v2_idx = uid_to_idx[v2_uid]

            # Reference coverage between v1 and v2.
            v1v2_cov = float(cov_dict.get(v2_uid, 0.0))

            # ---- Select v3 using interpolation constraints,
            #      angle-only scoring, and
            #      coverage(v1, v3) >= coverage(v1, v2) ----
            forbid = {v1_uid, v2_uid}
            v3_uid = pick_v3_by_visible_union(
                v1_uid=v1_uid,
                v2_uid=v2_uid,
                v1_idx=seed_idx,
                v2_idx=v2_idx,
                pool_uids=pool_uids,
                forbid=forbid,
                min_union_cov=MIN_UNION_COV,
            )
            if v3_uid is None:
                tried_seed[seed_idx] = True
                continue
            v3_idx = uid_to_idx[v3_uid]

            chosen = [v1_uid, v2_uid, v3_uid]

            # ------------------------------------------------------------
            # compute v3 visible mask at 504 resolution
            # ------------------------------------------------------------
            p1, _ = load_depth_pack(v1_uid)
            p2, _ = load_depth_pack(v2_uid)
            p3, _ = load_depth_pack(v3_uid)
            
            v3_mask_504 = None
            
            if p1 is not None and p2 is not None and p3 is not None:
            
                depth3 = p3["depth504"].unsqueeze(0)
                sky3   = p3["sky504"].unsqueeze(0)
            
                # v3 -> v1
                m_31 = backward_warp_valid_occl_mask_batch_cuda(
                    tgt_depth_bhw=depth3,
                    src_depth_bhw=p1["depth504"].unsqueeze(0),
                    src_K_b33=p1["K504"].unsqueeze(0),
                    src_w2c_b34=p1["w2c"].unsqueeze(0),
                    tgt_K_b33=p3["K504"].unsqueeze(0),
                    tgt_w2c_b34=p3["w2c"].unsqueeze(0),
                    tgt_sky_bhw=sky3,
                    LOG_THR=LOG_THR,
                    SKY_THR=SKY_THR,
                )
            
                # v3 -> v2
                m_32 = backward_warp_valid_occl_mask_batch_cuda(
                    tgt_depth_bhw=depth3,
                    src_depth_bhw=p2["depth504"].unsqueeze(0),
                    src_K_b33=p2["K504"].unsqueeze(0),
                    src_w2c_b34=p2["w2c"].unsqueeze(0),
                    tgt_K_b33=p3["K504"].unsqueeze(0),
                    tgt_w2c_b34=p3["w2c"].unsqueeze(0),
                    tgt_sky_bhw=sky3,
                    LOG_THR=LOG_THR,
                    SKY_THR=SKY_THR,
                )
            
                nosky = sky3 < SKY_THR
                v3_mask_504 = (m_31 | m_32) & nosky

            # ---- SAVE SET ----
            set_name = f"{i0:03}{i1:03}{saved:02d}"
            set_dir = os.path.join(out_root, set_name)
            os.makedirs(set_dir, exist_ok=True)

            meta = {
                "set_name": set_name,
                "i0": i0,
                "i1": i1,
                "set_idx": saved,
                "views": [],
                "roles": {
                    "context": [0, 1],
                    "target": [2],
                },
                "picked": {
                    "v1_uid": int(v1_uid),
                    "v2_uid": int(v2_uid),
                    "v3_uid": int(v3_uid),
                    #"cov_v1v2": float(v1v2_cov),
                    #"cov_v1v3": float(cov_dict.get(v3_uid, 0.0)),
                    "d12": float(dist[seed_idx, v2_idx]),
                    "a12": float(ang[seed_idx, v2_idx]),
                    "d13": float(dist[seed_idx, v3_idx]),
                    "d23": float(dist[v2_idx, v3_idx]),
                    "a13": float(ang[seed_idx, v3_idx]),
                    "a23": float(ang[v2_idx, v3_idx]),
                }
            }

            for vi, iid in enumerate(chosen):
                im = images[iid]
                cam = cameras[im.camera_id]
                img_path = id_to_path[iid]

                with Image.open(img_path) as im_pil:
                    img = im_pil.convert("RGB")

                img_proc, sc, left, top, new_w, new_h, orig_w, orig_h = resize_and_center_crop_with_params(img, out_size_save)
                fx, fy, cx, cy, cam_model = colmap_cam_to_intrinsics(cam)
                fx2, fy2, _, _ = adjust_intrinsics_for_resize_crop(fx, fy, cx, cy, sc, left, top)

                png_name = f"view_{vi}.png"
                npz_name = f"view_{vi}.npz"

                img_proc.save(
                    os.path.join(set_dir, png_name),
                    format="PNG",
                    compress_level=png_compress_level,
                    optimize=True
                )

                np.savez(
                    os.path.join(set_dir, npz_name),
                    image_id=np.int32(iid),
                    fx=np.float32(fx2),
                    fy=np.float32(fy2),
                    qvec=np.asarray(im.qvec, dtype=np.float32),
                    tvec=np.asarray(im.tvec, dtype=np.float32),
                )

                role = "context" if vi in (0, 1) else "target"

                # ---- save original 504 depth ----
                p_depth, _ = load_depth_pack(iid)
                depth_name = f"view_{vi}_depth.npy"
                sky_name = f"view_{vi}_sky.npy"
                
                if p_depth is not None:
                    depth504 = p_depth["depth504"].detach().cpu().numpy()
                    np.save(
                        os.path.join(set_dir, depth_name),
                        depth504.astype(np.float32)
                    )
                    sky = p_depth["sky504"].detach().cpu().numpy()
                    np.save(
                        os.path.join(set_dir, sky_name),
                        sky.astype(np.float32)
                    )
                else:
                    depth_name = ""

                meta["views"].append({
                    "vi": vi,
                    "role": role,
                    "colmap_image_id": int(iid),
                    "colmap_image_name": str(im.name),
                    "camera_id": int(im.camera_id),
                    "src_image_path": img_path,
                    "depth_npz": id_to_depth_path.get(iid, ""),
                    "png": png_name,
                    "npz": npz_name,
                    "orig_size": [int(orig_w), int(orig_h)],
                    "resize_scale": float(sc),
                    "resized_size": [int(new_w), int(new_h)],
                    "crop_left_top": [int(left), int(top)],
                    "fx_fy_out": [float(fx2), float(fy2)],
                    "camera_model": str(cam_model),
                })

            # save v3 visible mask (504 resolution)
            if v3_mask_504 is not None:
                mask_np = v3_mask_504[0].detach().cpu().numpy().astype(np.uint8)
                np.save(
                    os.path.join(set_dir, "view_2_mask.npy"),
                    mask_np
                )

            with open(os.path.join(set_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # ---- update remaining / degree ----
            # NOTE: remove ONLY seed (v1) from remaining to avoid view exhaustion
            seed_rem_idx = int(uid_to_idx[v1_uid])
            remaining_mask[seed_rem_idx] = False
            
            # Update degrees as if we removed only the seed node from the graph
            deg_dec = (adj_full[seed_rem_idx] | adj_full[:, seed_rem_idx]).astype(np.int32)
            degree -= deg_dec
            degree[degree < 0] = 0
            degree[~remaining_mask] = -1

            saved += 1
            set_dirs.append(set_dir)
            made_one = True
            break

        if not made_one:
            break

    status = "ok" if saved > 0 else "skip:no_complete_sets"
    return {
        "status": status,
        "i0": i0,
        "i1": i1,
        "saved_sets": saved,
        "set_dirs": set_dirs,
        "bad_images": bad_images_all,
    }


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, default="/path/to/MegaScenes")
    ap.add_argument("--depth_root", type=str, default="", help="default: <root_dir>/depths")
    ap.add_argument("--out_root", type=str, default="/path/to/viewsets")

    ap.add_argument("--out_size_save", type=int, default=504)
    ap.add_argument("--out_size_warp", type=int, default=252)
    ap.add_argument("--png_compress_level", type=int, default=0)

    # Kept for backward compatibility (this script always outputs exactly three views).
    ap.add_argument("--k_set", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--min_view_angle", type=float, default=10.0) # 10.
    ap.add_argument("--max_view_angle", type=float, default=80.0)

    ap.add_argument("--min_baseline_ratio", type=float, default=0.05) # 0.05
    ap.add_argument("--max_baseline_ratio", type=float, default=0.80)

    ap.add_argument("--mincov", type=float, default=0.5)
    ap.add_argument("--cov_stride", type=int, default=1)

    ap.add_argument("--min_union_cov", type=float, default=0.9)

    ap.add_argument("--top_m", type=int, default=64)
    ap.add_argument("--pair_batch", type=int, default=64)
    ap.add_argument("--max_candidates_per_seed", type=int, default=256)
    ap.add_argument("--max_sets_per_scene", type=int, default=100)

    ap.add_argument("--eps_dup_baseline", type=float, default=1e-3)

    ap.add_argument("--log_depth_thr", type=float, default=0.05)

    ap.add_argument("--sky_thr", type=float, default=0.3)
    ap.add_argument("--min_nosky_ratio", type=float, default=0.7)

    ap.add_argument("--i0_start", type=int, required=True)
    ap.add_argument("--i0_end", type=int, required=True)
    ap.add_argument("--i1_start", type=int, default=0)
    ap.add_argument("--i1_end", type=int, default=999)

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--bad_images_log", type=str, default="")
    return ap.parse_args()

def enumerate_scenes(root_dir: str, i0_start: int, i0_end: int, i1_start: int, i1_end: int):
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

def append_bad_images_jsonl(path: str, bad_list: List[dict]):
    if (path is None) or (path == "") or (len(bad_list) == 0):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a") as f:
        for x in bad_list:
            f.write(json.dumps(x) + "\n")

def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    depth_root = args.depth_root if args.depth_root else os.path.join(args.root_dir, "depths")

    scenes = enumerate_scenes(args.root_dir, args.i0_start, args.i0_end, args.i1_start, args.i1_end)
    if len(scenes) == 0:
        print("[WARN] no scenes found in the given range (reconstruct/* exists check).")
        return

    args_dict = dict(
        root_dir=args.root_dir,
        depth_root=depth_root,
        out_root=args.out_root,

        out_size_save=args.out_size_save,
        out_size_warp=args.out_size_warp,
        png_compress_level=args.png_compress_level,

        k_set=args.k_set,
        seed=args.seed,

        min_view_angle=args.min_view_angle,
        max_view_angle=args.max_view_angle,

        min_baseline_ratio=args.min_baseline_ratio,
        max_baseline_ratio=args.max_baseline_ratio,

        mincov=args.mincov,
        cov_stride=args.cov_stride,

        min_union_cov=args.min_union_cov,

        top_m=args.top_m,
        pair_batch=args.pair_batch,
        max_candidates_per_seed=args.max_candidates_per_seed,
        max_sets_per_scene=args.max_sets_per_scene,

        eps_dup_baseline=args.eps_dup_baseline,

        log_depth_thr=args.log_depth_thr,

        sky_thr=args.sky_thr,
        min_nosky_ratio=args.min_nosky_ratio,
    )

    if args.dry_run:
        print(f"[DRY RUN] scenes: {len(scenes)}")
        print("first 10 scenes:", scenes[:10])
        print("depth_root:", depth_root)
        return

    total_saved_sets = 0
    stats_ok = 0
    skips = {}

    bad_log_path = args.bad_images_log or os.path.join(args.out_root, "_bad_images.jsonl")

    pbar = tqdm(scenes, total=len(scenes), desc="Scenes (single-GPU)", ncols=120)
    for (i0, i1) in pbar:
        res = process_scene_worker(args_dict, i0, i1)

        status = res["status"]
        saved = int(res.get("saved_sets", 0))
        total_saved_sets += saved

        append_bad_images_jsonl(bad_log_path, res.get("bad_images", []))

        if status == "ok":
            stats_ok += 1
        else:
            skips[status] = skips.get(status, 0) + 1

        pbar.set_postfix_str(f"saved_sets={total_saved_sets} ok={stats_ok} last={status}")

    print("\n==== DONE ====")
    print("out_root:", args.out_root)
    print("depth_root:", depth_root)
    print("total saved sets:", total_saved_sets)
    print("scenes ok:", stats_ok)
    print("bad images log:", bad_log_path)
    if skips:
        print("skips:")
        for k, v in sorted(skips.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()