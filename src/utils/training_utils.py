import torch
import torch.nn.functional as F

def linear_transform(pts1, pts2, W):
    # pts1: B x N x 3
    # pts2: B x N x 3
    # W: B x N
    # scale_bias: B x 4
    # pts2 = scale_bias[0] * pts1 + scale_bias[1:4]

    B, N, _ = pts1.shape

    pts1_ = pts1[:,:,:,None] # B x N x 3 x 1
    pts2_ = pts2[:,:,:,None] # B x N x 3 x 1

    ones = torch.ones_like(pts1_[:,:,0:1,:]) # B x N x 1 x 1
    zeros = torch.zeros_like(pts1_[:,:,0:1,:]) # B x N x 1 x 1

    pts_x = torch.cat([pts1_[:,:,0:1,:], ones, zeros, zeros], 3) # B x N x 1 x 4
    pts_y = torch.cat([pts1_[:,:,1:2,:], zeros, ones, zeros], 3) # B x N x 1 x 4
    pts_z = torch.cat([pts1_[:,:,2:3,:], zeros, zeros, ones], 3) # B x N x 1 x 4

    X = torch.cat([pts_x, pts_y, pts_z], 2) # B x N x 3 x 4
    XT = X.transpose(2,3) # B x N x 4 x 3

    A = (W[:,:,None,None] * (XT @ X)).sum(dim=1)
    eps = 1e-6
    I = torch.eye(4, device=A.device).unsqueeze(0)  # 1 x 4 x 4
    A = A + eps * I
    B = (W[:,:,None,None] * (XT @ pts2_)).sum(dim=1).squeeze(-1) # B x 4

    scale_bias = torch.linalg.solve(A,B) # B x 4

    return scale_bias

def backward_warp_depth(
    tgt_depth_bhw: torch.Tensor,      # (B,H,W)
    src_depth_bhw: torch.Tensor,      # (B,H,W)
    src_K_b33: torch.Tensor,          # (B,3,3)
    src_w2c_b34: torch.Tensor,        # (B,3,4)
    tgt_K_b33: torch.Tensor,          # (B,3,3)
    tgt_w2c_b34: torch.Tensor,        # (B,3,4)
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

    return zs_flat.reshape(B,H,W), zsrc_flat.reshape(B,H,W)

def depth2points(depths, intrinsics, w2c):
    """
    depths:     (B, H, W)
    intrinsics: (B, 3, 3)
    w2c:        (B, 3, 4)

    returns:
        points_world: (B, H, W, 3)
    """

    B, H, W = depths.shape
    device = depths.device

    # ---- 1. pixel grid 作成 ----
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )  # (H, W)

    xs = xs.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    ys = ys.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

    # ---- 2. intrinsics 取り出し ----
    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)

    # ---- 3. camera座標系へ変換 ----
    z = depths
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z

    points_cam = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)

    # ---- 4. world座標系へ変換 ----
    R = w2c[:, :, :3]   # (B, 3, 3)
    t = w2c[:, :, 3:]   # (B, 3, 1)

    # world = R^T (cam - t)
    R_inv = R.transpose(1, 2)
    t = t.view(B, 1, 1, 3)

    points_cam = points_cam - t
    points_world = torch.einsum("bij,bhwj->bhwi", R_inv, points_cam)

    return points_world

def compute_weight_two_views(
    depths: torch.Tensor,      # (B,2,H,W)
    intrinsics: torch.Tensor,          # (B,2,3,3)
    extrinsics: torch.Tensor,        # (B,2,3,4)
    skys : torch.Tensor, # (B, 2, H,W)
    nosky_thr : float = 0.3,
    gamma : float = 10.,
):
    weight_0 = _compute_weight(
        depths[:,0],
        depths[:,1],
        intrinsics[:,1],
        extrinsics[:,1,:3],
        intrinsics[:,0],
        extrinsics[:,0,:3],
        skys[:,0],
        nosky_thr,
        gamma,
    )

    weight_1 = _compute_weight(
        depths[:,1],
        depths[:,0],
        intrinsics[:,0],
        extrinsics[:,0,:3],
        intrinsics[:,1],
        extrinsics[:,1,:3],
        skys[:,1],
        nosky_thr,
        gamma,
    )

    return torch.stack([weight_0, weight_1], dim=1)

def _compute_weight(
    tgt_depth_bhw: torch.Tensor,      # (B,H,W)
    src_depth_bhw: torch.Tensor,      # (B,H,W)
    src_K_b33: torch.Tensor,          # (B,3,3)
    src_w2c_b34: torch.Tensor,        # (B,3,4)
    tgt_K_b33: torch.Tensor,          # (B,3,3)
    tgt_w2c_b34: torch.Tensor,        # (B,3,4)
    tgt_skys : torch.Tensor, # (B,H,W))
    nosky_thr : float = 0.3,
    gamma : float = 10.,
):
    d1, d2 = backward_warp_depth(
        tgt_depth_bhw,
        src_depth_bhw,
        src_K_b33,
        src_w2c_b34,
        tgt_K_b33,
        tgt_w2c_b34,
    )
    
    nosky_mask = tgt_skys < 0.3

    log_err = (torch.log(d1.clamp_min(1e-6)) - torch.log(d2.clamp_min(1e-6))).abs()
    weight = torch.exp(-log_err*gamma) * nosky_mask

    return weight