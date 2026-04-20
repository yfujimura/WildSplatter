from typing import Optional, Dict, Any, List

import numpy as np
import torch

# ---- qvec -> R (COLMAP)
def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    COLMAP quaternion qvec = [qw, qx, qy, qz]
    returns R (3,3)
    """
    qw, qx, qy, qz = qvec.astype(np.float64)
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,         2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,         1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)
    return R

def w2c_3x4_from_qt(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    COLMAP image pose is world->cam: x_cam = R x_world + t
    """
    R = qvec_to_rotmat(qvec)
    t = tvec.astype(np.float64).reshape(3, 1)
    Rt = np.concatenate([R, t], axis=1)  # (3,4)
    return Rt

def collate_viewsets(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Make batch dict:
      image: (B,N,3,H,W)
      intrinsics: (B,N,3,3)
      extrinsics: (B,N,3,4)
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    intrs  = torch.stack([b["intrinsics"] for b in batch], dim=0)
    extrs  = torch.stack([b["extrinsics"] for b in batch], dim=0)
    depths  = torch.stack([b["depth"] for b in batch], dim=0)
    skys  = torch.stack([b["sky"] for b in batch], dim=0)
    masks  = torch.stack([b["mask"] for b in batch], dim=0)

    out = {
        "image": images,
        "intrinsics": intrs,
        "extrinsics": extrs,
        "depth": depths,
        "sky": skys,
        "mask": masks,
    }
    # optional
    if "meta" in batch[0]:
        out["meta"] = [b.get("meta", {}) for b in batch]
        out["set_dir"] = [b.get("set_dir", "") for b in batch]
    return out