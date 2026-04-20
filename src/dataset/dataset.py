import os
import glob
import json
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from src.dataset.utils import w2c_3x4_from_qt


class MegaScenesDataset(Dataset):
    """
    Reads directories like:
      viewsets/00012300/
        view_0.png
        view_0.npz (fx,fy,qvec,tvec,...)
        ...
        meta.json

    New:
      - image_size: resize images to (H,W) (or int for square). None -> keep original.
      - train/val split is done by SCENE, not by set_dir.
        Scene key is derived from meta.json (i0,i1) if present, otherwise from set_name prefix.
    """
    def __init__(
        self,
        root: str,
        mode: str = "train",                 # "train" or "val"
        val_ratio: float = 0.1,
        split_seed: int = 0,
        n_views: Optional[int] = None,
        image_ext: str = "png",
        shuffle_views: bool = True,
        return_meta: bool = False,
        dtype_images: torch.dtype = torch.float32,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,   # e.g. 504 or (252,252); None -> no resize
        image_interp: str = "bilinear",                              # "nearest"|"bilinear"|"bicubic"
        antialias: bool = True,                                      # torchvision resize antialias
    ):
        assert mode in ("train", "val")
        assert 0.0 <= val_ratio <= 1.0

        self.root = root
        self.mode = mode
        self.val_ratio = val_ratio
        self.split_seed = split_seed

        self.n_ctx = 2
        self.n_views = n_views
        self.image_ext = image_ext
        self.shuffle_views = shuffle_views
        self.return_meta = return_meta
        self.dtype_images = dtype_images

        # ---- image resizing ----
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size  # None or (H,W)

        interp_map = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
        }
        if image_interp not in interp_map:
            raise ValueError(f"image_interp must be one of {list(interp_map.keys())}, got {image_interp}")
        self.image_interp = interp_map[image_interp]
        self.antialias = bool(antialias)

        # ---- collect all set dirs ----
        all_dirs: List[str] = []
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if not os.path.isdir(p):
                continue
            if os.path.exists(os.path.join(p, "view_0.npz")):
                all_dirs.append(p)

        if len(all_dirs) == 0:
            raise RuntimeError(f"No viewset directories found under: {root}")

        # ---- group by scene (so split is scene-wise) ----
        scene_to_dirs: Dict[str, List[str]] = {}
        for set_dir in all_dirs:
            scene_key = self._infer_scene_key(set_dir)
            scene_to_dirs.setdefault(scene_key, []).append(set_dir)

        scene_keys = sorted(scene_to_dirs.keys())

        # ---- deterministic scene split ----
        rng = np.random.RandomState(split_seed)
        scene_indices = np.arange(len(scene_keys))
        rng.shuffle(scene_indices)

        n_val_scenes = int(round(len(scene_keys) * val_ratio))
        val_scene_idx = set(scene_indices[:n_val_scenes].tolist())
        train_scene_idx = set(scene_indices[n_val_scenes:].tolist())

        if mode == "train":
            chosen_scene_idx = train_scene_idx
        else:
            chosen_scene_idx = val_scene_idx

        chosen_dirs: List[str] = []
        for si, sk in enumerate(scene_keys):
            if si in chosen_scene_idx:
                chosen_dirs.extend(scene_to_dirs[sk])

        self.set_dirs = sorted(chosen_dirs)

        print(
            f"[MegaScenesDataset] mode={mode} "
            f"sets={len(self.set_dirs)} (total_sets={len(all_dirs)}) "
            f"scenes={len(chosen_scene_idx)} (total_scenes={len(scene_keys)}) "
            f"val_ratio={val_ratio} split_seed={split_seed} "
            f"image_size={self.image_size}"
        )

    def __len__(self):
        return len(self.set_dirs)

    def _infer_scene_key(self, set_dir: str) -> str:
        """
        Prefer meta.json's i0,i1. Fallback to set_name prefix like '00012300' -> i0=000,i1=123.
        Return: f"{i0:03}{i1:03}" (6 chars).
        """
        meta_path = os.path.join(set_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, "r"))
                if ("i0" in meta) and ("i1" in meta):
                    i0 = int(meta["i0"])
                    i1 = int(meta["i1"])
                    return f"{i0:03}{i1:03}"
            except Exception:
                pass

        # fallback: directory name (set_name) like "00012300" => i0=000 i1=123
        base = os.path.basename(set_dir.rstrip("/"))
        digits = "".join([c for c in base if c.isdigit()])
        if len(digits) >= 6:
            return digits[:6]
        # last resort: whole dirname
        return base

    def _load_one_view(self, set_dir: str, vi: int) -> Dict[str, Any]:
        img_path = os.path.join(set_dir, f"view_{vi}.{self.image_ext}")
        depth_path = os.path.join(set_dir, f"view_{vi}_depth.npy")
        sky_path = os.path.join(set_dir, f"view_{vi}_sky.npy")
        npz_path = os.path.join(set_dir, f"view_{vi}.npz")
    
        # ---- load image ONCE ----
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w0, h0 = im.size                    # original size
            img_t = TF.to_tensor(im).to(self.dtype_images)  # (3,H0,W0)

        depth = np.load(depth_path).astype(np.float32)
        sky = np.load(sky_path).astype(np.float32)
    
        # ---- optional resize ----
        if self.image_size is not None:
            img_t = TF.resize(
                img_t,
                size=list(self.image_size),
                interpolation=self.image_interp,
                antialias=self.antialias,
            )
    
        # ---- load camera params ----
        data = np.load(npz_path)
        fx = np.float32(data["fx"])
        fy = np.float32(data["fy"])
        qvec = data["qvec"].astype(np.float64)
        tvec = data["tvec"].astype(np.float64)
    
        # ---- scale intrinsics if resized ----
        H, W = int(img_t.shape[1]), int(img_t.shape[2])
    
        if self.image_size is not None:
            sx = W / float(w0)
            sy = H / float(h0)
            fx = np.float32(fx * sx)
            fy = np.float32(fy * sy)
    
        # ---- principal point: image center ----
        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5
    
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
    
        Rt = w2c_3x4_from_qt(qvec, tvec).astype(np.float32)

        if vi > 1:
            mask_path = os.path.join(set_dir, f"view_3_mask.npy")
            mask = np.load(mask_path).astype(np.float32)
        else:
            mask = np.ones((H,W)).astype(np.float32)
            

        return {
            "image": img_t,                         # (3,H,W)
            "intrinsics": torch.from_numpy(K),      # (3,3)
            "extrinsics": torch.from_numpy(Rt),     # (3,4)
            "depth": torch.from_numpy(depth),
            "sky": torch.from_numpy(sky),
            "mask": torch.from_numpy(mask),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        set_dir = self.set_dirs[idx]

        npz_files = sorted(glob.glob(os.path.join(set_dir, "view_*.npz")))
        vis = [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in npz_files]

        if self.shuffle_views:
            #perm = torch.randperm(len(vis)).tolist()
            #vis = [vis[i] for i in perm]
            perm = torch.randperm(self.n_ctx).tolist()
            vis = [vis[i] for i in perm] + vis[self.n_ctx:]

        if self.n_views is not None:
            vis = vis[: self.n_views]

        views = [self._load_one_view(set_dir, vi) for vi in vis]

        images = torch.stack([v["image"] for v in views], dim=0)      # (N,3,H,W)
        intrs  = torch.stack([v["intrinsics"] for v in views], dim=0) # (N,3,3)
        extrs  = torch.stack([v["extrinsics"] for v in views], dim=0) # (N,3,4)
        depths = torch.stack([v["depth"] for v in views], dim=0)
        skys = torch.stack([v["sky"] for v in views], dim=0)
        masks = torch.stack([v["mask"] for v in views], dim=0)

        out = {
            "image": images,
            "intrinsics": intrs,
            "extrinsics": extrs,
            "depth": depths,
            "sky": skys,
            "mask": masks,
        }

        if self.return_meta:
            meta_path = os.path.join(set_dir, "meta.json")
            out["meta"] = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            out["set_dir"] = set_dir
            out["scene_key"] = self._infer_scene_key(set_dir)

        return out