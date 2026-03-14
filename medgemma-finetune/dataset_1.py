# dataset.py
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


NPY_EXTS = (".npy",)


def list_npys_recursively(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(NPY_EXTS):
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def _random_resized_crop_params(
    H: int,
    W: int,
    scale: Tuple[float, float] = (0.7, 1.0),
) -> Tuple[int, int, int, int]:
    """
    为正方形输入（H==W）做简单版 RandomResizedCrop：
    - 随机选一个 crop 边长 = round(H * s), s ~ U(scale)
    - 随机左上角 (i, j)
    返回: (i, j, h, w)
    """
    assert H > 0 and W > 0
    s = random.uniform(scale[0], scale[1])
    crop = int(round(H * s))
    crop = max(1, min(crop, H, W))
    i = random.randint(0, H - crop)
    j = random.randint(0, W - crop)
    return i, j, crop, crop

def rescale_z(images_zyx, target_depth, is_mask_image=False, verbose=False):
    # print("Resizing dim z")
    resize_x = 1.0
    resize_y = target_depth/images_zyx.shape[0]
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    return res

def random_resized_crop_4d(
    x: torch.Tensor,
    out_size: int,
    scale: Tuple[float, float] = (0.7, 1.0),
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    x: (S, C, H, W)  对所有 slice 用同一个 crop 参数
    return: (S, C, out_size, out_size)
    """
    assert x.ndim == 4, f"Expect (S,C,H,W), got {x.shape}"
    S, C, H, W = x.shape
    i, j, h, w = _random_resized_crop_params(H, W, scale=scale)
    x = x[:, :, i:i+h, j:j+w]  # (S,C,h,w)
    x = F.interpolate(x, size=(out_size, out_size), mode=mode, align_corners=False)
    return x


class NPYVolumeDataset(Dataset):
    """
    每个 item = 一个 volume (.npy)

    train:
      - 随机取连续 num_slices（默认12）
      - 做 random resized crop（scale 0.8~1.0）再 resize 回 out_size

    val/test:
      - 固定取“前 num_slices + 后 num_slices” => 2*num_slices（默认24）
      - 不做 crop 增强，只确保尺寸/归一化一致
    """
    def __init__(
        self,
        covid_root: str,
        noncovid_root: str,
        num_slices: int = 12,
        out_size: int = 448,
        mmap: bool = True,
        split: str = "train",               # "train" or "val"
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        assume_uint8_0_255: bool = True,
    ):
        self.paths = []
        self.labels = []

        covid_npys = list_npys_recursively(covid_root)
        noncovid_npys = list_npys_recursively(noncovid_root)
        if len(covid_npys) == 0:
            raise RuntimeError(f"No npy found under covid_root: {covid_root}")
        if len(noncovid_npys) == 0:
            raise RuntimeError(f"No npy found under noncovid_root: {noncovid_root}")

        self.paths.extend(covid_npys)
        self.labels.extend([1] * len(covid_npys))
        self.paths.extend(noncovid_npys)
        self.labels.extend([0] * len(noncovid_npys))

        self.num_slices = int(num_slices)
        self.out_size = int(out_size)
        self.mmap = bool(mmap)
        self.split = split
        self.crop_scale = crop_scale
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)  # broadcast to (S,3,H,W)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.assume_uint8_0_255 = bool(assume_uint8_0_255)

    def __len__(self):
        return len(self.paths)

    def _load_volume(self, p: str):
        if self.mmap:
            vol = np.load(p, mmap_mode="r")
        else:
            vol = np.load(p)
        if vol.ndim != 3:
            raise ValueError(f"Expect 3D volume, got shape={vol.shape} at {p}")
        return vol

    # def _pick_slices_train(self, vol: np.ndarray) -> np.ndarray:
    #     D, H, W = vol.shape
    #     if D < self.num_slices:
    #         raise ValueError(f"Volume depth {D} < num_slices {self.num_slices} at {self.paths}")
    #     start = random.randint(0, D - self.num_slices)
    #     return vol[start:start + self.num_slices]

    # def _pick_slices_val(self, vol: np.ndarray) -> np.ndarray:
    #     D, H, W = vol.shape
    #     if D < 2 * self.num_slices:
    #         raise ValueError(f"Volume depth {D} < 2*num_slices {2*self.num_slices} for val at {self.paths}")
    #     first = vol[:self.num_slices]
    #     last = vol[-self.num_slices:]
    #     return np.concatenate([first, last], axis=0)  # (2S,H,W)

    #原本随机抽取,现在为全深度均匀
    def _pick_slices_train(self, vol: np.ndarray) -> np.ndarray:
        D, H, W = vol.shape
        if D < self.num_slices:
            raise ValueError(f"Volume depth {D} < num_slices {self.num_slices} at {self.paths}")

        # --- Minimal but effective upgrade ---
        # Instead of sampling a *continuous* window (highly redundant),
        # sample approximately-uniform indices across depth with small random jitter.
        # This tends to help CT classification when lesions occupy only a subset of slices.
        base = np.linspace(0, D - 1, self.num_slices)

        # jitter magnitude: small relative to the spacing
        spacing = (D - 1) / max(1, (self.num_slices - 1))
        jitter = max(1, int(round(spacing * 0.25)))

        idx = []
        for b in base:
            j = random.randint(-jitter, jitter)
            ii = int(round(b)) + j
            ii = 0 if ii < 0 else (D - 1 if ii > D - 1 else ii)
            idx.append(ii)

        idx = np.array(sorted(idx), dtype=np.int64)
        return vol[idx]



    def _pick_slices_val(self, vol: np.ndarray) -> np.ndarray:
        D, H, W = vol.shape
        if D < 2 * self.num_slices:
            raise ValueError(f"Volume depth {D} < 2*num_slices {2*self.num_slices} for val at {self.paths}")

        # Deterministic uniform sampling over the entire volume for validation.
        # (Avoid missing mid-lung lesions when only taking head/tail slices.)
        idx = np.linspace(0, D - 1, 2 * self.num_slices).round().astype(np.int64)
        idx = np.clip(idx, 0, D - 1)
        return vol[idx]  # (2S,H,W)

    #中间2slices
    # def _pick_slices_val(self, vol: np.ndarray) -> np.ndarray:
    #     D, H, W = vol.shape
    #     S2 = 2 * self.num_slices
    #     if D < S2:
    #         raise ValueError(f"Volume depth {D} < 2*num_slices {S2} for val at {self.paths}")

    #     # Take the middle consecutive 2S slices (center crop along depth).
    #     start = (D - S2) // 2
    #     end = start + S2
    #     return vol[start:end]  # (2S, H, W)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = int(self.labels[idx])

        vol = self._load_volume(p)  # (D,H,W)

        if self.split == "train":
            window = self._pick_slices_train(vol)      # (S,H,W)
        else:
            window = self._pick_slices_val(vol)        # (2S,H,W)

        window = np.array(window, copy=True)

        x = torch.from_numpy(window).float().unsqueeze(1)  # (S,1,H,W)

        # - 如果是 uint8 0~255 -> /255
        if self.assume_uint8_0_255:
            x = x / 255.0

        x = x.repeat(1, 3, 1, 1)  # (S,3,H,W)

        # resize 到 out_size（保证输入统一）
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)

        # 训练增强：random resized crop，再 resize 回 out_size
        if self.split == "train":
            x = random_resized_crop_4d(
                x,
                out_size=self.out_size,
                scale=self.crop_scale,
                mode="bilinear",
            )

        # Normalize: (x - mean) / std
        x = (x - self.mean) / self.std

        return {
            "pixel_values": x,                             # (S,3,448,448) or (2S,3,448,448)
            "labels": torch.tensor(y, dtype=torch.long),
        }


def volume_collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples], dim=0)  # (B,S,3,H,W)
    labels = torch.stack([ex["labels"] for ex in examples], dim=0)              # (B,)
    return {"pixel_values": pixel_values, "labels": labels}