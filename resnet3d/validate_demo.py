#!/usr/bin/env python
# coding: utf-8
"""
Infer ONLY ONE npy with EXACT preprocessing as dataset1 (val path):
- rescale_z to img_depth (default 64)
- normalize (mean=0.3529, std=0.2983)

Run:
python infer_one_like_dataset_val.py \
  --npy /remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/train/covid/ct_scan_354.npy \
  --ckpt /remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m4-nomix-class-bai-v100-batch6/epoch038_f10.9732.pkl \
  --model resnest50_3D \
  --gpu 0
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import cv2

# ---- project imports ----
CVPR26_ROOT = "/remote-home/share/25-jianfabai/cvpr2026"
sys.path.insert(0, CVPR26_ROOT)
from models.ResNet import SupConResNet


# ====== copied from dataset1 (exact) ======
def normalize(image: np.ndarray) -> np.ndarray:
    image = image / 255.0
    mean = 0.3529
    std = 0.2983
    image = 1.0 * (image - mean) / std
    return image


def rescale_z(images_zyx: np.ndarray, target_depth: int, is_mask_image: bool = False) -> np.ndarray:
    resize_x = 1.0
    resize_y = target_depth / images_zyx.shape[0]
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    # NOTE: cv2.resize treats first two dims as (y,x); here y=z so fy scales depth
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
    return res
# =========================================


def load_ckpt_like_training(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt

    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "")] = v
    model.load_state_dict(new_sd, strict=False)


@torch.no_grad()
def infer_one(npy_path: str, model: nn.Module, device: torch.device, img_depth: int = 64):
    arr = np.load(npy_path)  # expected (D,H,W), uint8 0~255 (your data min/max like 20~255)
    if arr.ndim != 3:
        raise ValueError(f"Expect (D,H,W), got {arr.shape}")
    print(f"[npy path]={npy_path}")
    print(f"[raw] shape={arr.shape} dtype={arr.dtype} min/max={arr.min()}/{arr.max()}")

    # dataset1: if self.img_depth != 128 -> rescale_z to 64
    if img_depth != 128:
        arr = rescale_z(arr, img_depth)
        print(f"[after rescale_z] shape={arr.shape} dtype={arr.dtype} min/max={arr.min()}/{arr.max()}")

    # dataset1 val: torch.from_numpy(normalize(img_array)).float()
    arr_norm = normalize(arr).astype(np.float32)
    print(f"[after normalize] dtype={arr_norm.dtype} min/max={arr_norm.min():.4f}/{arr_norm.max():.4f}")

    x = torch.from_numpy(arr_norm).float()          # (D,H,W)
    x = x.unsqueeze(0).unsqueeze(1).to(device)      # (1,1,D,H,W)

    _, feat, logits = model(x)
    prob = F.softmax(logits, dim=1)
    pred = int(prob.argmax(dim=1).item())

    print(f"[logits] {logits.detach().cpu().numpy()}")
    print(f"[prob]   {prob.detach().cpu().numpy()}")
    print(f"[pred]   {pred}")
    return pred, prob.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", default="/remote-home/share/25-jianfabai/cvpr2026/somework/lung_region_extraction/norm/valid/covid/ct_scan_63.npy", type=str)
    # parser.add_argument("--npy", default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/covid1/ct_scan_63.npy", type=str)
    # parser.add_argument("--npy", default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/train/covid/ct_scan_356.npy", type=str)
    parser.add_argument("--ckpt", default="/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m4-nomix-class-bai-v100-batch6/epoch038_f10.9732.pkl", type=str)
    parser.add_argument("--model", default="resnest50_3D", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--img_depth", default=64, type=int, help="must match dataset1: default 64")
    args = parser.parse_args()

    assert os.path.exists(args.npy), args.npy
    assert os.path.exists(args.ckpt), args.ckpt

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # build model (match your training signature)
    net = SupConResNet(
        name=args.model,
        ipt_dim=1,
        head="mlp",
        feat_dim=128,
        n_classes=2,
        supcon=False,
    )

    # load ckpt like your script (before DataParallel)
    load_ckpt_like_training(net, args.ckpt)

    # wrap like your script
    net = nn.DataParallel(net).to(device).eval()

    infer_one(args.npy, net, device, img_depth=args.img_depth)


if __name__ == "__main__":
    main()