#!/usr/bin/env python
# coding: utf-8
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CVPR26_ROOT = "/remote-home/share/25-jianfabai/cvpr2026"
sys.path.insert(0, CVPR26_ROOT)

from models.ResNet import SupConResNet  # 你的工程路径

import cv2

import os, re

def parse_case_from_npy_path(npy_path: str):
    # 例: .../challenge1/norm/valid/covid/ct_scan_0.npy
    parts = npy_path.replace("\\", "/").split("/")
    fname = os.path.basename(npy_path)
    sample = os.path.splitext(fname)[0]  # ct_scan_0

    # 尝试找 split 和 cls
    split = "unknown"
    cls = "unknown"
    for i, p in enumerate(parts):
        if p in ("train", "valid", "val", "test"):
            split = "valid" if p == "val" else p
            if i + 1 < len(parts):
                cls = parts[i + 1]  # covid / non-covid
            break
    return split, cls, sample

def make_out_paths(out_root: str, ckpt_path: str, npy_path: str):
    split, cls, sample = parse_case_from_npy_path(npy_path)

    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]  # epoch038_f10.9732
    case_dir = os.path.join(out_root, ckpt_tag, split, cls, sample)
    os.makedirs(case_dir, exist_ok=True)

    cam_f32_path = os.path.join(case_dir, f"{sample}_cam_float32.npy")
    cam_u8_path  = os.path.join(case_dir, f"{sample}_cam_uint8.npy")
    overlay_dir  = os.path.join(case_dir, "overlay_slices")
    gif_path     = os.path.join(case_dir, f"{sample}_overlay.gif")
    mp4_path     = os.path.join(case_dir, f"{sample}_overlay.mp4")
    return case_dir, cam_f32_path, cam_u8_path, overlay_dir, gif_path, mp4_path

def strip_module_prefix(state_dict):
    # 去掉 'module.' 前缀
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_ckpt_to_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded. missing={len(missing)}, unexpected={len(unexpected)}")
    return ckpt


def find_last_linear(model: nn.Module):
    """
    尽量找到用于分类的最后一层 Linear（fc/classifier）
    返回: linear_layer
    """
    # 常见命名优先
    for name in ["fc", "classifier", "head", "classify"]:
        if hasattr(model, name) and isinstance(getattr(model, name), nn.Linear):
            return getattr(model, name)

    # 递归找最后一个 Linear
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("No nn.Linear found in model.")
    return last


@torch.no_grad()
def build_cam_3d_from_featmap_and_fc(feat_map, fc_layer: nn.Linear, class_idx: int):
    """
    feat_map: [B, C, D, H, W]
    fc_layer: nn.Linear(C, num_classes) (必须输入维度=feat_map通道数)
    class_idx: int
    return:
      cam_low: [B, 1, D, H, W] (low-res)
    """
    # fc 权重: [num_classes, C]
    W = fc_layer.weight[class_idx]  # [C]
    # 加权求和：sum_c (W[c] * feat_map[:,c,:,:,:])
    cam = (feat_map * W.view(1, -1, 1, 1, 1)).sum(dim=1, keepdim=True)  # [B,1,D,H,W]
    cam = F.relu(cam)
    # 归一化到[0,1]
    cam_min = cam.amin(dim=(2,3,4), keepdim=True)
    cam_max = cam.amax(dim=(2,3,4), keepdim=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
    return cam


def overlay_and_save_slices(ct_vol_u8, cam_vol_01, out_dir, step=5, alpha=0.45):
    """
    ct_vol_u8: [D,H,W] uint8 (0-255)
    cam_vol_01: [D,H,W] float in [0,1]
    """
    os.makedirs(out_dir, exist_ok=True)
    D, H, W = ct_vol_u8.shape

    # 为 show_cam_on_image 准备 0-1 float 的底图
    ct_01 = ct_vol_u8.astype(np.float32) / 255.0

    for z in range(0, D, step):
        base = ct_01[z]                      # [H,W] float32 0-1
        mask = cam_vol_01[z]                 # [H,W] float32 0-1

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # BGR uint8
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255.0                                # [H,W,3] float32 0-1

        base_rgb = np.stack([base, base, base], axis=-1)                      # [H,W,3]
        cam_img = heatmap * alpha + base_rgb * (1 - alpha)
        cam_img = cam_img / (cam_img.max() + 1e-6)
        cam_img_u8 = np.uint8(255 * cam_img)
        cam_img_bgr = cv2.cvtColor(cam_img_u8, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(out_dir, f"slice_{z:03d}.png"), cam_img_bgr)

import imageio
import numpy as np
import cv2
import os

def make_overlay_frame(ct_slice_u8, cam_slice_01, alpha=0.45):
    """
    ct_slice_u8: [H,W] uint8
    cam_slice_01: [H,W] float32 in [0,1]
    return: RGB uint8 [H,W,3]
    """
    base = ct_slice_u8.astype(np.float32) / 255.0  # [0,1]
    base_rgb = np.stack([base, base, base], axis=-1)  # [H,W,3]

    heat = cv2.applyColorMap(np.uint8(255 * cam_slice_01), cv2.COLORMAP_JET)  # BGR uint8
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0   # RGB [0,1]

    overlay = heat * alpha + base_rgb * (1 - alpha)
    overlay = overlay / (overlay.max() + 1e-6)
    return (overlay * 255).astype(np.uint8)

def save_overlay_gif(vol_u8, cam_up_01, gif_path, fps=10, step=1, alpha=0.45):
    """
    vol_u8: [D,H,W] uint8
    cam_up_01: [D,H,W] float32 in [0,1]
    step: 1 表示每层都保存；5 表示每5层一帧（更小）
    """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    frames = []
    for z in range(0, vol_u8.shape[0], step):
        frame = make_overlay_frame(vol_u8[z], cam_up_01[z], alpha=alpha)  # RGB uint8
        frames.append(frame)
    imageio.mimsave(gif_path, frames, duration=1.0/fps)  # GIF 用 duration
    print("Saved GIF:", gif_path, "frames:", len(frames))

def save_overlay_mp4(vol_u8, cam_up_01, mp4_path, fps=10, step=1, alpha=0.45):
    os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
    frames = []
    for z in range(0, vol_u8.shape[0], step):
        frame = make_overlay_frame(vol_u8[z], cam_up_01[z], alpha=alpha)  # RGB uint8
        frames.append(frame)
    imageio.mimsave(mp4_path, frames, fps=fps)  # mp4 用 fps
    print("Saved MP4:", mp4_path, "frames:", len(frames))

def normalize_like_dataset1(vol_u8: np.ndarray):
    vol_f = vol_u8.astype(np.float32) / 255.0
    mean, std = 0.3529, 0.2983
    vol_f = (vol_f - mean) / std
    return vol_f


def main():
    ckpt_path = "/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m4-nomix-class-bai-v100-batch6/epoch038_f10.9732.pkl"
    # npy_path  = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/covid/ct_scan_0.npy"
    # npy_path = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/non-covid/ct_scan_0.npy"
    npy_path = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/covid/ct_scan_58.npy"

    out_root = "/remote-home/share/25-jianfabai/cvpr2026/somework/cam3d_out"
    os.makedirs(out_root, exist_ok=True)

    # === 这里要和你训练时一致 ===
    backbone_name = "resnest50_3D"
    ipt_dim = 1          # 只有CT一个通道
    n_classes = 2
    supcon = False       # 你这版训练脚本默认 supcon=False（分类训练）

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) build model
    model = SupConResNet(name=backbone_name, ipt_dim=ipt_dim, head="mlp",
                         feat_dim=128, n_classes=n_classes, supcon=supcon).to(device)
    model.eval()

    # 2) load ckpt
    load_ckpt_to_model(model, ckpt_path)

    # 3) load volume
    vol = np.load(npy_path)   # 期望 [128,256,256]
    assert vol.ndim == 3, vol.shape
    D, H, W = vol.shape
    print("[npy] shape:", vol.shape, "dtype:", vol.dtype)

    # 你截图中单个ct_scan_x.npy 大约8MB，基本是 uint8
    # 如果不是uint8，这里也统一映射到 0-255 便于可视化
    if vol.dtype != np.uint8:
        v = vol.astype(np.float32)
        v = (v - v.min()) / (v.max() - v.min() + 1e-6)
        vol_u8 = (v * 255).astype(np.uint8)
    else:
        vol_u8 = vol

    # 网络输入用 float32
    # x = torch.from_numpy(vol_u8.astype(np.float32) / 255.0)[None, None].to(device)  # [1,1,D,H,W]
    # x = torch.from_numpy(vol_u8.astype(np.float32))[None, None].to(device)
    # print("input min/max:", x.min().item(), x.max().item())

    # 保留 vol_u8 用于可视化叠加；x 用 normalized 用于推理
    vol_u8 = vol if vol.dtype == np.uint8 else ((vol - vol.min())/(vol.max()-vol.min()+1e-6)*255).astype(np.uint8)

    x_np = normalize_like_dataset1(vol_u8)                # float32, 归一化后
    x = torch.from_numpy(x_np)[None, None].to(device)     # [1,1,D,H,W]

    print("input min/max(after normalize):", x.min().item(), x.max().item())

    # 4) forward 得到 feat_map 和 logits
    feat_map, feat, logits = model(x)  # feat_map: [1,C,d,h,w] logits:[1,2]
    prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_cls = int(prob.argmax())
    print("Pred class =", pred_cls, "prob =", prob)
    print("feat_map shape =", tuple(feat_map.shape))

    # 5) 找到分类用的最后 Linear（重要：其in_features要等于feat_map通道数）
    fc = find_last_linear(model.encoder) if hasattr(model, "encoder") else find_last_linear(model)
    print("found last linear:", fc)

    C = feat_map.shape[1]
    if fc.in_features != C:
        raise RuntimeError(f"FC in_features ({fc.in_features}) != feat_map channels ({C}). "
                           f"说明 feat_map 不是最后一层用于分类的特征图，需要换成更靠后的 feature map。")

    # 6) CAM（低分辨率）
    cam_low = build_cam_3d_from_featmap_and_fc(feat_map, fc, pred_cls)  # [1,1,d,h,w]

    # 7) 上采样到原始大小 [1,1,128,256,256]
    cam_up = F.interpolate(cam_low, size=(D, H, W), mode="trilinear", align_corners=False)
    cam_up = cam_up[0,0].detach().cpu().numpy()  # [D,H,W] in [0,1]
    print("cam_up shape =", cam_up.shape, "dtype =", cam_up.dtype)

    # # 8) 保存 3D CAM（两份：float32 和 uint8）
    # cam_f32_path = os.path.join(out_root, "ct_scan_0_cam_float32.npy")
    # np.save(cam_f32_path, cam_up.astype(np.float32))
    # print("Saved:", cam_f32_path, "size_bytes=", os.path.getsize(cam_f32_path))

    # cam_u8_path = os.path.join(out_root, "ct_scan_0_cam_uint8.npy")
    # np.save(cam_u8_path, (cam_up * 255).astype(np.uint8))
    # print("Saved:", cam_u8_path, "size_bytes=", os.path.getsize(cam_u8_path))

    # # 9) 叠加保存切片图（每5层一张，像你师姐那样）
    # overlay_dir = os.path.join(out_root, "overlay_slices_ct_scan_0")
    # overlay_and_save_slices(vol_u8, cam_up, overlay_dir, step=5, alpha=0.45)
    # print("Saved overlays to:", overlay_dir)

    case_dir, cam_f32_path, cam_u8_path, overlay_dir, gif_path, mp4_path = make_out_paths(out_root, ckpt_path, npy_path)
    print("Outputs will be saved to:", case_dir)
    # np.save(cam_f32_path, cam_up.astype(np.float32))
    # print("Saved:", cam_f32_path, "size_bytes=", os.path.getsize(cam_f32_path))
    np.save(cam_u8_path, (cam_up * 255).astype(np.uint8))
    print("Saved:", cam_u8_path, "size_bytes=", os.path.getsize(cam_u8_path))
    overlay_and_save_slices(vol_u8, cam_up, overlay_dir, step=5, alpha=0.45)
    print("Saved overlays to:", overlay_dir)
    #step=1：128 帧，动图最完整; step=5：只有 26 帧，文件小很多
    save_overlay_gif(vol_u8, cam_up, gif_path, fps=10, step=1, alpha=0.45)
    save_overlay_mp4(vol_u8, cam_up, mp4_path, fps=10, step=1, alpha=0.45)


if __name__ == "__main__":
    main()