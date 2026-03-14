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

from models.ResNet import SupConResNet
import cv2
import imageio
import re

# =========================
# 你原来的工具函数（基本不动）
# =========================
def parse_case_from_npy_path(npy_path: str):
    parts = npy_path.replace("\\", "/").split("/")
    fname = os.path.basename(npy_path)
    sample = os.path.splitext(fname)[0]
    split, cls = "unknown", "unknown"
    for i, p in enumerate(parts):
        if p in ("train", "valid", "val", "test"):
            split = "valid" if p == "val" else p
            if i + 1 < len(parts):
                cls = parts[i + 1]
            break
    return split, cls, sample

def make_out_paths(out_root: str, ckpt_path: str, npy_path: str):
    split, cls, sample = parse_case_from_npy_path(npy_path)
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    case_dir = os.path.join(out_root, ckpt_tag, split, cls, sample)
    os.makedirs(case_dir, exist_ok=True)
    cam_f32_path = os.path.join(case_dir, f"{sample}_cam_float32.npy")
    cam_u8_path  = os.path.join(case_dir, f"{sample}_cam_uint8.npy")
    overlay_dir  = os.path.join(case_dir, "overlay_slices")
    gif_path     = os.path.join(case_dir, f"{sample}_overlay.gif")
    mp4_path     = os.path.join(case_dir, f"{sample}_overlay.mp4")
    return case_dir, cam_f32_path, cam_u8_path, overlay_dir, gif_path, mp4_path

def strip_module_prefix(state_dict):
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

def overlay_and_save_slices(ct_vol_u8, cam_vol_01, out_dir, step=5, alpha=0.45):
    os.makedirs(out_dir, exist_ok=True)
    D, H, W = ct_vol_u8.shape
    ct_01 = ct_vol_u8.astype(np.float32) / 255.0

    for z in range(0, D, step):
        base = ct_01[z]
        mask = cam_vol_01[z]

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255.0

        base_rgb = np.stack([base, base, base], axis=-1)
        cam_img = heatmap * alpha + base_rgb * (1 - alpha)
        cam_img = cam_img / (cam_img.max() + 1e-6)
        cam_img_u8 = np.uint8(255 * cam_img)
        cam_img_bgr = cv2.cvtColor(cam_img_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"slice_{z:03d}.png"), cam_img_bgr)

def make_overlay_frame(ct_slice_u8, cam_slice_01, alpha=0.45):
    base = ct_slice_u8.astype(np.float32) / 255.0
    base_rgb = np.stack([base, base, base], axis=-1)

    heat = cv2.applyColorMap(np.uint8(255 * cam_slice_01), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = heat * alpha + base_rgb * (1 - alpha)
    overlay = overlay / (overlay.max() + 1e-6)
    return (overlay * 255).astype(np.uint8)

def save_overlay_gif(vol_u8, cam_up_01, gif_path, fps=10, step=1, alpha=0.45):
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    frames = []
    for z in range(0, vol_u8.shape[0], step):
        frames.append(make_overlay_frame(vol_u8[z], cam_up_01[z], alpha=alpha))
    imageio.mimsave(gif_path, frames, duration=1.0 / fps)
    print("Saved GIF:", gif_path, "frames:", len(frames))

def save_overlay_mp4(vol_u8, cam_up_01, mp4_path, fps=10, step=1, alpha=0.45):
    os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
    frames = []
    for z in range(0, vol_u8.shape[0], step):
        frames.append(make_overlay_frame(vol_u8[z], cam_up_01[z], alpha=alpha))
    imageio.mimsave(mp4_path, frames, fps=fps)
    print("Saved MP4:", mp4_path, "frames:", len(frames))

def normalize_like_dataset1(vol_u8: np.ndarray):
    vol_f = vol_u8.astype(np.float32) / 255.0
    mean, std = 0.3529, 0.2983
    return (vol_f - mean) / std

def find_last_conv3d(model: nn.Module):
    last_name, last_layer = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            last_name, last_layer = name, m
    if last_layer is None:
        raise RuntimeError("No nn.Conv3d found in model.")
    print(f"[GradCAM] using last Conv3d layer: {last_name} -> {last_layer}")
    return last_layer

# =========================
# 关键：3D Grad-CAM 计算器
# =========================
# class GradCAM3D:
#     """
#     基于 feature map (A) 与其梯度 dY/dA 计算 3D Grad-CAM
#     这里直接对 model(x) 返回的 feat_map 注册 hook，因此无需知道 encoder 具体层名。
#     """
#     def __init__(self):
#         self.feat = None
#         self.grad = None
#         self._h1 = None
#         self._h2 = None

#     def _forward_hook(self, module, inp, out):
#         # out: feat_map tensor [B,C,d,h,w]
#         self.feat = out

#     def _backward_hook(self, module, grad_in, grad_out):
#         # grad_out[0]: dY/d(out)
#         self.grad = grad_out[0]

#     def attach_to_tensor(self, feat_tensor: torch.Tensor):
#         """
#         PyTorch 对 Tensor 的 hook：更稳，不依赖具体 module。
#         """
#         self.feat = feat_tensor
#         # 反向时拿梯度
#         feat_tensor.retain_grad()

#     def compute(self, class_logit: torch.Tensor):
#         """
#         class_logit: shape [B] 或标量（通常取 logits[:, class_idx]）
#         return cam_low: [B,1,d,h,w] 归一化到[0,1]
#         """
#         # 反传得到 self.feat.grad
#         class_logit.sum().backward(retain_graph=True)

#         grad = self.feat.grad          # [B,C,d,h,w]
#         feat = self.feat               # [B,C,d,h,w]

#         if grad is None:
#             raise RuntimeError("Grad is None. Ensure retain_grad() and backward() are called.")

#         # α_c = mean_{d,h,w}(grad)
#         weights = grad.mean(dim=(2, 3, 4), keepdim=True)   # [B,C,1,1,1]

#         cam = (weights * feat).sum(dim=1, keepdim=True)    # [B,1,d,h,w]
#         cam = F.relu(cam)

#         # normalize per-sample
#         cam_min = cam.amin(dim=(2,3,4), keepdim=True)
#         cam_max = cam.amax(dim=(2,3,4), keepdim=True)
#         cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
#         return cam

class GradCAM3D:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None  # [B,C,d,h,w]
        self.gradients = None    # [B,C,d,h,w]
        self.hook = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # out: [B,C,d,h,w], must be in graph
            self.activations = out
            # 用 register_hook 捕获反向梯度（最稳）
            out.register_hook(self._save_grad)

        self.hook = self.target_layer.register_forward_hook(forward_hook)

    def _save_grad(self, grad):
        self.gradients = grad

    def remove_hooks(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """
        返回 cam_low: [B,1,d,h,w] (0-1 normalized)
        """
        self.model.zero_grad(set_to_none=True)
        self.gradients = None
        self.activations = None

        feat_map, feat, logits = self.model(x)  # 你的 forward
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]  # [B]
        score.sum().backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Failed to get activations/gradients. "
                "This usually means target_layer is not on the path to logits."
            )

        # Grad-CAM: α = mean(grad) over (d,h,w)
        weights = self.gradients.mean(dim=(2,3,4), keepdim=True)  # [B,C,1,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B,1,d,h,w]
        cam = F.relu(cam)

        # normalize per sample
        cam_min = cam.amin(dim=(2,3,4), keepdim=True)
        cam_max = cam.amax(dim=(2,3,4), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

        return cam, logits.detach(), class_idx


def main():
    ckpt_path = "/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m4-nomix-class-bai-v100-batch6/epoch038_f10.9732.pkl"
    npy_path  = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/covid/ct_scan_58.npy"

    out_root = "/remote-home/share/25-jianfabai/cvpr2026/somework/gradcam3d_out"
    os.makedirs(out_root, exist_ok=True)

    backbone_name = "resnest50_3D"
    ipt_dim = 1
    n_classes = 2
    supcon = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SupConResNet(
        name=backbone_name,
        ipt_dim=ipt_dim,
        head="mlp",
        feat_dim=128,
        n_classes=n_classes,
        supcon=supcon
    ).to(device)
    model.eval()
    load_ckpt_to_model(model, ckpt_path)

    # ===== load volume =====
    vol = np.load(npy_path)
    assert vol.ndim == 3, vol.shape
    D, H, W = vol.shape
    print("[npy] shape:", vol.shape, "dtype:", vol.dtype)

    # vol_u8 for visualization
    vol_u8 = vol if vol.dtype == np.uint8 else ((vol - vol.min())/(vol.max()-vol.min()+1e-6)*255).astype(np.uint8)

    # normalized input
    x_np = normalize_like_dataset1(vol_u8).astype(np.float32)
    x = torch.from_numpy(x_np)[None, None].to(device)  # [1,1,D,H,W]
    x.requires_grad_(True)  # 不是必须，但有时你也想看 input-saliency

    print("input min/max(after normalize):", x.min().item(), x.max().item())

    # ===== forward (NO torch.no_grad) =====
    model.zero_grad(set_to_none=True)
    # with torch.no_grad():
    #     feat_map, feat, logits = model(x)  # feat_map: [B,C,d,h,w]
    # prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    # pred_cls = int(prob.argmax())
    # print("Pred class =", pred_cls, "prob =", prob)
    # print("feat_map shape =", tuple(feat_map.shape))

    # # 你也可以手动指定想解释的类别：
    # # target_cls = 0  # e.g., covid
    # target_cls = pred_cls

    # ===== Grad-CAM on feat_map =====
    # cam_engine = GradCAM3D()
    # cam_engine.attach_to_tensor(feat_map)  # retain grad on tensor

    # target_logit = logits[:, target_cls]   # [B]
    # cam_low = cam_engine.compute(target_logit)  # [B,1,d,h,w]

    # 选择 target layer：最后一个 Conv3d（最通用）
    target_layer = find_last_conv3d(model)
    gradcam = GradCAM3D(model, target_layer)

    # 计算 Grad-CAM（class_idx=None 表示用预测类；你也可指定 0/1）
    cam_low, logits_det, used_cls = gradcam(x, class_idx=None)
    # cam_low, logits_det, used_cls = gradcam(x, class_idx=1)
    print("[GradCAM] used class:", used_cls, "prob:", torch.softmax(logits_det, dim=1).cpu().numpy())

    # upsample to original [D,H,W]
    cam_up = F.interpolate(cam_low, size=(D, H, W), mode="trilinear", align_corners=False)
    cam_up = cam_up[0, 0].detach().cpu().numpy().astype(np.float32)  # [D,H,W] in [0,1]
    print("gradcam_up shape =", cam_up.shape, "dtype =", cam_up.dtype)

    # ===== save =====
    case_dir, cam_f32_path, cam_u8_path, overlay_dir, gif_path, mp4_path = make_out_paths(out_root, ckpt_path, npy_path)
    print("Outputs will be saved to:", case_dir)

    np.save(cam_f32_path, cam_up.astype(np.float32))
    print("Saved:", cam_f32_path, "size_bytes=", os.path.getsize(cam_f32_path))

    np.save(cam_u8_path, (cam_up * 255).astype(np.uint8))
    print("Saved:", cam_u8_path, "size_bytes=", os.path.getsize(cam_u8_path))

    overlay_and_save_slices(vol_u8, cam_up, overlay_dir, step=5, alpha=0.45)
    print("Saved overlays to:", overlay_dir)

    save_overlay_gif(vol_u8, cam_up, gif_path, fps=10, step=1, alpha=0.45)
    save_overlay_mp4(vol_u8, cam_up, mp4_path, fps=10, step=1, alpha=0.45)


if __name__ == "__main__":
    main()