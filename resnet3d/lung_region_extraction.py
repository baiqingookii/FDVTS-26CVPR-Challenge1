import os
import cv2
import numpy as np
import re
from tqdm import tqdm

def numeric_key(name: str):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else name

def get_global_bbox_from_volume(vol, thr=22, extra_pad=0):
    proj = np.max(vol, axis=0).astype(np.uint8)  # (H,W)

    mask = (proj > thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, vol.shape[2]-1, vol.shape[1]-1)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    H, W = proj.shape
    x1 = max(0, x1 - extra_pad); y1 = max(0, y1 - extra_pad)
    x2 = min(W - 1, x2 + extra_pad); y2 = min(H - 1, y2 + extra_pad)
    return (x1, y1, x2, y2)

def crop_scale_center_crop_slice(img, bbox, out_size):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2+1, x1:x2+1].astype(np.uint8)
    h, w = crop.shape

    # 等比缩放：短边 >= out_size
    scale = out_size / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 中心裁剪 out_size x out_size
    y0 = (new_h - out_size) // 2
    x0 = (new_w - out_size) // 2
    out = resized[y0:y0+out_size, x0:x0+out_size]
    return out

def process_one_npy_keep_size(in_npy_path, out_npy_path, thr=22, extra_pad=0):
    vol = np.load(in_npy_path)
    if vol.ndim != 3:
        raise ValueError(f"Expect 3D npy (D,H,W), got {vol.shape} @ {in_npy_path}")

    in_dtype = vol.dtype
    vol_u8 = vol.astype(np.uint8)

    D, H, W = vol_u8.shape
    out_size = H  

    bbox = get_global_bbox_from_volume(vol_u8, thr=thr, extra_pad=extra_pad)
    out_vol = np.empty((D, out_size, out_size), dtype=np.uint8)

    for i in range(D):
        out_vol[i] = crop_scale_center_crop_slice(vol_u8[i], bbox, out_size=out_size)

    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)

    np.save(out_npy_path, out_vol)

    return bbox, (D, H, W), out_vol.shape, out_size

def batch_process_folder_keep_size(in_dir, out_dir, thr=22, extra_pad=0, overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(in_dir) if f.endswith(".npy")]
    files = sorted(files, key=numeric_key)

    print(f"[INPUT ] {in_dir}  ({len(files)} npy)")
    print(f"[OUTPUT] {out_dir}")
    print(f"[PARAM ] thr={thr} extra_pad={extra_pad} overwrite={overwrite}")
    print("[MODE  ] keep per-file size (D,H,W) -> (D,H,W)")

    for fn in tqdm(files):
        in_path = os.path.join(in_dir, fn)
        out_path = os.path.join(out_dir, fn)

        if (not overwrite) and os.path.exists(out_path):
            continue

        bbox, in_shape, out_shape, out_size = process_one_npy_keep_size(
            in_path, out_path, thr=thr, extra_pad=extra_pad
        )

    print("Done.")

if __name__ == "__main__":
    batch_process_folder_keep_size(
        "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/test",
        "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/test_lungex",
        thr=22, extra_pad=0, overwrite=False
    )

    # train
    # batch_process_folder_keep_size(
    #     "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/train/covid",
    #     "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/train/covid1b",
    #     thr=22, extra_pad=0, overwrite=False
    # )

    # batch_process_folder_keep_size(
    #     "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/train/non-covid",
    #     "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/train/non-covid1b",
    #     thr=22, extra_pad=0, overwrite=False
    # )

    # # valid
    # batch_process_folder_keep_size(
    #     "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/valid/covid",
    #     "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/valid/covid1b",
    #     thr=22, extra_pad=0, overwrite=False
    # )
    # batch_process_folder_keep_size(
    #     "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/valid/non-covid",
    #     "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/valid/non-covid1b",
    #     thr=22, extra_pad=0, overwrite=False
    # )