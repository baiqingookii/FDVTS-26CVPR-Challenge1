import numpy as np
import imageio

# arr = np.load("/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/train/covid/ct_scan_0.npy")  # (D,H,W)
# arr = np.load("/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/train/covid/ct_scan_362.npy")
arr = np.load("/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/non-covid1b/ct_scan_58.npy")
# 可选：降采样减少体积
frames = [arr[i].astype(np.uint8) for i in range(arr.shape[0])]
imageio.mimsave("/remote-home/share/25-jianfabai/cvpr2026/somework/dataset_gif/valid_non-covid1b_ct_scan_58.gif", frames, fps=10)
print("saved")

# import os
# import numpy as np
# import imageio

# # 输入根目录（npy）
# IN_ROOT = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid"
# # 输出目录（gif）
# OUT_DIR = "/remote-home/share/25-jianfabai/cvpr2026/somework/dataset_gif"
# os.makedirs(OUT_DIR, exist_ok=True)

# # 8 个误分类样本： (scan_id, true_label, pred_label)
# misclassified = [
#     ("ct_scan_32",  1, 0),  # center 0
#     ("ct_scan_67",  0, 1),  # center 1
#     ("ct_scan_68",  0, 1),
#     ("ct_scan_58",  1, 0),
#     ("ct_scan_76",  1, 0),
#     ("ct_scan_148", 0, 1),  # center 3
#     ("ct_scan_110", 1, 0),
#     ("ct_scan_88",  1, 0),
# ]

# def npy_to_gif(npy_path: str, gif_path: str, fps: int = 10):
#     arr = np.load(npy_path)  # (D,H,W) or similar
#     # 若数据不是 0-255 的 uint8，这里做一次稳妥的归一化到 0-255
#     if arr.dtype != np.uint8:
#         a = arr.astype(np.float32)
#         mn, mx = float(a.min()), float(a.max())
#         if mx > mn:
#             a = (a - mn) / (mx - mn) * 255.0
#         else:
#             a = np.zeros_like(a)
#         arr_u8 = a.astype(np.uint8)
#     else:
#         arr_u8 = arr

#     frames = [arr_u8[i] for i in range(arr_u8.shape[0])]
#     imageio.mimsave(gif_path, frames, fps=fps)

# for scan_id, true_lab, pred_lab in misclassified:
#     subdir = "covid" if true_lab == 1 else "non-covid"
#     npy_path = os.path.join(IN_ROOT, subdir, f"{scan_id}.npy")
#     gif_path = os.path.join(OUT_DIR, f"valid_{subdir}_{scan_id}.gif")

#     if not os.path.exists(npy_path):
#         print(f"[SKIP] not found: {npy_path}")
#         continue

#     try:
#         npy_to_gif(npy_path, gif_path, fps=10)
#         print(f"[OK] {scan_id} true={true_lab} pred={pred_lab} -> {gif_path}")
#     except Exception as e:
#         print(f"[ERR] {scan_id} -> {e}")