import os
import shutil

src_dir = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/train/covid1b"
dst_dir = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/valid/covid1b"

os.makedirs(dst_dir, exist_ok=True)

start_id = 350
end_id = 388

# start_id = 0
# end_id = 127

missing = []
copied = 0

for i in range(start_id, end_id + 1):
    fname = f"ct_scan_{i}.npy"
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    if not os.path.isfile(src_path):
        missing.append(fname)
        continue

    shutil.copy2(src_path, dst_path)  # 保留时间戳等元信息；文件名不变
    copied += 1

print(f"Copied {copied} files to: {dst_dir}")
if missing:
    print(f"Missing {len(missing)} files:")
    for m in missing:
        print(" -", m)