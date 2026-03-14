# from pathlib import Path

# ROOT = Path("/remote-home/share/25-jianfabai/cvpr2026/challenge1/original")
# SUBDIRS = ["train", "valid"]

# DRY_RUN = False  # 先预演，确认后改 False

# def delete_dot_underscore_jpg(base: Path):
#     files = list(base.rglob("._*.jpg")) + list(base.rglob("._*.JPG"))
#     print(f"[SCAN] {base}  Found {len(files)} files like ._*.jpg")
#     for p in files[:10]:
#         print(" -", p)

#     if DRY_RUN:
#         print("  [DRY_RUN] No files deleted.")
#         return

#     deleted = 0
#     for p in files:
#         try:
#             p.unlink()
#             deleted += 1
#         except Exception as e:
#             print(f"  [FAIL] {p} -> {e}")
#     print(f"  [DONE] Deleted {deleted} files")

# for sd in SUBDIRS:
#     delete_dot_underscore_jpg(ROOT / sd)

# print("DRY_RUN =", DRY_RUN)

from pathlib import Path

ROOT = Path("/remote-home/share/25-jianfabai/cvpr2026/challenge1/original")
# SUBDIRS = ["train", "valid"]
SUBDIRS = ["test"]

DRY_RUN = False  # 先预演，确认后改 False
# DRY_RUN = True

def delete_dot_underscore_jpg(base: Path):
    files = list(base.rglob("._*"))
    print(f"[SCAN] {base}  Found {len(files)} files like ._*.jpg")
    for p in files[:10]:
        print(" -", p)

    if DRY_RUN:
        print("  [DRY_RUN] No files deleted.")
        return

    deleted = 0
    for p in files:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"  [FAIL] {p} -> {e}")
    print(f"  [DONE] Deleted {deleted} files")

for sd in SUBDIRS:
    delete_dot_underscore_jpg(ROOT / sd)

print("DRY_RUN =", DRY_RUN)