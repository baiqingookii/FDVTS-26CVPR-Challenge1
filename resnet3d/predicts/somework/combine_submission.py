#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv

CSV_PATHS = {
    "c2m5_best": "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/tta/c2m5-1b-3090-nomix_epoch075_f10.9681_test_lungex_tta10_results/covid_best_thr.csv",
    "med7": "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/stimulate_submission/medsiglip/0.5_thr_vote_7.csv",
    "med4": "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/stimulate_submission/medsiglip/0.5_thr_vote_4.csv",
    "med5": "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/stimulate_submission/medsiglip/0.5_thr_vote_5.csv",
    "tta5": "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/stimulate_submission/tta/best_thr_vote_5.csv",
    "tta6": "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/stimulate_submission/tta/best_thr_vote_6.csv",
    "trans3": "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/stimulate_submission/trans_flatten/best_thr_vote_3.csv",
}

OUT_CSV = "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/stimulate_submission/combine/submission2.csv"

SOURCE_RANGES = {
    "source0": [(0, 249), (343, 641)],
    "source1": [(250, 262), (642, 942)],
    "source2": [(943, 1187)],
    "source3": [(263, 342), (1188, 1487)],
}

def normalize_name_to_index(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    m = re.search(r"ct_scan_(\d+)", s)
    if m:
        return int(m.group(1))

    if s.isdigit():
        return int(s)

    return None


def read_positive_set(csv_path: str):
    pos_set = set()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return pos_set

    start_idx = 0
    first_row = rows[0]
    if len(first_row) > 0:
        first_cell = str(first_row[0]).strip().lower()
        if first_cell in {"npy_name", "filename", "file_name", "name"}:
            start_idx = 1

    for row in rows[start_idx:]:
        if not row:
            continue
        idx = normalize_name_to_index(row[0])
        if idx is not None:
            pos_set.add(idx)

    return pos_set


def in_ranges(x: int, ranges):
    for l, r in ranges:
        if l <= x <= r:
            return True
    return False


def filter_by_ranges(idx_set, ranges):
    return {x for x in idx_set if in_ranges(x, ranges)}


def idx_to_name(idx: int):
    return f"ct_scan_{idx}.npy"

c2m5_best = read_positive_set(CSV_PATHS["c2m5_best"])
med7 = read_positive_set(CSV_PATHS["med7"])
med4 = read_positive_set(CSV_PATHS["med4"])
med5 = read_positive_set(CSV_PATHS["med5"])
tta5 = read_positive_set(CSV_PATHS["tta5"])
tta6 = read_positive_set(CSV_PATHS["tta6"])
trans3 = read_positive_set(CSV_PATHS["trans3"])

print(f"[INFO] c2m5_best: {len(c2m5_best)}")
print(f"[INFO] med7     : {len(med7)}")
print(f"[INFO] med4     : {len(med4)}")
print(f"[INFO] med5     : {len(med5)}")
print(f"[INFO] tta5     : {len(tta5)}")
print(f"[INFO] tta6     : {len(tta6)}")
print(f"[INFO] trans3   : {len(trans3)}")

# source0: med7 ∪ c2m5_best
source0_pos_raw = med7 | c2m5_best

# source1: tta6 ∩ med5
source1_pos_raw = tta6 & med5

# source2: tta5 ∩ med7
source2_pos_raw = tta5 & med7

# source3: tta6 ∩ med4 ∩ trans3
source3_pos_raw = tta6 & med4 & trans3

source0_pos = filter_by_ranges(source0_pos_raw, SOURCE_RANGES["source0"])
source1_pos = filter_by_ranges(source1_pos_raw, SOURCE_RANGES["source1"])
source2_pos = filter_by_ranges(source2_pos_raw, SOURCE_RANGES["source2"])
source3_pos = filter_by_ranges(source3_pos_raw, SOURCE_RANGES["source3"])

print(f"[INFO] source0_pos after range filter: {len(source0_pos)}")
print(f"[INFO] source1_pos after range filter: {len(source1_pos)}")
print(f"[INFO] source2_pos after range filter: {len(source2_pos)}")
print(f"[INFO] source3_pos after range filter: {len(source3_pos)}")

final_pos = source0_pos | source1_pos | source2_pos | source3_pos
final_pos = sorted(final_pos)

print(f"[INFO] final merged covid count: {len(final_pos)}")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["npy_name"])
    for idx in final_pos:
        writer.writerow([idx_to_name(idx)])

print(f"[OK] 已保存最终csv: {OUT_CSV}")

base_dir = os.path.dirname(OUT_CSV)

for src_name, src_set in [
    ("source0", source0_pos),
    ("source1", source1_pos),
    ("source2", source2_pos),
    ("source3", source3_pos),
]:
    out_path = os.path.join(base_dir, f"{src_name}_latest.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["npy_name"])
        for idx in sorted(src_set):
            writer.writerow([idx_to_name(idx)])
    print(f"[OK] 已保存 {src_name}: {out_path}")