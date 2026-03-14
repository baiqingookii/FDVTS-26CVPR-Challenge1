#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import re
from collections import Counter, defaultdict



# SUMMARY_FILES = [
#     ("/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/tta/summary_notta_covid_best_thr.csv", "best_thr"),
#     ("/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/tta/summary_notta_covid_thr_0.5.csv", "thr_0.5"),
# ]

# 1
# SUMMARY_FILES = [
#     ("/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/valid_notta-test_tta/summary_notta_covid_best_thr.csv", "best_thr"),
#     ("/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/valid_notta-test_tta/summary_notta_covid_thr_0.5.csv", "thr_0.5"),
# ]

SUMMARY_FILES = [
    ("/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/notta/summary_notta_covid_best_thr.csv", "best_thr"),
    ("/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/notta/summary_notta_covid_thr_0.5.csv", "thr_0.5"),
]

# 2
# OUTPUT_ROOT = "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/tta/summary"
# OUTPUT_ROOT = "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/valid_notta-test_tta/summary"
OUTPUT_ROOT = "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/notta/summary"


def natural_key(s):
    """
    让 ct_scan_2.npy 排在 ct_scan_10.npy 前面
    """
    m = re.match(r"^(.*?)(\d+)(\D*)$", s)
    if m:
        return (m.group(1), int(m.group(2)), m.group(3))
    return (s, -1, "")


def parse_summary_csv(summary_csv_path):
    included_folders = []
    common_items = []
    non_common_by_folder = defaultdict(list)

    with open(summary_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            section = row[0].strip()

            if section == "included_folder":
                included_folders.append(row[1].strip())

            elif section == "common_npy_item":
                # 格式: common_npy_item,,ct_scan_0.npy
                if len(row) >= 3:
                    item = row[2].strip()
                    if item:
                        common_items.append(item)

            elif section == "non_common_npy_item":
                # 格式: non_common_npy_item,folder,,ct_scan_0.npy
                if len(row) >= 4:
                    folder = row[1].strip()
                    item = row[3].strip()
                    if folder and item:
                        non_common_by_folder[folder].append(item)

    return included_folders, common_items, non_common_by_folder


def vote_from_summary(summary_csv_path, output_dir):
    included_folders, common_items, non_common_by_folder = parse_summary_csv(summary_csv_path)

    if not included_folders:
        print(f"[WARN] 未解析到 included_folder: {summary_csv_path}")
        return

    total_models = len(included_folders)

    # 统计票数
    vote_counter = Counter()

    # common_npy 直接记 total_models 票
    for item in common_items:
        vote_counter[item] = total_models

    # non_common_npy_item 每出现一次 +1
    for folder, items in non_common_by_folder.items():
        for item in items:
            vote_counter[item] += 1

    # 建立票数桶
    vote_buckets = defaultdict(list)
    for item, vote in vote_counter.items():
        vote_buckets[vote].append(item)

    os.makedirs(output_dir, exist_ok=True)

    # 写 meta 信息
    meta_path = os.path.join(output_dir, "vote_summary_info.csv")
    with open(meta_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["summary_csv", summary_csv_path])
        writer.writerow(["included_folder_count", total_models])

        for vote in sorted(vote_buckets.keys(), reverse=True):
            writer.writerow([f"vote_{vote}_count", len(vote_buckets[vote])])

    # 分票数输出
    for vote in range(total_models, 0, -1):
        items = sorted(vote_buckets.get(vote, []), key=natural_key)
        out_csv = os.path.join(output_dir, f"vote_{vote}.csv")
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["npy_name"])
            for item in items:
                writer.writerow([item])

    print(f"[OK] {summary_csv_path}")
    print(f"     included_folder数: {total_models}")
    for vote in range(total_models, 0, -1):
        print(f"     vote_{vote}: {len(vote_buckets.get(vote, []))}")
    print(f"     输出目录: {output_dir}")


def main():
    for summary_csv_path, subfolder_name in SUMMARY_FILES:
        output_dir = os.path.join(OUTPUT_ROOT, subfolder_name)
        vote_from_summary(summary_csv_path, output_dir)


if __name__ == "__main__":
    main()