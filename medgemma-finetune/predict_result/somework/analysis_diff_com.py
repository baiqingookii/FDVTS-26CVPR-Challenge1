#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import re
from collections import OrderedDict

INPUT_DIRS = [
    # "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/transformer_block/3d_ori-and-1b_dataset_cls_lr5e-5_scale0.7-1_24_448_448_12_24_checkpoint-27",
    # "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/transformer_2vision_block/3d_ori-and-1b_dataset_cls_lr5e-5_scale0.7-1_24_448_448_12_24_checkpoint-27",
    # "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/flatten_block/3d_ori-and-1b_dataset_cls_lr5e-5_scale0.7-1_24_448_448_12_24_checkpoint-27",
    "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/test_lungex/3d_1_dataset_mean_lr5e-5_scale0.7-1_area-checkpoint-2132",
    "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/test_lungex/3d_1_dataset_mean_lr5e-5_scale0.7-1_ori_and_1b-checkpoint-972",
    "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/test_lungex/3d_1_dataset1_lse_lr5e-5_scale0.7-1_64_448_448_12_32-checkpoint-779",
    "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/test_lungex/3d_continue-checkpoint-1566",
    "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/test_lungex/3d-checkpoint-1312",
    "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/test_lungex/128*256*256_3d_1_ori-and-1b_dataset1_mean_lr5e-5_scale0.7-1-checkpoint-1836",
    "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/test_lungex/real_24*448*448_3d_1_ori-and-1b_dataset1_mean_lr5e-5_scale0.7-1-checkpoint-1105",
]

OUTPUT_DIR = "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/somework/analysis_csv/cvpr/analysis_diff_com"


def natural_key(s):
    """
    让 ct_scan_2.npy 排在 ct_scan_10.npy 前面
    """
    m = re.match(r"^(.*?)(\d+)(\D*)$", s)
    if m:
        return (m.group(1), int(m.group(2)), m.group(3))
    return (s, -1, "")


def read_single_column_csv(csv_path):
    """
    读取单列csv，返回去重后的列表
    """
    items = []
    seen = set()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            val = row[0].strip()
            if not val:
                continue
            if val not in seen:
                seen.add(val)
                items.append(val)

    return items


def build_folder_to_csv(input_dirs, target_name):
    folder_to_csv = OrderedDict()

    for folder_path in input_dirs:
        if not os.path.isdir(folder_path):
            print(f"[WARN] 目录不存在，跳过: {folder_path}")
            continue

        csv_path = os.path.join(folder_path, target_name)
        if not os.path.isfile(csv_path):
            print(f"[WARN] 未找到 {target_name}，跳过: {folder_path}")
            continue

        # 用 basename 作为展示名；若重名则补全路径避免覆盖
        display_name = os.path.basename(folder_path.rstrip("/"))
        if display_name in folder_to_csv:
            display_name = folder_path.rstrip("/")

        folder_to_csv[display_name] = csv_path

    return folder_to_csv


def analyze_csv_group(input_dirs, target_name, output_name):
    folder_to_csv = build_folder_to_csv(input_dirs, target_name)

    if not folder_to_csv:
        print(f"[WARN] 没有找到任何 {target_name}")
        return

    folder_to_set = OrderedDict()
    for folder, csv_path in folder_to_csv.items():
        items = read_single_column_csv(csv_path)
        folder_to_set[folder] = set(items)

    all_sets = list(folder_to_set.values())

    # 1) 所有给定文件夹共同拥有的 npy
    common_set = set.intersection(*all_sets) if all_sets else set()
    common_list = sorted(common_set, key=natural_key)

    # 2) 每个文件夹中“不在共同交集里”的 npy
    folder_to_non_common = OrderedDict()
    for folder, current_set in folder_to_set.items():
        non_common_set = current_set - common_set
        folder_to_non_common[folder] = sorted(non_common_set, key=natural_key)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        # 参与统计的文件夹
        writer.writerow(["section", "folder", "value"])
        for folder, csv_path in folder_to_csv.items():
            writer.writerow(["included_folder", folder, csv_path])

        writer.writerow([])

        # 共同部分
        writer.writerow(["section", "count", "value"])
        writer.writerow(["common_npy", len(common_list), ""])
        for item in common_list:
            writer.writerow(["common_npy_item", "", item])

        writer.writerow([])

        # 每个文件夹中的“非共同部分”
        writer.writerow(["section", "folder", "count", "value"])
        for folder, item_list in folder_to_non_common.items():
            writer.writerow(["non_common_npy", folder, len(item_list), ""])
            for item in item_list:
                writer.writerow(["non_common_npy_item", folder, "", item])
            writer.writerow([])

    print(f"[OK] {target_name} -> {output_path}")
    print(f"     参与文件夹数: {len(folder_to_csv)}")
    print(f"     共同npy数: {len(common_list)}")
    for folder, item_list in folder_to_non_common.items():
        print(f"     {folder} 非共同npy数: {len(item_list)}")


def main():
    analyze_csv_group(
        INPUT_DIRS,
        # "covid_thr_0.5.csv",
        # transformer&flatten block
        # "thr_0.5_covid.csv",
        # medsiglip
        "thr0.5_covid.csv",
        "medsiglip_summary_covid_thr_0.5.csv"
        # transformer&flatten block
        # "trans_flat_summary_covid_thr_0.5.csv"
    )

    analyze_csv_group(
        INPUT_DIRS,
        # "covid_best_thr.csv",
        "best_thr_covid.csv",
        # medsiglip
        "medsiglip_summary_covid_best_thr.csv"
        # transformer&flatten block
        # "trans_flat_summary_covid_best_thr.csv"
    )


if __name__ == "__main__":
    main()