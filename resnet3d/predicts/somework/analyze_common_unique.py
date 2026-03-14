
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import re
from collections import OrderedDict

ROOT_DIR = "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/notta"


def natural_key(s):
    """
    让 ct_scan_2.npy 排在 ct_scan_10.npy 前面
    """
    m = re.match(r"^(.*?)(\d+)(\D*)$", s)
    if m:
        return (m.group(1), int(m.group(2)), m.group(3))
    return (s, -1, "")


def read_single_column_csv(csv_path):
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


def find_target_csvs(root_dir, target_name):
    """
    遍历 root_dir 下一级子文件夹，寻找 target_name
    返回: OrderedDict {folder_name: csv_path}
    """
    result = OrderedDict()

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        target_path = os.path.join(folder_path, target_name)
        if os.path.isfile(target_path):
            result[folder_name] = target_path

    return result


def analyze_csv_group(root_dir, target_name, output_name):
    folder_to_csv = find_target_csvs(root_dir, target_name)

    if not folder_to_csv:
        print(f"[WARN] 没有找到任何 {target_name}")
        return

    folder_to_set = OrderedDict()
    for folder, csv_path in folder_to_csv.items():
        items = read_single_column_csv(csv_path)
        folder_to_set[folder] = set(items)

    all_sets = list(folder_to_set.values())

    # 1) 所有文件夹共同拥有的 npy
    common_set = set.intersection(*all_sets) if all_sets else set()
    common_list = sorted(common_set, key=natural_key)

    # 2) 每个文件夹中“不在共同交集里”的 npy
    #    即：当前文件夹集合 - common_set
    folder_to_non_common = OrderedDict()
    for folder, current_set in folder_to_set.items():
        non_common_set = current_set - common_set
        folder_to_non_common[folder] = sorted(non_common_set, key=natural_key)

    output_path = os.path.join(root_dir, output_name)

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
    print(f"     文件夹数: {len(folder_to_csv)}")
    print(f"     共同npy数: {len(common_list)}")
    for folder, item_list in folder_to_non_common.items():
        print(f"     {folder} 非共同npy数: {len(item_list)}")


def main():
    analyze_csv_group(
        ROOT_DIR,
        "covid_thr_0.5.csv",
        "summary_notta_covid_thr_0.5.csv"
    )

    analyze_csv_group(
        ROOT_DIR,
        "covid_best_thr.csv",
        "summary_notta_covid_best_thr.csv"
    )



if __name__ == "__main__":
    main()