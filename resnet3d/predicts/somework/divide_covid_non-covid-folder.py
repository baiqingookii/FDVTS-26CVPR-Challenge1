import os
import re
import pandas as pd

# 总目录
# root_dir = "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/valid_notta-test_tta"
root_dir = "/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/notta"


def extract_num(name):
    m = re.search(r"ct_scan_(\d+)\.npy$", str(name))
    return int(m.group(1)) if m else float("inf")


def process_one_csv(input_csv):
    # 输出目录：原 csv 所在文件夹
    out_dir = os.path.dirname(input_csv)

    # 读取 csv 文件名（不带路径）
    input_name = os.path.basename(input_csv)         # 例如 predict_with_best_thr.csv
    input_stem = os.path.splitext(input_name)[0]     # 例如 predict_with_best_thr

    # 从文件名中提取后缀
    suffix = None
    if input_stem.endswith("best_thr"):
        suffix = "best_thr"
    else:
        m = re.search(r"(thr_[^/\\]+)$", input_stem)
        if m:
            suffix = m.group(1)

    # 兜底
    if suffix is None:
        suffix = "result"

    # 输出文件名
    covid_out = os.path.join(out_dir, f"covid_{suffix}.csv")
    noncovid_out = os.path.join(out_dir, f"non-covid_{suffix}.csv")

    # 读取原始结果
    df = pd.read_csv(input_csv)

    # 检查必要列
    required_cols = {"id", "pred_class"}
    if not required_cols.issubset(df.columns):
        print(f"[Skip] 缺少必要列: {input_csv}")
        return

    # 只保留文件名，如 ct_scan_1062.npy
    df["ct_scan"] = df["id"].apply(lambda x: os.path.basename(str(x)))

    # 分组
    covid_df = df[df["pred_class"] == "covid"][["ct_scan"]].copy()
    noncovid_df = df[df["pred_class"] == "non-covid"][["ct_scan"]].copy()

    # 排序
    covid_df = covid_df.sort_values(by="ct_scan", key=lambda s: s.map(extract_num))
    noncovid_df = noncovid_df.sort_values(by="ct_scan", key=lambda s: s.map(extract_num))

    # 输出：每行一个 ct_scan，不带索引，不带表头
    covid_df.to_csv(covid_out, index=False, header=False)
    noncovid_df.to_csv(noncovid_out, index=False, header=False)

    print(f"[Done] {input_csv}")
    print("       covid     ->", covid_out, "num =", len(covid_df))
    print("       non-covid ->", noncovid_out, "num =", len(noncovid_df))


def main():
    # 需要处理的目标文件名
    target_files = ["predict_with_best_thr.csv", "predict_with_thr_0.5.csv"]

    # 遍历 tta 下各个子文件夹
    for subname in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, subname)
        if not os.path.isdir(subdir):
            continue

        for fname in target_files:
            input_csv = os.path.join(subdir, fname)
            if os.path.isfile(input_csv):
                process_one_csv(input_csv)
            else:
                print(f"[Skip] 不存在: {input_csv}")


if __name__ == "__main__":
    main()