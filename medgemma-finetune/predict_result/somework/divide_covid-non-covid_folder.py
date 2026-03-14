import os
import re
import pandas as pd

# 根目录
root_dir = "/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/test_lungex"

def extract_num(name):
    """
    从 ct_scan_1062.npy 中提取数字 1062，用于数值排序
    """
    m = re.search(r"ct_scan_(\d+)\.npy$", str(name))
    return int(m.group(1)) if m else float("inf")


def get_suffix_from_filename(filename):
    """
    根据输入文件名识别后缀：
    infer_predictions_best_thr.csv -> best_thr
    infer_predictions_thr0.5.csv   -> thr0.5
    infer_predictions_thr_0.5.csv  -> thr0.5
    """
    stem = os.path.splitext(os.path.basename(filename))[0]

    if stem.endswith("best_thr"):
        return "best_thr"
    if re.search(r"thr0\.?5$", stem):
        return "thr0.5"
    if re.search(r"thr_0\.?5$", stem):
        return "thr0.5"

    return None


def process_one_csv(input_csv):
    out_dir = os.path.dirname(input_csv)
    suffix = get_suffix_from_filename(input_csv)

    if suffix is None:
        print(f"[Skip] 无法识别后缀: {input_csv}")
        return

    covid_out = os.path.join(out_dir, f"{suffix}_covid.csv")
    noncovid_out = os.path.join(out_dir, f"{suffix}_non-covid.csv")

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[Skip] 读取失败: {input_csv} | {e}")
        return

    required_cols = {"filename", "pred_label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"[Skip] 缺少必要列 {missing_cols}: {input_csv}")
        return

    # 统一小写，避免大小写问题
    pred_label = df["pred_label"].astype(str).str.strip().str.lower()

    covid_df = df[pred_label == "covid"][["filename"]].copy()
    noncovid_df = df[pred_label.isin(["non-covid", "non_covid", "noncovid"])][["filename"]].copy()

    covid_df = covid_df.sort_values(by="filename", key=lambda s: s.map(extract_num))
    noncovid_df = noncovid_df.sort_values(by="filename", key=lambda s: s.map(extract_num))

    # 输出每行一个文件名，不带表头、不带索引
    covid_df.to_csv(covid_out, index=False, header=False)
    noncovid_df.to_csv(noncovid_out, index=False, header=False)

    print(f"[Done] {input_csv}")
    print(f"       covid     -> {covid_out}  num = {len(covid_df)}")
    print(f"       non-covid -> {noncovid_out}  num = {len(noncovid_df)}")


def main():
    target_files = {
        "infer_predictions_best_thr.csv",
        "infer_predictions_thr0.5.csv",
        "infer_predictions_thr_0.5.csv",   # 兼容另一种命名
    }

    for subname in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, subname)

        if not os.path.isdir(subdir):
            continue

        found_any = False
        for fname in target_files:
            fpath = os.path.join(subdir, fname)
            if os.path.isfile(fpath):
                process_one_csv(fpath)
                found_any = True

        if not found_any:
            print(f"[Skip] 未找到目标 csv: {subdir}")


if __name__ == "__main__":
    main()