#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from safetensors.torch import load_file as safe_load
# ---- import your dataset + collate from the same repo you used in training ----
import sys
medgemma_ROOT = "/remote-home/share/25-jianfabai/medgemma-finetune"
sys.path.insert(0, medgemma_ROOT)
from dataset import NPYVolumeDataset, volume_collate_fn  # must match training

from transformers import AutoImageProcessor, AutoModelForImageClassification


class SliceMeanPoolClassifier(nn.Module):
    def __init__(self, base_model, chunk_size=3):
        super().__init__()
        self.base = base_model
        self.chunk_size = chunk_size

    def forward(self, pixel_values=None, labels=None):
        # pixel_values: (B,S,C,H,W)
        B, S, C, H, W = pixel_values.shape
        flat = pixel_values.reshape(B * S, C, H, W)

        logits_chunks = []
        for i in range(0, B * S, self.chunk_size):
            out = self.base(pixel_values=flat[i:i + self.chunk_size])
            logits_chunks.append(out.logits)

        logits = torch.cat(logits_chunks, dim=0)      # (B*S,2)
        logits = logits.reshape(B, S, -1).mean(dim=1) # (B,2)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def load_csv_as_table(csv_path: str, label: int):
    """
    Accept your CSV schema:
      - ct_scan_name (e.g., ct_scan_0 / ct_scan_0.npy / subdir/ct_scan_0)
      - data_centre  (e.g., 0/1/2/3)
    """
    df = pd.read_csv(csv_path)

    # ---- path/name column ----
    if "ct_scan_name" in df.columns:
        name_col = "ct_scan_name"
    else:
        # fallback to common names
        name_col = None
        for c in ["path", "filepath", "file_path", "file", "filename", "npy", "npy_path"]:
            if c in df.columns:
                name_col = c
                break
        if name_col is None:
            raise ValueError(f"[{csv_path}] cannot find scan-name/path column. Columns={list(df.columns)}")

    # ---- center column ----
    if "data_centre" in df.columns:
        center_col = "data_centre"
    elif "data_center" in df.columns:
        center_col = "data_center"
    elif "center" in df.columns:
        center_col = "center"
    else:
        raise ValueError(f"[{csv_path}] cannot find center column. Columns={list(df.columns)}")

    out = pd.DataFrame({
        "path_raw": df[name_col].astype(str),
        "data_center": df[center_col].astype(str),
        "label": int(label),
    })
    return out

def _resolve_path(p: str, root_if_relative: str):
    """
    Accept:
      - absolute path
      - relative path
      - bare name like 'ct_scan_0'  (auto add '.npy')
      - name like 'ct_scan_0.npy'
    """
    if p is None:
        return None
    p = str(p).strip()

    # if no extension, assume npy
    if not p.lower().endswith(".npy"):
        p_npy = p + ".npy"
    else:
        p_npy = p

    # 1) absolute path exists
    if os.path.isfile(p_npy):
        return p_npy

    # 2) root + basename
    cand = os.path.join(root_if_relative, os.path.basename(p_npy))
    if os.path.isfile(cand):
        return cand

    # 3) root + relative path
    cand2 = os.path.join(root_if_relative, p_npy)
    if os.path.isfile(cand2):
        return cand2

    # return best guess for debugging
    return cand2

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/covid1b")
    parser.add_argument("--val_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/non-covid1b")
    # parser.add_argument("--val_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/covid1b_64_448_448")
    # parser.add_argument("--val_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/non-covid1b_64_448_448")

    parser.add_argument("--csv_pos", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/validation_covid1.csv")  # validation_covid1.csv
    parser.add_argument("--csv_neg", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/validation_non_covid.csv")  # validation_non_covid.csv

    parser.add_argument("--model_id", type=str, default="/root/medsiglip-448")  # for image_processor stats
    # parser.add_argument("--model_dir", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_mean_lr5e-5_scale0.7-1/checkpoint-1728")  # your fine-tuned checkpoint dir
    parser.add_argument("--model_dir", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_mean_lr5e-5_scale0.7-1_ori_and_1b/checkpoint-972")  # your fine-tuned checkpoint dir

    # 最下面的save_path记得也改一下
    parser.add_argument("--num_slices", type=int, default=12)
    # parser.add_argument("--num_slices", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--assume_uint8_0_255", action="store_true", default=True)
    parser.add_argument("--mmap", action="store_true", default=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--thr_steps", type=int, default=1001)  # threshold sweep granularity
    #如果不用滑动窗口就删
    # parser.add_argument("--mw", action="store_true", default=True,
    #                     help="multi-window eval on slice dimension (3 windows: first/mid/last)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}  fp16={args.fp16}")

    # ---- read CSVs ----
    df_pos = load_csv_as_table(args.csv_pos, label=1)
    df_neg = load_csv_as_table(args.csv_neg, label=0)
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    # resolve actual npy paths using corresponding roots
    resolved = []
    for i, row in df.iterrows():
        root = args.val_covid_root if row["label"] == 1 else args.val_noncovid_root
        rp = _resolve_path(row["path_raw"], root)
        resolved.append(rp)
    df["path"] = resolved

    missing = df[~df["path"].map(os.path.isfile)]
    if len(missing) > 0:
        print("[error] Some npy paths do not exist. Showing first 20:")
        print(missing.head(20)[["path_raw", "path", "label", "data_center"]].to_string(index=False))
        raise FileNotFoundError("Some npy paths are missing. Fix CSV paths or roots first.")

    print(f"[data] loaded {len(df)} samples from CSVs.")
    print(df["label"].value_counts().to_string())
    print(df["data_center"].value_counts().to_string())

    # ---- build the SAME val dataset pipeline as training ----
    image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    size = image_processor.size["height"]
    mean = image_processor.image_mean
    std = image_processor.image_std
    print(f"[pre] size={size} mean={mean} std={std}")

    # Build a val dataset from the folders (same as training),
    # then map path->index to pick exactly the CSV-listed samples.
    val_ds_full = NPYVolumeDataset(
        args.val_covid_root,
        args.val_noncovid_root,
        num_slices=args.num_slices,
        out_size=size,
        split="val",
        mean=mean,
        std=std,
        mmap=args.mmap,
        assume_uint8_0_255=args.assume_uint8_0_255,
    )

    # Try to access underlying file list in NPYVolumeDataset
    # Common patterns: .paths / .files / .samples
    cand_attrs = ["paths", "files", "samples", "file_paths", "npy_paths"]
    file_list = None
    for a in cand_attrs:
        if hasattr(val_ds_full, a):
            file_list = getattr(val_ds_full, a)
            break
    if file_list is None:
        raise AttributeError(
            "Cannot find file list attribute in NPYVolumeDataset. "
            "Please open dataset.py and expose a list of file paths (e.g., self.paths)."
        )

    # normalize to absolute paths
    file_list = [os.path.abspath(p) for p in list(file_list)]
    path2idx = {p: i for i, p in enumerate(file_list)}

    # create subset indices in the order of df
    indices = []
    for p in df["path"].tolist():
        ap = os.path.abspath(p)
        if ap not in path2idx:
            raise KeyError(f"CSV path not found in val_ds_full file list: {ap}")
        indices.append(path2idx[ap])

    class SubsetDS(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base = base
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.base[self.indices[i]]

    val_ds = SubsetDS(val_ds_full, indices)

    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=volume_collate_fn,
        pin_memory=True,
    )

    # ---- load model (fine-tuned weights) ----
    id2label = {0: "non-covid", 1: "covid"}
    label2id = {"non-covid": 0, "covid": 1}

    # base_model = AutoModelForImageClassification.from_pretrained(
    #     args.model_dir,  # load your fine-tuned checkpoint
    #     num_labels=2,
    #     id2label=id2label,
    #     label2id=label2id,
    #     problem_type="single_label_classification",
    #     ignore_mismatched_sizes=True,
    # )

    # 1) 用 medsiglip 的 config/processor 初始化结构（一定要是图像模型）
    base_model = AutoModelForImageClassification.from_pretrained(
        args.model_id,                  # 例如 /root/medsiglip-448
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    model = SliceMeanPoolClassifier(base_model).to(device)

    # 2) 把 checkpoint 的权重覆盖进去（checkpoint_dir 里有 model.safetensors）
    ckpt_path = os.path.join(args.model_dir, "model.safetensors")  # args.model_dir = .../checkpoint-697
    state = safe_load(ckpt_path)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load] missing keys:", len(missing))
    print("[load] unexpected keys:", len(unexpected))
    if len(unexpected) > 0:
        print("  unexpected examples:", unexpected[:20])
    if len(missing) > 0:
        print("  missing examples:", missing[:20])

    model.eval()

    # ---- inference ----
    all_prob = []
    all_label = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].cpu().numpy().astype(int)

        if args.fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(pixel_values=pixel_values)
        else:
            out = model(pixel_values=pixel_values)

        logits = out["logits"]
        prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()  # prob(covid)

        all_prob.append(prob)
        all_label.append(labels)

    # move window
    # for batch in loader:
    #     pixel_values = batch["pixel_values"].to(device, non_blocking=True)  # (B,S,3,H,W)
    #     labels = batch["labels"].cpu().numpy().astype(int)

    #     B, S, C, H, W = pixel_values.shape
    #     s12 = args.num_slices  # 训练时的12
    #     assert S == 2 * s12, f"Expect val S={2*s12}, got {S}. (Check dataset.py val behavior)"

    #     # 3个窗口：前12 / 中12 / 后12
    #     win_first = pixel_values[:, 0:s12]           # (B,12,3,H,W)
    #     mid_start = (S - s12) // 2                   # 24->6
    #     win_mid   = pixel_values[:, mid_start:mid_start + s12]  # (B,12,3,H,W)
    #     win_last  = pixel_values[:, S - s12:S]       # (B,12,3,H,W)

    #     if args.fp16 and device.type == "cuda":
    #         with torch.autocast(device_type="cuda", dtype=torch.float16):
    #             if args.mw:
    #                 logit1 = model(pixel_values=win_first)["logits"]
    #                 logit2 = model(pixel_values=win_mid)["logits"]
    #                 logit3 = model(pixel_values=win_last)["logits"]
    #                 logits = (logit1 + logit2 + logit3) / 3.0
    #             else:
    #                 logits = model(pixel_values=pixel_values)["logits"]
    #     else:
    #         if args.mw:
    #             logit1 = model(pixel_values=win_first)["logits"]
    #             logit2 = model(pixel_values=win_mid)["logits"]
    #             logit3 = model(pixel_values=win_last)["logits"]
    #             logits = (logit1 + logit2 + logit3) / 3.0
    #         else:
    #             logits = model(pixel_values=pixel_values)["logits"]

        # prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()  # prob(covid)

        # all_prob.append(prob)
        # all_label.append(labels)


    all_prob = np.concatenate(all_prob, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    assert len(all_prob) == len(df)

    # ---- metrics: best acc by threshold sweep + AUC ----
    auc = roc_auc_score(all_label, all_prob)

    # threshold sweep [0,1]
    thr_grid = np.linspace(0.0, 1.0, args.thr_steps)
    best_acc = -1.0
    best_thr = 0.5
    for thr in thr_grid:
        pred = (all_prob >= thr).astype(int)
        acc = accuracy_score(all_label, pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    # predictions at best threshold
    pred_best = (all_prob >= best_thr).astype(int)

    # per-center F1 and macro_f1 across centers
    df_out = df.copy()
    df_out["prob_covid"] = all_prob
    df_out["pred"] = pred_best

    centers = sorted(df_out["data_center"].unique().tolist())
    center_f1 = {}
    for c in centers:
        sub = df_out[df_out["data_center"] == c]
        # handle edge case: if a center has only one class, f1_score may warn; set zero_division=0
        center_f1[c] = f1_score(sub["label"].values, sub["pred"].values, zero_division=0)

    macro_f1 = float(np.mean([center_f1[c] for c in centers])) if len(centers) > 0 else 0.0

    # misclassified samples
    wrong = df_out[df_out["pred"] != df_out["label"]].copy()
    # wrong = wrong.sort_values("prob_covid", ascending=False)
    wrong["_center_int"] = pd.to_numeric(wrong["data_center"], errors="coerce")
    wrong = wrong.sort_values(["_center_int", "path"], ascending=[True, True])

    # ---- print summary ----
    print("\n================= SUMMARY =================")
    print(f"Best ACC: {best_acc:.6f}")
    print(f"Best THR: {best_thr:.4f}")
    print(f"ROC-AUC : {auc:.6f}")
    print("\nPer-center F1:")
    for c in centers:
        print(f"  {c}: {center_f1[c]:.6f}")
    print(f"\nMacro-F1 (over centers): {macro_f1:.6f}")

    # print("\n================= MISCLASSIFIED =================")
    # if len(wrong) == 0:
    #     print("No misclassified samples.")
    # else:
    #     show_cols = ["path", "data_center", "label", "pred", "prob_covid"]
    #     print(wrong[show_cols].to_string(index=False))
    print("\n================= MISCLASSIFIED (grouped by center) =================")
    if len(wrong) == 0:
        print("No misclassified samples.")
    else:
        show_cols = ["data_center", "path", "label", "pred", "prob_covid"]
        with pd.option_context("display.max_colwidth", 200, "display.width", 220, "display.expand_frame_repr", False):
            for c, sub in wrong.groupby("data_center", sort=True):
                print(f"\n--- center {c} (n={len(sub)}) ---")
                print(sub[show_cols].to_string(index=False))

    # also save a csv for further analysis
    # save_path = os.path.join(args.model_dir, "val_csv_predictions_mean.csv")
    save_path = os.path.join(args.model_dir, "val_csv_predictions_24_mean.csv")
    df_out.to_csv(save_path, index=False)
    print(f"\n[done] saved per-sample predictions to: {save_path}")


if __name__ == "__main__":
    main()