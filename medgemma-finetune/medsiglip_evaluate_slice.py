import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from safetensors.torch import load_file as safe_load

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

    def forward(self, pixel_values=None, labels=None, return_slice_probs=False):
        """
        pixel_values: (B,S,C,H,W)
        return:
          - logits: (B,2)               总体logits（对S个slice的logits做mean）
          - slice_probs: (B,S)          每个slice属于covid(class=1)的概率
        """
        B, S, C, H, W = pixel_values.shape
        flat = pixel_values.reshape(B * S, C, H, W)

        logits_chunks = []
        for i in range(0, B * S, self.chunk_size):
            out = self.base(pixel_values=flat[i:i + self.chunk_size])
            logits_chunks.append(out.logits)

        slice_logits = torch.cat(logits_chunks, dim=0)            # (B*S,2)
        slice_logits = slice_logits.reshape(B, S, -1)             # (B,S,2)

        logits = slice_logits.mean(dim=1)                         # (B,2)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        out_dict = {"loss": loss, "logits": logits}

        if return_slice_probs:
            slice_probs = torch.softmax(slice_logits, dim=-1)[..., 1]  # (B,S)
            out_dict["slice_probs"] = slice_probs

        return out_dict


def load_csv_as_table(csv_path: str, label: int):
    df = pd.read_csv(csv_path)

    # ---- path/name column ----
    if "ct_scan_name" in df.columns:
        name_col = "ct_scan_name"
    else:
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

    return cand2


def get_model_save_dir(model_dir: str):
    """
    例如:
    /.../save_model/3d_xxx/checkpoint-972
    -> 3d_xxx
    """
    model_name = os.path.basename(os.path.dirname(model_dir.rstrip("/")))
    out_dir = os.path.join(
        "/remote-home/share/25-jianfabai/medgemma-finetune/log/medsiglip_valid_slice_log",
        model_name
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, model_name


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/covid1b")
    parser.add_argument("--val_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/non-covid1b")

    parser.add_argument("--csv_pos", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/validation_covid1.csv")
    parser.add_argument("--csv_neg", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/validation_non_covid.csv")

    parser.add_argument("--model_id", type=str, default="/root/medsiglip-448")
    parser.add_argument("--model_dir", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_mean_lr5e-5_scale0.7-1_ori_and_1b/checkpoint-972")

    parser.add_argument("--num_slices", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--assume_uint8_0_255", action="store_true", default=True)
    parser.add_argument("--mmap", action="store_true", default=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--thr_steps", type=int, default=1001)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}  fp16={args.fp16}")

    out_dir, model_name = get_model_save_dir(args.model_dir)
    print(f"[save] model_name={model_name}")
    print(f"[save] output_dir={out_dir}")

    # ---- read CSVs ----
    df_pos = load_csv_as_table(args.csv_pos, label=1)
    df_neg = load_csv_as_table(args.csv_neg, label=0)
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    # resolve actual npy paths using corresponding roots
    resolved = []
    for _, row in df.iterrows():
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

    base_model = AutoModelForImageClassification.from_pretrained(
        args.model_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    model = SliceMeanPoolClassifier(base_model).to(device)

    ckpt_path = os.path.join(args.model_dir, "model.safetensors")
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
    all_slice_probs = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].cpu().numpy().astype(int)

        if args.fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(pixel_values=pixel_values, return_slice_probs=True)
        else:
            out = model(pixel_values=pixel_values, return_slice_probs=True)

        logits = out["logits"]                                   # (B,2)
        slice_probs = out["slice_probs"]                         # (B,S)

        prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        slice_prob_np = slice_probs.detach().cpu().numpy()

        all_prob.append(prob)
        all_label.append(labels)
        all_slice_probs.append(slice_prob_np)

    all_prob = np.concatenate(all_prob, axis=0)                 # (N,)
    all_label = np.concatenate(all_label, axis=0)               # (N,)
    all_slice_probs = np.concatenate(all_slice_probs, axis=0)   # (N,S)

    assert len(all_prob) == len(df)
    assert all_slice_probs.shape[0] == len(df)

    num_actual_slices = all_slice_probs.shape[1]
    print(f"[info] detected slice count per sample = {num_actual_slices}")

    # ---- metrics: best acc by threshold sweep + AUC ----
    auc = roc_auc_score(all_label, all_prob)

    thr_grid = np.linspace(0.0, 1.0, args.thr_steps)
    best_acc = -1.0
    best_thr = 0.5
    for thr in thr_grid:
        pred = (all_prob >= thr).astype(int)
        acc = accuracy_score(all_label, pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    pred_best = (all_prob >= best_thr).astype(int)

    # ---- pack output ----
    df_out = df.copy()
    df_out["prob_covid"] = all_prob
    df_out["pred"] = pred_best

    # 添加每个切片的概率列
    for s in range(num_actual_slices):
        df_out[f"slice_prob_{s:02d}"] = all_slice_probs[:, s]

    centers = sorted(df_out["data_center"].unique().tolist())
    center_f1 = {}
    for c in centers:
        sub = df_out[df_out["data_center"] == c]
        center_f1[c] = f1_score(sub["label"].values, sub["pred"].values, zero_division=0)

    macro_f1 = float(np.mean([center_f1[c] for c in centers])) if len(centers) > 0 else 0.0

    wrong = df_out[df_out["pred"] != df_out["label"]].copy()
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

    print("\n================= MISCLASSIFIED (grouped by center) =================")
    if len(wrong) == 0:
        print("No misclassified samples.")
    else:
        show_cols = ["data_center", "path", "label", "pred", "prob_covid"]
        with pd.option_context("display.max_colwidth", 200, "display.width", 220, "display.expand_frame_repr", False):
            for c, sub in wrong.groupby("data_center", sort=True):
                print(f"\n--- center {c} (n={len(sub)}) ---")
                print(sub[show_cols].to_string(index=False))

    # ---- save all / pos / neg ----
    all_csv = os.path.join(out_dir, "all_samples.csv")
    covid_csv = os.path.join(out_dir, "covid.csv")
    noncovid_csv = os.path.join(out_dir, "non-covid.csv")
    wrong_csv = os.path.join(out_dir, "misclassified.csv")

    # 建议按 center + path 排序，便于看
    df_save = df_out.copy()
    df_save["_center_int"] = pd.to_numeric(df_save["data_center"], errors="coerce")
    df_save = df_save.sort_values(["label", "_center_int", "path"], ascending=[False, True, True]).drop(columns=["_center_int"])

    df_save.to_csv(all_csv, index=False)
    df_save[df_save["label"] == 1].to_csv(covid_csv, index=False)
    df_save[df_save["label"] == 0].to_csv(noncovid_csv, index=False)
    wrong.to_csv(wrong_csv, index=False)

    print(f"\n[done] saved all samples      to: {all_csv}")
    print(f"[done] saved covid samples    to: {covid_csv}")
    print(f"[done] saved non-covid samples to: {noncovid_csv}")
    print(f"[done] saved misclassified    to: {wrong_csv}")


if __name__ == "__main__":
    main()