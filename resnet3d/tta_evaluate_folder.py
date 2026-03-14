#!/usr/bin/env python
# coding: utf-8

import os
import re
import csv
import glob
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

import sys
CVPR26_ROOT = "/remote-home/share/25-jianfabai/cvpr2026"
sys.path.insert(0, CVPR26_ROOT)

from models.ResNet import SupConResNet
from dataset1_hjltest import (
    Lung3D_eccv_patient_supcon as hjl_Lung3D_eccv_patient_supcon,
    augment,
    normalize,
    normalize_hu,
    rescale_z,
    rescale_gao,
)

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())


# -------------------------
# Utilities
# -------------------------
def seed_everything(seed: int):
    import random
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


def scan_num(name: str) -> int:
    name = os.path.basename(name).replace(".npy", "")
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 10**18


def strip_or_add_module_prefix(ckpt_state: dict, model_state: dict) -> dict:
    ckpt_keys = list(ckpt_state.keys())
    model_keys = list(model_state.keys())
    if len(ckpt_keys) == 0:
        return ckpt_state

    ckpt_has_module = ckpt_keys[0].startswith("module.")
    model_has_module = model_keys[0].startswith("module.")

    if ckpt_has_module and (not model_has_module):
        return {k.replace("module.", "", 1): v for k, v in ckpt_state.items()}

    if (not ckpt_has_module) and model_has_module:
        return {"module." + k: v for k, v in ckpt_state.items()}

    return ckpt_state


def load_center_maps():
    df_covid = pd.read_csv("/remote-home/share/25-jianfabai/cvpr2026/validation_covid1.csv")
    df_noncovid = pd.read_csv("/remote-home/share/25-jianfabai/cvpr2026/validation_non_covid.csv")
    name2center_covid = dict(zip(df_covid["ct_scan_name"], df_covid["data_centre"]))
    name2center_non = dict(zip(df_noncovid["ct_scan_name"], df_noncovid["data_centre"]))
    return name2center_covid, name2center_non


def get_center(scan_name: str, true_label: int, name2center_covid: dict, name2center_non: dict):
    if int(true_label) == 1:
        return name2center_covid.get(scan_name)
    else:
        return name2center_non.get(scan_name)


def find_best_thr_for_macro_f1(y_true, probs, scan_names, thr_grid=None):
    covid_prob = probs[:, 1]

    if thr_grid is None:
        uniq = np.unique(covid_prob)
        thr_grid = np.concatenate(([0.0], uniq, [1.0]))

    best = {
        "thr": 0.5,
        "macro_f1": -1.0,
        "acc": None,
        "auc": None,
        "recall": None,
        "precision": None,
        "center_f1_scores": None,
        "final_score": None,
        "y_pred": None,
    }

    for thr in thr_grid:
        y_pred = (covid_prob >= thr).astype(np.int64)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        if macro_f1 > best["macro_f1"]:
            best["thr"] = float(thr)
            best["macro_f1"] = float(macro_f1)

    acc, recall, precision, f1_macro, auc_score, center_f1_scores, final_score, y_pred = \
        compute_metrics_and_centers_by_thr(y_true, probs, scan_names, best["thr"])

    best.update({
        "acc": acc,
        "auc": auc_score,
        "recall": recall,
        "precision": precision,
        "center_f1_scores": center_f1_scores,
        "final_score": final_score,
        "y_pred": y_pred,
        "macro_f1": f1_macro,
    })
    return best


def compute_metrics_and_centers_by_thr(
    y_true: np.ndarray,
    probs: np.ndarray,
    scan_names,
    thr: float,
):
    covid_prob = probs[:, 1]
    y_pred = (covid_prob >= thr).astype(np.int64)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None, labels=[0, 1])
    precision = precision_score(y_true, y_pred, average=None, labels=[0, 1])
    f1_macro = f1_score(y_true, y_pred, average="macro")
    auc_score = roc_auc_score(y_true, covid_prob)

    print(f"\n[Global@thr={thr:.4f}] acc={acc*100:.4f}  auc={auc_score:.6f}  f1_macro={f1_macro:.6f}")
    print("recall(%)[non,covid]:", recall * 100.0)
    print("precision(%)[non,covid]:", precision * 100.0)

    name2center_covid, name2center_non = load_center_maps()
    center_items = defaultdict(list)

    for i, scan_name in enumerate(scan_names):
        t = int(y_true[i])
        p = int(y_pred[i])
        center = get_center(scan_name, t, name2center_covid, name2center_non)
        if center is None:
            continue
        center_items[center].append((scan_name, t, p))

    center_f1_scores = {}

    def _center_sort_key(x):
        s = str(x)
        if s.isdigit():
            return (0, int(s))
        return (1, s)

    for center in sorted(center_items.keys(), key=_center_sort_key):
        items = sorted(center_items[center], key=lambda x: scan_num(x[0]))
        y_t = [t for _, t, _ in items]
        y_p = [p for _, _, p in items]
        f1_covid = f1_score(y_t, y_p, pos_label=1)
        f1_non = f1_score(y_t, y_p, pos_label=0)
        avg_f1 = (f1_covid + f1_non) / 2.0
        center_f1_scores[center] = avg_f1
        print(f"[center {center}] F1_covid={f1_covid:.4f}  F1_non={f1_non:.4f}  Avg={avg_f1:.4f}  n={len(y_t)}")

    final_score = sum(center_f1_scores.values()) / max(len(center_f1_scores), 1)
    print(f">>> Final Averaged F1 Score Across Centers: {final_score:.6f}")

    return acc, recall, precision, f1_macro, auc_score, center_f1_scores, final_score, y_pred

def write_val_csv_threshold(scan_names, y_true, probs, thr, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    covid_prob = probs[:, 1]
    pred = (covid_prob >= thr).astype(np.int64)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "true_label",
            "true_class",
            "pred",
            "pred_class",
            "prob_non-covid",
            "correct",
            "prob_covid",
            "prob_non-covid:prob_covid",
            "threshold_used",
        ])
        for name, yt, yp, pr in zip(scan_names, y_true, pred, probs):
            writer.writerow([
                name,
                int(yt),
                "covid" if int(yt) == 1 else "non-covid",
                int(yp),
                "covid" if int(yp) == 1 else "non-covid",
                int(int(yt) == int(yp)),
                f"{pr[0]:.6f}",
                f"{pr[1]:.6f}",
                f"{pr[0]:.6f}:{pr[1]:.6f}",
                f"{thr:.6f}",
            ])

# -------------------------
# Custom folder dataset
# -------------------------
class FolderNPYTTAInferenceDataset(Dataset):
    """
    对文件夹里的 npy 复现 dataset1_hjltest.py 的 inference 预处理：
      - np.load
      - if 'study' in ID: normalize_hu + rescale_gao
      - if img_depth == 64: rescale_z(..., 64)
      - augment(...)
      - normalize(...)
    """
    def __init__(self, npy_dir, recursive=False, img_depth=64):
        self.npy_dir = npy_dir
        self.img_depth = img_depth

        if recursive:
            self.datalist = sorted(glob.glob(os.path.join(npy_dir, "**", "*.npy"), recursive=True))
        else:
            self.datalist = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))

        if len(self.datalist) == 0:
            raise FileNotFoundError(f"No .npy found in {npy_dir}")

        print(f"[Predict Dataset] found {len(self.datalist)} npy files")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        img = self.datalist[index]
        ID = img

        img_array = np.load(img)

        if "study" in ID:
            img_array = normalize_hu(img_array)
            img_array = rescale_gao(img_array)

        if self.img_depth == 64:
            img_array = rescale_z(img_array, self.img_depth)

        # tta 1
        # img_array = augment(
        #     img_array,
        #     ifhalfcrop=False,
        #     ifrandom_resized_crop=True,
        #     ifflip=False,
        #     ifrotate=False,
        #     ifcontrast=True,
        #     ifswap=False,
        #     filling_value=0,
        # )
        img_array = torch.from_numpy(normalize(img_array)).float()

        return img_array, 0, -1, ID


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def infer_one_pass(net, loader, device):
    """
    Return:
      probs (N,2)
      labels (N,)
      scan_names: list[str]
      scan_paths: list[str]
    """
    net.eval()
    probs_list, labels_list = [], []
    scan_names, scan_paths = [], []

    pbar = tqdm(loader, ascii=True)
    for data, _, label, id_list in pbar:
        data = data.unsqueeze(1)  # [B,1,D,H,W]
        data = data.float().to(device, non_blocking=True)

        out = net(data)
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            pred_logits = out[2]
        elif isinstance(out, (tuple, list)):
            pred_logits = out[-1]
        else:
            pred_logits = out

        prob = F.softmax(pred_logits, dim=1)

        probs_list.append(prob.detach().cpu().numpy())
        labels_list.append(np.array(label))

        for s in id_list:
            full_path = str(s)
            base = os.path.basename(full_path).replace(".npy", "")
            scan_names.append(base)
            scan_paths.append(full_path)

    probs = np.concatenate(probs_list, axis=0).astype(np.float32) if probs_list else np.zeros((0, 2), np.float32)
    labels = np.concatenate(labels_list, axis=0).astype(np.int64) if labels_list else np.zeros((0,), np.int64)
    return probs, labels, scan_names, scan_paths


def tta_predict(net, loader, device, tta=3, seed=0):
    probs_tta = []
    names_ref, paths_ref, labels_ref = None, None, None

    for t in range(tta):
        seed_everything(seed + t)

        probs, labels, scan_names, scan_paths = infer_one_pass(net, loader, device)
        probs_tta.append(probs)

        if names_ref is None:
            names_ref = scan_names
            paths_ref = scan_paths
            labels_ref = labels
        else:
            assert names_ref == scan_names, "TTA sample order changed"
            assert paths_ref == scan_paths, "TTA path order changed"

    probs_mean = np.mean(np.stack(probs_tta, axis=0), axis=0)
    return probs_mean, labels_ref, names_ref, paths_ref


# -------------------------
# CSV writers
# -------------------------
def write_pred_csv_threshold(scan_paths, probs, thr, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    covid_prob = probs[:, 1]
    pred = (covid_prob >= thr).astype(np.int64)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "pred",
            "pred_class",
            "prob_non-covid",
            "prob_covid",
            "prob_non-covid:prob_covid",
            "threshold_used",
        ])
        for path, p, pr in zip(scan_paths, pred, probs):
            writer.writerow([
                path,
                int(p),
                "covid" if int(p) == 1 else "non-covid",
                f"{pr[0]:.6f}",
                f"{pr[1]:.6f}",
                f"{pr[0]:.6f}:{pr[1]:.6f}",
                f"{thr:.6f}",
            ])


# -------------------------
# Main
# -------------------------
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", "-bs", default=4, type=int)
    # tta 2
    # p.add_argument("--tta", default=10, type=int)
    p.add_argument("--tta", default=1, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--seed", default=0, type=int)

    p.add_argument("--ckpt", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m5-nomix-class-bai-v100-batch6/epoch005_f10.9700.pkl")
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-1b-3090-nomix/epoch075_f10.9681.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-ori_and_1b-3090-nomix/epoch073_f10.9710.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-1b-3090-nomix-rotate/epoch091_f10.9594.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-b2-v100-5-1b-nomix-rotate/epoch021_f10.9649.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-ori_and_1b-mix_v100_batch6/epoch042_f10.9567.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m5-nomix-class-bai-v100-batch6/epoch005_f10.9700.pkl
    # p.add_argument("--predict-dir", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1-iccv25/norm/test_lungex")
    p.add_argument("--predict-dir", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/test_lungex")
    p.add_argument("--recursive", action="store_true")

    # p.add_argument("--save-dir", type=str, default=f"/remote-home/share/25-jianfabai/cvpr2026/predicts/iccv/tta/cvpr-pretrain-c2m5-nomix-class-bai-v100_test_lungex_tta10_results_{TIMESTAMP}")
    p.add_argument("--save-dir", type=str, default=f"/remote-home/share/25-jianfabai/cvpr2026/predicts/cvpr/tta/cvpr-pretrain-c2m5-nomix-class-bai-v100_test_lungex_tta10_results_{TIMESTAMP}")

    return p.parse_args()


def main():
    args = build_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[CKPT]", args.ckpt)

    net = SupConResNet(
        name="resnest50_3D",
        ipt_dim=1,
        head="mlp",
        feat_dim=128,
        n_classes=2,
        supcon=True,
    ).to(device)
    net.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    state = strip_or_add_module_prefix(state, net.state_dict())
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"[Load] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    print(f"  missing: {missing}")

    # -------------------------
    # 1) validation set -> get best threshold
    # -------------------------
    val_ds = hjl_Lung3D_eccv_patient_supcon(
        train=False,
        val=True,
        inference=False,
        n_classes=2,
        supcon=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_probs, val_labels, val_names, val_paths = tta_predict(
        net, val_loader, device, tta=args.tta, seed=args.seed
    )

    best = find_best_thr_for_macro_f1(val_labels, val_probs, val_names)

    print("\n================= VALIDATION SUMMARY =================")
    print(f"Best ACC: {best['acc']:.6f}")
    print(f"Best THR: {best['thr']:.4f}")
    print(f"ROC-AUC : {best['auc']:.6f}")
    print(f"Macro-F1: {best['macro_f1']:.6f}")
    print(f"Final Averaged F1 Score Across Centers: {best['final_score']:.6f}")

    best_thr = best["thr"]

    # -------------------------
    # save validation predictions
    # -------------------------
    val_csv_best = os.path.join(args.save_dir, "val_predict_with_best_thr.csv")
    val_csv_05 = os.path.join(args.save_dir, "val_predict_with_thr_0.5.csv")

    write_val_csv_threshold(val_names, val_labels, val_probs, best_thr, val_csv_best)
    write_val_csv_threshold(val_names, val_labels, val_probs, 0.5, val_csv_05)

    print("Saved:", val_csv_best)
    print("Saved:", val_csv_05)

    # 保存一份 val 的结果，方便你核对
    os.makedirs(args.save_dir, exist_ok=True)
    val_thr_txt = os.path.join(args.save_dir, "best_threshold.txt")
    with open(val_thr_txt, "w") as f:
        f.write(f"best_thr={best_thr:.6f}\n")
        f.write(f"best_acc={best['acc']:.6f}\n")
        f.write(f"best_auc={best['auc']:.6f}\n")
        f.write(f"best_macro_f1={best['macro_f1']:.6f}\n")
        f.write(f"best_center_avg_f1={best['final_score']:.6f}\n")
    print("Saved:", val_thr_txt)

    # -------------------------
    # 2) folder npy -> 3x TTA predict
    # -------------------------
    pred_ds = FolderNPYTTAInferenceDataset(
        npy_dir=args.predict_dir,
        recursive=args.recursive,
        img_depth=64
    )
    pred_loader = DataLoader(
        pred_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pred_probs, _, pred_names, pred_paths = tta_predict(
        net, pred_loader, device, tta=args.tta, seed=args.seed
    )

    order = np.argsort([scan_num(x) for x in pred_names])
    pred_names = [pred_names[i] for i in order]
    pred_paths = [pred_paths[i] for i in order]
    pred_probs = pred_probs[order]

    # -------------------------
    # 3) write 2 csv
    # -------------------------
    csv_best = os.path.join(args.save_dir, "predict_with_best_thr.csv")
    csv_05 = os.path.join(args.save_dir, "predict_with_thr_0.5.csv")

    write_pred_csv_threshold(pred_paths, pred_probs, best_thr, csv_best)
    write_pred_csv_threshold(pred_paths, pred_probs, 0.5, csv_05)

    print("Saved:", csv_best)
    print("Saved:", csv_05)


if __name__ == "__main__":
    main()