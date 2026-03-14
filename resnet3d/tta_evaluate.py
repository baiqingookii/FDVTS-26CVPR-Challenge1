#!/usr/bin/env python
# coding: utf-8
"""
DDP + TTA evaluation for Challenge1 (covid vs non-covid)
...
"""

import os
import re
import csv
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

from models.ResNet import SupConResNet
from dataset1_hjltest import Lung3D_eccv_patient_supcon as hjl_Lung3D_eccv_patient_supcon
from dataset1_xjltest import Lung3D_eccv_patient_supcon as xjl_Lung3D_eccv_patient_supcon


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


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
    name = name.replace(".npy", "")
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
    """rank0: load once"""
    df_covid = pd.read_csv("/remote-home/share/25-jianfabai/cvpr2026/validation_covid1.csv")
    df_noncovid = pd.read_csv("/remote-home/share/25-jianfabai/cvpr2026/validation_non_covid.csv")
    name2center_covid = dict(zip(df_covid["ct_scan_name"], df_covid["data_centre"]))
    name2center_non = dict(zip(df_noncovid["ct_scan_name"], df_noncovid["data_centre"]))
    return name2center_covid, name2center_non


def get_center(scan_name: str, true_label: int, name2center_covid: dict, name2center_non: dict):
    # true_label==1 => covid list; true_label==0 => noncovid list
    if int(true_label) == 1:
        return name2center_covid.get(scan_name)
    else:
        return name2center_non.get(scan_name)


def compute_metrics_and_centers(y_true: np.ndarray, probs: np.ndarray, scan_names: list[str]):
    """
    scan_names: ['ct_scan_0', ...] 用于中心映射
    """
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    auc_score = roc_auc_score(y_true, probs[:, 1])

    print(f"\n[Global] acc={acc*100:.4f}  auc={auc_score:.6f}  f1_macro={f1_macro:.6f}")
    print("recall(%):", recall * 100.0)
    print("precision(%):", precision * 100.0)

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


def write_val_csv_with_probs(scan_paths, y_true, y_pred, probs, save_path):
    """id 列写完整路径"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "pred", "prob_non-covid:prob_covid", "covid_prob"])
        for path, t, p, pr in zip(scan_paths, y_true, y_pred, probs):
            writer.writerow([path, int(t), int(p), f"{pr[0]:.6f}:{pr[1]:.6f}", f"{pr[1]:.6f}"])


def write_wrong_by_center_csv(rows, save_path):
    """
    rows: list of (center, path, label, pred, prob_str, covid_prob)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["center", "id", "label", "pred", "prob_non-covid:prob_covid", "covid_prob"])
        writer.writerows(rows)


@torch.no_grad()
def infer_one_pass(net, loader, device):
    net.eval()
    probs_list, labels_list = [], []
    scan_names, scan_paths = [], []

    show_pbar = (not dist.is_initialized()) or (dist.get_rank() == 0)
    pbar = tqdm(loader, ascii=True, disable=not show_pbar)

    for data, _, label, id_list in pbar:
        data = data.unsqueeze(1)  # [B,1,D,H,W]
        data = data.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)

        out = net(data)

        if isinstance(out, (tuple, list)) and len(out) >= 3:
            pred_logits = out[2]
        elif isinstance(out, (tuple, list)):
            pred_logits = out[-1]
        else:
            pred_logits = out

        prob = F.softmax(pred_logits, dim=1)  # [B,2]

        probs_list.append(prob.detach().cpu().numpy())
        labels_list.append(label.detach().cpu().numpy())

        for s in id_list:
            full_path = str(s)

            base = full_path.split("/")[-1]
            base = base.replace(".npy", "")  # scan_name
            scan_names.append(base)
            scan_paths.append(full_path)

    probs = np.concatenate(probs_list, axis=0).astype(np.float32) if probs_list else np.zeros((0, 2), np.float32)
    labels = np.concatenate(labels_list, axis=0).astype(np.int64) if labels_list else np.zeros((0,), np.int64)
    return probs, labels, scan_names, scan_paths


def all_gather_object(obj):
    world = dist.get_world_size()
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, obj)
    return gathered


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", "-bs", default=4, type=int, help="per-rank batch size")
    p.add_argument("--tta", default=3, type=int, help="TTA repeats")
    p.add_argument("--num-workers", default=2, type=int, help="dataloader workers per rank")
    p.add_argument("--seed", default=0, type=int, help="seed")
    p.add_argument("--dataset", default="hjl", choices=["hjl", "xjl"], help="dataset class to use")
    p.add_argument("--ckpt", default="/remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-ori_and_1b-mix_v100_batch6/epoch042_f10.9567.pkl", type=str)
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m5-nomix-class-bai-v100-batch6/epoch005_f10.9700.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-1b-3090-nomix/epoch075_f10.9681.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-1b-3090-nomix-rotate/epoch091_f10.9594.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-b2-v100-5-1b-nomix-rotate/epoch021_f10.9649.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-ori_and_1b-3090-nomix/epoch073_f10.9710.pkl
    # /remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-ori_and_1b-mix_v100_batch6/epoch042_f10.9567.pkl
    p.add_argument("--out-csv", default="/remote-home/share/25-jianfabai/cvpr2026/predicts/c2m5-ori_and_1b-mix_v100_batch6-evaluate.csv", type=str)
    return p.parse_args()


def main():
    args = build_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"[DDP] world_size={world}, visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"[Args] batch_size(per-rank)={args.batch_size}, tta={args.tta}")
        print(f"[Ckpt] {args.ckpt}")

    seed_everything(args.seed)

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

    if rank == 0:
        print(f"[Load] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if len(missing) > 50:
            print("[WARN] Too many missing keys. Check ckpt/model mismatch!")

    if args.dataset == "hjl":
        ds = hjl_Lung3D_eccv_patient_supcon(train=False, val=True, inference=False, n_classes=2, supcon=True)
    else:
        ds = xjl_Lung3D_eccv_patient_supcon(train=False, val=True, inference=False, n_classes=2, supcon=True)

    sampler = DistributedSampler(ds, shuffle=False, drop_last=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    local_probs_tta = []
    local_names_ref = None
    local_paths_ref = None
    local_labels_ref = None

    for t in range(args.tta):
        seed_everything(args.seed + t + 1000 * rank)
        sampler.set_epoch(t)

        probs, labels, scan_names, scan_paths = infer_one_pass(net, loader, device)
        local_probs_tta.append(probs)

        if local_names_ref is None:
            local_names_ref = scan_names
            local_paths_ref = scan_paths
            local_labels_ref = labels
        else:
            assert local_names_ref == scan_names, 
            assert local_paths_ref == scan_paths, 
            assert np.all(local_labels_ref == labels), 

    probs_local = np.mean(np.stack(local_probs_tta, axis=0), axis=0)

    # gather: (names, paths, labels, probs)
    gathered = all_gather_object((local_names_ref, local_paths_ref, local_labels_ref, probs_local))

    if rank == 0:
        names_all, paths_all = [], []
        labels_all_list, probs_all_list = [], []

        for n_i, p_i, lab_i, pr_i in gathered:
            names_all += list(n_i)
            paths_all += list(p_i)
            labels_all_list.append(np.asarray(lab_i))
            probs_all_list.append(np.asarray(pr_i))

        labels_all = np.concatenate(labels_all_list, axis=0)
        probs_all = np.concatenate(probs_all_list, axis=0)

        # sort by scan_num(name)
        order = np.argsort([scan_num(x) for x in names_all])
        names_all = [names_all[i] for i in order]
        paths_all = [paths_all[i] for i in order]
        labels_all = labels_all[order]
        probs_all = probs_all[order]

        # metrics
        acc, recall, precision, f1_macro, auc_score, center_f1_scores, final_score, y_pred = \
            compute_metrics_and_centers(labels_all, probs_all, names_all)

        # write main csv (id = full path)
        write_val_csv_with_probs(paths_all, labels_all, y_pred, probs_all, args.out_csv)
        print("Saved:", args.out_csv)

        # write wrong samples by center
        name2center_covid, name2center_non = load_center_maps()
        wrong_rows = []
        for name, path, t, p, pr in zip(names_all, paths_all, labels_all, y_pred, probs_all):
            if int(t) == int(p):
                continue
            center = get_center(name, int(t), name2center_covid, name2center_non)
            if center is None:
                center = "NA"
            wrong_rows.append([
                center,
                path,
                int(t),
                int(p),
                f"{pr[0]:.6f}:{pr[1]:.6f}",
                f"{pr[1]:.6f}",
            ])

        # sort wrong rows by center then scan_num(name extracted from path)
        def _center_key(x):
            s = str(x[0])
            if s.isdigit():
                return (0, int(s))
            return (1, s)

        def _scan_from_path(path):
            base = str(path).split("/")[-1].replace(".npy", "")
            return scan_num(base)

        wrong_rows = sorted(wrong_rows, key=lambda r: (_center_key(r), _scan_from_path(r[1])))

        wrong_path = os.path.splitext(args.out_csv)[0] + "_wrong.csv"
        write_wrong_by_center_csv(wrong_rows, wrong_path)
        print("Saved wrong samples:", wrong_path)
        print(f"[Wrong] count={len(wrong_rows)}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()