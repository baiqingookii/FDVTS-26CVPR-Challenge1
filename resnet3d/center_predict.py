#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CVPR26_ROOT = "/remote-home/share/25-jianfabai/cvpr2026"
import sys
sys.path.insert(0, CVPR26_ROOT)
sys.stdout.reconfigure(line_buffering=True)
from models.ResNet import SupConResNet
from dataset1 import rescale_z, normalize

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def ct_scan_sort_key(path):
    fname = os.path.basename(path)
    stem = os.path.splitext(fname)[0]
    return natural_key(stem)

class TestNPYDataset(Dataset):
    def __init__(self, root_dir, img_depth=64, recursive=True):
        self.root_dir = root_dir
        self.img_depth = img_depth
        self.samples = []

        if recursive:
            for dirpath, _, filenames in os.walk(root_dir):
                # for fn in sorted(filenames):
                for fn in sorted(filenames, key=natural_key):
                    if fn.endswith(".npy"):
                        self.samples.append(os.path.join(dirpath, fn))
        else:
            # for fn in sorted(os.listdir(root_dir)):
            for fn in sorted(os.listdir(root_dir), key=natural_key):
                fpath = os.path.join(root_dir, fn)
                if os.path.isfile(fpath) and fn.endswith(".npy"):
                    self.samples.append(fpath)

        print(f"[Dataset] root = {root_dir}")
        print(f"[Dataset] num_samples = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath = self.samples[idx]
        fname = os.path.basename(fpath)
        stem = os.path.splitext(fname)[0]

        img_array = np.load(fpath).astype(np.float32)  # (D,H,W)

        if self.img_depth != 128:
            img_array = rescale_z(img_array, self.img_depth)

        img_array = torch.from_numpy(normalize(img_array)).float()  # (D,H,W)

        return img_array, fname, stem, fpath


def collate_fn(batch):
    imgs, fnames, stems, fpaths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # (B,D,H,W)
    return imgs, list(fnames), list(stems), list(fpaths)


def build_model(backbone_name="resnest50_3D", n_classes=4, supcon=False):
    model = SupConResNet(
        name=backbone_name,
        ipt_dim=1,
        head='mlp',
        feat_dim=128,
        n_classes=n_classes,
        supcon=supcon
    )
    return model


def strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        new_state[k.replace("module.", "")] = v
    return new_state


def load_ckpt(model, ckpt_path, device="cpu"):
    print(f"[Load] ckpt = {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "net" in ckpt:
        state_dict = ckpt["net"]
    else:
        state_dict = ckpt

    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    print("[Load] checkpoint loaded successfully")
    return model


def forward_model(model, imgs):
    out = model(imgs)
    if isinstance(out, (list, tuple)):
        if len(out) == 3:
            _, features, logits = out
        elif len(out) == 2:
            features, logits = out
        else:
            raise RuntimeError(f"Unexpected model output length: {len(out)}")
    else:
        raise RuntimeError("Unexpected model output type.")
    return features, logits


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    rows = []
    rows_simple = []
    for imgs, fnames, stems, fpaths in loader:
        imgs = imgs.unsqueeze(1).to(device, non_blocking=True)  # (B,1,D,H,W)

        _, logits = forward_model(model, imgs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

        for fname, stem, fpath, pred, prob in zip(fnames, stems, fpaths, preds, probs):
            rows.append([
                fname,
                stem,
                fpath,
                int(pred),
                float(prob[0]),
                float(prob[1]),
                float(prob[2]),
                float(prob[3]),
                float(np.max(prob)),
            ])
            rows_simple.append([stem, int(pred)])
    return rows, rows_simple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str,
                        default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/test_lungex")
    # parser.add_argument("--test_root", type=str,
    #                     default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/covid1b")
    parser.add_argument("--ckpt", type=str,
                        default="/remote-home/share/25-jianfabai/cvpr2026/center/resnet3d/linear_center_cls_from_c2m5_ori_and_1b/best_epoch026_f10.9114.pkl")
    # parser.add_argument("--outdir", type=str,
    #                     default="/remote-home/share/25-jianfabai/cvpr2026/center/predict_result")
    parser.add_argument("--outdir", type=str,
                        default="/remote-home/share/25-jianfabai/cvpr2026/center/predict_result/1b_predict/test_dataset")
    parser.add_argument("--model", type=str, default="resnest50_3D")
    parser.add_argument("--img_depth", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--recursive", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    folder_name = os.path.basename(os.path.normpath(args.test_root))
    out_csv = os.path.join(args.outdir, f"{folder_name}.csv")
    out_simple_csv = os.path.join(args.outdir, f"{folder_name}_simple.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    dataset = TestNPYDataset(
        root_dir=args.test_root,
        img_depth=args.img_depth,
        recursive=args.recursive
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = build_model(backbone_name=args.model, n_classes=4, supcon=False)
    model = load_ckpt(model, args.ckpt, device="cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    rows,rows_simple = predict(model, loader, device)

    rows = sorted(rows, key=lambda x: natural_key(x[0]))
    rows_simple = sorted(rows_simple, key=lambda x: natural_key(x[0]))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "ct_scan_name",
            "full_path",
            "pred_center",
            "prob_center_0",
            "prob_center_1",
            "prob_center_2",
            "prob_center_3",
            "max_prob"
        ])
        writer.writerows(rows)
    
    with open(out_simple_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ct_scan_name", "pred_center"])
        writer.writerows(rows_simple)

    print(f"[OK] saved: {out_csv}")
    print(f"[OK] num_predictions: {len(rows)}")    
    print(f"[OK] saved: {out_simple_csv}")

    center_rows = {0: [], 1: [], 2: [], 3: []}
    for ct_scan_name, pred_center in rows_simple:
        center_rows[pred_center].append([ct_scan_name, pred_center])

    for c in range(4):
        center_csv = os.path.join(args.outdir, f"center_{c}.csv")
        center_rows[c] = sorted(center_rows[c], key=lambda x: natural_key(x[0]))
        with open(center_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ct_scan_name", "pred_center"])
            writer.writerows(center_rows[c])
        print(f"[OK] saved: {center_csv}, num = {len(center_rows[c])}")



if __name__ == "__main__":
    main()