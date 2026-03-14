#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load trained MedSigLIP + slice transformer model, run validation/test inference,
sweep threshold on validation set by 4-center macro F1, and save CSV outputs.

Extra outputs for test set:
- best_thr_covid.csv
- best_thr_non-covid.csv
- thr_0.5_covid.csv
- thr_0.5_non-covid.csv

Each of the 4 files contains one ct_scan_name per line, naturally sorted.
"""

import os
import json
import math
import re
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from safetensors.torch import load_file as safe_load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

NPY_EXTS = (".npy",)


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def list_npys_recursively(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(NPY_EXTS):
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def _get_vision_encoder(base_model):
    if hasattr(base_model, "vision_model"):
        return base_model.vision_model
    if hasattr(base_model, "model") and hasattr(base_model.model, "vision_model"):
        return base_model.model.vision_model
    return base_model


class SliceTransformerClassifier(nn.Module):
    def __init__(
        self,
        base_model,
        chunk_size=4,
        max_slices=64,
        n_layers=2,
        n_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        use_cls_token=True,
        label_smoothing=0.0,
        pool="cls",
    ):
        super().__init__()
        self.base_model = base_model
        self.chunk_size = chunk_size
        self.label_smoothing = float(label_smoothing)
        self.max_slices = max_slices
        self.use_cls_token = use_cls_token
        assert pool in {"cls", "mean"}
        self.pool = pool

        for p in self.base_model.parameters():
            p.requires_grad = False

        D = None
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "hidden_size"):
            D = self.base_model.config.hidden_size
        if D is None:
            vision = _get_vision_encoder(self.base_model)
            if hasattr(vision, "config") and hasattr(vision.config, "hidden_size"):
                D = vision.config.hidden_size
        if D is None:
            D = 768
        self.hidden_dim = D

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (1 if use_cls_token else 0) + max_slices, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=n_heads,
            dim_feedforward=int(self.hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slice_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, 2)

    @property
    def vision(self):
        return _get_vision_encoder(self.base_model)

    @torch.no_grad()
    def _extract_slice_features(self, flat_pixel_values):
        out = self.vision(
            pixel_values=flat_pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            feats = out.last_hidden_state[:, 0]
        else:
            raise RuntimeError("Vision encoder output has neither pooler_output nor last_hidden_state.")
        return feats

    def forward(self, pixel_values=None, labels=None):
        B, S, C, H, W = pixel_values.shape
        assert S <= self.max_slices, f"S={S} exceeds max_slices={self.max_slices}"

        flat = pixel_values.reshape(B * S, C, H, W)

        feats_chunks = []
        for i in range(0, B * S, self.chunk_size):
            feats = self._extract_slice_features(flat[i:i + self.chunk_size])
            feats_chunks.append(feats)
        feats = torch.cat(feats_chunks, dim=0)

        x = feats.view(B, S, -1)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embed[:, :(1 + S), :]
        else:
            x = x + self.pos_embed[:, :S, :]

        x = self.pos_drop(x)
        x = self.slice_encoder(x)
        x = self.norm(x)

        if self.pool == "cls":
            pooled = x[:, 0] if self.use_cls_token else x.mean(dim=1)
        else:
            pooled = x[:, 1:].mean(dim=1) if self.use_cls_token else x.mean(dim=1)

        logits = self.head(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        return {"loss": loss, "logits": logits}


class NPYSubsetDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        labels: Optional[List[int]],
        centers: Optional[List[int]],
        out_size: int,
        num_slices: int = 12,
        mmap: bool = True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        assume_uint8_0_255: bool = True,
    ):
        self.paths = paths
        self.labels = labels
        self.centers = centers
        self.out_size = int(out_size)
        self.num_slices = int(num_slices)
        self.mmap = bool(mmap)
        self.assume_uint8_0_255 = bool(assume_uint8_0_255)

        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def _load_volume(self, p: str):
        vol = np.load(p, mmap_mode="r") if self.mmap else np.load(p)
        if vol.ndim != 3:
            raise ValueError(f"Expect 3D volume, got shape={vol.shape} at {p}")
        return vol

    def _pick_slices_val(self, vol: np.ndarray) -> np.ndarray:
        D, H, W = vol.shape
        need = 2 * self.num_slices
        if D < need:
            raise ValueError(f"Volume depth {D} < required {need}")
        first = vol[:self.num_slices]
        last = vol[-self.num_slices:]
        return np.concatenate([first, last], axis=0)

    def __getitem__(self, idx):
        p = self.paths[idx]
        vol = self._load_volume(p)
        window = self._pick_slices_val(vol)
        window = np.array(window, copy=True)

        x = torch.from_numpy(window).float().unsqueeze(1)  # (S,1,H,W)
        if self.assume_uint8_0_255:
            x = x / 255.0
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        item = {
            "pixel_values": x,
            "path": p,
            "ct_scan_name": os.path.splitext(os.path.basename(p))[0],
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if self.centers is not None:
            item["center"] = int(self.centers[idx])

        return item


@dataclass
class InferCollator:
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            "pixel_values": torch.stack([ex["pixel_values"] for ex in examples], dim=0),
            "path": [ex["path"] for ex in examples],
            "ct_scan_name": [ex["ct_scan_name"] for ex in examples],
        }
        if "labels" in examples[0]:
            batch["labels"] = torch.stack([ex["labels"] for ex in examples], dim=0)
        if "center" in examples[0]:
            batch["center"] = [ex["center"] for ex in examples]
        return batch


def build_name_to_path(root: str) -> Dict[str, str]:
    paths = list_npys_recursively(root)
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def build_val_dataset_from_csv(
    covid_root: str,
    noncovid_root: str,
    csv_pos: str,
    csv_neg: str,
    out_size: int,
    num_slices: int,
    mean,
    std,
    mmap=True,
    assume_uint8_0_255=True,
) -> NPYSubsetDataset:
    pos_df = pd.read_csv(csv_pos)
    neg_df = pd.read_csv(csv_neg)

    required_cols = {"ct_scan_name", "data_centre"}
    if not required_cols.issubset(pos_df.columns):
        raise ValueError(f"{csv_pos} must contain columns: {required_cols}")
    if not required_cols.issubset(neg_df.columns):
        raise ValueError(f"{csv_neg} must contain columns: {required_cols}")

    pos_map = build_name_to_path(covid_root)
    neg_map = build_name_to_path(noncovid_root)

    paths, labels, centers = [], [], []

    missing_pos = []
    for _, row in pos_df.iterrows():
        name = str(row["ct_scan_name"])
        center = int(row["data_centre"])
        if name not in pos_map:
            missing_pos.append(name)
            continue
        paths.append(pos_map[name])
        labels.append(1)
        centers.append(center)

    missing_neg = []
    for _, row in neg_df.iterrows():
        name = str(row["ct_scan_name"])
        center = int(row["data_centre"])
        if name not in neg_map:
            missing_neg.append(name)
            continue
        paths.append(neg_map[name])
        labels.append(0)
        centers.append(center)

    if missing_pos:
        print(f"[warn] missing positive samples: {len(missing_pos)}")
        print("[warn] first missing pos:", missing_pos[:10])
    if missing_neg:
        print(f"[warn] missing negative samples: {len(missing_neg)}")
        print("[warn] first missing neg:", missing_neg[:10])

    print(f"[val] matched samples: {len(paths)}")
    print(f"[val] positives: {sum(labels)}  negatives: {len(labels)-sum(labels)}")

    return NPYSubsetDataset(
        paths=paths,
        labels=labels,
        centers=centers,
        out_size=out_size,
        num_slices=num_slices,
        mmap=mmap,
        mean=mean,
        std=std,
        assume_uint8_0_255=assume_uint8_0_255,
    )


def build_test_dataset(
    test_root: str,
    out_size: int,
    num_slices: int,
    mean,
    std,
    mmap=True,
    assume_uint8_0_255=True,
) -> NPYSubsetDataset:
    paths = list_npys_recursively(test_root)
    if len(paths) == 0:
        raise RuntimeError(f"No npy found under test_root: {test_root}")

    print(f"[test] total npy files: {len(paths)}")

    return NPYSubsetDataset(
        paths=paths,
        labels=None,
        centers=None,
        out_size=out_size,
        num_slices=num_slices,
        mmap=mmap,
        mean=mean,
        std=std,
        assume_uint8_0_255=assume_uint8_0_255,
    )


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    rows = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        out = model(pixel_values=pixel_values)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        bs = len(batch["ct_scan_name"])
        for i in range(bs):
            row = {
                "path": batch["path"][i],
                "ct_scan_name": batch["ct_scan_name"][i],
                "prob_covid": float(probs[i]),
            }
            if "labels" in batch:
                row["label"] = int(batch["labels"][i].item())
            if "center" in batch:
                row["data_centre"] = int(batch["center"][i])
            rows.append(row)

    return pd.DataFrame(rows)


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def evaluate_threshold(df: pd.DataFrame, thr: float) -> Dict[str, Any]:
    tmp = df.copy()
    tmp["pred_label"] = (tmp["prob_covid"].values >= thr).astype(int)

    per_center = {}
    centers = sorted(tmp["data_centre"].unique().tolist())

    for c in centers:
        sub = tmp[tmp["data_centre"] == c]
        f1 = binary_f1(sub["label"].values.astype(int), sub["pred_label"].values.astype(int))
        per_center[int(c)] = f1

    macro_f1 = float(np.mean(list(per_center.values()))) if len(per_center) > 0 else 0.0
    overall_f1 = binary_f1(tmp["label"].values.astype(int), tmp["pred_label"].values.astype(int))
    acc = float((tmp["label"].values.astype(int) == tmp["pred_label"].values.astype(int)).mean())

    return {
        "threshold": float(thr),
        "per_center_f1": per_center,
        "macro_f1_4centers": macro_f1,
        "overall_f1": overall_f1,
        "accuracy": acc,
    }


def find_best_threshold(val_df: pd.DataFrame, num_thresholds: int = 1001) -> Dict[str, Any]:
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    best = None

    for thr in thresholds:
        info = evaluate_threshold(val_df, float(thr))
        if best is None:
            best = info
            continue

        better = False
        if info["macro_f1_4centers"] > best["macro_f1_4centers"]:
            better = True
        elif math.isclose(info["macro_f1_4centers"], best["macro_f1_4centers"], rel_tol=0, abs_tol=1e-12):
            if info["overall_f1"] > best["overall_f1"]:
                better = True
            elif math.isclose(info["overall_f1"], best["overall_f1"], rel_tol=0, abs_tol=1e-12):
                if abs(info["threshold"] - 0.5) < abs(best["threshold"] - 0.5):
                    better = True

        if better:
            best = info

    return best


def apply_threshold_and_format(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    out = df.copy()
    out["threshold"] = float(thr)
    out["pred_label"] = (out["prob_covid"].values >= thr).astype(int)
    out["pred_class"] = out["pred_label"].map({1: "covid", 0: "non-covid"})
    return out


def save_name_only_csv(df: pd.DataFrame, pred_class: str, out_csv: str):
    sub = df[df["pred_class"] == pred_class].copy()
    names = sorted(sub["ct_scan_name"].astype(str).tolist(), key=natural_key)
    pd.DataFrame(names).to_csv(out_csv, index=False, header=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="/root/medsiglip-448")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/remote-home/share/25-jianfabai/medgemma-finetune/save_model/transformer_block/3d_ori-and-1b_dataset_cls_lr5e-5_scale0.7-1/checkpoint-27"
    )

    parser.add_argument(
        "--val_covid_root",
        type=str,
        default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/covid1b"
    )
    parser.add_argument(
        "--val_noncovid_root",
        type=str,
        default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/non-covid1b"
    )

    parser.add_argument(
        "--csv_pos",
        type=str,
        default="/remote-home/share/25-jianfabai/cvpr2026/validation_covid1.csv"
    )
    parser.add_argument(
        "--csv_neg",
        type=str,
        default="/remote-home/share/25-jianfabai/cvpr2026/validation_non_covid.csv"
    )

    parser.add_argument(
        "--test_root",
        type=str,
        default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/test_lungex"
        # default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/covid1b"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        # default="/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/transformer_block_eval/3d_ori-and-1b_dataset_cls_lr5e-5_scale0.7-1_24_448_448_12_24_checkpoint-27"
        default="/remote-home/share/25-jianfabai/medgemma-finetune/predict_result/cvpr/transformer_block_eval/3d_ori-and-1b_dataset_cls_lr5e-5_scale0.7-1_24_448_448_12_24_checkpoint-27"
    )

    parser.add_argument("--num_slices", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--mmap", action="store_true", default=True)
    parser.add_argument("--assume_uint8_0_255", action="store_true", default=True)

    # parser.add_argument("--max_slices", type=int, default=64)
    parser.add_argument("--max_slices", type=int, default=24)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_cls_token", action="store_true", default=True)
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    size = image_processor.size["height"]
    mean = image_processor.image_mean
    std = image_processor.image_std
    print(f"[processor] size={size}, mean={mean}, std={std}")

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

    model = SliceTransformerClassifier(
        base_model=base_model,
        chunk_size=args.chunk_size,
        max_slices=args.max_slices,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        use_cls_token=args.use_cls_token,
        label_smoothing=0.0,
        pool=args.pool,
    )

    ckpt_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = safe_load(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load] ckpt:", ckpt_path)
    print("[load] missing:", len(missing))
    print("[load] unexpected:", len(unexpected))
    if len(missing) > 0:
        print("[load] missing examples:", missing[:20])
    if len(unexpected) > 0:
        print("[load] unexpected examples:", unexpected[:20])

    model.to(device)
    model.eval()

    val_ds = build_val_dataset_from_csv(
        covid_root=args.val_covid_root,
        noncovid_root=args.val_noncovid_root,
        csv_pos=args.csv_pos,
        csv_neg=args.csv_neg,
        out_size=size,
        num_slices=args.num_slices,
        mean=mean,
        std=std,
        mmap=args.mmap,
        assume_uint8_0_255=args.assume_uint8_0_255,
    )

    test_ds = build_test_dataset(
        test_root=args.test_root,
        out_size=size,
        num_slices=args.num_slices,
        mean=mean,
        std=std,
        mmap=args.mmap,
        assume_uint8_0_255=args.assume_uint8_0_255,
    )

    collator = InferCollator()

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    print("[run] validation inference ...")
    val_df = run_inference(model, val_loader, device)
    print(f"[run] validation done: {len(val_df)} samples")

    print("[run] test inference ...")
    test_df = run_inference(model, test_loader, device)
    print(f"[run] test done: {len(test_df)} samples")

    best_info = find_best_threshold(val_df, num_thresholds=1001)
    best_thr = float(best_info["threshold"])

    print("\n================= SUMMARY =================")
    print(f"Best THR: {best_thr:.4f}")
    print("Per-center F1:")
    for c, f1 in best_info["per_center_f1"].items():
        print(f"  {c}: {f1:.6f}")
    print(f"Macro-F1 (over 4 centers): {best_info['macro_f1_4centers']:.6f}")
    print(f"Overall F1: {best_info['overall_f1']:.6f}")
    print(f"Accuracy : {best_info['accuracy']:.6f}")

    val_thr05 = apply_threshold_and_format(val_df, 0.5)
    val_best = apply_threshold_and_format(val_df, best_thr)
    test_thr05 = apply_threshold_and_format(test_df, 0.5)
    test_best = apply_threshold_and_format(test_df, best_thr)

    val_thr05_path = os.path.join(args.output_dir, "val_predictions_thr_0.5.csv")
    val_best_path = os.path.join(args.output_dir, "val_predictions_best_thr.csv")
    test_thr05_path = os.path.join(args.output_dir, "test_predictions_thr_0.5.csv")
    test_best_path = os.path.join(args.output_dir, "test_predictions_best_thr.csv")
    summary_path = os.path.join(args.output_dir, "threshold_summary.json")

    val_thr05.to_csv(val_thr05_path, index=False)
    val_best.to_csv(val_best_path, index=False)
    test_thr05.to_csv(test_thr05_path, index=False)
    test_best.to_csv(test_best_path, index=False)

    # extra split csv for test set
    best_thr_covid_csv = os.path.join(args.output_dir, "best_thr_covid.csv")
    best_thr_noncovid_csv = os.path.join(args.output_dir, "best_thr_non-covid.csv")
    thr05_covid_csv = os.path.join(args.output_dir, "thr_0.5_covid.csv")
    thr05_noncovid_csv = os.path.join(args.output_dir, "thr_0.5_non-covid.csv")

    save_name_only_csv(test_best, "covid", best_thr_covid_csv)
    save_name_only_csv(test_best, "non-covid", best_thr_noncovid_csv)
    save_name_only_csv(test_thr05, "covid", thr05_covid_csv)
    save_name_only_csv(test_thr05, "non-covid", thr05_noncovid_csv)

    summary = {
        "checkpoint_dir": args.checkpoint_dir,
        "model_id": args.model_id,
        "num_slices_front_back": args.num_slices,
        "actual_total_slices_used": 2 * args.num_slices,
        "best_threshold": best_thr,
        "best_result": best_info,
        "files": {
            "val_predictions_thr_0.5": val_thr05_path,
            "val_predictions_best_thr": val_best_path,
            "test_predictions_thr_0.5": test_thr05_path,
            "test_predictions_best_thr": test_best_path,
            "best_thr_covid": best_thr_covid_csv,
            "best_thr_non-covid": best_thr_noncovid_csv,
            "thr_0.5_covid": thr05_covid_csv,
            "thr_0.5_non-covid": thr05_noncovid_csv,
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[done] saved files:")
    print(" ", val_thr05_path)
    print(" ", val_best_path)
    print(" ", test_thr05_path)
    print(" ", test_best_path)
    print(" ", best_thr_covid_csv)
    print(" ", best_thr_noncovid_csv)
    print(" ", thr05_covid_csv)
    print(" ", thr05_noncovid_csv)
    print(" ", summary_path)


if __name__ == "__main__":
    main()