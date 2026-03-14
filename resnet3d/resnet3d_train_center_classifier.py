#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import time
import random
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

CVPR26_ROOT = "/remote-home/share/25-jianfabai/cvpr2026"
import sys
sys.path.insert(0, CVPR26_ROOT)
sys.stdout.reconfigure(line_buffering=True)
from models.ResNet import SupConResNet
from dataset1 import rescale_z, normalize



def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class CenterNPYDataset(Dataset):
    """
    读取结构:
        root/
            center_0/
                *.npy
            center_1/
                *.npy
            center_2/
                *.npy
            center_3/
                *.npy
    """
    def __init__(self, root_dir, img_depth=64):
        self.root_dir = root_dir
        self.img_depth = img_depth
        self.samples = []

        center_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for d in center_dirs:
            m = re.search(r"center[_-]?(\d+)", d)
            if m is None:
                print(f"[Warn] skip folder: {d}")
                continue
            center_label = int(m.group(1))
            folder = os.path.join(root_dir, d)

            for fn in sorted(os.listdir(folder)):
                if fn.endswith(".npy"):
                    fpath = os.path.join(folder, fn)
                    self.samples.append((fpath, center_label, fn))

        print(f"[Dataset] {root_dir}")
        print(f"[Dataset] num_samples = {len(self.samples)}")
        print(f"[Dataset] label_dist = {Counter([x[1] for x in self.samples])}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label, fname = self.samples[idx]

        img_array = np.load(fpath).astype(np.float32)   # (D,H,W)

        if self.img_depth != 128:
            img_array = rescale_z(img_array, self.img_depth)

        img_array = torch.from_numpy(normalize(img_array)).float()   # (D,H,W)

        return img_array, label, fname


def collate_fn(batch):
    imgs, labels, fnames = zip(*batch)
    imgs = torch.stack(imgs, dim=0)          # (B,D,H,W)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels, list(fnames)


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


def load_pretrained_backbone(model, ckpt_path, device="cpu"):
    print(f"[Load] ckpt = {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "net" in ckpt:
        state_dict = ckpt["net"]
    else:
        state_dict = ckpt

    state_dict = strip_module_prefix(state_dict)

    model_dict = model.state_dict()
    loadable = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped.append(k)

    model_dict.update(loadable)
    model.load_state_dict(model_dict, strict=False)

    print(f"[Load] loaded params: {len(loadable)}")
    print(f"[Load] skipped params: {len(skipped)}")
    if len(skipped) > 0:
        print("[Load] first skipped keys:")
        for x in skipped[:20]:
            print("   ", x)

    return model


def set_trainable_params(model, mode="linear"):
    for p in model.parameters():
        p.requires_grad = False

    if mode == "linear":
        enabled = 0
        for name, p in model.named_parameters():
            if (
                "encoder.fc" in name or
                "encoder.classifier" in name or
                name.endswith(".fc.weight") or
                name.endswith(".fc.bias") or
                "classifier.weight" in name or
                "classifier.bias" in name
            ):
                p.requires_grad = True
                enabled += p.numel()
        print(f"[Trainable] linear probe only")
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        print(f"[Trainable] full finetune")
    else:
        raise ValueError("mode must be 'linear' or 'full'")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Params] trainable = {trainable}, total = {total}")


def set_bn_eval(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()

def set_linear_probe_mode(model):
    if isinstance(model, nn.DataParallel):
        model.module.encoder.eval()
        model.module.encoder.apply(set_bn_eval)
    else:
        model.encoder.eval()
        model.encoder.apply(set_bn_eval)

def get_classifier_params(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return params


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


def evaluate(model, loader, device, criterion, save_csv_path=None):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_names = []
    running_loss = 0.0
    n = 0

    with torch.no_grad():
        for imgs, labels, fnames in loader:
            imgs = imgs.unsqueeze(1).to(device, non_blocking=True)   # (B,1,D,H,W)
            labels = labels.to(device, non_blocking=True)

            _, logits = forward_model(model, imgs)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n += bs

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_names.extend(fnames)

    avg_loss = running_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    if save_csv_path is not None:
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "true_center", "pred_center", "prob_c0", "prob_c1", "prob_c2", "prob_c3"])
            for name, yt, yp, prob in zip(all_names, all_labels, all_preds, all_probs):
                row = [name, yt, yp] + [float(x) for x in prob]
                writer.writerow(row)

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "cm": cm,
        "labels": all_labels,
        "preds": all_preds,
    }


def save_checkpoint(path, model, optimizer, epoch, best_metric, args):
    state = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_macro_f1": best_metric,
        "args": vars(args)
    }
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_root", type=str,
                        default="/remote-home/share/25-jianfabai/cvpr2026/center/train_by_center")
    parser.add_argument("--valid_root", type=str,
                        default="/remote-home/share/25-jianfabai/cvpr2026/center/valid_by_center")
    parser.add_argument("--ckpt", type=str,
                        default="/remote-home/share/25-jianfabai/cvpr2026/checkpoints/c2m5-ori_and_1b-3090-nomix/epoch073_f10.9710.pkl")
    parser.add_argument("--outdir", type=str,
                        default="/remote-home/share/25-jianfabai/cvpr2026/center/resnet3d/linear_center_cls_from_c2m5_ori_and_1b")
    parser.add_argument("--model", type=str, default="resnest50_3D")
    parser.add_argument("--img_depth", type=int, default=64)
    # parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_mode", type=str, default="linear", choices=["linear", "full"])
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ===== Dataset =====
    train_set = CenterNPYDataset(args.train_root, img_depth=args.img_depth)
    valid_set = CenterNPYDataset(args.valid_root, img_depth=args.img_depth)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ===== Model =====
    model = build_model(backbone_name=args.model, n_classes=4, supcon=False)
    model = load_pretrained_backbone(model, args.ckpt, device="cpu")
    set_trainable_params(model, mode=args.train_mode)

    model = nn.DataParallel(model)
    model = model.to(device)

    # ===== Loss =====
    if args.use_class_weight:
        train_labels = [x[1] for x in train_set.samples]
        cnt = Counter(train_labels)
        weights = []
        for i in range(4):
            weights.append(len(train_labels) / max(cnt.get(i, 1), 1))
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum() * len(weights)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print("[Loss] class_weights =", class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ===== Optimizer =====
    optimizer = torch.optim.Adam(
        get_classifier_params(model),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_f1 = -1.0
    best_epoch = -1

    summary_path = os.path.join(args.outdir, "train_log.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("start training\n")

    for epoch in range(1, args.epochs + 1):
        model.train()

        if args.train_mode == "linear":
            set_linear_probe_mode(model)

        running_loss = 0.0
        all_labels = []
        all_preds = []
        n = 0

        t0 = time.time()

        for imgs, labels, _ in train_loader:
            imgs = imgs.unsqueeze(1).to(device, non_blocking=True)   # (B,1,D,H,W)
            labels = labels.to(device, non_blocking=True)

            _, logits = forward_model(model, imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n += bs

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())

        train_loss = running_loss / max(n, 1)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average="macro")

        valid_csv = os.path.join(args.outdir, f"valid_epoch{epoch:03d}.csv")
        val_res = evaluate(model, valid_loader, device, criterion, save_csv_path=valid_csv)

        epoch_time = time.time() - t0

        log_str = (
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_res['loss']:.4f} val_acc={val_res['acc']:.4f} val_f1={val_res['macro_f1']:.4f} | "
            f"time={epoch_time:.1f}s"
        )
        print(log_str)

        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")
            f.write(str(val_res["cm"]) + "\n")

        if epoch % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.outdir, f"epoch{epoch:03d}.pkl"),
                model, optimizer, epoch, best_f1, args
            )

        if val_res["macro_f1"] > best_f1:
            best_f1 = val_res["macro_f1"]
            best_epoch = epoch
            best_path = os.path.join(args.outdir, f"best_epoch{epoch:03d}_f1{best_f1:.4f}.pkl")
            save_checkpoint(best_path, model, optimizer, epoch, best_f1, args)

            print("[Best] confusion matrix:")
            print(val_res["cm"])
            print("[Best] classification report:")
            print(classification_report(val_res["labels"], val_res["preds"], digits=4))

    final_res = evaluate(
        model, valid_loader, device, criterion,
        save_csv_path=os.path.join(args.outdir, "valid_final.csv")
    )

    result_json = {
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "final_val_loss": final_res["loss"],
        "final_val_acc": final_res["acc"],
        "final_val_macro_f1": final_res["macro_f1"],
        "final_confusion_matrix": final_res["cm"].tolist(),
    }
    with open(os.path.join(args.outdir, "result_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print("\n========== Training Finished ==========")
    print(f"best_epoch = {best_epoch}")
    print(f"best_macro_f1 = {best_f1:.4f}")
    print("final confusion matrix:")
    print(final_res["cm"])
    print(classification_report(final_res["labels"], final_res["preds"], digits=4))


if __name__ == "__main__":
    main()