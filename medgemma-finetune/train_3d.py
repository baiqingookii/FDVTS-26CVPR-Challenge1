#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from transformers import EarlyStoppingCallback
import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize, InterpolationMode
import sys, time
sys.stdout.reconfigure(line_buffering=True)
from datasets import Dataset, Image
import evaluate
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from torchvision.transforms import (
    Compose,
    CenterCrop,
    Resize,
    ToTensor,
    Normalize,
    InterpolationMode,
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys, time
medgemma_ROOT = "/remote-home/share/25-jianfabai/medgemma-finetune"
sys.path.insert(0, medgemma_ROOT)
from transformers import TrainerCallback
from dataset_1 import NPYVolumeDataset, volume_collate_fn
# from dataset import NPYVolumeDataset, volume_collate_fn


NPY_EXTS = (".npy",)

def list_npys_recursively(root: str):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(NPY_EXTS):
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)

class EmptyCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

    def on_save(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()


# class SliceMeanPoolClassifier(nn.Module):
#     def __init__(self, base_model, chunk_size=3):
#         super().__init__()
#         self.base = base_model
#         self.chunk_size = chunk_size

#     # ✅ 让 Trainer 能调用到
#     # def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
#     #     if hasattr(self.base, "gradient_checkpointing_enable"):
#     #         return self.base.gradient_checkpointing_enable(
#     #             gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
#     #         )

#     # def gradient_checkpointing_disable(self):
#     #     if hasattr(self.base, "gradient_checkpointing_disable"):
#     #         return self.base.gradient_checkpointing_disable()

#     def forward(self, pixel_values=None, labels=None):
#         B, S, C, H, W = pixel_values.shape
#         flat = pixel_values.reshape(B * S, C, H, W)

#         logits_chunks = []
#         for i in range(0, B * S, self.chunk_size):
#             out = self.base(pixel_values=flat[i:i+self.chunk_size])
#             logits_chunks.append(out.logits)

#         logits = torch.cat(logits_chunks, dim=0)   # (B*S,2)
#         # logits = logits.reshape(B, S, -1).mean(dim=1)
#         #修改
#         logits = logits.reshape(B, S, -1)
#         logits = torch.logsumexp(logits, dim=1) - math.log(S)   # 等价于 soft-max pooling

#         loss = None
#         if labels is not None:
#             loss = F.cross_entropy(logits, labels)
#         return {"loss": loss, "logits": logits}

class SliceMeanPoolClassifier(nn.Module):
    def __init__(self, base_model, chunk_size=3, label_smoothing: float = 0.0, pool: str = "lse"):
        super().__init__()
        self.base = base_model
        self.chunk_size = chunk_size
        self.label_smoothing = float(label_smoothing)
        assert pool in {"mean", "max", "lse"}
        self.pool = pool

    def forward(self, pixel_values=None, labels=None):
        B, S, C, H, W = pixel_values.shape
        flat = pixel_values.reshape(B * S, C, H, W)

        logits_chunks = []
        for i in range(0, B * S, self.chunk_size):
            out = self.base(pixel_values=flat[i:i+self.chunk_size])
            logits_chunks.append(out.logits)

        logits = torch.cat(logits_chunks, dim=0)   # (B*S,2)
        logits = logits.reshape(B, S, -1)

        # --- Minimal but high-impact improvement ---
        # Mean over slices tends to dilute sparse-lesion signals in CT.
        # Use LogSumExp pooling (a smooth "soft-max" MIL pooling) by default.
        if self.pool == "mean":
            logits = logits.mean(dim=1)
        elif self.pool == "max":
            logits = logits.max(dim=1).values
        else:  # "lse"
            logits = torch.logsumexp(logits, dim=1) - math.log(S)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        return {"loss": loss, "logits": logits}

def list_images_recursively(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def build_dataset(covid_root: str, noncovid_root: str) -> Dataset:
    covid_imgs = list_images_recursively(covid_root)
    noncovid_imgs = list_images_recursively(noncovid_root)

    if len(covid_imgs) == 0:
        raise RuntimeError(f"No images found under covid_root: {covid_root}")
    if len(noncovid_imgs) == 0:
        raise RuntimeError(f"No images found under noncovid_root: {noncovid_root}")

    # label: 1=covid, 0=non-covid
    images = covid_imgs + noncovid_imgs
    labels = [1] * len(covid_imgs) + [0] * len(noncovid_imgs)

    ds = Dataset.from_dict({"image": images, "label": labels})
    ds = ds.cast_column("image", Image())  # decode as PIL at runtime
    return ds

# def volume_collate_fn(examples):
#     pixel_values = torch.stack([ex["pixel_values"] for ex in examples], dim=0)  # (B,32,3,448,448)
#     labels = torch.stack([ex["labels"] for ex in examples], dim=0)              # (B,)
#     return {"pixel_values": pixel_values, "labels": labels}

@dataclass
class DataCollator:
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples], dim=0)
        labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    #输入错误
    # parser.add_argument("--train_covid_root", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/train/covid_ori_and_1b")
    # parser.add_argument("--train_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/train/non-covid_ori_and_1b")
    # parser.add_argument("--val_covid_root", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/covid_ori_and_1b")
    # parser.add_argument("--val_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/non-covid_ori_and_1b")

    parser.add_argument("--train_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/train/covid1b_64_448_448")
    parser.add_argument("--train_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/train/non-covid1b_64_448_448")
    parser.add_argument("--val_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/covid1b_64_448_448")
    parser.add_argument("--val_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/non-covid1b_64_448_448")

    # parser.add_argument("--train_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/train/covid_ori_and_1b")
    # parser.add_argument("--train_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/train/non-covid_ori_and_1b")
    # parser.add_argument("--val_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/covid1b")
    # parser.add_argument("--val_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/non-covid1b")

    parser.add_argument("--model_id", type=str, default="/root/medsiglip-448")
    parser.add_argument("--output_dir", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/save_model/64*448*448_3d_1_1b_dataset1_mean_lr5e-5_scale0.7-1")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-5)
    # parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--logging_steps", type=int, default=40)
    parser.add_argument("--eval_steps", type=int, default=40)
    parser.add_argument("--warmup_steps", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--fp16", action="store_true")

    # By default don't push to hub (avoid auth issues on servers)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--early_stop_patience", type=int, default=16)
    # parser.add_argument("--early_stop_patience", type=int, default=6)
    # parser.add_argument("--pool", type=str, default="lse", choices=["mean", "max", "lse"])
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "max", "lse"])

    args = parser.parse_args()
    set_seed(args.seed)

    # 1) Build datasets
    # train_ds = NPYVolumeDataset(args.train_covid_root, args.train_noncovid_root, num_slices=12, out_size=448)
    # val_ds   = NPYVolumeDataset(args.val_covid_root, args.val_noncovid_root, num_slices=12, out_size=448)

    image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    size = image_processor.size["height"]  # typically 448
    # size = 256  # typically 448
    mean = image_processor.image_mean      # typically 0.5
    std = image_processor.image_std        # typically 0.5
    print(f"size:{size}, mean:{mean}, std:{std}")

    train_ds = NPYVolumeDataset(
        args.train_covid_root,
        args.train_noncovid_root,
        num_slices=12,               # 训练 S=12
        out_size=size,
        split="train",
        # crop_scale=(0.8, 1.0),       
        crop_scale=(0.7, 1.0), 
        mean=mean,
        std=std,
        mmap=True,
        assume_uint8_0_255=True,     # 如果 norm npy 已经不是 0~255，这里要改
    )

    val_ds = NPYVolumeDataset(
        args.val_covid_root,
        args.val_noncovid_root,
        num_slices=12,               # ✅ val 会变成 2*num_slices=24
        out_size=size,
        split="val",                 # ✅ 不做 crop 增强
        mean=mean,
        std=std,
        mmap=True,
        assume_uint8_0_255=True,
    )


    print(f"[data] train: {len(train_ds)}  val: {len(val_ds)}")
    print(f"[data] train covid_root={args.train_covid_root}")
    print(f"[data] train noncovid_root={args.train_noncovid_root}")
    print(f"[data] val covid_root={args.val_covid_root}")
    print(f"[data] val noncovid_root={args.val_noncovid_root}")

    transform = Compose([
        Resize((size, size), interpolation=InterpolationMode.BILINEAR),
        ToTensor(),                        # [0,255] -> [0,1]
        Normalize(mean=mean, std=std),      # -> roughly [-1,1] if mean=std=0.5
    ])

    # def preprocess(examples):
    #     # CenterCrop(max(image.size)) can "square-pad" effectively
    #     pixel_values = []
    #     for img in examples["image"]:
    #         img = img.convert("RGB")
    #         img = CenterCrop(max(img.size))(img)
    #         pixel_values.append(transform(img))
    #     examples["pixel_values"] = pixel_values
    #     return examples


    # 3) Model
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
    #再加一条“硬保命”：gradient checkpointing（非常推荐）这条对 ViT/SigLIP 特别有效，显存能明显下降：
    # base_model.gradient_checkpointing_enable()

    # model = SliceMeanPoolClassifier(base_model)
    model = SliceMeanPoolClassifier(
        base_model,
        # chunk_size=3,
        chunk_size=2,
        label_smoothing=args.label_smoothing,
        pool=args.pool,
    )

    # 4) Metrics (Accuracy + ROC-AUC)
    accuracy = evaluate.load("/remote-home/share/25-jianfabai/medgemma-finetune/metric/accuracy.py")
    roc_auc = evaluate.load("/remote-home/share/25-jianfabai/medgemma-finetune/metric/roc_auc.py")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # softmax prob of class 1 for AUC
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        prob_pos = probs[:, 1]

        out = {}
        out.update(accuracy.compute(predictions=preds, references=labels))
        # roc_auc expects "prediction_scores" for binary
        out["roc_auc"] = roc_auc.compute(
            prediction_scores=prob_pos,
            references=labels
        )["roc_auc"]
        return out

    # 5) Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,

        logging_steps=args.logging_steps,

        # evaluation_strategy="epoch",   # ✅ 每个 epoch eval 一次
        eval_strategy="epoch",
        save_strategy="epoch",         # ✅ 每个 epoch 存一次（但会被下面规则裁剪）
        save_total_limit=1,            # ✅ 只保留一个checkpoint

        load_best_model_at_end=True,   # ✅ 训练结束自动加载最佳模型
        # metric_for_best_model="accuracy",
        metric_for_best_model="roc_auc",
        greater_is_better=True,

        # warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=0,
        lr_scheduler_type="cosine",

        report_to=["tensorboard"],
        fp16=args.fp16,

        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,

        push_to_hub=args.push_to_hub,
        ddp_find_unused_parameters=True,

        # gradient_checkpointing=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=volume_collate_fn,
        compute_metrics=compute_metrics,
        # callbacks=[EmptyCacheCallback()],
        callbacks=[
            EmptyCacheCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stop_patience,
                early_stopping_threshold=2e-5,
                # early_stopping_threshold=1e-4,
            ),
        ],
    )

    # 6) Train & Save
    trainer.train()
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)

    print(f"[done] saved to: {args.output_dir}")


if __name__ == "__main__":
    main()