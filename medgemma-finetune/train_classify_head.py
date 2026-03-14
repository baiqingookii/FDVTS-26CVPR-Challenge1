import math
from transformers import EarlyStoppingCallback
import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from safetensors.torch import load_file as safe_load
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
from dataset import NPYVolumeDataset, volume_collate_fn


def _get_vision_encoder(base_model):
    """
    Try best-effort to locate the vision backbone inside a HF image classification model.
    Works for many ViT/SigLIP-like wrappers.
    """
    if hasattr(base_model, "vision_model"):
        return base_model.vision_model
    if hasattr(base_model, "model") and hasattr(base_model.model, "vision_model"):
        return base_model.model.vision_model
    # fallback: maybe base_model itself behaves like vision encoder
    return base_model

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


# class SliceTransformerClassifier(nn.Module):
# class SliceMeanClassifier(nn.Module):
#     def __init__(
#         self,
#         base_model,
#         chunk_size=2,
#         max_slices=24,
#         n_layers=2,
#         n_heads=8,
#         mlp_ratio=4.0,
#         dropout=0.1,
#         use_cls_token=True,
#         label_smoothing=0.0,
#         pool="cls",   # "cls" or "mean"
#     ):
#         super().__init__()
#         self.base_model = base_model
#         self.chunk_size = chunk_size
#         self.label_smoothing = float(label_smoothing)

#         # ---- freeze whole base model (MedSigLIP) ----
#         for p in self.base_model.parameters():
#             p.requires_grad = False

#         # self.vision = _get_vision_encoder(self.base_model)

#         # ---- infer hidden size (D) robustly ----
#         D = None
#         if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "hidden_size"):
#             D = self.base_model.config.hidden_size
#         if D is None and hasattr(self.vision, "config") and hasattr(self.vision.config, "hidden_size"):
#             D = self.vision.config.hidden_size
#         if D is None:
#             # will be set on first forward if needed
#             D = 768
#         self.hidden_dim = D

#         self.max_slices = max_slices
#         # self.use_cls_token = use_cls_token
#         # assert pool in {"cls", "mean"}
#         # self.pool = pool

#         # # ---- positional embeddings ----
#         # # If using cls_token, sequence length = 1 + S
#         # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim)) if use_cls_token else None
#         # self.pos_embed = nn.Parameter(
#         #     torch.zeros(1, (1 if use_cls_token else 0) + max_slices, self.hidden_dim)
#         # )
#         # nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         # if self.cls_token is not None:
#         #     nn.init.trunc_normal_(self.cls_token, std=0.02)

#         # self.pos_drop = nn.Dropout(dropout)

#         # # ---- slice-level TransformerEncoder ----
#         # encoder_layer = nn.TransformerEncoderLayer(
#         #     d_model=self.hidden_dim,
#         #     nhead=n_heads,
#         #     dim_feedforward=int(self.hidden_dim * mlp_ratio),
#         #     dropout=dropout,
#         #     activation="gelu",
#         #     batch_first=True,   # important: (B, T, D)
#         #     norm_first=True,
#         # )
#         # self.slice_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

#         # self.norm = nn.LayerNorm(self.hidden_dim)
#         self.head = nn.Linear(self.hidden_dim, 2)

#     @property
#     def vision(self):
#         return _get_vision_encoder(self.base_model)

#     @torch.no_grad()
#     def _extract_slice_features(self, flat_pixel_values):
#         """
#         flat_pixel_values: (B*S, 3, H, W)
#         returns: (B*S, D)
#         """
#         # Ask for hidden states if available; but pooled is usually enough.
#         out = self.vision(pixel_values=flat_pixel_values, output_hidden_states=False, return_dict=True)

#         if hasattr(out, "pooler_output") and out.pooler_output is not None:
#             feats = out.pooler_output  # (N, D)
#         elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
#             feats = out.last_hidden_state[:, 0]  # CLS token
#         else:
#             raise RuntimeError("Vision encoder output has neither pooler_output nor last_hidden_state.")
#         return feats

#     def forward(self, pixel_values=None, labels=None):
#         """
#         pixel_values: (B, S, 3, H, W)
#         """
#         B, S, C, H, W = pixel_values.shape
#         assert S <= self.max_slices, f"S={S} exceeds max_slices={self.max_slices}"

#         flat = pixel_values.reshape(B * S, C, H, W)

#         # ---- chunked feature extraction to save memory ----
#         feats_chunks = []
#         for i in range(0, B * S, self.chunk_size):
#             # NOTE: base is frozen; keep no_grad for speed & memory
#             feats = self._extract_slice_features(flat[i:i+self.chunk_size])
#             feats_chunks.append(feats)
#         feats = torch.cat(feats_chunks, dim=0)  # (B*S, D)

#         # x = feats.view(B, S, -1)  # (B, S, D)

#         # # ---- add cls token + pos embedding ----
#         # if self.use_cls_token:
#         #     cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
#         #     #修改
#         #     # cls = self.cls_token.repeat(B, 1, 1)   # ✅ 替代 expand
#         #     x = torch.cat([cls, x], dim=1)          # (B,1+S,D)
#         #     x = x + self.pos_embed[:, : (1 + S), :]
#         # else:
#         #     x = x + self.pos_embed[:, :S, :]

#         # x = self.pos_drop(x)

#         # #修改
#         # x = x.contiguous()  # ✅ 进 encoder 前做一次
#         # # ---- transformer over slices ----
#         # x = self.slice_encoder(x)  # (B, T, D)
#         # x = self.norm(x)

#         # # ---- pooling ----
#         # if self.pool == "cls":
#         #     if not self.use_cls_token:
#         #         # fallback
#         #         pooled = x.mean(dim=1)
#         #     else:
#         #         pooled = x[:, 0]
#         # else:  # "mean"
#         #     if self.use_cls_token:
#         #         pooled = x[:, 1:].mean(dim=1)
#         #     else:
#         #         pooled = x.mean(dim=1)

#         # logits = self.head(pooled)  # (B,2)
#         x = feats.view(B, S, -1)      # (B, S, D)
#         pooled = x.mean(dim=1)        # (B, D)
#         logits = self.head(pooled)    # (B, 2)

#         loss = None
#         if labels is not None:
#             loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

#         return {"loss": loss, "logits": logits}

class SliceFlattenClassifier(nn.Module):
    def __init__(
        self,
        base_model,
        chunk_size=2,
        max_slices=24,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.chunk_size = chunk_size
        self.label_smoothing = float(label_smoothing)
        self.max_slices = max_slices

        # freeze whole base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        # infer hidden size
        D = None
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "hidden_size"):
            D = self.base_model.config.hidden_size
        if D is None and hasattr(self.vision, "config") and hasattr(self.vision.config, "hidden_size"):
            D = self.vision.config.hidden_size
        if D is None:
            D = 768
        self.hidden_dim = D

        in_dim = self.max_slices * self.hidden_dim
        self.norm = nn.LayerNorm(in_dim)
        self.head = nn.Linear(in_dim, 2)

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
            feats = out.pooler_output           # (N, D)
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            feats = out.last_hidden_state[:, 0] # (N, D)
        else:
            raise RuntimeError("Vision encoder output has neither pooler_output nor last_hidden_state.")
        return feats

    def forward(self, pixel_values=None, labels=None):
        """
        pixel_values: (B, S, 3, H, W)
        """
        B, S, C, H, W = pixel_values.shape
        assert S <= self.max_slices, f"S={S} exceeds max_slices={self.max_slices}"

        flat = pixel_values.reshape(B * S, C, H, W)

        feats_chunks = []
        for i in range(0, B * S, self.chunk_size):
            feats = self._extract_slice_features(flat[i:i+self.chunk_size])
            feats_chunks.append(feats)

        feats = torch.cat(feats_chunks, dim=0)   # (B*S, D)
        x = feats.view(B, S, -1)                 # (B, S, D)

        # 如果 val 可能是少于 max_slices，这里补零到固定长度
        if S < self.max_slices:
            pad = x.new_zeros(B, self.max_slices - S, x.size(-1))
            x = torch.cat([x, pad], dim=1)       # (B, max_slices, D)

        flat_feats = x.reshape(B, -1)            # (B, max_slices*D)
        flat_feats = self.norm(flat_feats)
        logits = self.head(flat_feats)           # (B, 2)

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
    # parser.add_argument("--train_covid_root", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/train/covid_ori_and_1b")
    # parser.add_argument("--train_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/train/non-covid_ori_and_1b")
    # parser.add_argument("--val_covid_root", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/covid_ori_and_1b")
    # parser.add_argument("--val_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid/non-covid_ori_and_1b")
    # parser.add_argument("--train_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/train/covid1b_64_448_448")
    # parser.add_argument("--train_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/train/non-covid1b_64_448_448")
    # parser.add_argument("--val_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/covid1b_64_448_448")
    # parser.add_argument("--val_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/non-covid1b_64_448_448")
    parser.add_argument("--train_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/train/covid_ori_and_1b")
    parser.add_argument("--train_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/train/non-covid_ori_and_1b")
    parser.add_argument("--val_covid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/covid1b")
    parser.add_argument("--val_noncovid_root", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/valid/non-covid1b")


    parser.add_argument("--model_id", type=str, default="/root/medsiglip-448")
    parser.add_argument("--model_dir", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_mean_lr5e-5_scale0.7-1_ori_and_1b/checkpoint-972")  
    
    parser.add_argument("--output_dir", type=str, default="/remote-home/share/25-jianfabai/medgemma-finetune/save_model/flat_block_classify/3d_ori-and-1b_dataset_mean_lr5e-5_scale0.7-1-lr5e-6")

    parser.add_argument("--epochs", type=int, default=60)
    # parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr", type=float, default=5e-6)
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
    parser.add_argument("--early_stop_patience", type=int, default=8)
    # parser.add_argument("--early_stop_patience", type=int, default=6)
    # parser.add_argument("--pool", type=str, default="lse", choices=["mean", "max", "lse"])
    # parser.add_argument("--pool", type=str, default="mean", choices=["mean", "max", "lse"])

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
        num_slices=24,               # 训练 S=12
        out_size=size,
        split="train",
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

    # ===== 新增：检查一个样本 =====
    sample = train_ds[0]
    x = sample["pixel_values"]   # shape: (S, 3, H, W)
    print("single sample shape:", x.shape)
    print(
        "single sample min/max/mean/std:",
        x.min().item(),
        x.max().item(),
        x.mean().item(),
        x.std().item(),
    )
    print("single sample label:", sample["labels"] if "labels" in sample else sample["label"])


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

    model = SliceFlattenClassifier(
        base_model,
        chunk_size=4,
        max_slices=24,
        label_smoothing=args.label_smoothing,
    )

    ckpt_path = os.path.join(args.model_dir, "model.safetensors")  # args.model_dir = .../checkpoint-697
    state = safe_load(ckpt_path)

    #查询
    # keys = list(state.keys())
    # print("ckpt key examples:", keys[:30])
    # mkeys = list(model.state_dict().keys())
    # print("model key examples:", mkeys[:30])

    model_state = model.state_dict()
    remapped = {}
    for k, v in state.items():
        # 只要 backbone（跳过旧的分类头）
        # if k.startswith("base.classifier."):
        #     continue
        if k.startswith("base."):
            nk = "base_model." + k[len("base."):]   # base.xxx -> base_model.xxx
            # 只加载新模型里存在且shape一致的
            if nk in model_state and v.shape == model_state[nk].shape:
                remapped[nk] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    print("[remap-load] loaded keys:", len(remapped))
    print("[remap-load] missing:", len(missing))
    print("[remap-load] unexpected:", len(unexpected))
    print("  missing ex:", missing[:10])
    print("  unexpected ex:", unexpected[:10])
    # missing, unexpected = model.load_state_dict(state, strict=False)
    # print("[load] missing keys:", len(missing))
    # print("[load] unexpected keys:", len(unexpected))
    # if len(unexpected) > 0:
    #     print("  unexpected examples:", unexpected[:20])
    # if len(missing) > 0:
    #     print("  missing examples:", missing[:20])

    #添加
    for p in model.parameters():
        if p.requires_grad and not p.data.is_contiguous():
            p.data = p.data.contiguous()

    print("===== Trainable parameters =====")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)

    print("model hidden_dim =", model.hidden_dim)

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
        metric_for_best_model="accuracy",
        # metric_for_best_model="roc_auc",
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
        # ddp_find_unused_parameters=True,
        #修改
        ddp_find_unused_parameters=False,

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