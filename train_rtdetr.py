#!/usr/bin/env python3
"""
Stage 2 (Upgrade): Train RT-DETR on CARLA collected data.

RT-DETR = ResNet backbone (CNN) + Transformer encoder-decoder (attention).
Uses pretrained COCO weights and fine-tunes on our CARLA dataset.

Usage:
    python train_rtdetr.py
    python train_rtdetr.py --epochs 30 --batch-size 4 --lr 0.0002

Comparison with Faster R-CNN:
    python train_detector.py    → pure CNN, ~2 hours, baseline
    python train_rtdetr.py      → CNN + Transformer, ~5-8 hours, better accuracy
"""

import argparse
import glob
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_rtdetr")


# Our class mapping
CLASS_NAMES = {
    0: "vehicle",
    1: "pedestrian",
    2: "rider",
    3: "bicycle",
    4: "motorcycle",
    5: "bus",
    6: "truck",
}


class CARLADetectionDatasetRTDETR(Dataset):
    """
    Dataset for RT-DETR training from CARLA collected data.

    RT-DETR (via HuggingFace) expects COCO-format annotations:
        - boxes in [cx, cy, w, h] normalized to [0, 1]
        - class_labels as a list of ints (0-indexed, no background class)
    """

    def __init__(
        self,
        run_dirs: list[str],
        processor: RTDetrImageProcessor,
        min_visibility: float = 0.2,
        max_objects: int = 100,
    ):
        self.processor = processor
        self.min_visibility = min_visibility
        self.max_objects = max_objects

        # Index all frames
        self.samples = []
        for run_dir in run_dirs:
            run_path = Path(run_dir)
            ann_dir = run_path / "annotations"
            rgb_dir = run_path / "rgb"
            if not ann_dir.exists() or not rgb_dir.exists():
                continue

            ann_files = sorted(ann_dir.glob("*.json"))
            for ann_file in ann_files:
                frame_name = ann_file.stem
                rgb_path = rgb_dir / f"{frame_name}.png"
                if not rgb_path.exists():
                    rgb_path = rgb_dir / f"{frame_name}.jpg"
                    if not rgb_path.exists():
                        continue

                self.samples.append({
                    "rgb_path": str(rgb_path),
                    "ann_path": str(ann_file),
                })

        print(f"[RT-DETR Dataset] Loaded {len(self.samples)} samples from {len(run_dirs)} runs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image as RGB
        image = cv2.imread(sample["rgb_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Load annotation
        with open(sample["ann_path"]) as f:
            ann = json.load(f)

        # Extract boxes and labels in COCO format
        boxes = []
        class_labels = []

        for actor in ann.get("actors", []):
            bbox_2d = actor.get("bbox_2d")
            if bbox_2d is None:
                continue

            vis = actor.get("visibility", 0.0)
            if vis < self.min_visibility:
                continue

            x_min, y_min, x_max, y_max = bbox_2d

            if x_max <= x_min or y_max <= y_min:
                continue
            if x_max - x_min < 2 or y_max - y_min < 2:
                continue

            # Convert to COCO format: [cx, cy, w, h] normalized
            cx = ((x_min + x_max) / 2.0) / w
            cy = ((y_min + y_max) / 2.0) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h

            # Clamp to valid range
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))

            boxes.append([cx, cy, bw, bh])
            class_labels.append(actor["class_id"])

        # Limit objects
        boxes = boxes[:self.max_objects]
        class_labels = class_labels[:self.max_objects]

        # Build COCO-format annotation for the processor
        annotations = {
            "image_id": idx,
            "annotations": [
                {
                    "bbox": [
                        boxes[i][0] - boxes[i][2] / 2,  # x_min normalized
                        boxes[i][1] - boxes[i][3] / 2,  # y_min normalized
                        boxes[i][2],  # width normalized
                        boxes[i][3],  # height normalized
                    ],
                    "category_id": class_labels[i],
                    "area": boxes[i][2] * boxes[i][3] * w * h,
                    "iscrowd": 0,
                }
                for i in range(len(boxes))
            ],
        }

        # Process image and annotations through RT-DETR processor
        result = self.processor(
            images=image,
            annotations=[annotations],
            return_tensors="pt",
        )

        # Squeeze batch dimension (processor adds it)
        pixel_values = result["pixel_values"].squeeze(0)
        labels = result["labels"][0]  # dict with class_labels, boxes, etc.

        return pixel_values, labels


def collate_fn(batch):
    """Custom collate: stack pixel values, keep labels as list."""
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    loss_components = {}
    num_batches = 0

    for batch_idx, batch in enumerate(data_loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        # Skip if no valid labels
        valid = [l for l in labels if l["class_labels"].shape[0] > 0]
        if len(valid) == 0:
            continue

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Track individual losses
        if hasattr(outputs, "loss_dict"):
            for k, v in outputs.loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v.item()

        if (batch_idx + 1) % 20 == 0:
            avg = total_loss / num_batches
            logger.info(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(data_loader)} | Loss: {avg:.4f}")

    if num_batches == 0:
        return {"total_loss": 0.0}

    avg_losses = {k: v / num_batches for k, v in loss_components.items()}
    avg_losses["total_loss"] = total_loss / num_batches
    return avg_losses


@torch.no_grad()
def evaluate(model, processor, data_loader, device):
    model.eval()
    total_images = 0
    total_gt = 0
    total_preds = 0
    high_conf = 0

    for batch in data_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values)

        # Post-process predictions
        target_sizes = torch.tensor(
            [[pixel_values.shape[2], pixel_values.shape[3]]] * pixel_values.shape[0],
            device=device,
        )
        results = processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )

        for pred, gt in zip(results, labels):
            total_images += 1
            total_gt += gt["class_labels"].shape[0]
            total_preds += len(pred["scores"])
            high_conf += (pred["scores"] > 0.5).sum().item()

    return {
        "total_images": total_images,
        "total_gt_boxes": total_gt,
        "total_detections": total_preds,
        "high_conf_detections": high_conf,
        "avg_detections_per_image": total_preds / max(total_images, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR on CARLA data")
    parser.add_argument("--dataset-dir", type=str, default="./dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--lr-backbone", type=float, default=0.00002)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_rtdetr")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--model-name", type=str, default="PekingU/rtdetr_r50vd",
                        help="HuggingFace model checkpoint")
    args = parser.parse_args()

    # Find runs
    dataset_root = Path(args.dataset_dir)
    run_dirs = sorted(glob.glob(str(dataset_root / "run_*")))
    if not run_dirs:
        logger.error(f"No run directories found in {dataset_root}")
        return

    logger.info(f"Found {len(run_dirs)} runs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load RT-DETR processor and model
    logger.info(f"Loading RT-DETR from {args.model_name}...")
    processor = RTDetrImageProcessor.from_pretrained(args.model_name)

    num_classes = len(CLASS_NAMES)  # 7 (no background class for RT-DETR)
    id2label = {i: name for i, name in CLASS_NAMES.items()}
    label2id = {name: i for i, name in CLASS_NAMES.items()}

    model = RTDetrForObjectDetection.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create dataset
    full_dataset = CARLADetectionDatasetRTDETR(
        run_dirs=run_dirs,
        processor=processor,
    )

    # Train/val split
    total = len(full_dataset)
    val_size = int(total * args.val_split)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Optimizer — AdamW with separate backbone LR (standard for DETR-family)
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                     if "backbone" not in n and p.requires_grad],
         "lr": args.lr},
        {"params": [p for n, p in model.named_parameters()
                     if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0001)

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Resume
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training
    logger.info("=" * 60)
    logger.info("Starting RT-DETR training")
    logger.info(f"  Epochs: {start_epoch} → {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  LR: {args.lr} (backbone: {args.lr_backbone})")
    logger.info(f"  Classes: {num_classes}")
    logger.info("=" * 60)

    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_losses['total_loss']:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.0f}s"
        )

        # Evaluate
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_results = evaluate(model, processor, val_loader, device)
            logger.info(
                f"  [Eval] GT: {eval_results['total_gt_boxes']} | "
                f"Detections: {eval_results['total_detections']} | "
                f"Avg/img: {eval_results['avg_detections_per_image']:.1f}"
            )
            train_losses["eval"] = eval_results

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"rtdetr_epoch_{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": train_losses,
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

        history.append({"epoch": epoch, **train_losses})

    # Save final
    final_path = os.path.join(args.checkpoint_dir, "rtdetr_final.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved: {final_path}")

    # Also save as HuggingFace format for easy loading
    hf_path = os.path.join(args.checkpoint_dir, "rtdetr_hf")
    model.save_pretrained(hf_path)
    processor.save_pretrained(hf_path)
    logger.info(f"HuggingFace model saved: {hf_path}")

    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
