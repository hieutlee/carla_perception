#!/usr/bin/env python3
"""
Stage 2: Train Faster R-CNN on CARLA collected data.

Usage:
    # Train on all runs in the dataset folder:
    python train_detector.py

    # Custom settings:
    python train_detector.py --epochs 20 --batch-size 4 --lr 0.005

    # Resume from checkpoint:
    python train_detector.py --resume checkpoints/frcnn_epoch_005.pth
"""

import argparse
import glob
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset.detection_dataset import CARLADetectionDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def get_model(num_classes: int, pretrained_backbone: bool = True) -> torch.nn.Module:
    """
    Build Faster R-CNN with a ResNet-50-FPN backbone.

    Uses ImageNet-pretrained backbone weights, but replaces the
    classification head with one matching our number of classes.

    Args:
        num_classes: Number of classes INCLUDING background.
            e.g., 8 = 7 object classes + 1 background.
        pretrained_backbone: Use ImageNet-pretrained ResNet-50.
    """
    # Load model with pretrained backbone (not pretrained detection head)
    if pretrained_backbone:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)

    # Replace the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def collate_fn(batch):
    """
    Custom collate for detection — images stay as a list (variable-size targets).
    Faster R-CNN expects a list of images, not a stacked tensor.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch, return average losses."""
    model.train()

    total_loss = 0.0
    loss_components = {"loss_classifier": 0, "loss_box_reg": 0, "loss_objectness": 0, "loss_rpn_box_reg": 0}
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip batches where all images have no boxes
        # (Faster R-CNN crashes on empty targets)
        valid_targets = [t for t in targets if t["boxes"].shape[0] > 0]
        if len(valid_targets) == 0:
            continue

        # Filter to only images with valid targets
        valid_pairs = [(img, t) for img, t in zip(images, targets) if t["boxes"].shape[0] > 0]
        images = [p[0] for p in valid_pairs]
        targets = [p[1] for p in valid_pairs]

        # Forward pass — Faster R-CNN returns losses in train mode
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Track losses
        total_loss += losses.item()
        for k in loss_components:
            if k in loss_dict:
                loss_components[k] += loss_dict[k].item()
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % 20 == 0:
            avg = total_loss / num_batches
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(data_loader)} | "
                f"Loss: {avg:.4f}"
            )

    if num_batches == 0:
        return {"total_loss": 0.0}

    avg_losses = {k: v / num_batches for k, v in loss_components.items()}
    avg_losses["total_loss"] = total_loss / num_batches
    return avg_losses


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Simple evaluation: run inference and compute average detection confidence.
    For proper mAP evaluation, use COCO evaluator (added later).
    """
    model.eval()

    total_detections = 0
    total_images = 0
    total_gt_boxes = 0
    high_conf_detections = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        predictions = model(images)

        for pred, gt in zip(predictions, targets):
            total_images += 1
            total_gt_boxes += gt["boxes"].shape[0]

            # Count detections above confidence threshold
            if "scores" in pred:
                scores = pred["scores"]
                total_detections += len(scores)
                high_conf_detections += (scores > 0.5).sum().item()

    return {
        "total_images": total_images,
        "total_gt_boxes": total_gt_boxes,
        "total_detections": total_detections,
        "high_conf_detections": high_conf_detections,
        "avg_detections_per_image": total_detections / max(total_images, 1),
        "avg_high_conf_per_image": high_conf_detections / max(total_images, 1),
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses: dict,
    path: str,
):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on CARLA data")
    parser.add_argument("--dataset-dir", type=str, default="./dataset",
                        help="Root directory containing run_XXX folders")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data for validation")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval-every", type=int, default=3,
                        help="Run evaluation every N epochs")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    # Find all run directories
    dataset_root = Path(args.dataset_dir)
    run_dirs = sorted(glob.glob(str(dataset_root / "run_*")))

    if not run_dirs:
        logger.error(f"No run directories found in {dataset_root}")
        return

    logger.info(f"Found {len(run_dirs)} runs: {[Path(r).name for r in run_dirs]}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataset
    full_dataset = CARLADetectionDataset(
        run_dirs=run_dirs,
        target_format="frcnn",
        min_visibility=0.2,
    )

    num_classes = 8
    logger.info(f"Number of classes (incl. background): {num_classes}")

    # Train/val split
    total = len(full_dataset)
    val_size = int(total * args.val_split)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"Train: {train_size} samples | Val: {val_size} samples")

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Build model
    model = get_model(num_classes, pretrained_backbone=True)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer — SGD with momentum works best for Faster R-CNN
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # Learning rate scheduler — step down at epochs 8 and 12
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[8, 12],
        gamma=0.1,
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training history
    history = []

    # =============================================
    # Training loop
    # =============================================
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info(f"  Epochs: {start_epoch} → {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Classes: {num_classes}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_losses['total_loss']:.4f} | "
            f"cls: {train_losses.get('loss_classifier', 0):.4f} | "
            f"box: {train_losses.get('loss_box_reg', 0):.4f} | "
            f"obj: {train_losses.get('loss_objectness', 0):.4f} | "
            f"rpn: {train_losses.get('loss_rpn_box_reg', 0):.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.0f}s"
        )

        # Evaluate
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_results = evaluate(model, val_loader, device)
            logger.info(
                f"  [Eval] GT boxes: {eval_results['total_gt_boxes']} | "
                f"Detections: {eval_results['total_detections']} | "
                f"High-conf (>0.5): {eval_results['high_conf_detections']} | "
                f"Avg/img: {eval_results['avg_high_conf_per_image']:.1f}"
            )
            train_losses["eval"] = eval_results

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"frcnn_epoch_{epoch:03d}.pth")
            save_checkpoint(model, optimizer, epoch, train_losses, ckpt_path)

        history.append({"epoch": epoch, **train_losses})

    # Save final model (just weights, smaller file)
    final_path = os.path.join(args.checkpoint_dir, "frcnn_final.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved: {final_path}")

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"Training history saved: {history_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
