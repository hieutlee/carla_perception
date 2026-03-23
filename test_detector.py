#!/usr/bin/env python3
"""
Run inference with a trained Faster R-CNN on CARLA images.

Usage:
    # Test on a single image:
    python test_detector.py --checkpoint checkpoints/frcnn_final.pth --image dataset/run_000_Town10HD_Opt/rgb/000100.png

    # Test on a whole run (generates annotated video):
    python test_detector.py --checkpoint checkpoints/frcnn_final.pth --run dataset/run_000_Town10HD_Opt --video

    # Test on a single frame with ground truth comparison:
    python test_detector.py --checkpoint checkpoints/frcnn_final.pth --run dataset/run_000_Town10HD_Opt --frame 100
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


CLASS_NAMES = {
    1: "vehicle",
    2: "pedestrian",
    3: "rider",
    4: "bicycle",
    5: "motorcycle",
    6: "bus",
    7: "truck",
}

# Prediction colors (BGR for OpenCV)
PRED_COLORS = {
    1: (0, 255, 0),      # vehicle - green
    2: (0, 0, 255),      # pedestrian - red
    3: (255, 165, 0),    # rider - orange
    4: (255, 255, 0),    # bicycle - cyan
    5: (255, 0, 255),    # motorcycle - magenta
    6: (0, 255, 255),    # bus - yellow
    7: (128, 128, 255),  # truck - light red
}

GT_COLOR = (255, 200, 0)  # ground truth boxes in light blue


def load_model(checkpoint_path: str, num_classes: int = 8) -> torch.nn.Module:
    """Load trained Faster R-CNN from checkpoint."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Try loading as full checkpoint first, then as state_dict
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    confidence_threshold: float = 0.5,
) -> dict:
    """Run inference on a single image."""
    # Convert BGR to RGB, normalize
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    predictions = model([img_tensor])[0]

    # Filter by confidence
    keep = predictions["scores"] >= confidence_threshold
    return {
        "boxes": predictions["boxes"][keep].cpu().numpy(),
        "labels": predictions["labels"][keep].cpu().numpy(),
        "scores": predictions["scores"][keep].cpu().numpy(),
    }


def draw_predictions(
    image: np.ndarray,
    predictions: dict,
    ground_truth: dict = None,
) -> np.ndarray:
    """Draw predicted and optionally ground truth boxes on image."""
    vis = image.copy()

    # Draw ground truth first (so predictions draw on top)
    if ground_truth is not None:
        for actor in ground_truth.get("actors", []):
            bbox = actor.get("bbox_2d")
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), GT_COLOR, 1)
            cv2.putText(vis, f"GT:{actor.get('class_name', '?')}",
                        (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, GT_COLOR, 1)

    # Draw predictions
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        label_int = int(label)
        color = PRED_COLORS.get(label_int, (255, 255, 255))
        class_name = CLASS_NAMES.get(label_int, f"cls{label_int}")

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = f"{class_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


def test_single_image(model, image_path, device, confidence):
    """Test on a single image and display."""
    image = cv2.imread(image_path)
    preds = predict(model, image, device, confidence)
    vis = draw_predictions(image, preds)

    print(f"Detections: {len(preds['boxes'])}")
    for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        name = CLASS_NAMES.get(int(label), f"cls{label}")
        print(f"  {name}: {score:.3f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

    cv2.imshow("Predictions", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_frame_with_gt(model, run_dir, frame_idx, device, confidence):
    """Test a frame and compare predictions with ground truth."""
    run_path = Path(run_dir)
    fname = f"{frame_idx:06d}"

    image = cv2.imread(str(run_path / "rgb" / f"{fname}.png"))
    ann_path = run_path / "annotations" / f"{fname}.json"

    gt = None
    if ann_path.exists():
        with open(ann_path) as f:
            gt = json.load(f)

    preds = predict(model, image, device, confidence)
    vis = draw_predictions(image, preds, gt)

    num_gt = len([a for a in gt.get("actors", []) if a.get("bbox_2d")]) if gt else 0
    print(f"Ground truth: {num_gt} objects | Predictions: {len(preds['boxes'])} detections")

    cv2.imshow(f"Frame {frame_idx} — GT (thin blue) vs Predictions (colored)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_prediction_video(model, run_dir, device, confidence, fps=10, max_frames=None):
    """Generate a video with model predictions overlaid."""
    run_path = Path(run_dir)
    rgb_dir = run_path / "rgb"
    ann_dir = run_path / "annotations"

    rgb_files = sorted(rgb_dir.glob("*.png"))
    if max_frames:
        rgb_files = rgb_files[:max_frames]

    if not rgb_files:
        print(f"No images found in {rgb_dir}")
        return

    first_img = cv2.imread(str(rgb_files[0]))
    h, w = first_img.shape[:2]

    output_path = str(run_path / "prediction_video.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i, rgb_path in enumerate(rgb_files):
        image = cv2.imread(str(rgb_path))
        preds = predict(model, image, device, confidence)

        # Load GT if available
        ann_path = ann_dir / f"{rgb_path.stem}.json"
        gt = None
        if ann_path.exists():
            with open(ann_path) as f:
                gt = json.load(f)

        vis = draw_predictions(image, preds, gt)

        # Add info text
        num_preds = len(preds["boxes"])
        num_gt = len([a for a in gt.get("actors", []) if a.get("bbox_2d")]) if gt else 0
        info = f"Frame {i} | Pred: {num_preds} | GT: {num_gt}"
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(vis)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(rgb_files)} frames")

    writer.release()
    print(f"Prediction video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test trained Faster R-CNN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--run", type=str, default=None, help="Run directory")
    parser.add_argument("--frame", type=int, default=None, help="Frame index to test")
    parser.add_argument("--video", action="store_true", help="Generate prediction video")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.num_classes)
    model.to(device)
    print(f"Model loaded from {args.checkpoint}")

    if args.image:
        test_single_image(model, args.image, device, args.confidence)
    elif args.run and args.frame is not None:
        test_frame_with_gt(model, args.run, args.frame, device, args.confidence)
    elif args.run and args.video:
        generate_prediction_video(model, args.run, device, args.confidence,
                                  fps=args.fps, max_frames=args.max_frames)
    else:
        print("Specify --image, --run with --frame, or --run with --video")


if __name__ == "__main__":
    main()
