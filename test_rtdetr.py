#!/usr/bin/env python3
"""
Test RT-DETR on CARLA images. Generates prediction videos
comparable to test_detector.py (Faster R-CNN) for side-by-side evaluation.

Usage:
    # Single frame with ground truth comparison:
    python test_rtdetr.py --checkpoint checkpoints_rtdetr/rtdetr_hf --run dataset/run_000_Town10HD_Opt --frame 100

    # Generate prediction video:
    python test_rtdetr.py --checkpoint checkpoints_rtdetr/rtdetr_hf --run dataset/run_000_Town10HD_Opt --video
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

CLASS_NAMES = {
    0: "vehicle",
    1: "pedestrian",
    2: "rider",
    3: "bicycle",
    4: "motorcycle",
    5: "bus",
    6: "truck",
}

PRED_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 165, 0),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 128, 255),
}

GT_COLOR = (255, 200, 0)


def load_model(checkpoint_path, device):
    """Load RT-DETR from HuggingFace saved format."""
    processor = RTDetrImageProcessor.from_pretrained(checkpoint_path)
    model = RTDetrForObjectDetection.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def predict(model, processor, image_rgb, device, confidence=0.5):
    """Run inference on a single RGB image."""
    inputs = processor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    h, w = image_rgb.shape[:2]
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=confidence, target_sizes=target_sizes
    )[0]

    return {
        "boxes": results["boxes"].cpu().numpy(),
        "labels": results["labels"].cpu().numpy(),
        "scores": results["scores"].cpu().numpy(),
    }


def draw_predictions(image, predictions, ground_truth=None):
    vis = image.copy()

    if ground_truth is not None:
        for actor in ground_truth.get("actors", []):
            bbox = actor.get("bbox_2d")
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), GT_COLOR, 1)
            cv2.putText(vis, f"GT:{actor.get('class_name', '?')}",
                        (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, GT_COLOR, 1)

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


def main():
    parser = argparse.ArgumentParser(description="Test RT-DETR on CARLA data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(args.checkpoint, device)
    print(f"RT-DETR loaded from {args.checkpoint}")

    if args.run and args.frame is not None:
        run_path = Path(args.run)
        fname = f"{args.frame:06d}"
        image = cv2.imread(str(run_path / "rgb" / f"{fname}.png"))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_path = run_path / "annotations" / f"{fname}.json"
        gt = None
        if ann_path.exists():
            with open(ann_path) as f:
                gt = json.load(f)

        preds = predict(model, processor, image_rgb, device, args.confidence)
        vis = draw_predictions(image, preds, gt)

        num_gt = len([a for a in gt.get("actors", []) if a.get("bbox_2d")]) if gt else 0
        print(f"GT: {num_gt} | Predictions: {len(preds['boxes'])}")

        cv2.imshow("RT-DETR Predictions", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.run and args.video:
        run_path = Path(args.run)
        rgb_files = sorted((run_path / "rgb").glob("*.png"))
        if args.max_frames:
            rgb_files = rgb_files[:args.max_frames]

        first_img = cv2.imread(str(rgb_files[0]))
        h, w = first_img.shape[:2]

        output_path = str(run_path / "rtdetr_prediction_video.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

        for i, rgb_path in enumerate(rgb_files):
            image = cv2.imread(str(rgb_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            preds = predict(model, processor, image_rgb, device, args.confidence)

            ann_path = run_path / "annotations" / f"{rgb_path.stem}.json"
            gt = None
            if ann_path.exists():
                with open(ann_path) as f:
                    gt = json.load(f)

            vis = draw_predictions(image, preds, gt)

            num_preds = len(preds["boxes"])
            num_gt = len([a for a in gt.get("actors", []) if a.get("bbox_2d")]) if gt else 0
            cv2.putText(vis, f"RT-DETR | Pred: {num_preds} | GT: {num_gt}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            writer.write(vis)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(rgb_files)} frames")

        writer.release()
        print(f"Video saved: {output_path}")
    else:
        print("Specify --run with --frame or --video")


if __name__ == "__main__":
    main()
