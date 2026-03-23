"""
Visualization utilities for the CARLA perception dataset.

Use these to sanity-check your collected data before training:
    python -m utils.visualization --run dataset/run_000_Town10HD --frame 100
    python -m utils.visualization --run dataset/run_000_Town10HD --video --fps 10
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np


# Class colors (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),      # vehicle - green
    1: (0, 0, 255),      # pedestrian - red
    2: (255, 165, 0),    # rider - orange
    3: (255, 255, 0),    # bicycle - cyan
    4: (255, 0, 255),    # motorcycle - magenta
    5: (0, 255, 255),    # bus - yellow
    6: (128, 128, 255),  # truck - light red
}


def draw_annotations(
    image: np.ndarray,
    annotation: dict,
    show_ids: bool = True,
    show_class: bool = True,
    show_distance: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw 2D bounding boxes and metadata on an image.

    Returns a copy with annotations drawn.
    """
    vis = image.copy()

    for actor in annotation.get("actors", []):
        bbox = actor.get("bbox_2d")
        if bbox is None:
            continue

        x_min, y_min, x_max, y_max = [int(v) for v in bbox]
        class_id = actor["class_id"]
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        visibility = actor.get("visibility", 0.0)

        # Scale alpha by visibility for visual occlusion feedback
        alpha = max(0.3, min(1.0, visibility))

        # Draw bbox
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color, line_thickness)

        # Build label text
        parts = []
        if show_ids:
            parts.append(f"ID:{actor['id']}")
        if show_class:
            parts.append(actor.get("class_name", f"cls{class_id}"))
        if show_distance:
            dist = actor.get("distance", 0)
            parts.append(f"{dist:.0f}m")

        label = " | ".join(parts)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            vis,
            (x_min, y_min - th - 6),
            (x_min + tw + 4, y_min),
            color, -1
        )
        cv2.putText(
            vis, label,
            (x_min + 2, y_min - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )

    # Draw ego info
    ego = annotation.get("ego", {})
    ego_vel = ego.get("velocity", [0, 0, 0])
    ego_speed = np.sqrt(sum(v**2 for v in ego_vel)) * 3.6  # m/s → km/h
    num_actors = len([a for a in annotation.get("actors", []) if a.get("bbox_2d")])

    info_text = f"Frame: {annotation.get('frame', '?')} | Ego: {ego_speed:.0f} km/h | Actors: {num_actors}"
    cv2.putText(
        vis, info_text,
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )

    return vis


def draw_trajectories_bev(
    trajectories: dict,
    canvas_size: int = 800,
    meters_per_pixel: float = 0.5,
    highlight_ids: list[int] = None,
) -> np.ndarray:
    """
    Draw bird's-eye view of all actor trajectories.

    Useful for validating trajectory data before prediction training.
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    center = canvas_size // 2

    # Compute trajectory bounds for auto-centering
    all_x, all_y = [], []
    for actor_id, history in trajectories.items():
        for h in history:
            all_x.append(h["x"])
            all_y.append(h["y"])

    if len(all_x) == 0:
        return canvas

    cx = np.mean(all_x)
    cy = np.mean(all_y)

    for actor_id_str, history in trajectories.items():
        actor_id = int(actor_id_str) if isinstance(actor_id_str, str) else actor_id_str

        if len(history) < 2:
            continue

        # Map world coords to canvas
        points = []
        for h in history:
            px = int(center + (h["x"] - cx) / meters_per_pixel)
            py = int(center + (h["y"] - cy) / meters_per_pixel)
            points.append((px, py))

        # Choose color
        if highlight_ids and actor_id in highlight_ids:
            color = (0, 255, 255)  # yellow for highlighted
            thickness = 2
        else:
            # Hash actor ID to get a consistent color
            np.random.seed(actor_id % 10000)
            color = tuple(int(c) for c in np.random.randint(80, 255, 3))
            thickness = 1

        # Draw trajectory as polyline
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)

        # Draw start point (circle) and end point (arrow)
        cv2.circle(canvas, points[0], 3, color, -1)
        if len(points) >= 2:
            cv2.arrowedLine(canvas, points[-2], points[-1], color, thickness + 1)

    # Draw scale bar
    bar_length_m = 50  # meters
    bar_length_px = int(bar_length_m / meters_per_pixel)
    cv2.line(canvas, (20, canvas_size - 30), (20 + bar_length_px, canvas_size - 30), (255, 255, 255), 2)
    cv2.putText(
        canvas, f"{bar_length_m}m",
        (20, canvas_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    return canvas


def visualize_frame(run_dir: str, frame_idx: int = 0):
    """Load and display a single annotated frame."""
    run_path = Path(run_dir)
    fname = f"{frame_idx:06d}"

    rgb_path = run_path / "rgb" / f"{fname}.png"
    ann_path = run_path / "annotations" / f"{fname}.json"

    if not rgb_path.exists() or not ann_path.exists():
        print(f"Frame {frame_idx} not found in {run_dir}")
        return

    image = cv2.imread(str(rgb_path))
    with open(ann_path) as f:
        annotation = json.load(f)

    vis = draw_annotations(image, annotation)

    cv2.imshow(f"Frame {frame_idx}", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_video(
    run_dir: str,
    output_path: str = None,
    fps: int = 10,
    max_frames: int = None,
):
    """Generate annotated video from a collection run."""
    run_path = Path(run_dir)
    ann_dir = run_path / "annotations"
    rgb_dir = run_path / "rgb"

    ann_files = sorted(ann_dir.glob("*.json"))
    if max_frames:
        ann_files = ann_files[:max_frames]

    if not ann_files:
        print(f"No annotations found in {run_dir}")
        return

    # Get image dimensions from first frame
    first_frame = ann_files[0].stem
    first_img = cv2.imread(str(rgb_dir / f"{first_frame}.png"))
    h, w = first_img.shape[:2]

    if output_path is None:
        output_path = str(run_path / "annotated_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, ann_file in enumerate(ann_files):
        frame_name = ann_file.stem
        rgb_path = rgb_dir / f"{frame_name}.png"

        if not rgb_path.exists():
            continue

        image = cv2.imread(str(rgb_path))
        with open(ann_file) as f:
            annotation = json.load(f)

        vis = draw_annotations(image, annotation)
        writer.write(vis)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(ann_files)} frames")

    writer.release()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Path to collection run directory")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    parser.add_argument("--video", action="store_true", help="Generate annotated video")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--bev", action="store_true", help="Show BEV trajectory plot")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit video frames")
    args = parser.parse_args()

    if args.video:
        generate_video(args.run, fps=args.fps, max_frames=args.max_frames)
    elif args.bev:
        traj_path = Path(args.run) / "trajectories" / "trajectories.json"
        with open(traj_path) as f:
            trajectories = json.load(f)
        bev = draw_trajectories_bev(trajectories)
        cv2.imshow("BEV Trajectories", bev)
        cv2.waitKey(0)
    else:
        visualize_frame(args.run, args.frame)
