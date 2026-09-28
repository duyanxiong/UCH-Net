from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(r"D:\Master\datasets\WRF-Dataset")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def images_in(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


metadata = json.loads((ROOT / "data.json").read_text(encoding="utf-8"))
categories = sorted(metadata["categories"], key=lambda item: item["id"])
print("CATEGORIES", [(item["id"], item["name"]) for item in categories])
print("JSON_COUNTS", len(metadata["images"]), len(metadata["annotations"]))

for split in ("train", "val"):
    rgb_files = images_in(ROOT / "images" / split)
    depth_by_stem = {p.stem: p for p in images_in(ROOT / "images_depth" / split)}
    labels_dir = ROOT / "labels" / split

    unreadable_rgb = []
    unreadable_depth = []
    shape_mismatches = []
    depth_channels = Counter()
    rgb_shapes = Counter()
    depth_dtypes = Counter()

    for index, rgb_path in enumerate(rgb_files, 1):
        depth_path = depth_by_stem[rgb_path.stem]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            unreadable_rgb.append(rgb_path.name)
            continue
        if depth is None:
            unreadable_depth.append(depth_path.name)
            continue
        rgb_shapes[(rgb.shape[1], rgb.shape[0])] += 1
        depth_channels[1 if depth.ndim == 2 else depth.shape[2]] += 1
        depth_dtypes[str(depth.dtype)] += 1
        if depth.shape[:2] != rgb.shape[:2]:
            shape_mismatches.append((rgb_path.name, rgb.shape[:2], depth.shape[:2]))
        if index % 1000 == 0:
            print("IMAGE_PROGRESS", split, index, len(rgb_files), flush=True)

    class_counts = Counter()
    invalid_lines = []
    nonpositive_boxes = []
    total_boxes = 0
    coordinate_min = math.inf
    coordinate_max = -math.inf
    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            try:
                values = [float(field) for field in fields]
            except ValueError:
                invalid_lines.append((label_path.name, line_number, "not numeric"))
                continue
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                invalid_lines.append((label_path.name, line_number, f"fields={len(values)}"))
                continue
            class_value, x, y, width, height = values
            class_id = int(class_value)
            if class_value != class_id:
                invalid_lines.append((label_path.name, line_number, f"non-integer class={class_value}"))
                continue
            class_counts[class_id] += 1
            total_boxes += 1
            coordinate_min = min(coordinate_min, x, y, width, height)
            coordinate_max = max(coordinate_max, x, y, width, height)
            if not all(0 <= value <= 1 for value in (x, y, width, height)):
                invalid_lines.append((label_path.name, line_number, "coordinate out of [0,1]"))
            if width <= 0 or height <= 0:
                nonpositive_boxes.append((label_path.name, line_number))

    print(
        "IMAGE_CHECK",
        split,
        {
            "pairs": len(rgb_files),
            "unreadable_rgb": len(unreadable_rgb),
            "unreadable_depth": len(unreadable_depth),
            "shape_mismatches": len(shape_mismatches),
            "rgb_shapes": dict(rgb_shapes),
            "depth_channels": dict(depth_channels),
            "depth_dtypes": dict(depth_dtypes),
        },
    )
    print(
        "LABEL_CHECK",
        split,
        {
            "files": len(label_files),
            "background_images": len(rgb_files) - len(label_files),
            "boxes": total_boxes,
            "class_counts": dict(sorted(class_counts.items())),
            "coordinate_min": coordinate_min,
            "coordinate_max": coordinate_max,
            "invalid_lines": len(invalid_lines),
            "nonpositive_boxes": len(nonpositive_boxes),
        },
    )
    if unreadable_rgb or unreadable_depth or shape_mismatches or invalid_lines or nonpositive_boxes:
        print(
            "PROBLEM_SAMPLES",
            {
                "rgb": unreadable_rgb[:3],
                "depth": unreadable_depth[:3],
                "shape": shape_mismatches[:3],
                "labels": invalid_lines[:3],
                "boxes": nonpositive_boxes[:3],
            },
        )
