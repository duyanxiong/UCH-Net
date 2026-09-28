"""Command-line training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .trainer import MidPriorTrainer


def build_parser():
    project_root = Path(__file__).resolve().parents[2]
    checkout_config = project_root / "configs" / "model.yaml"
    packaged_config = Path(__file__).resolve().parent / "configs" / "model.yaml"
    default_config = checkout_config if checkout_config.exists() else packaged_config
    parser = argparse.ArgumentParser(description="Train YOLO11-MM Mid-Prior on paired RGB-X images")
    parser.add_argument("--data", required=True, help="Dataset YAML")
    parser.add_argument("--model", default=str(default_config))
    parser.add_argument("--scale", choices=("n", "s", "m", "l", "x"), default="n")
    parser.add_argument("--weights", help="Portable .pth state-dict exported from the original project")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="mid-prior")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    trainer = MidPriorTrainer(
        scale=args.scale,
        initial_weights=args.weights,
        overrides={
            "model": str(Path(args.model).resolve()),
            "data": str(Path(args.data).resolve()),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "project": args.project,
            "name": args.name,
            "cache": args.cache,
            "amp": not args.no_amp,
            "plots": not args.no_plots,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "exist_ok": True,
            "task": "detect",
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()

