"""Ultralytics trainer adapter for paired RGB-X detection."""

from __future__ import annotations

from copy import copy

from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first

from .data import PairedRGBXDataset
from .model import MidPriorDetectionModel
from .weights import load_public_weights


def _modality_dirs(data):
    mapping = data.get("modality") or data.get("modalities") or {}
    used = data.get("modality_used") or data.get("models") or ["rgb", "x"]
    if not isinstance(used, (list, tuple)) or len(used) != 2 or "rgb" not in used:
        raise ValueError("data YAML must define modality_used: [rgb, <x-modality>]")
    x_name = next(name for name in used if name != "rgb")
    return mapping.get("rgb", "images"), mapping.get(x_name, f"images_{x_name}")


class MidPriorValidator(DetectionValidator):
    @staticmethod
    def _rgb_batch(batch):
        result = dict(batch)
        result["img"] = batch["img"][:, :3]
        return result

    def plot_val_samples(self, batch, ni):
        super().plot_val_samples(self._rgb_batch(batch), ni)

    def plot_predictions(self, batch, preds, ni):
        super().plot_predictions(self._rgb_batch(batch), preds, ni)


class MidPriorTrainer(DetectionTrainer):
    def __init__(self, *args, scale="n", initial_weights=None, **kwargs):
        self.model_scale = scale
        self.initial_weights = initial_weights
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = MidPriorDetectionModel(
            cfg,
            nc=self.data["nc"],
            scale=self.model_scale,
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        if self.initial_weights:
            load_public_weights(model, self.initial_weights, strict=True)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        rgb_dir, x_dir = _modality_dirs(self.data)
        stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return PairedRGBXDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task="detect",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            rgb_dir=rgb_dir,
            x_dir=x_dir,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train" and not getattr(dataset, "rect", False)
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MidPriorValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"][:, :3],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

