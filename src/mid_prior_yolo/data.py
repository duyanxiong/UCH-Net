"""Paired RGB-X dataset with synchronized geometry."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.augment import Compose, Format, LetterBox, RandomFlip, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER


class MultiModalFormat(Format):
    """Convert HWC to CHW without reversing all six modality channels."""

    def _format_img(self, img):
        if img.ndim < 3:
            img = np.expand_dims(img, -1)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class MultiModalLetterBox(LetterBox):
    """Run OpenCV padding per modality because copyMakeBorder supports at most four channels."""

    def __call__(self, labels=None, image=None):
        labels = {} if labels is None else labels
        source = labels.get("img") if image is None else image
        if source.ndim != 3 or source.shape[2] != 6:
            return super().__call__(labels=labels, image=image)
        target_shape = labels.get("rect_shape", self.new_shape)
        rgb_labels = dict(labels)
        rgb_labels["img"] = source[:, :, :3]
        rgb_result = super().__call__(labels=rgb_labels)
        x_transform = LetterBox(
            new_shape=target_shape,
            auto=self.auto,
            scale_fill=self.scale_fill,
            scaleup=self.scaleup,
            center=self.center,
            stride=self.stride,
        )
        auxiliary = x_transform(image=source[:, :, 3:])
        rgb_result["img"] = np.concatenate((rgb_result["img"], auxiliary), axis=2)
        return rgb_result


class MultiModalRandomPerspective(RandomPerspective):
    """Apply one sampled transform matrix to RGB and X using separate OpenCV calls."""

    def affine_transform(self, img, border):
        if img.ndim != 3 or img.shape[2] != 6:
            return super().affine_transform(img, border)
        rgb, auxiliary = img[:, :, :3], img[:, :, 3:]
        rgb, matrix, scale = super().affine_transform(rgb, border)
        if self.perspective:
            auxiliary = cv2.warpPerspective(
                auxiliary, matrix, dsize=self.size, borderValue=(114, 114, 114)
            )
        else:
            auxiliary = cv2.warpAffine(
                auxiliary, matrix[:2], dsize=self.size, borderValue=(114, 114, 114)
            )
        return np.concatenate((rgb, auxiliary), axis=2), matrix, scale


class RGBOnlyHSV:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain, self.sgain, self.vgain = hgain, sgain, vgain

    def __call__(self, labels):
        image = labels["img"]
        if image.ndim != 3 or image.shape[2] != 6 or not any((self.hgain, self.sgain, self.vgain)):
            return labels
        rgb, auxiliary = image[:, :, :3].copy(), image[:, :, 3:]
        gains = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV))
        values = np.arange(256, dtype=gains.dtype)
        hsv = cv2.merge(
            (
                cv2.LUT(hue, ((values * gains[0]) % 180).astype(np.uint8)),
                cv2.LUT(sat, np.clip(values * gains[1], 0, 255).astype(np.uint8)),
                cv2.LUT(val, np.clip(values * gains[2], 0, 255).astype(np.uint8)),
            )
        )
        labels["img"] = np.concatenate((cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), auxiliary), axis=2)
        return labels


class PairedRGBXDataset(YOLODataset):
    """Load same-stem RGB and X images and concatenate them as [RGB, X]."""

    def __init__(self, *args, rgb_dir="images", x_dir="images_x", **kwargs):
        self.rgb_dir = Path(rgb_dir)
        self.x_dir = Path(x_dir)
        super().__init__(*args, **kwargs)
        self._validate_pairs()

    def _paired_path(self, rgb_file):
        rgb_path = Path(rgb_file)
        root = Path(self.data["path"])
        try:
            relative = rgb_path.resolve().relative_to(root.resolve())
            parts = list(relative.parts)
            rgb_parts = self.rgb_dir.parts
            if tuple(parts[: len(rgb_parts)]) == rgb_parts:
                parts[: len(rgb_parts)] = self.x_dir.parts
                candidate = root.joinpath(*parts)
            else:
                candidate = root / self.x_dir / rgb_path.parent.name / rgb_path.name
        except (ValueError, OSError):
            candidate = rgb_path.parent.parent.parent / self.x_dir / rgb_path.parent.name / rgb_path.name
        if candidate.exists():
            return candidate
        for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            alternative = candidate.with_suffix(suffix)
            if alternative.exists():
                return alternative
        return candidate

    def _validate_pairs(self):
        missing = [str(self._paired_path(path)) for path in self.im_files[:100] if not self._paired_path(path).exists()]
        if missing:
            sample = "\n".join(f"  - {path}" for path in missing[:3])
            raise FileNotFoundError(f"Missing {len(missing)} paired X image(s) in the first 100 samples:\n{sample}")
        LOGGER.info(f"RGB-X pairs verified: {min(len(self.im_files), 100)}/{min(len(self.im_files), 100)} sampled")

    @staticmethod
    def _three_channels(image, path):
        if image is None:
            raise ValueError(f"Could not read X image: {path}")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        if image.shape[2] == 4:
            return image[:, :, :3]
        if image.shape[2] != 3:
            raise ValueError(f"X image must have 1, 3, or 4 channels: {path} has {image.shape[2]}")
        return image

    def load_multimodal_image(self, index):
        rgb, _, _ = self.load_image(index)
        x_path = self._paired_path(self.im_files[index])
        auxiliary = self._three_channels(cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED), x_path)
        if auxiliary.shape[:2] != rgb.shape[:2]:
            auxiliary = cv2.resize(auxiliary, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if auxiliary.dtype != rgb.dtype:
            auxiliary = np.clip(auxiliary, 0, 255).astype(rgb.dtype)
        return np.concatenate((rgb, auxiliary), axis=2)

    def __getitem__(self, index):
        label = self.get_image_and_label(index)
        label["img"] = self.load_multimodal_image(index)
        return self.transforms(label)

    def build_transforms(self, hyp=None):
        if self.augment:
            transforms = Compose(
                [
                    MultiModalRandomPerspective(
                        degrees=hyp.degrees,
                        translate=hyp.translate,
                        scale=hyp.scale,
                        shear=hyp.shear,
                        perspective=hyp.perspective,
                        pre_transform=MultiModalLetterBox(new_shape=(self.imgsz, self.imgsz)),
                    ),
                    RGBOnlyHSV(hyp.hsv_h, hyp.hsv_s, hyp.hsv_v),
                    RandomFlip(direction="vertical", p=hyp.flipud),
                    RandomFlip(direction="horizontal", p=hyp.fliplr),
                ]
            )
        else:
            transforms = Compose([MultiModalLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            MultiModalFormat(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,
            )
        )
        return transforms

