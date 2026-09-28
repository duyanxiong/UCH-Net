"""Fixed graph builder for the published YOLO11-MM Mid-Prior architecture."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from ultralytics.nn.modules import C2PSA, C3k2, Concat, Conv, SPPF
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import initialize_weights

from .modules import C3k2AdditiveBlock, DetectLSCD, PriorCGAFusionV2


SUPPORTED_SCALES = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


def _load_config(config):
    if isinstance(config, dict):
        return deepcopy(config)
    with Path(config).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class MidPriorDetectionModel(BaseModel):
    """Six-channel RGB-X detector with stable, checkpoint-compatible layer indices."""

    def __init__(self, config, nc=None, scale=None, verbose=True):
        super().__init__()
        self.yaml = _load_config(config)
        self.scale = scale or self.yaml.get("scale", "n")
        scales = self.yaml.get("scales", SUPPORTED_SCALES)
        if self.scale not in scales:
            raise ValueError(f"Unknown scale {self.scale!r}; choose one of {tuple(scales)}")
        self.yaml["scale"] = self.scale
        if nc is not None:
            self.yaml["nc"] = int(nc)
        self.nc = int(self.yaml["nc"])
        self.depth, self.width, self.max_channels = scales[self.scale]
        self.model = self._build_graph()
        self.save = [3, 4, 6, 10, 17, 21, 23, 26, 29, 32, 35, 38]
        self.names = {index: str(index) for index in range(self.nc)}
        self.inplace = True
        self.end2end = False
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.model[-1].stride = self.stride.clone()
        self.model[-1].bias_init()
        initialize_weights(self)
        if verbose:
            LOGGER.info(f"MidPriorDetectionModel scale={self.scale}, nc={self.nc}, input=RGB(3)+X(3)")
            self.info()

    def _channels(self, value):
        return make_divisible(min(value, self.max_channels) * self.width, 8)

    def _repeats(self, value):
        return max(round(value * self.depth), 1) if value > 1 else value

    def _c3k(self, yaml_value):
        return bool(yaml_value or self.scale in "mlx")

    def _build_graph(self):
        c = self._channels
        n2 = self._repeats(2)
        modules = [
            Conv(3, c(64), 3, 2),
            Conv(c(64), c(128), 3, 2),
            C3k2(c(128), c(256), n2, self._c3k(False), 0.25),
            Conv(c(256), c(256), 3, 2),
            C3k2(c(256), c(512), n2, self._c3k(False), 0.25),
            Conv(c(512), c(512), 3, 2),
            C3k2(c(512), c(512), n2, self._c3k(True)),
            Conv(c(512), c(1024), 3, 2),
            C3k2(c(1024), c(1024), n2, self._c3k(True)),
            SPPF(c(1024), c(1024), 5),
            C2PSA(c(1024), c(1024), n2),
            Conv(3, c(64), 3, 2),
            Conv(c(64), c(128), 3, 2),
            C3k2(c(128), c(256), n2, self._c3k(False), 0.25),
            Conv(c(256), c(256), 3, 2),
            C3k2(c(256), c(512), n2, self._c3k(False), 0.25),
            Conv(c(512), c(512), 3, 2),
            C3k2(c(512), c(512), n2, self._c3k(True)),
            Conv(c(512), c(1024), 3, 2),
            C3k2(c(1024), c(1024), n2, self._c3k(True)),
            SPPF(c(1024), c(1024), 5),
            C2PSA(c(1024), c(1024), n2),
            PriorCGAFusionV2(c(512), c(512), c(256), c(512)),
            C3k2AdditiveBlock(c(512), c(1024), n2, self._c3k(True)),
            PriorCGAFusionV2(c(1024), c(1024), c(256), c(1024)),
            C3k2AdditiveBlock(c(1024), c(1024), n2, self._c3k(True)),
            C2PSA(c(1024), c(1024), 1),
            nn.Upsample(None, 2, "nearest"),
            Concat(1),
            C3k2AdditiveBlock(c(1024) + c(1024), c(512), n2, self._c3k(False)),
            nn.Upsample(None, 2, "nearest"),
            Concat(1),
            C3k2AdditiveBlock(c(512) + c(512), c(256), n2, self._c3k(False)),
            Conv(c(256), c(256), 3, 2),
            Concat(1),
            C3k2(c(256) + c(512) + c(1024), c(512), n2, self._c3k(False)),
            Conv(c(512), c(512), 3, 2),
            Concat(1),
            C3k2AdditiveBlock(c(512) + c(1024), c(1024), n2, self._c3k(True)),
            DetectLSCD(self.nc, c(256), (c(256), c(512), c(1024))),
        ]
        for index, module in enumerate(modules):
            module.i = index
            module.type = module.__class__.__name__
            module.np = sum(parameter.numel() for parameter in module.parameters())
        return nn.ModuleList(modules)

    @staticmethod
    def _split_modalities(x):
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] == 3:
            return x, x
        if x.shape[1] != 6:
            raise ValueError(f"Expected 6 channels [RGB(3), X(3)], got {x.shape[1]}")
        return x[:, :3], x[:, 3:]

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if augment:
            LOGGER.warning("Augmented inference is not implemented; using one forward pass.")
        m = self.model
        rgb, aux = self._split_modalities(x)

        r0 = m[0](rgb)
        r1 = m[1](r0)
        r2 = m[2](r1)
        r3 = m[3](r2)
        r4 = m[4](r3)
        r5 = m[5](r4)
        r6 = m[6](r5)
        r7 = m[7](r6)
        r8 = m[8](r7)
        r9 = m[9](r8)
        r10 = m[10](r9)

        x11 = m[11](aux)
        x12 = m[12](x11)
        x13 = m[13](x12)
        x14 = m[14](x13)
        x15 = m[15](x14)
        x16 = m[16](x15)
        x17 = m[17](x16)
        x18 = m[18](x17)
        x19 = m[19](x18)
        x20 = m[20](x19)
        x21 = m[21](x20)

        fused_p4 = m[23](m[22]([r6, x17, r3]))
        fused_p5 = m[26](m[25](m[24]([r10, x21, r3])))
        p4_td = m[29](m[28]([m[27](fused_p5), fused_p4]))
        p3_out = m[32](m[31]([m[30](p4_td), r4]))
        p4_out = m[35](m[34]([m[33](p3_out), p4_td, fused_p4]))
        p5_out = m[38](m[37]([m[36](p4_out), fused_p5]))
        return m[39]([p3_out, p4_out, p5_out])

    def init_criterion(self):
        return v8DetectionLoss(self)

    def _apply(self, fn):
        self = super()._apply(fn)
        head = self.model[-1]
        self.stride = fn(self.stride)
        head.stride = fn(head.stride)
        head.anchors = fn(head.anchors)
        head.strides = fn(head.strides)
        return self

