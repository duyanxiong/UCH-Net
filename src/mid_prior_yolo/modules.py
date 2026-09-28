"""Modules used exclusively by the Mid-Prior detector.

The implementations mirror the corresponding modules in the research project so
that layer indices and state-dict keys remain compatible.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, DFL
from ultralytics.nn.modules.block import C3k
from ultralytics.nn.modules.conv import autopad
from ultralytics.utils.tal import dist2bbox, make_anchors


class MlpCASViT(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class LocalIntegration(nn.Module):
    def __init__(self, dim, ratio=1.0, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1),
        )

    def forward(self, x):
        return self.network(x)


class AdditiveTokenMixer(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.0):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=attn_bias)
        self.oper_q = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.oper_k = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q, k = self.oper_q(q), self.oper_k(k)
        return self.proj_drop(self.proj(self.dwc(q + k) * v))


class AdditiveBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        attn_bias=False,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        if drop_path:
            raise ValueError("This model uses drop_path=0; non-zero stochastic depth is intentionally not bundled.")
        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MlpCASViT(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class C3kAdditiveBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        hidden = int(c2 * e)
        self.m = nn.Sequential(*(AdditiveBlock(hidden) for _ in range(n)))


class C3k2AdditiveBlock(nn.Module):
    """C3k2 wrapper whose inner blocks use additive attention."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3kAdditiveBlock(self.c, self.c, 2, shortcut, g)
            if c3k
            else AdditiveBlock(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))


class RegionalPhysicalPriorExtractor(nn.Module):
    def __init__(self, in_channels, dim, spatial_kernels=(3, 5), block_sizes=(1, 2)):
        super().__init__()
        self.block_sizes = block_sizes
        self.ref_embed = Conv(in_channels, dim, k=1, s=1)
        self.spatial_branches = nn.ModuleList(
            Conv(dim, dim, k=k, s=1, p=k // 2, g=dim) for k in spatial_kernels
        )
        self.block_proj = nn.ModuleList(Conv(dim, dim, k=1, s=1) for _ in block_sizes)
        self.global_proj = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.SiLU())
        self.pos_proj = nn.Sequential(nn.Conv2d(2, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU())
        prior_channels = dim * (1 + len(spatial_kernels) + len(block_sizes) + 2)
        self.fuse = Conv(prior_channels, dim, k=1, s=1)
        self.to_params = nn.Conv2d(dim, dim * 2, 1)

    @staticmethod
    def _make_coord(batch, height, width, device, dtype):
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy = yy.view(1, 1, height, 1).expand(batch, 1, height, width)
        xx = xx.view(1, 1, 1, width).expand(batch, 1, height, width)
        return torch.cat((xx, yy), dim=1)

    def forward(self, ref, target_size=None):
        if target_size is not None and ref.shape[-2:] != target_size:
            ref = F.interpolate(ref, size=target_size, mode="bilinear", align_corners=False)
        ref = self.ref_embed(ref)
        batch, _, height, width = ref.shape
        features = [ref]
        features.extend(branch(ref) for branch in self.spatial_branches)
        for block_size, projection in zip(self.block_sizes, self.block_proj):
            pooled = F.adaptive_avg_pool2d(ref, (block_size, block_size))
            pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
            features.append(projection(pooled))
        avg_light = F.adaptive_avg_pool2d(ref, 1)
        max_light = F.adaptive_max_pool2d(ref, 1)
        features.append(self.global_proj(torch.cat((avg_light, max_light), 1)).expand(-1, -1, height, width))
        features.append(self.pos_proj(self._make_coord(batch, height, width, ref.device, ref.dtype)))
        context = self.fuse(torch.cat(features, dim=1))
        return self.to_params(context), context


class CGAPixelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, guidance):
        batch, channels, height, width = x.shape
        pair = torch.stack((x, guidance), dim=2).reshape(batch, 2 * channels, height, width)
        return self.sigmoid(self.conv(pair))


class CGASpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect")

    def forward(self, x):
        average = torch.mean(x, dim=1, keepdim=True)
        maximum, _ = torch.max(x, dim=1, keepdim=True)
        return self.conv(torch.cat((average, maximum), dim=1))


class CGAChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return self.conv(self.gap(x))


class PriorCGAFusionV2(nn.Module):
    """Physical-prior RGB correction followed by pixel-level RGB/X fusion."""

    def __init__(self, c1, dim, ref_dim, y_channels=None, scale_strength=0.5, bias_strength=0.1):
        super().__init__()
        self.scale_strength = scale_strength
        self.bias_strength = bias_strength
        y_channels = y_channels or dim
        self.x_proj = Conv(c1, dim, k=1, s=1)
        self.y_proj = Conv(y_channels, dim, k=1, s=1) if y_channels != dim else nn.Identity()
        self.prior_extractor = RegionalPhysicalPriorExtractor(ref_dim, dim)
        # Kept for exact checkpoint compatibility with the research implementation.
        self.sa = CGASpatialAttention()
        self.ca = CGAChannelAttention(dim)
        self.prior_attn = nn.Conv2d(dim, dim, 1)
        self.pa = CGAPixelAttention(dim)
        self.out_conv = Conv(dim, dim, k=3, s=1)
        nn.init.zeros_(self.prior_attn.weight)
        nn.init.zeros_(self.prior_attn.bias)

    def forward(self, data):
        x, y, ref = data
        target_size = x.shape[-2:]
        x = self.x_proj(x)
        if y.shape[-2:] != target_size:
            y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
        y = self.y_proj(y)
        params, _ = self.prior_extractor(ref, target_size=target_size)
        raw_scale, raw_bias = params.chunk(2, dim=1)
        gain = 1.0 + self.scale_strength * torch.tanh(raw_scale)
        bias = self.bias_strength * torch.tanh(raw_bias)
        x_corrected = x * gain + bias
        initial = x_corrected + y
        gate = self.pa(initial, initial)
        return self.out_conv(x_corrected * gate + y * (1.0 - gate))


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class ConvGN(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class DetectLSCD(nn.Module):
    """Lightweight shared-convolution detection head."""

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.conv = nn.ModuleList(nn.Sequential(ConvGN(x, hidc, 3)) for x in ch)
        self.share_conv = nn.Sequential(ConvGN(hidc, hidc, 3, g=hidc), ConvGN(hidc, hidc, 1))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        x = list(x)
        for i in range(self.nl):
            x[i] = self.share_conv(self.conv[i](x[i]))
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:
            return x
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (value.transpose(0, 1) for value in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        decoded = self.decode_bboxes(box)
        output = torch.cat((decoded, cls.sigmoid()), 1)
        return output if self.export else (output, x)

    def bias_init(self):
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

