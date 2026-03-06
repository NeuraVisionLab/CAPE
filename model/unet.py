"""
2D UNet for distance map regression (CREMI with CAPE).
Output is ReLU-applied to get non-negative distance map.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        is_first: bool,
        n_convs: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True,
        pooling: str = "max",
    ):
        super().__init__()
        layers = []
        if not is_first:
            in_channels = out_channels // 2
            layers.append(nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
        if batch_norm:
            layers.append(nn.GroupNorm(8, out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
            if batch_norm:
                layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(p=dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_convs: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True,
        upsampling: str = "bilinear",
    ):
        super().__init__()
        out_channels = in_channels // 2
        if upsampling in ("nearest", "bilinear"):
            self.upsampling = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsampling),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True),
            )
        else:
            self.upsampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
        if batch_norm:
            layers.append(nn.GroupNorm(8, out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
            if batch_norm:
                layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(p=dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsampling(x)
        x = torch.cat([skip, x], dim=1)
        return self.layers(x)


class UNet(nn.Module):
    """2D UNet for distance map regression (single channel output, ReLU)."""

    def __init__(
        self,
        in_channels: int = 1,
        m_channels: int = 32,
        out_channels: int = 1,
        n_convs: int = 2,
        n_levels: int = 3,
        dropout: float = 0.1,
        batch_norm: bool = True,
        upsampling: str = "bilinear",
        pooling: str = "max",
    ):
        super().__init__()
        assert n_levels >= 1 and n_convs >= 1
        channels = [2**i * m_channels for i in range(n_levels + 1)]
        self.down_path = nn.ModuleList()
        self.down_path.append(DownBlock2D(in_channels, channels[0], True, n_convs, dropout, batch_norm, pooling))
        for i in range(1, n_levels):
            self.down_path.append(DownBlock2D(None, channels[i], False, n_convs, dropout, batch_norm, pooling))
        self.bottom = nn.Sequential(
            DownBlock2D(None, channels[n_levels], False, n_convs, dropout, batch_norm, pooling)
        )
        self.up_path = nn.ModuleList()
        for c in reversed(channels[1:]):
            self.up_path.append(UpBlock2D(c, n_convs, dropout, batch_norm, upsampling))
        self.last = nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmaps = [self.down_path[0](x)]
        for i in range(1, len(self.down_path)):
            fmaps.append(self.down_path[i](fmaps[-1]))
        x = self.bottom(fmaps[-1])
        for i, block in enumerate(self.up_path):
            x = block(x, fmaps[len(fmaps) - i - 2])
        x = self.last(x)
        return F.relu(x)
