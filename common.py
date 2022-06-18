#!/usr/bin/env python3

from typing import Callable, Iterable
from functools import partial
import torch as th
nn = th.nn
F = nn.functional


class Conv2D(nn.Module):
    def __init__(self, c_in: int, c_out: int,
                 kernel_size: int = 5,
                 padding: int = 2):
        super().__init__()
        conv = nn.Conv2d(c_in, c_out,
                         kernel_size=kernel_size,
                         padding=padding)
        relu = nn.ReLU(inplace=True)
        self.layer = nn.Sequential(conv, relu)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layer(x)


class ConvT2D(nn.Module):
    def __init__(self, c_in: int, c_out: int,
                 stride: int = 2,
                 kernel_size: int = 5,
                 padding: int = 1):
        super().__init__()
        conv = nn.ConvTranspose2d(c_in, c_out,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding)
        relu = nn.ReLU(inplace=True)
        self.layer = nn.Sequential(conv, relu)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layer(x)


class PositionEmbedding(nn.Module):
    def __init__(self,
                 num_channels: int,
                 spatial_shape: Iterable[int]):
        super().__init__()
        self.linear = nn.Linear(2 * len(spatial_shape), num_channels)  # 2+2
        grid = th.stack(
            th.meshgrid(
                * [th.linspace(0.0, 1.0, d, requires_grad=False)
                   for d in spatial_shape]),
            dim=-1)  # HxWxD
        grid = th.cat([grid, 1 - grid], dim=-1)  # HxWx(2xD)
        self.register_buffer('grid', grid)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """assumes CHW layout for input/output."""
        return x + self.linear(self.grid).permute(2, 0, 1)
