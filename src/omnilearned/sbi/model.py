"""Smooth fixed-mask global-attention velocity for density benchmarks.

This is a separately trained architecture, not a reinterpretation of existing
PET checkpoints. No top-k neighborhoods or coordinate-derived masks are used.
"""

import math
import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("Width must be divisible by attention heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )

    def forward(self, x, mask):
        batch, particles, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(
            batch, particles, 3, self.heads, width // self.heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // self.heads)
        scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
        attention = scores.softmax(dim=-1)
        attended = (attention @ v).transpose(1, 2).reshape(batch, particles, width)
        x = (x + self.projection(attended)) * mask[..., None]
        return (x + self.mlp(self.norm2(x))) * mask[..., None]


class SmoothVelocity(nn.Module):
    def __init__(self, features=3, conditions=3, width=32, depth=2, heads=4):
        super().__init__()
        self.architecture = dict(
            features=features,
            conditions=conditions,
            width=width,
            depth=depth,
            heads=heads,
        )
        self.input = nn.Linear(features, width)
        self.condition = nn.Sequential(
            nn.Linear(conditions + 3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [AttentionBlock(width, heads) for _ in range(depth)]
        )
        self.output = nn.Linear(width, features)

    def forward(self, x, time, condition, mask):
        if not mask.any(dim=1).all():
            raise ValueError(
                "Attention requires at least one real constituent per event"
            )
        t = torch.stack(
            [time, torch.sin(math.pi * time), torch.cos(math.pi * time)], dim=-1
        )
        hidden = (
            self.input(torch.where(mask[..., None], x, 0.0))
            + self.condition(torch.cat([condition, t], dim=-1))[:, None]
        ) * mask[..., None]
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.output(hidden) * mask[..., None]
