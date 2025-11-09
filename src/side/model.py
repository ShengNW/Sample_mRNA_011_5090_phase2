"""Dual-branch CNN model with FiLM conditioning for tissue-specific modulation."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation for 1D feature maps."""

    def __init__(self, cond_dim: int, num_channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, num_channels)
        self.beta = nn.Linear(cond_dim, num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(cond).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1)
        return x * (1 + gamma) + beta


class ResidualBlock1D(nn.Module):
    """A simple residual block with Conv1d → BN → GELU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        return y + self.shortcut(x)


class BranchCNN(nn.Module):
    """One branch that processes either 5' or 3' UTR sequences."""

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        block_channels: Sequence[int],
        film_dim: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, stem_channels, kernel_size=1)
        self.blocks = nn.ModuleList()
        self.film = nn.ModuleList()
        prev = stem_channels
        for out_c in block_channels:
            self.blocks.append(ResidualBlock1D(prev, out_c))
            self.film.append(FiLMLayer(film_dim, out_c))
            prev = out_c
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.out_features = prev * 2

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block, film in zip(self.blocks, self.film):
            x = block(x)
            x = film(x, cond)
            x = F.gelu(x)
        avg = self.gap(x).squeeze(-1)
        mx = self.gmp(x).squeeze(-1)
        return torch.cat([avg, mx], dim=-1)


class DualBranchCNNFiLM(nn.Module):
    """Dual-branch FiLM conditioned CNN for regression."""

    def __init__(
        self,
        in_channels: int,
        num_organs: int,
        conv_channels: Iterable[int],
        stem_channels: int = 32,
        film_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        conv_channels = list(conv_channels)
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one element")
        self.tissue_embedding = nn.Embedding(num_organs, film_dim)
        self.branch5 = BranchCNN(in_channels, stem_channels, conv_channels, film_dim)
        self.branch3 = BranchCNN(in_channels, stem_channels, conv_channels, film_dim)
        fused_dim = self.branch5.out_features + self.branch3.out_features
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, utr5: torch.Tensor, utr3: torch.Tensor, organ_id: torch.Tensor) -> torch.Tensor:
        if utr5.dim() != 3:
            raise ValueError(f"utr5 must be [B,C,L], got {utr5.shape}")
        if utr3.dim() != 3:
            raise ValueError(f"utr3 must be [B,C,L], got {utr3.shape}")
        cond = self.tissue_embedding(organ_id)
        feat5 = self.branch5(utr5, cond)
        feat3 = self.branch3(utr3, cond)
        fused = torch.cat([feat5, feat3], dim=-1)
        return self.head(fused).squeeze(-1)
