"""
Shared PINN architecture for all spde-inne single-scalar problems.

ScalarLevyPINN: same architecture as Phase 2 PINNs but adapted for
Lévy-driven systems (wider network for fat-tailed targets).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding for (x, y, t)."""
    def __init__(self, num_features: int, scale: float = 2.0, in_dim: int = 3):
        super().__init__()
        B = scale * torch.randn(in_dim, num_features)
        self.register_buffer("B", B)

    def forward(self, x, y, t):
        v = torch.cat([x, y, t], dim=-1)
        proj = 2 * math.pi * v @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class _ScalarPINN(nn.Module):
    """MLP with Fourier features for u(x, y, t)."""
    def __init__(self, hidden_layers: int, hidden_dim: int,
                 num_fourier_features: int, fourier_scale: float):
        super().__init__()
        self.fourier = FourierFeatures(num_fourier_features, fourier_scale)
        in_dim = 2 * num_fourier_features
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        feat = self.fourier(x, y, t)
        return self.net(feat)


class ScalarLevyPINNSystem(nn.Module):
    """Single-scalar PINN system (u(x, y, t)).

    Used by Burgers-VG, RDA-VG, FracDiff-VG, AC-VG.
    Heston-VG uses HestonPINNSystem (2 channels) — see existing.
    """
    def __init__(self, cfg):
        super().__init__()
        self.net_u = _ScalarPINN(
            cfg.hidden_layers, cfg.hidden_dim,
            cfg.num_fourier_features, cfg.fourier_scale,
        )

    def forward_all(self, x, y, t):
        return self.net_u(x, y, t)
