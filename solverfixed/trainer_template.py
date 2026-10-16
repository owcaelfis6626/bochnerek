"""
Unified trainer template for spde-inne single-scalar Lévy problems.

Used by Burgers-VG, RDA-VG, AC-VG (and adapted for FracDiff-VG).

The trainer is parameterized by:
- cfg: ConfigPINN<Problem>VG or ConfigPINN<Problem>VGSpectral
- compute_residual: problem-specific PDE residual function
- ic_func: initial condition generator
- solve_func: FD solver
- mode: 'vanilla' or 'spectral'
"""

from __future__ import annotations
import os
import time
import numpy as np
import torch


# ─── Spectral helpers (same as Phase 2) ──────────────────────────────

def _spatial_psd_xy(field):
    Fx = torch.fft.rfft(field, dim=-3).abs() ** 2
    Fy = torch.fft.rfft(field, dim=-2).abs() ** 2
    psd_x = Fx.mean(dim=(-2, -1))
    psd_y = Fy.mean(dim=(-3, -1))
    K = min(psd_x.shape[-1], psd_y.shape[-1])
    return 0.5 * (psd_x[..., :K] + psd_y[..., :K])


def _temporal_psd(field):
    Ft = torch.fft.rfft(field, dim=-1).abs() ** 2
    return Ft.mean(dim=(-3, -2))


def _build_spectral_subgrid(cfg, grids, device):
    x_grid, y_grid, t_grid = grids
    Nx, Ny, Nt = len(x_grid), len(y_grid), len(t_grid)
    ix = torch.linspace(0, Nx-1, cfg.spec_Nx, device=device).round().long()
    iy = torch.linspace(0, Ny-1, cfg.spec_Ny, device=device).round().long()
    it = torch.linspace(0, Nt-1, cfg.spec_Nt, device=device).round().long()
    xg = torch.tensor(x_grid, dtype=torch.float32, device=device)[ix]
    yg = torch.tensor(y_grid, dtype=torch.float32, device=device)[iy]
    tg = torch.tensor(t_grid, dtype=torch.float32, device=device)[it]
    XX, YY, TT = torch.meshgrid(xg, yg, tg, indexing="ij")
    return (XX.reshape(-1, 1).contiguous(),
            YY.reshape(-1, 1).contiguous(),
            TT.reshape(-1, 1).contiguous(),
            ix, iy, it)


def _spectral_targets(field, ix, iy, it):
    with torch.no_grad():
        sub = field[ix][:, iy][:, :, it]
        sp = _spatial_psd_xy(sub.unsqueeze(0)).squeeze(0)
        tp = _temporal_psd(sub.unsqueeze(0)).squeeze(0)
        return {
            "log_sp": torch.log10(sp.clamp(min=1e-12)),
            "log_tp": torch.log10(tp.clamp(min=1e-12)),
            "var":    sub.var(),
        }


def _spectral_loss(pred, targets, eps=1e-12):
    sp = _spatial_psd_xy(pred.unsqueeze(0)).squeeze(0)
    tp = _temporal_psd(pred.unsqueeze(0)).squeeze(0)
    log_sp = torch.log10(sp.clamp(min=eps))
    log_tp = torch.log10(tp.clamp(min=eps))
    L_xy = ((log_sp - targets["log_sp"]) ** 2).mean()
    L_t = ((log_tp - targets["log_tp"]) ** 2).mean()
    var_p = pred.var()
    var_t = targets["var"]
    L_v = ((var_p - var_t) / (var_t + eps)) ** 2
    return L_xy, L_t, L_v


# ─── Sampling ─────────────────────────────────────────────────────────

def sample_collocation(cfg, n, device):
    x = torch.rand(n, 1, device=device) * (cfg.x_max - cfg.x_min) + cfg.x_min
    y = torch.rand(n, 1, device=device) * (cfg.y_max - cfg.y_min) + cfg.y_min
    t = torch.rand(n, 1, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    return x, y, t


def sample_ic(cfg, n, device):
    x = torch.rand(n, 1, device=device) * (cfg.x_max - cfg.x_min) + cfg.x_min
    y = torch.rand(n, 1, device=device) * (cfg.y_max - cfg.y_min) + cfg.y_min
    t = torch.full((n, 1), cfg.t_min, device=device)
    return x, y, t


def sample_bc_neumann(cfg, n, device):
    """Sample boundary points for Neumann BC."""
    t = torch.rand(n, 1, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min
    y_in = torch.rand(n, 1, device=device) * (cfg.y_max - cfg.y_min) + cfg.y_min
    x_in = torch.rand(n, 1, device=device) * (cfg.x_max - cfg.x_min) + cfg.x_min
    bnds = {}
    for tag, fixed_axis, fixed_val in [
        ("xmin", "x", cfg.x_min), ("xmax", "x", cfg.x_max),
        ("ymin", "y", cfg.y_min), ("ymax", "y", cfg.y_max),
    ]:
        if fixed_axis == "x":
            xb = torch.full((n, 1), fixed_val, device=device); yb = y_in.clone()
        else:
            xb = x_in.clone(); yb = torch.full((n, 1), fixed_val, device=device)
        tb = t.clone()
        xb.requires_grad_(True); yb.requires_grad_(True)
        bnds[tag] = (xb, yb, tb)
    return bnds
