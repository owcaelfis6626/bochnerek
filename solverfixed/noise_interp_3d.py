"""
3D bilinear+linear-in-time interpolation of noise fields for PINN residuals.

Given a discrete noise field on (Nx, Ny, Nt) FD grid, evaluate it at
arbitrary (x, y, t) collocation points via trilinear interpolation.
"""

from __future__ import annotations
import numpy as np
import torch


class NoiseInterpolator3D:
    """Trilinear interpolation of (Nx, Ny, Nt) noise on a regular grid.

    Used for both Gaussian noise and VG jump fields.
    """

    def __init__(self, field: np.ndarray, x_grid, y_grid, t_grid, device):
        self.device = device
        # Field shape: (Nx, Ny, Nt)
        self.field = torch.tensor(field, dtype=torch.float32, device=device)
        self.x_grid = torch.tensor(x_grid, dtype=torch.float32, device=device)
        self.y_grid = torch.tensor(y_grid, dtype=torch.float32, device=device)
        self.t_grid = torch.tensor(t_grid, dtype=torch.float32, device=device)
        self.Nx, self.Ny, self.Nt = field.shape
        self.x_min, self.x_max = float(x_grid[0]), float(x_grid[-1])
        self.y_min, self.y_max = float(y_grid[0]), float(y_grid[-1])
        self.t_min, self.t_max = float(t_grid[0]), float(t_grid[-1])

    def __call__(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """Trilinear interp at (x, y, t)."""
        x = x.detach()
        y = y.detach()
        t = t.detach()

        # Normalize to [0, N-1]
        ix_f = (x - self.x_min) / (self.x_max - self.x_min) * (self.Nx - 1)
        iy_f = (y - self.y_min) / (self.y_max - self.y_min) * (self.Ny - 1)
        it_f = (t - self.t_min) / (self.t_max - self.t_min) * (self.Nt - 1)

        ix_f = ix_f.clamp(0, self.Nx - 1).squeeze(-1)
        iy_f = iy_f.clamp(0, self.Ny - 1).squeeze(-1)
        it_f = it_f.clamp(0, self.Nt - 1).squeeze(-1)

        ix0 = ix_f.floor().long(); ix1 = (ix0 + 1).clamp(max=self.Nx - 1)
        iy0 = iy_f.floor().long(); iy1 = (iy0 + 1).clamp(max=self.Ny - 1)
        it0 = it_f.floor().long(); it1 = (it0 + 1).clamp(max=self.Nt - 1)

        wx = (ix_f - ix0.float()).unsqueeze(-1)
        wy = (iy_f - iy0.float()).unsqueeze(-1)
        wt = (it_f - it0.float()).unsqueeze(-1)

        c000 = self.field[ix0, iy0, it0].unsqueeze(-1)
        c001 = self.field[ix0, iy0, it1].unsqueeze(-1)
        c010 = self.field[ix0, iy1, it0].unsqueeze(-1)
        c011 = self.field[ix0, iy1, it1].unsqueeze(-1)
        c100 = self.field[ix1, iy0, it0].unsqueeze(-1)
        c101 = self.field[ix1, iy0, it1].unsqueeze(-1)
        c110 = self.field[ix1, iy1, it0].unsqueeze(-1)
        c111 = self.field[ix1, iy1, it1].unsqueeze(-1)

        c00 = c000 * (1 - wt) + c001 * wt
        c01 = c010 * (1 - wt) + c011 * wt
        c10 = c100 * (1 - wt) + c101 * wt
        c11 = c110 * (1 - wt) + c111 * wt

        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy

        return c0 * (1 - wx) + c1 * wx
