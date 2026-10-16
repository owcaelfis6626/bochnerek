"""
FD solver for 2D Stochastic Burgers + Variance-Gamma jumps.

    du/dt + u (du/dx + du/dy) = nu * Lap(u) + sigma * xi + jump_intensity * J_VG

Numerics:
- Diffusion: Peaceman-Rachford ADI
- Advection: 1st-order upwind (shock-robust)
- Gaussian noise: Itô-Euler
- VG jumps: explicit Itô-style addition
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from .config_burgers_vg_2d import ConfigBurgersVG2D

from .vg_jumps import generate_vg_jump_field


# ---------------------- ADI helpers ---------------------------------
def _tridiag_neumann(N, r):
    half_r = 0.5 * r
    diag = np.full(N, 1 + 2 * half_r); off = np.full(N - 1, -half_r)
    diag[0] = 1 + half_r; diag[-1] = 1 + half_r
    lhs = np.zeros((3, N))
    lhs[0, 1:] = off; lhs[1, :] = diag; lhs[2, :-1] = off
    d_rhs = np.full(N, 1 - 2 * half_r)
    d_rhs[0] = 1 - half_r; d_rhs[-1] = 1 - half_r
    o_rhs = half_r
    def rhs_mult(u):
        v = d_rhs * u
        v[:-1] += o_rhs * u[1:]; v[1:] += o_rhs * u[:-1]
        return v
    return lhs, rhs_mult


def _adi_step(u, D, dx, dy, dt, forcing):
    """Peaceman-Rachford ADI step for u_t = D(u_xx + u_yy) + forcing.

    Each half-step treats exactly ONE direction implicitly and the CROSS
    direction explicitly. (An earlier revision added the same direction's
    explicit diffusion on the right-hand side as well, `rhs_x(col) + yyc`, which
    expands to u^{n+1} = u^n + (3/2) D dt (Lx+Ly) u^n + dt f -- i.e. 3/2 D.
    Measured D_eff/D = 1.5001; see solver_fix/verify_adi.py.)
    """
    Nx, Ny = u.shape
    rx = D * dt / dx ** 2; ry = D * dt / dy ** 2
    lhs_x, rhs_x = _tridiag_neumann(Nx, rx)
    lhs_y, rhs_y = _tridiag_neumann(Ny, ry)
    u_half = np.empty_like(u)
    for j in range(Ny):
        col = u[:, j]
        if 0 < j < Ny - 1:
            yyc = ry * 0.5 * (u[:, j+1] - 2*u[:, j] + u[:, j-1])
        elif j == 0:
            yyc = ry * 0.5 * (u[:, j+1] - u[:, j])
        else:
            yyc = ry * 0.5 * (u[:, j-1] - u[:, j])
        rhs_vec = col + yyc + 0.5 * dt * forcing[:, j]
        u_half[:, j] = solve_banded((1, 1), lhs_x, rhs_vec)
    u_new = np.empty_like(u)
    for i in range(Nx):
        row = u_half[i, :]
        if 0 < i < Nx - 1:
            xxc = rx * 0.5 * (u_half[i+1, :] - 2*u_half[i, :] + u_half[i-1, :])
        elif i == 0:
            xxc = rx * 0.5 * (u_half[i+1, :] - u_half[i, :])
        else:
            xxc = rx * 0.5 * (u_half[i-1, :] - u_half[i, :])
        rhs_vec = row + xxc + 0.5 * dt * forcing[i, :]
        u_new[i, :] = solve_banded((1, 1), lhs_y, rhs_vec)
    return u_new


def _upwind_advection(u, dx, dy):
    pos = (u >= 0)
    bwd_x = np.zeros_like(u); fwd_x = np.zeros_like(u)
    bwd_x[1:, :] = (u[1:, :] - u[:-1, :]) / dx
    fwd_x[:-1, :] = (u[1:, :] - u[:-1, :]) / dx
    u_x = np.where(pos, bwd_x, fwd_x)
    bwd_y = np.zeros_like(u); fwd_y = np.zeros_like(u)
    bwd_y[:, 1:] = (u[:, 1:] - u[:, :-1]) / dy
    fwd_y[:, :-1] = (u[:, 1:] - u[:, :-1]) / dy
    u_y = np.where(pos, bwd_y, fwd_y)
    return u * (u_x + u_y)


def _ic_u(X, Y, cfg, rng):
    base = cfg.ic_amplitude * np.sin(np.pi * X) * np.sin(np.pi * Y)
    perturb = 0.05 * rng.standard_normal(X.shape)
    return base + perturb


def ic_u_func(X, Y, cfg, rng=None):
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    return _ic_u(X, Y, cfg, rng)


def solve_burgers_vg_2d(cfg, noise_u, vg_jumps=None, rng=None):
    """Solve 2D Burgers + VG jumps SPDE.

    noise_u contains rates; vg_jumps contains increments over cfg.dt.
    If vg_jumps is None, generate increments with seed cfg.seed + 10000.
    """
    Nx, Ny, Nt = cfg.Nx, cfg.Ny, cfg.Nt
    dx, dy, dt = cfg.dx, cfg.dy, cfg.dt
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    if vg_jumps is None:
        rng_vg = np.random.default_rng(cfg.seed + 10000)
        vg_jumps = generate_vg_jump_field(
            Nx, Ny, Nt, dt,
            sigma_vg=cfg.sigma_vg, nu_vg=cfg.nu_vg, theta_vg=cfg.theta_vg,
            rng=rng_vg,
        )

    # ADI integrates forcing rates, so convert increments exactly once.
    vg_rate = vg_jumps / dt

    x_grid = np.linspace(cfg.x_min, cfg.x_max, Nx)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, Ny)
    t_grid = np.linspace(cfg.t_min, cfg.t_max, Nt)
    X, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")

    U = np.zeros((Nx, Ny, Nt), dtype=np.float32)
    U[:, :, 0] = _ic_u(X, Yg, cfg, rng)

    for n in range(Nt - 1):
        u_n = U[:, :, n].astype(np.float64)
        advect = _upwind_advection(u_n, dx, dy)
        # Forcing = -advection + Gaussian noise + VG jump
        forcing = -advect + noise_u[:, :, n] + cfg.jump_intensity * vg_rate[:, :, n]
        u_np1 = _adi_step(u_n, cfg.nu_u, dx, dy, dt, forcing)
        u_np1 = np.clip(u_np1, -3.0, 3.0)
        U[:, :, n + 1] = u_np1.astype(np.float32)

    return x_grid, y_grid, t_grid, U
