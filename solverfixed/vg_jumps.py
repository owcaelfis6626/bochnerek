"""
Shared Variance-Gamma jump field generator for all spde-inne problems.

VG construction (Madan-Seneta, Madan-Carr-Chang):
    Per timestep, per spatial point:
        G ~ Gamma(shape=dt/nu, scale=nu)              (subordinator)
        Z ~ N(0, 1)
        J = theta * G + sigma * sqrt(G) * Z

This generates infinite-activity Lévy increments with:
- finite variation
- fat tails (excess kurtosis > 0)
- continuous distribution of jump sizes

Lévy measure (Madan-Seneta parameterization):
    nu_VG(dx) = (1 / (nu * |x|)) * exp(-A * |x|) dx
with  A = sqrt(theta^2 + 2 sigma^2 / nu) / sigma^2.
"""

from __future__ import annotations

import numpy as np


def generate_vg_jump_field(
    Nx: int, Ny: int, Nt: int,
    dt: float,
    sigma_vg: float = 0.12,
    nu_vg: float = 0.20,
    theta_vg: float = -0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate (Nx, Ny, Nt) VG jump increments.

    Parameters
    ----------
    Nx, Ny, Nt : grid dimensions
    dt : timestep size
    sigma_vg : diffusion of subordinated Brownian
    nu_vg : variance rate of Gamma subordinator (higher = larger jumps)
    theta_vg : drift of subordinated Brownian (negative = downward bias)
    rng : random generator

    Returns
    -------
    vg : (Nx, Ny, Nt) array of VG increments
    """
    if rng is None:
        rng = np.random.default_rng()

    out = np.zeros((Nx, Ny, Nt), dtype=np.float32)

    for k in range(Nt):
        G = rng.gamma(shape=dt / nu_vg, scale=nu_vg, size=(Nx, Ny))
        Z = rng.normal(0.0, 1.0, size=(Nx, Ny))
        out[:, :, k] = (theta_vg * G
                        + sigma_vg * np.sqrt(np.maximum(G, 1e-12)) * Z)

    return out


def vg_levy_A(sigma_vg: float, nu_vg: float, theta_vg: float) -> float:
    """Compute A parameter of VG Lévy density (for diagnostics)."""
    import math
    return (math.sqrt(theta_vg**2 + 2 * sigma_vg**2 / nu_vg)
            / sigma_vg**2)
