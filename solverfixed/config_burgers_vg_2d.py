"""
2D Stochastic Burgers + VG jumps configuration.

Equation:
    du/dt + u (du/dx + du/dy) = nu * Lap(u) + sigma * xi + jump_intensity * J_VG

This is the Lévy-driven analog of the standard 2D Burgers SPDE.
The combination of nonlinear advection + Lévy jumps creates rich
distributional structure with shock formation enhanced by jump
discontinuities.
"""

from dataclasses import dataclass


@dataclass
class ConfigBurgersVG2D:
    # -- Domain --------------------------------------------------------
    x_min: float = -1.0
    x_max: float =  1.0
    y_min: float = -1.0
    y_max: float =  1.0
    t_min: float =  0.0
    t_max: float =  0.5

    # -- Grid ----------------------------------------------------------
    Nx: int = 64
    Ny: int = 64
    Nt: int = 512

    # -- Burgers parameters --------------------------------------------
    nu_u:    float = 1.0e-2
    sigma_u: float = 0.05
    noise_type_u: str = "white"
    hurst_u: float = 0.7

    # -- VG jump parameters --------------------------------------------
    sigma_vg: float = 0.10        # smaller than Heston because shocks
    nu_vg: float = 0.20
    theta_vg: float = -0.03
    jump_intensity: float = 0.5   # smaller scale to avoid blowup

    # -- IC ------------------------------------------------------------
    ic_amplitude: float = 0.6

    # -- Misc ----------------------------------------------------------
    seed: int = 42
    output_dir: str = "results_burgers_vg_2d"

    @property
    def dx(self): return (self.x_max - self.x_min) / (self.Nx - 1)
    @property
    def dy(self): return (self.y_max - self.y_min) / (self.Ny - 1)
    @property
    def dt(self): return (self.t_max - self.t_min) / (self.Nt - 1)
