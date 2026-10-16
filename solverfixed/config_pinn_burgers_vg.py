"""PINN configs for Burgers + VG problem (vanilla and spectral)."""

from dataclasses import dataclass

from .config_burgers_vg_2d import ConfigBurgersVG2D


@dataclass
class ConfigPINNBurgersVG(ConfigBurgersVG2D):
    # -- Network --------------------------------------------------------
    hidden_layers: int = 4
    hidden_dim: int = 128         # wider for jump-induced roughness
    num_fourier_features: int = 64
    fourier_scale: float = 3.0

    # -- Training -------------------------------------------------------
    num_epochs: int = 5000
    warmup_epochs: int = 1000
    learning_rate: float = 1e-3
    lr_min: float = 1e-6
    batch_coll: int = 4096
    batch_ic:   int = 512
    batch_bc:   int = 512
    batch_data: int = 1024
    grad_clip:  float = 5.0

    # -- Loss weights ---------------------------------------------------
    w_pde:  float = 1.0
    w_ic:   float = 50.0
    w_bc:   float = 10.0
    w_data: float = 5.0

    # -- Burn-in --------------------------------------------------------
    # Extra physical time integrated BEFORE the reported window and then
    # discarded. At t = 0 the solution is still relaxing off its deterministic
    # initial condition, so assumption (A1) (stationarity) does not hold and a
    # variance-ratio score is partly grading that relaxation. 1.0 puts the
    # window on the variance plateau. Default 0.0 reproduces the original runs.
    burn_t: float = 0.0

    # -- Misc -----------------------------------------------------------
    log_every: int = 100
    output_dir: str = "results_pinn_burgers_vg"


@dataclass
class ConfigPINNBurgersVGSpectral(ConfigPINNBurgersVG):
    # -- Spectral sub-grid ---------------------------------------------
    spec_Nx: int = 24
    spec_Ny: int = 24
    spec_Nt: int = 96

    # -- Spectral loss weights -----------------------------------------
    w_spec_xy: float = 3.0
    w_spec_t:  float = 5.0
    w_var:     float = 50.0
    spec_every: int = 1

    output_dir: str = "results_pinn_burgers_vg_spectral"
