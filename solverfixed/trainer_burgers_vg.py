"""Trainer for Burgers + VG (vanilla / spectral)."""

from __future__ import annotations
import os, time
import dataclasses
import numpy as np
import torch

from .config_pinn_burgers_vg import ConfigPINNBurgersVG, ConfigPINNBurgersVGSpectral
from .burgers_vg_solver_2d import solve_burgers_vg_2d, ic_u_func
from .pde_residuals_burgers_vg import (compute_residual_burgers_vg, ic_loss, bc_loss_neumann)
from .vg_jumps import generate_vg_jump_field
from .pinn_models import ScalarLevyPINNSystem
from .noise_interp_3d import NoiseInterpolator3D
from .trainer_template import (
    _build_spectral_subgrid, _spectral_targets, _spectral_loss,
    sample_collocation, sample_ic, sample_bc_neumann,
)


def _build_realisation(cfg, seed, device):
    """One FD realisation on the window [cfg.t_min, cfg.t_max].

    With `cfg.burn_t > 0` the solver integrates that much extra physical time
    BEFORE the window and the transient is discarded. Collocation, IC, BC and
    the spectral sub-grid all key off cfg.t_min/cfg.t_max, so the surviving
    window is written back onto cfg here and nothing downstream needs to know
    about the burn-in. Writing it back is idempotent: cfg.Nt keeps the window
    length, so calling this twice on one cfg burns in once.
    """
    burn_t = float(getattr(cfg, "burn_t", 0.0))
    dt0 = cfg.dt
    n_burn = int(round(burn_t / dt0)) if burn_t > 0.0 else 0

    if n_burn:
        # dt is a computed PROPERTY, (t_max - t_min) / (Nt - 1). Extending Nt
        # without moving t_max would refine the step instead of lengthening the
        # interval, leaving the transient exactly where it was -- so t_max is
        # set from the captured dt0 and the result asserted.
        gen = dataclasses.replace(cfg, t_min=0.0, Nt=cfg.Nt + n_burn,
                                  t_max=(cfg.Nt + n_burn - 1) * dt0)
        assert abs(gen.dt - dt0) < 1e-12 * max(dt0, 1.0), "dt drifted"
    else:
        gen = cfg

    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal((gen.Nx, gen.Ny, gen.Nt)).astype(np.float32)
             * gen.sigma_u / np.sqrt(gen.dx * gen.dy * gen.dt))
    rng_vg = np.random.default_rng(seed + 10000)
    vg_inc = generate_vg_jump_field(
        gen.Nx, gen.Ny, gen.Nt, gen.dt,
        sigma_vg=gen.sigma_vg, nu_vg=gen.nu_vg, theta_vg=gen.theta_vg, rng=rng_vg)
    # Solvers consume increments; the differential residual consumes rates.
    vg_jumps = vg_inc / gen.dt
    x_grid, y_grid, t_grid, U_fd = solve_burgers_vg_2d(gen, noise, vg_jumps=vg_inc, rng=rng)

    if n_burn:
        noise, vg_jumps = noise[:, :, n_burn:], vg_jumps[:, :, n_burn:]
        U_fd, t_grid = U_fd[:, :, n_burn:], t_grid[n_burn:]
        cfg.t_min, cfg.t_max = float(t_grid[0]), float(t_grid[-1])

    interp = {
        "u":  NoiseInterpolator3D(noise, x_grid, y_grid, t_grid, device),
        "vg": NoiseInterpolator3D(vg_jumps, x_grid, y_grid, t_grid, device),
    }
    grids = (x_grid, y_grid, t_grid)
    fields = torch.tensor(U_fd, dtype=torch.float32, device=device)
    return interp, grids, fields


def train_one(cfg, seed: int, device: torch.device,
              mode: str = "vanilla", save_path: str | None = None,
              data_seed: int | None = None):
    """`data_seed` (optional) decouples the FD realisation from the network
    init/collocation seed — needed for deep-ensemble baselines (M inits on ONE
    realisation). Default None keeps the original behaviour (data_seed = seed)."""
    assert mode in ("vanilla", "spectral")
    torch.manual_seed(seed); np.random.seed(seed)

    ds = seed if data_seed is None else data_seed
    burn_in = float(getattr(cfg, "burn_t", 0.0)) > 0.0
    print(f"  [burgers-vg-{mode}] building realisation seed={ds}"
          f"{f' burn_t={cfg.burn_t}' if burn_in else ''}", flush=True)
    interp, grids, U_fd = _build_realisation(cfg, ds, device)
    if burn_in:
        print(f"  [burgers-vg-{mode}] window t in "
              f"[{cfg.t_min:.4f}, {cfg.t_max:.4f}]", flush=True)
    Nx, Ny, Nt = U_fd.shape

    system = ScalarLevyPINNSystem(cfg).to(device)
    print(f"  [burgers-vg-{mode}] params={sum(p.numel() for p in system.parameters()):,}", flush=True)

    opt = torch.optim.Adam(system.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=cfg.lr_min)

    x_grid_t = torch.tensor(grids[0], dtype=torch.float32, device=device)
    y_grid_t = torch.tensor(grids[1], dtype=torch.float32, device=device)

    if mode == "spectral":
        xs, ys, ts, ix_s, iy_s, it_s = _build_spectral_subgrid(cfg, grids, device)
        spec_targets = _spectral_targets(U_fd, ix_s, iy_s, it_s)
        print(f"  [burgers-vg-spectral] sub-grid: {cfg.spec_Nx}x{cfg.spec_Ny}x{cfg.spec_Nt}", flush=True)

    t0 = time.time()
    for ep in range(1, cfg.num_epochs + 1):
        system.train(); opt.zero_grad(set_to_none=True)

        # PDE residual
        xc, yc, tc = sample_collocation(cfg, cfg.batch_coll, device)
        if ep <= cfg.warmup_epochs:
            L_pde = torch.zeros((), device=device)
        else:
            R = compute_residual_burgers_vg(system, xc, yc, tc, interp["u"], interp["vg"], cfg)
            L_pde = cfg.w_pde * (R ** 2).mean()

        # IC
        if burn_in:
            # After a burn-in the window no longer opens on the analytic IC:
            # its initial data is the FD state at the new t_min. Sample that
            # slice on the grid, so the target is exact rather than interpolated.
            jx = torch.randint(0, Nx, (cfg.batch_ic,), device=device)
            jy = torch.randint(0, Ny, (cfg.batch_ic,), device=device)
            xi = x_grid_t[jx].view(-1, 1)
            yi = y_grid_t[jy].view(-1, 1)
            ti = torch.full_like(xi, cfg.t_min)
            ic_target = U_fd[jx, jy, 0].view(-1, 1)
        else:
            xi, yi, ti = sample_ic(cfg, cfg.batch_ic, device)
            ic_target = torch.tensor(
                ic_u_func(xi.detach().cpu().numpy().flatten(),
                          yi.detach().cpu().numpy().flatten(), cfg).reshape(-1, 1),
                dtype=torch.float32, device=device)
        L_ic = ic_loss(system.net_u, xi, yi, ti, ic_target)

        # BC
        bnds = sample_bc_neumann(cfg, cfg.batch_bc, device)
        L_bc = sum(bc_loss_neumann(system.net_u, xb, yb, tb,
                   "x" if tag.startswith("x") else "y")
                   for tag, (xb, yb, tb) in bnds.items())

        # Data
        n_d = cfg.batch_data
        ix = torch.randint(0, Nx, (n_d,), device=device)
        iy = torch.randint(0, Ny, (n_d,), device=device)
        it = torch.randint(0, Nt, (n_d,), device=device)
        xd = x_grid_t[ix].view(-1, 1); yd = y_grid_t[iy].view(-1, 1)
        td = (it.float() / max(Nt-1, 1)).view(-1, 1) * (cfg.t_max - cfg.t_min) + cfg.t_min
        with torch.no_grad():
            ut = U_fd[ix, iy, it].view(-1, 1)
        L_data = ((system.net_u(xd, yd, td) - ut) ** 2).mean()

        # Spectral (if applicable)
        if mode == "spectral" and ep % cfg.spec_every == 0:
            up_s = system.net_u(xs, ys, ts).reshape(cfg.spec_Nx, cfg.spec_Ny, cfg.spec_Nt)
            L_spec_xy, L_spec_t, L_var = _spectral_loss(up_s, spec_targets)
        else:
            L_spec_xy = torch.zeros((), device=device)
            L_spec_t = torch.zeros((), device=device)
            L_var = torch.zeros((), device=device)

        loss = (L_pde + cfg.w_ic*L_ic + cfg.w_bc*L_bc + cfg.w_data*L_data)
        if mode == "spectral":
            loss = loss + cfg.w_spec_xy*L_spec_xy + cfg.w_spec_t*L_spec_t + cfg.w_var*L_var

        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), cfg.grad_clip)
        opt.step(); sched.step()

        if ep % cfg.log_every == 0 or ep == 1:
            elapsed = time.time() - t0
            extra = ""
            if mode == "spectral":
                extra = f"  sxy={L_spec_xy.item():.2e} st={L_spec_t.item():.2e} var={L_var.item():.2e}"
            print(f"  [ep {ep:5d}/{cfg.num_epochs}] loss={loss.item():.3e} pde={L_pde.item():.2e} "
                  f"ic={L_ic.item():.2e} data={L_data.item():.2e}{extra} t={elapsed/60:.1f}m", flush=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"model": system.state_dict(), "cfg": cfg.__dict__,
                    "seed": seed, "mode": mode}, save_path)
        print(f"  [burgers-vg-{mode}] saved -> {save_path}", flush=True)
    return system, U_fd, grids
