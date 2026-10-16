"""Burgers PINN training with spectral sub-grids resampled each epoch."""
from __future__ import annotations
import os
import time

import numpy as np
import torch

from .config_pinn_burgers_vg import ConfigPINNBurgersVGSpectral  # noqa: F401
from .pinn_models import ScalarLevyPINNSystem
from .pde_residuals_burgers_vg import (
    compute_residual_burgers_vg, ic_loss, bc_loss_neumann,
)
from .burgers_vg_solver_2d import ic_u_func
from .trainer_template import (
    _spectral_targets, _spectral_loss,
    sample_collocation, sample_ic, sample_bc_neumann,
)
from .trainer_burgers_vg import _build_realisation


# CF compares standardised one-point value distributions. It does not
# determine the spatial arrangement or recover Fourier phases.
CF_THETA = torch.linspace(0.0, 6.0, 64)
CF_W = torch.exp(-0.5 * CF_THETA ** 2)


def cf_term(pred, ref, eps=1e-8):
    a = (pred.flatten() - pred.mean()) / (pred.std() + eps)
    b = (ref.flatten() - ref.mean()) / (ref.std() + eps)
    th = CF_THETA.to(pred.device)[:, None]
    w = CF_W.to(pred.device)
    ta = th * a[None, :]
    tb = th * b[None, :]
    d_re = torch.cos(ta).mean(1) - torch.cos(tb).mean(1)
    d_im = torch.sin(ta).mean(1) - torch.sin(tb).mean(1)
    return ((d_re ** 2 + d_im ** 2) * w).sum() / w.sum()


def _random_subgrid(cfg, grids, device, generator):
    x_grid, y_grid, t_grid = grids
    Nx, Ny, Nt = len(x_grid), len(y_grid), len(t_grid)
    ix = torch.randperm(Nx, generator=generator, device=device)[:cfg.spec_Nx].sort()[0]
    iy = torch.randperm(Ny, generator=generator, device=device)[:cfg.spec_Ny].sort()[0]
    it = torch.randperm(Nt, generator=generator, device=device)[:cfg.spec_Nt].sort()[0]
    xg = torch.tensor(x_grid, dtype=torch.float32, device=device)[ix]
    yg = torch.tensor(y_grid, dtype=torch.float32, device=device)[iy]
    tg = torch.tensor(t_grid, dtype=torch.float32, device=device)[it]
    XX, YY, TT = torch.meshgrid(xg, yg, tg, indexing="ij")
    return (XX.reshape(-1, 1).contiguous(),
            YY.reshape(-1, 1).contiguous(),
            TT.reshape(-1, 1).contiguous(),
            ix, iy, it)


def train_one_randsub(cfg, seed: int, device: torch.device,
                      mode: str = "spectral", save_path: str | None = None,
                      data_seed: int | None = None, resample_every: int = 1):
    assert mode in ("vanilla", "spectral")
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed + 1000)

    ds = seed if data_seed is None else data_seed
    burn_in = float(getattr(cfg, "burn_t", 0.0)) > 0.0
    interp, grids, U_fd = _build_realisation(cfg, ds, device)
    Nx, Ny, Nt = U_fd.shape

    system = ScalarLevyPINNSystem(cfg).to(device)
    print(f"  [burgers-vg-randsub] params="
          f"{sum(p.numel() for p in system.parameters()):,}", flush=True)

    opt = torch.optim.Adam(system.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=cfg.lr_min)

    x_grid_t = torch.tensor(grids[0], dtype=torch.float32, device=device)
    y_grid_t = torch.tensor(grids[1], dtype=torch.float32, device=device)

    xs, ys, ts, ix_s, iy_s, it_s = _random_subgrid(cfg, grids, device, gen)
    spec_targets = _spectral_targets(U_fd, ix_s, iy_s, it_s)

    t0 = time.time()
    for ep in range(1, cfg.num_epochs + 1):
        system.train(); opt.zero_grad(set_to_none=True)

        xc, yc, tc = sample_collocation(cfg, cfg.batch_coll, device)
        if ep <= cfg.warmup_epochs:
            L_pde = torch.zeros((), device=device)
        else:
            R = compute_residual_burgers_vg(system, xc, yc, tc,
                                            interp["u"], interp["vg"], cfg)
            L_pde = cfg.w_pde * (R ** 2).mean()

        if burn_in:
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

        bnds = sample_bc_neumann(cfg, cfg.batch_bc, device)
        L_bc = sum(bc_loss_neumann(system.net_u, xb, yb, tb,
                   "x" if tag.startswith("x") else "y")
                   for tag, (xb, yb, tb) in bnds.items())

        n_d = cfg.batch_data
        ix = torch.randint(0, Nx, (n_d,), device=device)
        iy = torch.randint(0, Ny, (n_d,), device=device)
        it = torch.randint(0, Nt, (n_d,), device=device)
        xd = x_grid_t[ix].view(-1, 1); yd = y_grid_t[iy].view(-1, 1)
        td = (it.float() / max(Nt - 1, 1)).view(-1, 1) \
            * (cfg.t_max - cfg.t_min) + cfg.t_min
        with torch.no_grad():
            ut = U_fd[ix, iy, it].view(-1, 1)
        L_data = ((system.net_u(xd, yd, td) - ut) ** 2).mean()

        if ep % resample_every == 0:
            xs, ys, ts, ix_s, iy_s, it_s = _random_subgrid(cfg, grids, device, gen)
            spec_targets = _spectral_targets(U_fd, ix_s, iy_s, it_s)
            ref_sub = U_fd[ix_s][:, iy_s][:, :, it_s]
        if mode == "spectral" and ep % cfg.spec_every == 0:
            up_s = system.net_u(xs, ys, ts).reshape(
                cfg.spec_Nx, cfg.spec_Ny, cfg.spec_Nt)
            L_spec_xy, L_spec_t, L_var = _spectral_loss(up_s, spec_targets)
            L_cf = cf_term(up_s, ref_sub) if getattr(cfg, "w_cf", 0.0) else \
                torch.zeros((), device=device)
        else:
            L_spec_xy = torch.zeros((), device=device)
            L_spec_t = torch.zeros((), device=device)
            L_var = torch.zeros((), device=device)
            L_cf = torch.zeros((), device=device)

        loss = (L_pde + cfg.w_ic * L_ic + cfg.w_bc * L_bc
                + cfg.w_data * L_data)
        if mode == "spectral":
            loss = loss + cfg.w_spec_xy * L_spec_xy \
                + cfg.w_spec_t * L_spec_t + cfg.w_var * L_var \
                + getattr(cfg, "w_cf", 0.0) * L_cf

        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), cfg.grad_clip)
        opt.step(); sched.step()

        if ep % cfg.log_every == 0 or ep == 1:
            elapsed = time.time() - t0
            extra = ""
            if mode == "spectral":
                extra = (f"  sxy={L_spec_xy.item():.2e} "
                         f"st={L_spec_t.item():.2e} var={L_var.item():.2e}")
            print(f"  [ep {ep:5d}/{cfg.num_epochs}] loss={loss.item():.3e} "
                  f"pde={L_pde.item():.2e} ic={L_ic.item():.2e} "
                  f"data={L_data.item():.2e}{extra} t={elapsed/60:.1f}m",
                  flush=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"model": system.state_dict(), "cfg": cfg.__dict__,
                    "seed": seed, "mode": mode}, save_path)
        print(f"  [burgers-vg-randsub] saved -> {save_path}", flush=True)
    return system, U_fd, grids