"""
PDE residual for Burgers + VG:
    R = u_t + u(u_x + u_y) - nu*Lap(u) - sigma*xi - jump_intensity*J_VG
"""

import torch


def _grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True)[0]


def compute_residual_burgers_vg(system, x, y, t, noise_u, vg_jumps, cfg):
    u = system.net_u(x, y, t)
    u_t = _grad(u, t)
    u_x = _grad(u, x); u_xx = _grad(u_x, x)
    u_y = _grad(u, y); u_yy = _grad(u_y, y)
    xi = noise_u(x, y, t)
    j_vg = vg_jumps(x, y, t)
    R = (u_t
         + u * (u_x + u_y)              # nonlinear advection
         - cfg.nu_u * (u_xx + u_yy)     # viscous diffusion
         - xi                           # Gaussian noise
         - cfg.jump_intensity * j_vg)   # VG jump term
    return R


# IC / BC / penalties

def ic_loss(net, x, y, t0, target):
    return torch.mean((net(x, y, t0) - target) ** 2)


def bc_loss_neumann(net, x, y, t, axis: str):
    u = net(x, y, t)
    u_d = _grad(u, x if axis == "x" else y)
    return torch.mean(u_d ** 2)
