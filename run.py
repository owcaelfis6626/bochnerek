"""Train and compare PINNs on one generated Burgers–VG realisation per seed."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from solverfixed.config_pinn_burgers_vg import ConfigPINNBurgersVGSpectral
from solverfixed.metrics import metrics
from solverfixed.trainer_burgers_vg_randsub import train_one_randsub


ARMS = {
    "data_only": dict(w_pde=0., w_ic=0., w_bc=0., w_spec_xy=0., w_spec_t=0., w_var=0.),
    "base": dict(w_spec_xy=0., w_spec_t=0., w_var=0.),
    "full": {},
    "cf": dict(w_cf=500.),
    "no_pde": dict(w_pde=0.),
    "no_icbc": dict(w_ic=0., w_bc=0.),
    "nophys": dict(w_pde=0., w_ic=0., w_bc=0.),
}


def configuration(profile, arm, epochs=None):
    cfg = ConfigPINNBurgersVGSpectral()
    cfg.burn_t = 1.0
    if profile == "demo":
        for key, value in dict(
            Nx=16, Ny=16, Nt=64, hidden_layers=2, hidden_dim=32,
            num_fourier_features=16, spec_Nx=8, spec_Ny=8, spec_Nt=16,
            batch_coll=128, batch_ic=64, batch_bc=32, batch_data=128,
            num_epochs=100, warmup_epochs=20, log_every=25,
        ).items():
            setattr(cfg, key, value)
    for key, value in ARMS[arm].items():
        setattr(cfg, key, value)
    if epochs is not None:
        cfg.num_epochs = epochs
        cfg.warmup_epochs = min(cfg.warmup_epochs, epochs // 5)
    return cfg


def predict(model, grids, shape, device):
    coords = [a.reshape(-1, 1) for a in torch.meshgrid(*[
        torch.as_tensor(g, dtype=torch.float32, device=device) for g in grids
    ], indexing="ij")]
    model.eval()
    with torch.inference_mode():
        chunks = [model.net_u(*(v[start:start + 65536] for v in coords)).cpu()
                  for start in range(0, coords[0].shape[0], 65536)]
    return torch.cat(chunks).reshape(shape)


def save_figure(path, fields, grids, time_value):
    reference = fields[0][1]
    limit = max(float(np.abs(field).max()) for _, field in fields)
    error_limit = max(float(np.abs(field - reference).max()) for _, field in fields)
    fig, axes = plt.subplots(2, len(fields), figsize=(4 * len(fields), 7), squeeze=False)
    for column, (label, field) in enumerate(fields):
        extent = [grids[0][0], grids[0][-1], grids[1][0], grids[1][-1]]
        im = axes[0, column].imshow(field.T, origin="lower", extent=extent,
                                    cmap="coolwarm", vmin=-limit, vmax=limit)
        err = axes[1, column].imshow((field - reference).T, origin="lower",
                                     extent=extent, cmap="coolwarm",
                                     vmin=-max(error_limit, 1e-12),
                                     vmax=max(error_limit, 1e-12))
        axes[0, column].set_title(label)
        axes[1, column].set_title("Error vs FD reference")
        for row in range(2):
            axes[row, column].set_xlabel("x")
            axes[row, column].set_ylabel("y")
        fig.colorbar(im, ax=axes[0, column], shrink=.8)
        fig.colorbar(err, ax=axes[1, column], shrink=.8)
    fig.suptitle(f"Final-time slice, t={time_value:.4f} (first seed)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("demo", "paper"), default="demo")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=["base", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, help="Override epochs; warmup is capped at epochs/5")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are refused")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu")
    output = args.output or Path("outputs") / (
        args.profile + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    root = Path(__file__).resolve().parent
    sources = [root / "run.py", *sorted((root / "solverfixed").glob("*.py"))]
    report = {
        "profile": args.profile,
        "evaluation": "Full grid of each run's own training realisation; no held-out drivers",
        "environment": dict(python=platform.python_version(), torch=torch.__version__,
                            numpy=np.__version__, scipy=scipy.__version__,
                            matplotlib=matplotlib.__version__, device=str(device),
                            threads=torch.get_num_threads()),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "complete": False,
        "runs": [],
    }
    reference_hashes = {}
    slices = []
    for seed in args.seeds:
        for arm in args.arms:
            cfg = configuration(args.profile, arm, args.epochs)
            print(f"\n[{arm}, seed={seed}] {args.profile}, {cfg.num_epochs} epochs, {device}", flush=True)
            start = time.perf_counter()
            model, reference, grids = train_one_randsub(
                cfg, seed, device, mode="spectral", resample_every=1)
            training_seconds = time.perf_counter() - start
            prediction = predict(model, grids, reference.shape, device)
            reference = reference.cpu()
            if not torch.isfinite(prediction).all():
                raise RuntimeError(f"Non-finite prediction: {arm}, seed={seed}")
            digest = hashlib.sha256(reference.numpy().tobytes()).hexdigest()
            if digest != reference_hashes.setdefault(seed, digest):
                raise RuntimeError("Compared arms generated different references")
            job = output / f"{arm}_s{seed}"
            job.mkdir()
            torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            cfg=vars(cfg), seed=seed, arm=arm), job / "checkpoint.pt")
            np.savez_compressed(job / "fields.npz", prediction=prediction.numpy(),
                                reference=reference.numpy(), x=grids[0], y=grids[1], t=grids[2])
            values = metrics(prediction, reference)
            report["runs"].append(dict(arm=arm, seed=seed, cfg=vars(cfg).copy(),
                                       training_seconds=training_seconds,
                                       reference_sha256=digest, metrics=values))
            (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"  variance ratio={values['var_ratio']:.4f}, relative L2={values['rel_l2']:.4f}", flush=True)
            if seed == args.seeds[0]:
                if not slices:
                    slices.append(("FD reference", reference[:, :, -1].numpy().copy()))
                slices.append((arm, prediction[:, :, -1].numpy().copy()))
            del model, reference, prediction
    save_figure(output / "comparison.png", slices, grids, grids[2][-1])
    report["complete"] = True
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"\nResults: {output.resolve()}")
    if args.profile == "demo":
        print("Demo scores are diagnostics, not a reproduction of the paper's table.")


if __name__ == "__main__":
    main()
