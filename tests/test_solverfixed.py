import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from solverfixed import burgers_vg_solver_2d as burgers
from solverfixed import trainer_burgers_vg as trainer
from solverfixed.config_pinn_burgers_vg import ConfigPINNBurgersVG, ConfigPINNBurgersVGSpectral
from solverfixed.pinn_models import ScalarLevyPINNSystem


class SolverContract(unittest.TestCase):
    def test_diffusion_eigenmode(self):
        nx, ny, dx, dy, dt, diffusion = 9, 7, .1, .2, .001, .3
        u = np.cos(np.pi * (np.arange(nx) + .5) / nx)[:, None] * np.cos(
            2 * np.pi * (np.arange(ny) + .5) / ny)[None, :]
        eigenvalues = (-4 * np.sin(np.pi / (2 * nx))**2 / dx**2,
                       -4 * np.sin(np.pi / ny)**2 / dy**2)
        gain = np.prod([(1 + .5 * dt * diffusion * value) /
                        (1 - .5 * dt * diffusion * value) for value in eigenvalues])
        actual = burgers._adi_step(u, diffusion, dx, dy, dt, np.zeros_like(u))
        np.testing.assert_allclose(actual, gain * u, atol=1e-14)

    def test_jump_increment_not_multiplied_by_dt_twice(self):
        for horizon in (.1, 1.):
            with self.subTest(horizon=horizon):
                cfg = ConfigPINNBurgersVG(Nx=5, Ny=4, Nt=3, t_max=horizon,
                                          nu_u=0., jump_intensity=.5)
                increments = np.full((5, 4, 3), .04, dtype=np.float32)
                with patch.object(burgers, "_ic_u", return_value=np.zeros((5, 4))):
                    field = burgers.solve_burgers_vg_2d(
                        cfg, np.zeros_like(increments), increments)[3]
                np.testing.assert_allclose(field[:, :, -1], .04, rtol=2e-7)

    def test_trainer_passes_increments_and_interpolates_rates(self):
        cfg = ConfigPINNBurgersVG(Nx=5, Ny=4, Nt=6)
        with patch.object(trainer, "solve_burgers_vg_2d", wraps=burgers.solve_burgers_vg_2d) as call:
            interpolators, _, _ = trainer._build_realisation(cfg, 0, torch.device("cpu"))
        increments = call.call_args.kwargs["vg_jumps"]
        np.testing.assert_array_equal(interpolators["vg"].field.numpy(), increments / cfg.dt)


class CommandLineRun(unittest.TestCase):
    def test_training_export_and_checkpoint_reload(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            process = subprocess.run(
                [sys.executable, "-B", str(root / "run.py"), "--epochs", "2",
                 "--arms", "data_only", "base", "full", "cf", "--output", str(output)],
                cwd=directory, capture_output=True, text=True, timeout=120,
                env={**os.environ, "MPLCONFIGDIR": str(Path(directory) / "matplotlib")})
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads((output / "results.json").read_text())
            previous_threads = torch.get_num_threads()
            self.addCleanup(torch.set_num_threads, previous_threads)
            torch.set_num_threads(report["environment"]["threads"])
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["runs"]), 4)
            self.assertGreater((output / "comparison.png").stat().st_size, 1000)
            self.assertEqual(len({r["reference_sha256"] for r in report["runs"]}), 1)
            for run in report["runs"]:
                job = output / (run["arm"] + "_s0")
                with np.load(job / "fields.npz") as arrays:
                    prediction = arrays["prediction"]
                    reference = arrays["reference"]
                    self.assertTrue(np.isfinite(prediction).all())
                    self.assertEqual(hashlib.sha256(reference.tobytes()).hexdigest(),
                                     run["reference_sha256"])
                    p, r = prediction.astype(np.float64), reference.astype(np.float64)
                    self.assertAlmostEqual(np.linalg.norm(p - r) / np.linalg.norm(r),
                                           run["metrics"]["rel_l2"], places=10)
                    self.assertAlmostEqual(p.var() / r.var(), run["metrics"]["var_ratio"], places=10)
                    checkpoint = torch.load(job / "checkpoint.pt", map_location="cpu", weights_only=True)
                    cfg = ConfigPINNBurgersVGSpectral()
                    for key, value in checkpoint["cfg"].items():
                        setattr(cfg, key, value)
                    model = ScalarLevyPINNSystem(cfg).eval()
                    model.load_state_dict(checkpoint["model"], strict=True)
                    coords = [a.reshape(-1, 1) for a in torch.meshgrid(*[
                        torch.tensor(arrays[key], dtype=torch.float32) for key in ("x", "y", "t")
                    ], indexing="ij")]
                    with torch.inference_mode():
                        reloaded = model.net_u(*coords).reshape(prediction.shape).numpy()
                    np.testing.assert_allclose(reloaded, prediction, rtol=1e-6, atol=1e-7)
            original_report = (output / "results.json").read_bytes()
            repeat = subprocess.run(
                [sys.executable, "-B", str(root / "run.py"), "--epochs", "1",
                 "--output", str(output)], cwd=directory, capture_output=True,
                text=True, timeout=30)
            self.assertEqual(repeat.returncode, 2, repeat.stdout + repeat.stderr)
            self.assertIn("Output already exists", repeat.stderr)
            self.assertEqual((output / "results.json").read_bytes(), original_report)


if __name__ == "__main__":
    unittest.main()
