# Solverfixed

A small, runnable extract of the **corrected Burgers–Variance-Gamma PINN experiment**. It compares a baseline PINN with additional log-power-spectrum and variance losses on a generated numerical reference.

The main observation motivating this project: recovering variance or power statistics can coexist with a worse field reconstruction. The code reports both variance ratio and relative field L2 error.

## Quick start

Use Python 3.10 or newer. No dataset, checkpoint, GPU, external account, or local research directory is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python run.py
```

The CPU wheel command is for Linux/Windows and avoids installing CUDA dependencies for the demo. On macOS, omit that command and install `requirements.txt` directly. On Windows, activate with `.venv\Scripts\activate` instead. For a GPU environment, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) instead of the CPU wheel command. Colab normally has PyTorch preinstalled.

The default run uses CPU, generates one reference, and trains `base` and `full` for 100 epochs each on a small grid. A new timestamped directory is created under `outputs/` on every run.

For a shorter execution check:

```bash
python run.py --epochs 2
```

For a three-way comparison:

```bash
python run.py --arms data_only base full
```

Outputs:

- `results.json`: complete configurations, environment, source/reference hashes, timing, and full-grid metrics.
- `comparison.png`: reference, predictions, and errors at the final time for the first seed.
- `<arm>_s<seed>/checkpoint.pt`: model weights, configuration, seed, and arm.
- `<arm>_s<seed>/fields.npz`: prediction, reference, and coordinate grids.

`--output PATH` selects a specific new directory. Existing directories are refused to protect previous results.

## Google Colab

Upload and unpack this repository's ZIP in Colab, then run these notebook cells from the unpacked project directory:

```python
%pip install -r requirements.txt
```

```python
!python run.py --output outputs/colab_demo
```

```python
from IPython.display import Image, display
display(Image("outputs/colab_demo/comparison.png"))
```

CPU is enough for the demo. For GPU, enable a GPU runtime and add `--device cuda`.

## Larger experiment

```bash
python run.py --profile paper --device cuda --arms data_only base full --seeds 0 1 2 3 4 5 6 7
```

The `paper` profile retains the original 64 × 64 × 512 grid, four hidden layers of width 128, 64 Fourier features, a 24 × 24 × 96 spectral sub-grid, 5,000 epochs, 1,000 warmup epochs, and 1.0 unit of burn-in time. It can run on CPU with `--device cpu`, but is substantially more expensive than the demo. No larger training campaign is started by installation or by the default command.

Available arms:

| Arm | Losses |
| --- | --- |
| `data_only` | Numerical-reference data |
| `base` | Data + PDE residual + initial/boundary conditions |
| `full` | Base + spatial/temporal log-power spectra + variance |
| `cf` | Full + standardised value-distribution characteristic-function loss |
| `no_pde` | Full without PDE residual |
| `no_icbc` | Full without initial/boundary-condition losses |
| `nophys` | Full without PDE/initial/boundary-condition losses |

Use all seven arms and eight seeds to run the full matched design. `paper_results/fresh_matched.csv` contains the 56 historical runs, included as reference data, not as newly reproduced results. Different dependency versions/devices may change training outcomes. `--epochs N` overrides the training budget and caps warmup at `N // 5`; such shortened runs are not the original protocol.

## What this experiment measures

- Each seed fits one fixed forcing realisation. Reference-grid samples supply the data, spectral, and variance targets; this is not training without reference data.
- Evaluation uses the full grid of that same realisation. It is not a held-out-driver or operator-generalisation test.
- The demo deliberately changes grid, network, batch sizes, and training budget. It checks the pipeline and illustrates the metrics; its scores do not reproduce the historical table or guarantee that `full` beats `base`.
- Random spectral sub-grids use sorted sampled grid indices. Their FFTs operate over those index arrays, not a uniformly sampled physical-frequency grid. Full-grid diagnostics are reported separately.
- The corrected reference uses Peaceman–Rachford ADI and converts VG increments to rates exactly once. The legacy reference's clipping to `[-3, 3]` is retained.
- CF compares marginal value distributions; it does not establish spatial phase recovery.

This extract covers the spectral-loss comparison. The later neural Picard experiments, FNO runs, proofs, and Lean formalisation are outside its scope.

## Files and provenance

`solverfixed/` contains the existing numerical solver, model, residuals, loss helpers, and trainer. `run.py` supplies the portable entry point and artifact export. `PROVENANCE.json` lists original source hashes and export hashes. Changes to copied core files are limited to package-relative imports and corrected descriptions; numerical/training calculations were retained.

## Checks

```bash
python -m unittest discover -s tests -v
```

The checks cover an analytic discrete diffusion eigenmode, jump scaling, increment/rate consistency, and an end-to-end CPU run with checkpoint reload and independently recomputed metrics.
