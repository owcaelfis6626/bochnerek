# Validation — 2026-10-16

- `python -m unittest discover -s tests -v`: **4 tests passed**. Includes four short training arms (`data_only`, `base`, `full`, `cf`), checkpoint reload, independent recomputation of variance ratio/L2, and refusal to overwrite an existing run.
- The full default CPU demo completed: two 100-epoch runs, generated reference data, checkpoints, field arrays, JSON metrics, and a PNG. The ZIP was then extracted into a temporary directory and the same default demo completed from outside the project directory, with `PYTHONPATH` removed.
- Original and exported core-file hashes match `PROVENANCE.json`. AST comparisons confirmed unchanged bodies for all 33 exported numerical/model/configuration/training functions and classes. Original research files were not modified.
- The installed versions satisfy all four dependency ranges: Python 3.14.7, PyTorch 2.13.0+cu130 (CPU execution), NumPy 1.26.4, SciPy 1.17.1, Matplotlib 3.10.9.

The portable demo's total training time was 3.55 seconds on this host, excluding imports, reference generation, evaluation, and plotting. Its seed-0 variance ratios were 0.0020 (`base`) and 0.8683 (`full`); relative L2 errors were 0.9994 and 1.3096. These are small-demo diagnostics, not the historical 56-run results.

Dependencies were supplied by the existing Python environment. A fresh dependency download/install, CUDA execution, and a full `paper`-profile campaign were not run. Results from the portable check are saved locally under `outputs/portable-validation/`; generated outputs are excluded from Git and the distributable ZIP.
