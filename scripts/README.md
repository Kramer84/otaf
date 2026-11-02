# Study scripts

Run from the repository root with the project environment (`.venv/bin/python scripts/<name>.py ...`). All scripts take `--seed`; use `--help` for the full options.

## `artifacts/`

Intermediate, regenerable files: the trained surrogates (`model*_surrogate.pth`) and the exported Pf grids (`*.npz`). Everything in it is git-ignored except its own `.gitignore`, so the folder exists on a fresh clone but is empty. The figures, CSVs and summaries of the thesis stay in this folder, next to the scripts.

## Pipeline

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `train_neural_networks.py` | model definitions | `artifacts/<model>_surrogate.pth` |
| 2 | `imprecise_bounds_analysis.py` | `artifacts/` surrogates (`--surrogate-dir`) | `OptimizationResults_<model>[_reduced].csv` (`--output-dir`) |
| 3 | `analyze_results.py` | the CSVs (`--input-dir`) | `Summary_<model>.md`, `pbox_<model>.png` (`--output-dir`) |
| – | `plot_pf_2d_4dof_model.py` | `artifacts/model1_4_dof_surrogate.pth` | `Pf_Grid_alpha_<alpha>.png` (`--output-dir`) |
| – | `optimize_tolerance_values_for_target_failure.py` | see `--help` | see `--help` |

## Pf grid figures (`plot_pf_2d_4dof_model.py`)

Evaluating the grid is the slow part (one GLD fit per grid point); choosing the colour scale and the red threshold is not. Export the grids once, then re-plot them with any plotting option:

```bash
# evaluate and export (bare name -> scripts/artifacts/pf_model1.npz)
python scripts/plot_pf_2d_4dof_model.py --alphas 0.0 0.01 0.025 0.05 0.075 \
    --grid-res 20 --mc-size 100000 --save-grids pf_model1

# re-plot without touching the surrogate; try other highlight settings
python scripts/plot_pf_2d_4dof_model.py --load-grids pf_model1 \
    --highlight-mode value --highlight-pct 0.005 --output-dir /tmp/pf_test
```

The GLD fit and CDF run on the GPU through `gldpy-gpu` (`--gld-backend cpu` for the NumPy/SciPy version; they agree to about 1e-5 relative, the GPU one is about 100x faster at 500k samples). `--alphas` with `--load-grids` selects among the stored alphas. The file also stores `grid_res`, `seed`, `mc_size` and the surrogate path. The figures are written to `--output-dir` (default: the current directory), so re-plotting with the default overwrites the tracked `Pf_Grid_alpha_*.png`.

### How the tracked `Pf_Grid_alpha_*.png` were (most likely) generated

The launch command was not recorded; this was inferred from the figures and a recomputation with `model1_4_dof_surrogate.pth`.

- Colour scale: `viridis`, 200 levels, from the smallest positive Pf (floored at 1e-5) to the largest Pf of each grid, so each alpha has its own scale. The bin spacing (`--norm-type`) cannot be read from the figures; `log` is the default.
- Red region: value mode, i.e. Pf >= P_max - pct * (P_max - P_min), with pct between 0.005 and 0.01 (it depends on the grid resolution: about 0.005 at 20x20, about 0.008 at 40x40). It is not a quantile.
- The exact grid resolution and Monte Carlo size of the original run are unknown. Re-plot from an exported grid to settle it.

### Exported grids (in `artifacts/`, not tracked)

Both were produced with the GPU backend, seed 0, model 1 surrogate, alphas 0.0 0.01 0.025 0.05 0.075:

| File | Grid | MC samples | Run time |
|---|---|---|---|
| `pf_model1_40x40_mc500k.npz` | 40x40 | 500 000 | 3.6 min |
| `pf_model1_50x50_mc1M.npz` | 50x50 | 1 000 000 | 7.2 min |

Their Pf ranges agree to within about 5 % at every alpha, so the grids are converged. The red-threshold percentage has not yet been scanned on them (open item: find the `--highlight-pct` that reproduces the tracked PNGs).

## GPU setup notes

- The GLD part uses `gldpy-gpu>=0.3.0` (import name `gldpy_gpu`). The scripts that do not use the GPU import its NumPy-compatible `GLD`.
- A GTX 10xx (compute capability 6.1) needs a torch build for CUDA 12.6 (`https://download.pytorch.org/whl/cu126`); the default wheel has no kernels for it and fails with `CUBLAS_STATUS_ARCH_MISMATCH`. Add the index in a local, uncommitted `[tool.uv.sources]` block of `pyproject.toml`.
- `uv sync` without `--inexact` removes extras that are not listed (pytest, ruff, ...); use `uv sync --inexact --extra surrogate --extra test --extra dev --extra viz`.
