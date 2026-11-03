# CLAUDE.md

OTAF (Open Tolerance Analysis Framework): Python library (`src/otaf`, ≥ 3.10) for statistical tolerance analysis of overconstrained 3D mechanical assemblies. Input dict → SymPy constraints → LP/MILP feasibility → failure probability (Monte Carlo/FORM/subset). Stack: numpy, scipy, sympy, openturns, pydantic, beartype; torch only for surrogates. Also holds the thesis study scripts (`scripts/`). Module map: `CODEBASE.md`; plan: `ROADMAP.md`.

## Commands

```bash
uv sync --extra dev                  # or: pip install -e .[test]   (venv: .venv)
pytest -m "not gpu"                  # what CI runs; includes doctests of src/otaf (see pyproject)
pytest -m "not slow" tests/<file>    # quick subset
ruff check src tests                 # CI lint (bug rules + numpy docstrings on src)
.venv/bin/python scripts/<name>.py --help   # study scripts, run from repo root
```

Snapshots: `OTAF_UPDATE_SNAPSHOTS=1 pytest tests/test_matrix_snapshots.py` regenerates `tests/data/matrix_snapshots/` (only when a numerics change is intended).

## Working rules

- Stay inside this folder. Read `CODEBASE.md` before exploring the tree; update it in the same commit when a module is added, moved or changes role.
- Facts vs hypotheses: state values with their source file; label guesses. If docs and files disagree, the files win and the doc gets fixed.
- Library code logs through `logging`, not `print`. Do not import `torch` from core code (`surrogate/` only).
- Heavy work (torch, GPU, big Monte Carlo) is `@pytest.mark.gpu` / `slow` and stays out of CI.
- Commit only when asked.

## Gotchas

- `pyproject.toml` has an uncommitted LOCAL block (`[tool.uv.sources]` torch → cu126 index, for the GTX 1070). Never commit it; stage `pyproject.toml` with `git add -p`.
- Do not edit generated outputs by hand: `scripts/OptimizationResults_*.csv`, `Summary_*.md`, `pbox_*.png`, `Pf_Grid_*.png`; `scripts/artifacts/` (surrogates, Pf grids) is git-ignored and regenerable. `scripts/old_results_may2026/` and `scripts/rerun_logs/` are untracked data.
- `uv.lock`, `_tests/`, `_tests.tmp/`, `*.npz`/`*.npy` are git-ignored (exceptions in `.gitignore`).
- `autotest.toml` configures an LLM test generator using the Mistral API; its output is not part of CI.

Last validated against the real files: 2026-11-04 (commands copied from `pyproject.toml` and `.github/workflows/pytest.yaml`, not re-run).
