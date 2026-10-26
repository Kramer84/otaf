# OTAF robustness roadmap

Goal: make OTAF correct, tested and documented enough for research use, and a
credible base for industrial studies (with clearly stated limits).

Compute constraint: CI and small machines run only light work: model
construction and Monte Carlo runs of at most a few thousand samples through the
scipy LP/MILP solvers. Anything using torch or a GPU (surrogates,
imprecise-probability sweeps) is marked `@pytest.mark.gpu` and runs elsewhere.

## Phase 0: Hygiene and safety net

- [x] Fix `__all__` missing commas in `model3`/`model4` (names were silently concatenated).
- [x] Restore `model2` credal-set functions (`eval_credal_set_constraints` had been lost; the other two always raised `NameError`). Regression tests added.
- [x] Align supported Python version (≥ 3.10) between README and `pyproject.toml`.
- [x] `CODEBASE.md` module map.
- [x] ruff in CI with a bug-catching rule set (`E9, F63, F7, F82, F811`); widen later.
- [x] pytest markers `slow` / `gpu`; CI runs `-m "not gpu"` on Python 3.10 and 3.12 with coverage report.
- [ ] Widen ruff rules gradually: `F401/F841` (unused imports/variables), `B` (mutable defaults: 7 sites), `W` (whitespace).

## Phase 1: Characterization tests (before changing any numerics)

- [ ] Golden-master snapshots: save `A_eq_Def, A_eq_Gap, K_eq, A_ub_Def, A_ub_Gap, K_ub` of models 1–4 to `tests/data/*.npz`; assert equality on each run.
- [ ] Analytical oracle: compare SOC failure probability of `model1` with the analytical reference (`notebooks/Ex1_Pocket/*Analytical_Ref*`) within Monte Carlo confidence bounds (≤ few thousand samples, fixed seed).
- [ ] Hand-coded baselines: use the manual models (`model3`/`model4`, `ManualImplementation4Pins2plates.ipynb`) as references. Their variable and constraint ordering differs from the automatic pipeline, so first write a mapping helper that matches matrices **up to row/column permutation and row scaling** (compare by variable label, not index).
- [ ] Physical-invariant tests: zero deviation ⇒ feasible (`s ≥ 0`); Pf non-decreasing in tolerance; invariance to part/surface relabelling; 2D models embedded in 3D give the same result.
- [ ] LP vs MILP agreement on the same samples.

## Phase 2: Cylinder–cylinder interfaces (top priority; core purpose of the library)

Known differences between the automatic path (`_interfaceLoopBuilder._cylinder_interaction_approximation`) and the hand-coded models:

| | Automatic | Hand-coded (`model3`) |
|---|---|---|
| Clearance | fixed `RADIUS` from input | diameter defects are variables: `rf/2 · (d_hole − d_pin)` |
| Polygon linearization | none | inscribed / mean / circumscribed factor `rf` |
| Contact sections | from fragile overlap cases in `_surfaceInteractionManager` (incl. a `NotImplementedError` branch) | both cylinder ends, explicit |

- [ ] Failing differential test: one pin/hole joint, automatic vs hand-coded matrices (via the Phase 1 mapping helper).
- [ ] Add diameter-defect variables for cylinders (ties into "intrinsic dimension variability").
- [ ] Add `LinearizationStrategy` to the automatic builder.
- [ ] Rewrite axial-overlap logic as explicit, unit-tested cases (no overlap / partial / contained / coincident ends).
- [ ] Acceptance: automatically generated `model3` (and `model4`) reproduces the hand-coded matrices and Pf.

## Phase 3: Documentation

- [ ] Normalise docstrings to numpy style (25 Google-style param lines, mostly `_surfaceInteractionManager.py`); document the 21 exception classes; enable ruff `D` rules.
- [ ] Run docstring examples as doctests on `src/`.
- [ ] Input-format reference page (all keys, `TYPE`, `CONSTRAINTS_D/G`, loop syntax), generated from the pydantic schema (Phase 4).
- [ ] Theory pages (SOC, slack variable, linearization) and a "Validity and limitations" page.

## Phase 4: API hardening

- [ ] **pydantic models for the input dictionary** (`SystemData`, `Part`, `Surface`, `Loops`, …): validate types, labels, frames (orthonormal), required keys per surface type (e.g. `RADIUS` for cylinders) up front with clear errors; keep accepting plain dicts.
- [ ] Replace `print` (54 calls) with `logging`.
- [ ] Explicit RNG seeds everywhere; named numerical tolerances instead of literals.
- [ ] Make solver failures explicit (status codes surfaced, not silently treated as failure).

## Phase 5: Release and validation

- [ ] Semantic versioning, `CHANGELOG.md`, Zenodo DOI per release (`CITATION.cff` exists).
- [ ] Verification report against literature cases (e.g. Dumas 2014) and the thesis results.
