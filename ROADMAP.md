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

- [x] Golden-master snapshots of the constraint matrices of models 1–4 (`tests/data/matrix_snapshots/`, regenerate with `OTAF_UPDATE_SNAPSHOTS=1`).
- [x] Order-independent comparison helper (`tests/matrix_utils.py`): matches columns by variable name, rows up to permutation.
- [x] **Model construction made independent of the Python hash seed.** Interface rows were ordered by `set` iteration, and in `model2` the direction kept for a gap used both ways (`GP1bB0P2bB0` / `GP2bB0P1bB0`) was a coin flip per process (30/60 seeds). One direction gave lever arms −5.2/+5 instead of ±5: at ×1.21 dispersion Pf 14.65 % vs 14.40 % (5/2000 samples differ). The choice is now deterministic and keeps the consistent (±5) variant.
- [x] Removed duplicated plane interaction loops (each was added twice and hidden by a `set`).
- [x] Analytical oracle for `model1`: SOC slack equals half the analytical clearance (`s = j/2`, to 1e-7) on every sample. Mutation check: catches sign, lever-arm and nominal-size errors.
- [x] Invariants on all models: nominal assembly feasible with positive margin; failures persist when the dispersion grows (follows from LP concavity); distribution labels match solver variables (models 1, 2, 4).
- [ ] **Project the overlap of facing planes instead of pairing points by proximity.** `SurfaceInteractionManager.is_point_facing` pairs points within 0.25 mm. With unequal faces (model2: heights 10 vs 10.2) one gap direction then places a contact at y = 10.2, outside the overlap [0, 10] (lever arm 5.2). With aligned points both directions agree to 1e-16, so the loop algebra itself is consistent. Also: faces differing by more than 0.25 mm silently lose the contact. Remove the strict xfail once fixed.
- [ ] Hand-coded baselines: map `ManualImplementation4Pins2plates.ipynb` and the 16-DOF analytical LP notebook onto the automatic models' variables and compare with `matrix_utils` (needs row scaling support).
- [ ] `model3` samples a reduced parameter space mapped by position (`sample_multiplier`); add explicit labels so the mapping can be checked by name.
- [ ] Remaining invariants: invariance to part/surface relabelling; 2D models embedded in 3D give the same result; LP vs MILP agreement.

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
- [ ] Make solver failures explicit: today a single failed `milp` in a batch returns `x=None` and crashes the whole batch with `ValueError: setting an array element with a sequence` (`uncertainty/_reliability.py`). Surface status codes per sample instead.
- [ ] `PointsNotOnPlaneError(...)` is instantiated but never raised in `_surfaceInteractionManager.process_plane_plane_interaction`, so non-planar point sets pass silently.
- [ ] `compute_gap_optimizations_on_sample_batch` defaults to `float32` output; the sign of `s` near 0 deserves `float64`.

## Phase 5: Release and validation

- [ ] Semantic versioning, `CHANGELOG.md`, Zenodo DOI per release (`CITATION.cff` exists).
- [ ] Verification report against literature cases (e.g. Dumas 2014) and the thesis results.
