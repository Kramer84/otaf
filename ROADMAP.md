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
- [x] **Contact points at the overlap of facing surfaces, not by proximity.** The 0.25 mm pairing radius (`is_point_facing`) is removed. Planes now intersect the two faces (segments in 2D, convex hulls in 3D: `otaf.geometry.planar_overlap_vertices`) and pair points at the overlap vertices. Cylinders use the same helper logic on their axial extents. In both cases an existing point at a vertex is reused (e.g. the origin) and new points are created otherwise. Before the fix, model2 with female faces 0.3 mm taller lost 4 of 8 contacts (Pf 5.35 % instead of 14.15 %), and 1 mm taller lost all of them (every solve unbounded).
- [x] **Interface loops no longer end with `JCor⁻¹`.** It re-applied the in-plane offset between the two surface origins after the gap rotation, so lever arms were off by that offset (model2: 0.1 mm) and depended on the gap direction. Both directions now give identical slack at every face height. Effect on model2 (×1.21): Pf 14.40 % → 14.20 %, |Δs| ≤ 0.0026. Models 1, 3 and 4 are unchanged.
- [x] Baselines and comparison notebook: `scripts/baselines/` (`pre-fix`, `post-overlap-fix`) and `notebooks/Baselines/Compare_To_Baseline.ipynb`.
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

- [x] Axial overlap pairing reviewed and tested (`tests/test_surface_interactions.py`). It is correct for same and opposed axes, with either cylinder longer and for staggered overlaps; the opposed-axes branch is complete despite its "I have begun here" comment.
- [x] An overlap end that coincides with an existing point (e.g. the cylinder origin) is now paired, reusing that point.
- [x] Origins may lie at a cylinder end (or a face end). When the reference pair of the loops is itself a contact, its constraint is now written too (`InterfaceLoopBuilder._contact_partners`).
- [x] Mixed interactions (plane–cylinder) raise `NotImplementedError` instead of being silently ignored.
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
- [x] `PointsNotOnPlaneError` is now raised for non-coplanar face points.
- [ ] `compute_gap_optimizations_on_sample_batch` defaults to `float32` output; the sign of `s` near 0 deserves `float64`.

## Phase 5: Release and validation

- [ ] Semantic versioning, `CHANGELOG.md`, Zenodo DOI per release (`CITATION.cff` exists).
- [ ] Verification report against literature cases (e.g. Dumas 2014) and the thesis results.
