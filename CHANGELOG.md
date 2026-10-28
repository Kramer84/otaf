# Changelog

All notable changes to OTAF are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor
version (0.x.0) may break the API or change numerical results; such changes are
listed under "Behaviour changes" first.

The version is defined once in `src/otaf/_version.py` (`otaf.__version__`);
`CITATION.cff` and the top entry below must match it (`tests/test_version.py`).

## [Unreleased]

## [0.3.0] - 2026-10-28

Robustness release: the work of `ROADMAP.md` phases 0 to 4 (pull requests #5 to
#12) and the verification report of phase 5. Results of the thesis examples are
reproduced by the current code (`docs/guide/verification.md`).

### Behaviour changes (read before upgrading)

Input dictionaries:

- **The input dictionary is validated before anything is built** (`otaf.schema`,
  #11). Dictionaries that used to be accepted, or failed later with a bare
  `KeyError`, now raise an `otaf.exceptions` error prefixed with the location of
  the problem, or `SystemDataValidationError` listing every problem found.
  Plain dicts are still accepted and are not converted.
- **`FRAME` is required on every surface** and must be a rotation matrix
  (orthonormal, determinant +1, to `otaf.constants.FRAME_ATOL`). It used to be
  checked only when the points were generated from `ORIGIN`.
- **`CONSTRAINTS_D` and `CONSTRAINTS_G` are one-entry lists** of a known value,
  e.g. `["PERFECT"]`, `["SLIDING"]`. A bare string, an empty or a longer list is
  rejected.
- **`CONSTRAINTS_G` must be the same on both surfaces of a contact**, including
  a side left to the default: `InconsistentContactConstraintsError` (was
  `KeyError: 'plane-plane-floating-sliding'`). Entries are compared
  case-insensitively.
- `TYPE` accepts only `plane` and `cylinder` (`otaf.constants.SUPPORTED_SURFACE_TYPES`);
  `cone` and `sphere` raise `UnsupportedSurfaceTypeError` up front.
- Cylinders in an interaction need `RADIUS` and `EXTENT_LOCAL`; a surface
  without `POINTS` needs `ORIGIN` and `INTERACTIONS`.
- Plane–cylinder interactions raise `NotImplementedError` (they were silently
  ignored, #7).
- Face points that are not coplanar raise `PointsNotOnPlaneError` (the error
  existed but was never raised, #6).
- A contact listed in `INTERACTIONS` that no compatibility loop crosses gets no
  contact condition, as before, but now emits `UncoveredContactWarning`
  (`InterfaceLoopHandling.uncovered_contacts()` lists them, #11).

Results:

- **Contact points are placed at the overlap of facing surfaces** (#6). The
  0.25 mm proximity pairing is gone: planes are intersected (segments in 2D,
  convex hulls in 3D) and points are paired at the overlap vertices; cylinders
  use the ends of their axial overlap. Contacts were lost before when facing
  faces differed in height by more than 0.25 mm (model2 with 0.3 mm taller
  female faces: Pf 5.35 % instead of 14.15 %; 1 mm taller: every solve
  unbounded).
- **Interface loops no longer end with `JCor⁻¹`** (#6). It shifted the lever
  arms by the in-plane offset between the two surface origins and made results
  depend on the gap direction. model2 at ×1.21 dispersion: Pf 14.40 % → 14.20 %
  (|Δs| ≤ 0.0026). Models 1, 3 and 4 are unchanged.
- **Model construction no longer depends on the Python hash seed** (#6).
  Interface rows were ordered by `set` iteration, and in model2 the direction
  kept for a gap used both ways was a coin flip per process (Pf 14.65 % or
  14.40 % at ×1.21).
- **Automatic pin/hole assemblies are feasible at nominal** (#7): the nominal
  transform of a cylinder–cylinder gap dropped the axial offset between the two
  surface origins, so loops through a pin blocked axially could not close.
- `model4`: the largest standard deviation of a plate rotation is the target
  divided by 50 mm (thesis normalization) instead of 37.5 mm: Pf 6.05 % → 5.40 %
  at the default tolerance (#7).
- `model4` evaluated the lower-plate holes at ±l2/2 instead of ±l3/2 (no effect
  with the default l2 = l3, #7).

Solvers and output:

- **A failed solve no longer aborts a batch** (#11). `milp_batch_sequential` and
  `compute_gap_optimizations_on_sample_batch` return a row of NaN for each
  sample without a solution, emit `SolverFailureWarning` (count per status) and
  return the status of every sample with `return_status=True`. Note that
  `NaN < 0` is `False`: count failed solves explicitly when estimating Pf.
- **`logging` instead of `print`** (#11). Library modules log through module
  loggers under `otaf` (with a `NullHandler`). `verbose=True` of the FORM, NAIS
  and subset functions now logs at INFO: configure `logging` to see it.
  `silence_prints` of `InterfaceLoopBuilder` is kept but has no effect (its
  messages are DEBUG). The two display helpers of `model4` still print.
- `compute_gap_optimizations_on_sample_batch` still defaults to `float32`
  output, the dtype of the torch surrogates trained on it; pass
  `dtype="float64"` for comparisons at the 1e-8 level (#12, documented, not
  changed).

### Added

- `otaf.__version__`, single-sourced from `src/otaf/_version.py`; this changelog.
- `otaf.schema`: pydantic models of the input dictionary (`SystemData`, `Part`,
  `Surface`, `ExtentLocal`, `Loops`), `validate_system_data`, JSON schema via
  `SystemData.model_json_schema()` (#11).
- Exceptions and warnings: `SystemDataValidationError`,
  `InconsistentContactConstraintsError`, `UnsupportedSurfaceTypeError`,
  `UncoveredContactWarning`, `SolverFailureWarning`;
  `otaf.uncertainty.MILP_STATUS_MESSAGES` (#11).
- Named tolerances and gap bounds in `otaf.constants` (`FRAME_ATOL`,
  `GEOMETRIC_ATOL`, `DIRECTION_ATOL`, `ZERO_NORM_ATOL`,
  `TAYLOR_COEFFICIENT_TOL`, `GAP_TRANSLATION_BOUND`, `GAP_ROTATION_BOUND`);
  values unchanged (#11).
- Cylinder diameter deviations: `"DIAMETER_DEVIATION": True` adds variables
  `d_d_<k>` to the radial clearance (#7).
- `otaf.geometry.LinearizationStrategy` (inscribed, mean, circumscribed
  polygon), taken by `InterfaceLoopHandling(linearization=...)` and by
  `model3`/`model4`, whose own enums it replaces (#7).
- `otaf.geometry.planar_overlap_vertices` (#6).
- `example_models.model4_auto`: the two-plate/four-pin assembly of `model4`
  built from an input dictionary, equal to `model4` sample by sample (#7).
  Credal-set functions for `model4_auto`; `model2` credal-set functions
  restored (`eval_credal_set_constraints` had been lost, the scaled variants
  raised `NameError`, #5).
- `otaf.common.set_seed` (NumPy, OpenTURNS and PyTorch at once) and `seed`
  arguments wherever the library samples (#7).
- User guide (`docs/guide/`): input format, theory, validity and limitations
  (#9), and a verification report: model3 against Dumas (2014), the thesis
  bounds against the scipy solver, the regression baselines
  (`scripts/verification/`).
- `model3.sample_multiplier_labels()`: names of the rows and columns of
  `sample_multiplier` (measured points to feature defects).
- Tests: golden-master snapshots of the constraint matrices of models 1–4,
  physical invariants, an analytical oracle for `model1`, cylinder interfaces,
  input validation, solver failures, docs examples; invariance to relabelling
  and loop order, LP vs MILP agreement, nesting of the polygon strategies;
  doctests of `src/otaf`; ruff and pytest (Python 3.10 and 3.12) in CI (#5 to
  #11).
- `scripts/baselines/`: reference cases and their per-sample slack, to measure
  how much a change moves results (#6).

### Changed

- Docstrings in NumPy style throughout, checked by ruff (`D` rules, #9).
- ruff also checks unused imports and variables, flake8-bugbear and whitespace.
  Mutable or computed default arguments became `None` or tuples (same values),
  e.g. `TransformationMatrix(initial=..., final=...)`, the lengths `L` of
  `model3`/`model4`, `heatmap(cbar_kw=...)`; exceptions raised while handling
  an `ImportError` now chain it (`raise ... from err`).
- Reference notebooks aligned with the thesis (parameters, normalized
  credal-set parametrization, seeds); thesis scripts take `--seed` (#7, #8).
- Before the roadmap work (2025-01 to 2026-06, not recorded change by change):
  explicit `__all__` in every subpackage instead of wildcard imports, plotting
  split into submodules, Python ≥ 3.10.

### Removed

- The (λ, 1 − λ) mixing of the credal set is deprecated in the notebooks; the
  credal set is the capability constraint per feature in the normalized space
  (#7).
- Before the roadmap work: score-function gradient estimators
  (`normal_score_mu`, `monte_carlo_non_compliancy_rate_gradient`, ...) and the
  unused basin-hopping optimizer.

### Fixed

- `__all__` of `model3`/`model4` missed commas, so names were silently
  concatenated (#5).
- Duplicated plane interaction loops (each was added twice, #6).
- `MissingSurfaceTypeKeyError` received part and surface swapped (#11).
- Wrong docstring examples (the median of `[2, 3, 4, 5]` is 3.5, #9).
- Documentation deployment (`linkify-it-py` missing, #10).

## [0.2.0] - never released

`0.2` was the version string of `main` from 2025-08 to 2026-10. It was never
tagged or archived on Zenodo; its changes are included in 0.3.0.

## [0.0.1-alpha] - 2025-01-13

First public snapshot, archived on Zenodo (DOI
[10.5281/zenodo.14639555](https://doi.org/10.5281/zenodo.14639555)).

[Unreleased]: https://github.com/Kramer84/otaf/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Kramer84/otaf/compare/v0.0.1-alpha...v0.3.0
[0.0.1-alpha]: https://github.com/Kramer84/otaf/releases/tag/v0.0.1-alpha
