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
- [x] Hand-coded references checked against the automatic models (`notebooks/REVIEW.md`): the four-pin model (`model4`, from `ManualImplementation4Pins2plates.ipynb`) sample by sample (`tests/test_cylinder_interfaces.py`); the 16-DOF analytical reference against `model2` in distribution (same Pf within Monte Carlo error).
- [x] Reference notebooks aligned with the revised thesis (Ch. 4): thesis parameters and normalized parametrization, credal sets from the model functions instead of the removed MI-SDOF classes, directional half-space constraints, seeds; stale API references fixed (all `otaf.*` names in the notebooks resolve). The 16-DOF analytical notebook is rebuilt (closed-form slack) and reproduces the thesis bounds of Ex2 (`notebooks/REVIEW.md`).
- [x] `model4` plate rotations normalized as in the thesis (`σ_max = target / 50`, the plate corner where the capability constraint is evaluated; was `/ 37.5`). Baseline `plate-capability`: `model4` Pf 6.05 % → 5.40 %, other cases unchanged.
- [ ] `model3` samples a reduced parameter space mapped by position (`sample_multiplier`); add explicit labels so the mapping can be checked by name.
- [ ] Remaining invariants: invariance to part/surface relabelling; 2D models embedded in 3D give the same result; LP vs MILP agreement.

## Phase 2: Cylinder–cylinder interfaces (top priority; core purpose of the library)

The automatic path now reproduces the hand-coded `model4` (two plates, four pins) sample by sample, for the three linearization strategies (`tests/test_cylinder_interfaces.py`, `example_models.model4_auto`).

| | Automatic (now) | Hand-coded (`model3`, `model4`) |
|---|---|---|
| Clearance | `RADIUS`, plus diameter deviations `d_d_<k>` when `DIAMETER_DEVIATION` is true | diameter variables: `rf/2 · (d_hole − d_pin)` |
| Polygon linearization | `InterfaceLoopHandling(linearization=LinearizationStrategy.…)` | inscribed / mean / circumscribed factor `rf` (same enum) |
| Contact sections | ends of the axial overlap (`SurfaceInteractionManager`) | both hole ends, explicit |

- [x] Axial overlap pairing reviewed and tested (`tests/test_surface_interactions.py`): same and opposed axes, either cylinder longer, staggered, no overlap (raises), overlap ends on existing points, origins at cylinder ends.
- [x] Mixed interactions (plane–cylinder) raise `NotImplementedError` instead of being silently ignored.
- [x] **Root cause of the failing automatic pin models:** the nominal transform of a cylinder–cylinder gap dropped the axial offset between the two surface origins (`GAP_TYPE_TO_NULLIFIED_NOMINAL_COMPONENTS`). With axial sliding blocked, the loops could not close and the nominal assembly was infeasible (`3D_Model_32DOF_Auto_GLD_EX2.ipynb`). The axial offset is now kept.
- [x] Diameter-deviation variables (`"DIAMETER_DEVIATION": True`), named `d_d_<k>` with the feature index of the cylinder's deviation matrix (pins, which have none, get new indices). Deviation variables may now appear only in interface constraints.
- [x] `LinearizationStrategy` in `otaf.geometry`, used by the automatic builder and by `model3`/`model4` (their separate enums made `model4` silently ignore a strategy passed from elsewhere).
- [x] Differential test: automatic vs hand-coded `model4` (max |Δs| ≈ 1e-8 over 60 samples, diameters included).
- [x] `model4` evaluated the lower-plate holes at ±l2/2 instead of ±l3/2 (no effect with the default l2 = l3).
- `model3` (30 DOF) stays hand-coded: it re-implements Dumas (2014) with its own parametrization (measured points), which the automatic pipeline does not use.
- [ ] Plane–cylinder contacts (later; they raise `NotImplementedError` meanwhile).

## Phase 3: Documentation

- [x] Docstrings in numpy style (the last Google-style parameter lists are converted), exception classes documented, ruff `D` rules (numpy convention) enabled on `src/` (tests exempt).
- [x] Docstring examples run as doctests on `src/otaf` (pytest config and CI; the torch-based `surrogate` package is not collected). Fixed examples that were wrong, not only failing: the 50th percentile of the positive values `[2, 3, 4, 5]` is 3.5, not 3.0. One example (`InterfaceLoopBuilder.get_interface_equations_facing_cylinders`) needs a fully built model and is kept as a skipped illustration; the model itself is exercised by `tests/test_cylinder_interfaces.py`.
- [x] Input-format reference page (`docs/guide/input_format.md`), written by hand from the validation code and `otaf.constants`, with a 2D example (`model1`) and a pin/hole example (`model4_auto`, one pin). `tests/test_docs_examples.py` runs both examples and checks them against the example models. To be regenerated from the pydantic schema in Phase 4.
- [x] Theory page (`docs/guide/theory.md`) and "Validity and limitations" page (`docs/guide/limitations.md`).

Found while writing the pages (docs only in this phase; fixed in Phase 4 unless noted, tests in `tests/test_input_schema.py`):

- [x] A contact listed in `INTERACTIONS` that no compatibility loop crosses got no contact condition, silently (removing loop `L1` of `model1` drops the two conditions of the right wall). Now `InterfaceLoopHandling` warns (`UncoveredContactWarning`; `uncovered_contacts()` lists them). The example models cover every contact.
- [x] `CONSTRAINTS_G` entries that differ on the two sides of a pair (including one side left to the default) gave a bare `KeyError: 'plane-plane-floating-sliding'`. Now `InconsistentContactConstraintsError`, raised by the schema for `INTERACTIONS` pairs and by `CompatibilityLoopHandling` for any loop gap. Entries are compared case-insensitively.
- [x] `MissingSurfaceTypeKeyError` was called with part and surface swapped. Now raised by the schema with the right order.
- [x] `FRAME` was validated only when the points were generated from `ORIGIN`. Now required and checked (rotation matrix, `otaf.constants.FRAME_ATOL`) for every surface.
- [x] `TYPE` accepted `cone` and `sphere`, which failed later with a `KeyError`. Now `UnsupportedSurfaceTypeError` up front (`otaf.constants.SUPPORTED_SURFACE_TYPES`).
- [x] Gap bounds (±3 mm, ±π/4) were literals in `get_gap_symbol_bounds`; now `otaf.constants.GAP_TRANSLATION_BOUND` / `GAP_ROTATION_BOUND` (same values; not yet a parameter).
- [ ] The default `circle_resolution` of `InterfaceLoopHandling` is 8 (up to 8 % error on the clearance); the pin models use 64.

## Phase 4: API hardening

- [x] **pydantic models for the input dictionary** (`otaf.schema`: `SystemData`, `Part`, `Surface`, `ExtentLocal`, `Loops`, `validate_system_data`). `AssemblyDataProcessor` validates up front: types, labels, frames (rotation matrices), required keys per surface (`RADIUS`/`EXTENT_LOCAL` for cylinders in an interaction, `ORIGIN`/`INTERACTIONS` without `POINTS`), one-entry `CONSTRAINTS_D`/`CONSTRAINTS_G` lists with known values, `BLOCK_*_G` axes, interaction and loop references, same-type contacts, consistent contact constraints. The existing `otaf.exceptions` are raised where they apply, prefixed by the location; other problems raise `SystemDataValidationError` listing all of them. Plain dicts are still accepted and not converted (validation only, results unchanged); a `SystemData` can be passed too. JSON schema: `SystemData.model_json_schema()`. `docs/guide/input_format.md` follows the models.
- [x] Replace `print` with `logging` (module loggers, `NullHandler` on `otaf`). `verbose=True` of the FORM/NAIS/subset functions now logs at INFO; `silence_prints` of `InterfaceLoopBuilder` is kept but has no effect (DEBUG messages). Kept as `print`: the two display helpers of `model4` (`print_kinematic_loops`, `print_interface_constraints`) and doctest examples.
- [x] Every random draw can be seeded: `otaf.common.set_seed` (NumPy, OpenTURNS, PyTorch) and `seed` arguments where the library samples.
- [x] Named numerical tolerances instead of literals: `otaf.constants` (`FRAME_ATOL`, `GEOMETRIC_ATOL`, `DIRECTION_ATOL`, `ZERO_NORM_ATOL`, `TAYLOR_COEFFICIENT_TOL`, gap bounds) used by geometry, contact pairing and the Taylor expansion; module constants for the mesh and hyper-surface samplers. Values unchanged.
- [x] Make solver failures explicit: a single failed `milp` in a batch returned `x=None` and crashed the whole batch with `ValueError: setting an array element with a sequence`. `milp_batch_sequential` and `compute_gap_optimizations_on_sample_batch` now give NaN rows for samples without a solution, warn (`SolverFailureWarning`, with the count per status) and return the status of every sample with `return_status=True` (`otaf.uncertainty.MILP_STATUS_MESSAGES`). Successful samples are unchanged (`tests/test_solver_failures.py`).
- [x] `PointsNotOnPlaneError` is now raised for non-coplanar face points.
- [ ] `compute_gap_optimizations_on_sample_batch` defaults to `float32` output; the sign of `s` near 0 deserves `float64`.

## Phase 5: Release and validation

- [ ] Semantic versioning, `CHANGELOG.md`, Zenodo DOI per release (`CITATION.cff` exists).
- [ ] Verification report against literature cases (e.g. Dumas 2014) and the thesis results.

## Phase 6: CAD to input dictionary

Goal: build the assembly dictionary from a CAD assembly (STEP first) instead of by hand. Depends on the Phase 4 schema (the converter's target and its JSON schema) and on plane–cylinder contacts only if the target assemblies need them.

How well the current format fits (reviewed against `docs/guide/input_format.md`):

- **Fits CAD:** the geometric keys of a surface (`TYPE`, `FRAME`, `ORIGIN`, `POINTS`/`CONTOUR_GLOBAL`, `RADIUS`, `EXTENT_LOCAL`) are what a B-rep gives for planar and cylindrical faces, in global coordinates. Extra keys are tolerated (checked: `META`, `CAD_ID` do not disturb the build), and surface labels continue past `z` (`aa`, `ab`, …).
- **Does not come from CAD:** `CONSTRAINTS_D` (needs GD&T/PMI, rarely present in STEP), `CONSTRAINTS_G` and `BLOCK_*_G` (contact kinematics, only partly in assembly mates), `DIAMETER_DEVIATION`, and above all `LOOPS` (a modelling choice: independent closed chains that cover every contact; an uncovered contact is silently dropped today). The dictionary also holds no tolerance values; distributions live in per-model functions, so a CAD import has nothing to fill there yet.
- **Under-specified for a generator:** the sign convention of the face normal (the code only needs facing planes to be anti-parallel; `model1` is not outward-normal consistently), units (mm assumed, not stated), the mapping between CAD names and labels (no traceability key defined), partial cylinders (a hole split in two half-faces has no angular extent), non-convex faces and faces with holes (only the convex hull of the overlap is used).
- Conclusion: usable as the geometric target, but not as the only interface. A converter needs an intermediate description and a user-in-the-loop step for the intent.

Items:

- [ ] Decide and document the conventions a generator must follow: normal direction, units, CAD-id traceability key (`META`/`CAD_ID`), label scheme. Put them in the Phase 4 schema.
- [ ] Neutral assembly description (parts, features with geometry and CAD ids, contact pairs, tolerance table) with a JSON schema; the dictionary is built from it.
- [ ] Automatic loop generation from the contact graph (`_assembly_topology.py` already builds a networkx graph): independent cycles that cross every contact, with a warning for contacts that cannot be covered. Useful for hand-written inputs too.
- [ ] STEP reader for planar and cylindrical faces (OpenCascade binding as an optional dependency): merge split cylinders, ignore other surface types with a report, sample contours.
- [ ] Contact detection (coincident or near faces within a tolerance) with a report the user confirms; derive `INTERACTIONS` and `SURFACE_DIRECTION`.
- [ ] Evaluate QIF (ISO 23952:2020, XML; free schemas at qifstandards.org) as the source of the tolerance side: it models features and characteristics (ASME GD&T and ISO GPS tolerances) and is aimed at model-based definition, so it is a candidate for `CONSTRAINTS_D` and for the tolerance values the dictionary lacks, and later for returning results. Open questions to settle with a spike on a sample file before committing to it: how much assembly structure (mates, contacts) a QIF file carries compared with STEP AP242, and whether its feature and characteristic identifiers can be matched to the B-rep faces.
- [ ] Tolerance input: a table, QIF or STEP PMI mapped to `CONSTRAINTS_D` and to the defect distributions; user-supplied `CONSTRAINTS_G`/`BLOCK_*_G` defaults per contact type.
- [ ] Acceptance tests by round trip: CAD of the `model1` pocket and of the two-plate/four-pin assembly must give dictionaries equivalent to `model1` and `model4_auto` (same constraint system by `tests/matrix_utils.py`).
- Out of scope at first: sections of 3D CAD into 2D models, non-planar/non-cylindrical contacts, assemblies where the contact topology changes with the deviations.
