# OTAF codebase map

Read this before exploring the tree. Update it in the same commit whenever a
module is added, moved, renamed, or changes responsibility.

## Pipeline (input dict → failure probability)

```
system_data (dict or otaf.schema.SystemData)
  └─ AssemblyDataProcessor            validate (otaf.schema), generate points, expand loops
       ├─ CompatibilityLoopHandling   compatibility loops → first-order equalities
       │    └─ FirstOrderMatrixExpansion, DeviationMatrix/GapMatrix/TransformationMatrix
       └─ InterfaceLoopHandling       contact (non-interpenetration) inequalities
            ├─ SurfaceInteractionManager   which points face which (plane–plane, cyl–cyl)
            └─ InterfaceLoopBuilder        builds the inequality expressions
  └─ SystemOfConstraintsAssemblyModel SymPy expressions → A_eq/A_ub/K matrices,
                                      LP (linprog) / MILP (milp) feasibility
  └─ otaf.uncertainty                 Monte Carlo / FORM / subset sampling on the
                                      slack variable s (s < 0 ⇒ assembly fails)
```

## Packages (`src/otaf/`)

| Path | Role |
|---|---|
| `__init__.py` | Eager-imports `constants`, `exceptions`, `common`, `geometry` and `__version__`; everything else is lazy (`__getattr__`). Re-exports the `_assembly_modeling` classes. Adds a `NullHandler` to the `otaf` logger: library code reports through `logging` (module loggers), not `print`. |
| `_version.py` | `__version__`, the single source of the version (read by `pyproject.toml` and `docs/conf.py`; `CITATION.cff` and `CHANGELOG.md` are checked against it by `tests/test_version.py`). |
| `_assembly_modeling/` | Core modelling engine (pipeline above). Private package; public names are re-exported from `otaf`. |
| `_assembly_modeling/_interface_constraints/` | Contact constraints. `InterfaceLoopHandling` warns about `INTERACTIONS` contacts that no loop crosses (`uncovered_contacts`). `SurfaceInteractionManager` pairs contact points at the overlap of two surfaces (`contactGapNames`); `InterfaceLoopBuilder` writes constraints at those pairs only. Cylinder pairs: radial clearance from `RADIUS`, plus diameter deviations `d_d_<k>` for cylinders with `DIAMETER_DEVIATION` (`InterfaceLoopBuilder.radial_clearance`), polygon chosen by `geometry.LinearizationStrategy`, 64 sides by default (`circle_resolution`). |
| `_assembly_modeling/_assembly_topology.py` | Assembly graph helpers (networkx). |
| `schema/` | pydantic models of the input dictionary (`SystemData`, `Part`, `Surface`, `ExtentLocal`, `Loops`) and `validate_system_data`, called by `AssemblyDataProcessor`. Validation only: the processor keeps working on its copy of the dict. Maps pydantic errors to `otaf.exceptions` (location-prefixed), else `SystemDataValidationError`. Lazy-loaded. |
| `constants/` | Regex patterns for labels (`P1aA0`, gap/deviation names), global-constraint presets (`3D`, `2D_NZ`, …), `SUPPORTED_SURFACE_TYPES`, and the named numerical tolerances and gap bounds (`FRAME_ATOL`, `GEOMETRIC_ATOL`, `DIRECTION_ATOL`, `GAP_TRANSLATION_BOUND`, …). |
| `common/` | Generic helpers: SymPy coefficient extraction, matrix-string parsing, nested dicts, tqdm/notebook utilities, `set_seed` (seeds NumPy, OpenTURNS and PyTorch at once). |
| `geometry/` | Points, planes, lines, cylinders, transforms; `_meshes.py` uses trimesh (optional). |
| `exceptions/` | Custom exceptions and warnings (`UncoveredContactWarning`, `SolverFailureWarning`), `_raise_missing_dependency`, timeout decorator. |
| `tolerances/` | Tolerance zone → defect standard deviation conversions (`sigma_delta_*`). |
| `distribution/` | Building OpenTURNS distributions for defect vectors. |
| `sampling/` | Low-discrepancy sequences, permutations, hyper-plane/surface samplers. |
| `uncertainty/` | Reliability: failure probability (FORM, NAIS, subset), batched gap optimisation on samples (`milp_batch_sequential`, `compute_gap_optimizations_on_sample_batch`: NaN rows and per-sample status for failed solves, `return_status=True`). |
| `optimization/` | Storage/tracking of optimisation history and credal-set constraints. |
| `capabilities/` | Process capability helpers. |
| `sensitivity/` | Sobol experiment construction and plotting. |
| `surrogate/` | **torch** neural surrogates for imprecise-probability studies. Heavy; never imported by core code. |
| `plotting/` | Matplotlib/trimesh plots (deviation domains, p-boxes, topology). |
| `example_models/` | Reference models as functions: `model1` 2D 4-DOF, `model2` 2D 16-DOF, `model3` 3D 30-DOF, `model4` 3D 50-DOF. `model3`/`model4` hand-code their constraint matrices (`model3` samples measured points, mapped to the solver variables by `sample_multiplier`, named by `sample_multiplier_labels()`); `model4_auto` (`models_3_D/_model_3D_pins_auto.py`) is the same assembly as `model4` from an input dictionary, checked against it in `tests/test_cylinder_interfaces.py`. Each model also provides its distribution (`get_distribution_params`) and the capability constraints of the thesis credal set (`eval_credal_set_constraints`, `get_scaled_credal_set_constraints_function`). |

## Outside `src/`

| Path | Role |
|---|---|
| `tests/` | pytest suite. Markers: `slow` (model construction, small MC), `gpu` (torch). CI runs `-m "not gpu"`. Docstring examples in `src/otaf` run as doctests (`--doctest-modules` in the pytest config; `surrogate/` is ignored because it imports torch). `conftest.py` caches built example models per session. |
| `tests/test_input_schema.py`, `tests/test_solver_failures.py` | Input validation (one test per Phase 3 finding) and per-sample solver failures. |
| `tests/matrix_utils.py` | Layout-independent comparison of constraint systems (columns by variable name, rows up to permutation). |
| `tests/data/matrix_snapshots/` | Golden-master matrices of models 1–4; regenerate with `OTAF_UPDATE_SNAPSHOTS=1 pytest tests/test_matrix_snapshots.py`. |
| `scripts/` | Thesis study scripts, which produced every figure of the thesis (training surrogates, imprecise bounds, Pf plots, result analysis), with their CSV/PNG/MD outputs. They take `--seed`; most import torch. |
| `scripts/verification/` | Verification report: `verification_cases.py` (Dumas 2014 published values, recovery of the thesis bounds from `scripts/OptimizationResults_*.csv`, reruns with the scipy solver, baseline comparison, rendering of the tables), `run_verification.py` (writes `docs/guide/verification_results.json` and `_verification_tables.md`; `--only`, `--render-only`). |
| `scripts/baselines/` | Reference cases (models 1–4, model2 with varied face heights, model4_auto) and their per-sample slack, saved as named baselines in `data/<tag>.{npz,json}` by `make_baseline.py`; used to measure how much a code change moves results. |
| `notebooks/` | Worked examples; `Ex1_Pocket/*Analytical_Ref*` and `AnalyticalModels/Analytical_2D_Reference_Assembly_16DOF.ipynb` are analytical references for the 4- and 16-DOF models (the latter also as `scripts/reference_checks/analytical_16dof_vs_model2.py`); `OptimizationBasedModels/ManualImplementation4Pins2plates.ipynb` is the manual pin/plate model; `Baselines/Compare_To_Baseline.ipynb` compares the current code with a saved baseline. `notebooks/REVIEW.md` records the checks of the reference notebooks. |
| `docs/` | Sphinx; API pages are regenerated by `.github/workflows/deploy_docs.yml`. |
| `docs/guide/` | Hand-written MyST pages: `input_format.md` (every key of the assembly dictionary, two worked examples), `theory.md` (loops, interface conditions, slack, circle linearization), `limitations.md` (validity, supported contacts, polygon error), `verification.md` (comparison with Dumas 2014, the thesis and the baselines; its tables are the generated `_verification_tables.md`, excluded from the toctree and included). Linked from `docs/index.rst`. |
| `tests/test_verification.py` | Keeps the verification page, its saved results and their sources (Dumas' tables, `scripts/Summary_*.md`, baselines) in step; cheap reruns marked `slow`. |
| `tests/test_version.py` | Version single source vs installed metadata, `CITATION.cff` and `CHANGELOG.md`. |
| `CHANGELOG.md` | Keep a Changelog; behaviour changes of each release in their own section. |
| `tests/test_docs_examples.py` | Runs the ```python blocks of `docs/guide/input_format.md` and checks them against `model1` / `model4_auto`; update both together. |
| `autotest.toml` | Config of the LLM test generator; its output (`_tests/`) is gitignored and not part of CI. |
| `ROADMAP.md` | Robustness roadmap (phases 0–5). |
