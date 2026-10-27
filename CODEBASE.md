# OTAF codebase map

Read this before exploring the tree. Update it in the same commit whenever a
module is added, moved, renamed, or changes responsibility.

## Pipeline (input dict → failure probability)

```
system_data (dict)
  └─ AssemblyDataProcessor            parse/validate parts, surfaces, points, loops
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
| `__init__.py` | Eager-imports `constants`, `exceptions`, `common`, `geometry`; everything else is lazy (`__getattr__`). Re-exports the `_assembly_modeling` classes. |
| `_assembly_modeling/` | Core modelling engine (pipeline above). Private package; public names are re-exported from `otaf`. |
| `_assembly_modeling/_interface_constraints/` | Contact constraints. `SurfaceInteractionManager` pairs contact points at the overlap of two surfaces (`contactGapNames`); `InterfaceLoopBuilder` writes constraints at those pairs only. Cylinder pairs: radial clearance from `RADIUS`, plus diameter deviations `d_d_<k>` for cylinders with `DIAMETER_DEVIATION` (`InterfaceLoopBuilder.radial_clearance`), polygon chosen by `geometry.LinearizationStrategy`. |
| `_assembly_modeling/_assembly_topology.py` | Assembly graph helpers (networkx). |
| `constants/` | Regex patterns for labels (`P1aA0`, gap/deviation names) and global-constraint presets (`3D`, `2D_NZ`, …). |
| `common/` | Generic helpers: SymPy coefficient extraction, matrix-string parsing, nested dicts, tqdm/notebook utilities, `set_seed` (seeds NumPy, OpenTURNS and PyTorch at once). |
| `geometry/` | Points, planes, lines, cylinders, transforms; `_meshes.py` uses trimesh (optional). |
| `exceptions/` | Custom exceptions, `_raise_missing_dependency`, timeout decorator. |
| `tolerances/` | Tolerance zone → defect standard deviation conversions (`sigma_delta_*`). |
| `distribution/` | Building OpenTURNS distributions for defect vectors. |
| `sampling/` | Low-discrepancy sequences, permutations, hyper-plane/surface samplers. |
| `uncertainty/` | Reliability: failure probability (FORM, NAIS, subset), batched gap optimisation on samples (`milp_batch_sequential`). |
| `optimization/` | Storage/tracking of optimisation history and credal-set constraints. |
| `capabilities/` | Process capability helpers. |
| `sensitivity/` | Sobol experiment construction and plotting. |
| `surrogate/` | **torch** neural surrogates for imprecise-probability studies. Heavy; never imported by core code. |
| `plotting/` | Matplotlib/trimesh plots (deviation domains, p-boxes, topology). |
| `example_models/` | Reference models as functions: `model1` 2D 4-DOF, `model2` 2D 16-DOF, `model3` 3D 30-DOF, `model4` 3D 50-DOF. `model3`/`model4` hand-code their constraint matrices; `model4_auto` (`models_3_D/_model_3D_pins_auto.py`) is the same assembly as `model4` from an input dictionary, checked against it in `tests/test_cylinder_interfaces.py`. Each model also provides its distribution (`get_distribution_params`) and the capability constraints of the thesis credal set (`eval_credal_set_constraints`, `get_scaled_credal_set_constraints_function`). |

## Outside `src/`

| Path | Role |
|---|---|
| `tests/` | pytest suite. Markers: `slow` (model construction, small MC), `gpu` (torch). CI runs `-m "not gpu"`. `conftest.py` caches built example models per session. |
| `tests/matrix_utils.py` | Layout-independent comparison of constraint systems (columns by variable name, rows up to permutation). |
| `tests/data/matrix_snapshots/` | Golden-master matrices of models 1–4; regenerate with `OTAF_UPDATE_SNAPSHOTS=1 pytest tests/test_matrix_snapshots.py`. |
| `scripts/` | Thesis study scripts, which produced every figure of the thesis (training surrogates, imprecise bounds, Pf plots, result analysis), with their CSV/PNG/MD outputs. They take `--seed`; most import torch. |
| `scripts/baselines/` | Reference cases (models 1–4, model2 with varied face heights, model4_auto) and their per-sample slack, saved as named baselines in `data/<tag>.{npz,json}` by `make_baseline.py`; used to measure how much a code change moves results. |
| `notebooks/` | Worked examples; `Ex1_Pocket/*Analytical_Ref*` and `AnalyticalModels/Analytical_2D_Reference_Assembly_16DOF.ipynb` are analytical references for the 4- and 16-DOF models (the latter also as `scripts/reference_checks/analytical_16dof_vs_model2.py`); `OptimizationBasedModels/ManualImplementation4Pins2plates.ipynb` is the manual pin/plate model; `Baselines/Compare_To_Baseline.ipynb` compares the current code with a saved baseline. `notebooks/REVIEW.md` records the checks of the reference notebooks. |
| `docs/` | Sphinx; API pages are regenerated by `.github/workflows/deploy_docs.yml`. |
| `autotest.toml` | Config of the LLM test generator; its output (`_tests/`) is gitignored and not part of CI. |
| `ROADMAP.md` | Robustness roadmap (phases 0–5). |
