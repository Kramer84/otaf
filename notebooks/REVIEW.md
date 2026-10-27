# Review of the reference notebooks

Checks of the notebooks that serve as references for the library (October 2026).
For each one: what was checked, how, and what was found. "Fixed" items are
changed in the notebook; the others are left to the author because they change
the reference results or are modelling choices.

## `Ex1_Pocket/Ex1_2D_4dof_Analytical_Ref.ipynb`

**The model is correct.** The sign conventions of the defects (translations and
rotations of the two facing planes, female frame rotated by 180°) were checked by
hand. The clearance is the smaller of the two face-end clearances; rigid motions of the
male part do not change it to first order, so it is the exact condition. The
automatic `model1` matches it (`s = j / 2`, `tests/test_physical_invariants.py`).

Open point:

- **The "max sigma" baseline is outside the capability constraint.** Both
  `sigma_u_max = t / 6` and `sigma_gamma_max = t / (6 y_max)` are used at once, so the
  standard deviation of the deviation at a face end is `sqrt(2) t / 6`: an
  effective capability of `1 / sqrt(2) ≈ 0.71`, not 1. The text calls this point
  "the boundary of the capability constraint". With the credal set of the optimisation
  notebook (`squared_sum=True`, `λ_u² + λ_γ² = 1`), every point has end
  deviations of standard deviation `t / 6`, and Pf is about an order of magnitude below
  the ≈ 5 % obtained here. The ≈ 5 % is a valid figure for this distribution, but it
  is not a point of the credal set.

## `Ex1_Pocket/Ex1_2D_4dof_Opti_Manual.ipynb`

**The model is correct.** The interface loops end with a pure rotation (`J4`), which
does not change the translation they measure. The automatic builder used to end with
the gap's nominal transform instead, which also contains a translation; that was the
lever-arm bug fixed in Phase 1.

Open points:

- Failed solves (`NaN`) are replaced by the worst slack of the batch,
  `np.nan_to_num(s_vals, nan=np.nanmax(s_vals)) * -1`. If every other sample passes,
  a failed solve is counted as a pass. Count failed solves explicitly instead.
- The λ sample is generated twice in cell 15; the first one is unused (harmless).

## `AnalyticalModels/Analytical_2D_Reference_Assembly_16DOF.ipynb`

**The model is correct.** The notebook writes 7 equalities on the 8 face-end
clearances and only tests feasibility. Row 5 forbids relative rotation of the parts,
which matches the sliding planar contact `e` of `model2`. To compare,
`scripts/reference_checks/analytical_16dof_vs_model2.py` solves the same equalities for
the largest common clearance, which is the slack OTAF maximises. It then compares this
slack with `model2` built with `hF = 10` (all faces 10 mm high, as in the notebook),
on independent samples of the same distribution:

| | analytical LP | OTAF `model2(hF=10)` |
|---|---|---|
| nominal slack | 0.100000 | 0.100000 |
| Pf (12 000 samples each) | 5.03 % | 5.07 % |
| slack quantiles 1 / 5 / 50 / 95 % | −0.0201 / −0.0001 / 0.0456 / 0.0868 | −0.0198 / −0.0003 / 0.0460 / 0.0878 |

The difference is within Monte Carlo noise: two-sample Kolmogorov–Smirnov p = 0.13.

Open points:

- **The λ parametrisation does not stay on the capability boundary.** It uses
  `e_pos = λ N(0, σ_pos)` and `e_ori = (1 − λ) N(0, σ_θ)`. With
  `(h / 2) σ_θ = σ_pos`, the standard deviation at a face end is
  `σ_pos sqrt(λ² + (1 − λ)²)`. That is `σ_pos` only for λ = 0 or 1, and falls to
  `σ_pos / √2` at λ = 0.5. The explored set therefore lies inside the capability
  constraint used by `model2.eval_credal_set_constraints`
  (`sqrt(σ_u² + (h/2)² σ_γ²) = target`) and by the Ex1 notebook (`squared_sum=True`).
  The upper Pf bound is then underestimated. With `sqrt(λ)` and `sqrt(1 − λ)` as
  factors, every point would lie on the boundary.
- The parameters differ from the defaults of `model2`: `t = 0.1`, `Cm = 0.3` and all
  faces 10 mm high, whereas `model2` uses `tol = 0.16`, `capa = 1` and female faces
  10.2 mm high. Compare with `model2` only with matching parameters.
- `method = "highs-ds"` is unused (`milp` has no method argument).
- Fixed: passing `defect_sample` to `get_failure_prob_MC` raised a `NameError`.
  The sample was never assigned.

## `OptimizationBasedModels/ManualImplementation4Pins2plates.ipynb`

**The model is correct.** Its equations are those of `example_models.model4`. The
automatic pipeline now builds the same assembly from an input dictionary
(`example_models.model4_auto`), and the two give the same slack sample by sample,
diameters included, for the circumscribed, inscribed and mean polygons
(`tests/test_cylinder_interfaces.py`). The cylinder constraint was also checked by
hand: the pin axis displacement at each hole end, projected on N directions, is at
most `rf (D_hole − D_pin) / 2`.

Fixed:

- The notebook failed at its third cell because it called
  `otaf.example_models.models_3_D.defect_vars`, `.get_kinematic_loop_expressions` and
  similar names, which now live in `otaf.example_models.model4`.
  Model construction (cells 1–8) runs again.
- The comment on the plate constraints said the plate corner is at
  `(x = 120, y = 50)`; it is at 50 mm from the plate centre along x and y, which is
  what the code uses.
- In `model4`, the holes of the lower plate were evaluated at ±l2/2 instead of
  their own ±l3/2. The default l2 = l3 hid it.

Open points:

- **The maximal standard deviation of the plate rotations violates the plate's own
  capability constraint.** `sigma_e_theta_plate = 2 tol / (EH + LB) / 6` gives
  `50 σ_θ = 0.22 tol` at the corner where `cons0`/`cons1` evaluate, against the
  target `tol / 6 ≈ 0.167 tol`. There is no effect on the 32-dimensional study,
  where plate deviations are zero, but there is one on any 50-dimensional use.
  `model4.get_distribution_params` has the same values.
- `NonlinearConstraint(lb=-1e-5, ub=1e-5)` makes the capability constraints
  equalities: every hole is exactly at the capability target. This is consistent
  with a credal set defined as "capability equal to the target", but then the
  minimum of Pf is also taken on that boundary. Check that this is intended.
- `np.random.seed(SEED)` does not seed OpenTURNS, so the training and test samples
  are not reproducible. Use `ot.RandomGenerator.SetSeed`.

## `OptimizationBasedModels/3D_Model_32DOF_Auto_GLD_EX2.ipynb`

The automatic version of the same assembly, which was infeasible at nominal (stored
output: "The problem is infeasible", inconsistent equalities). There were two causes:

- **Library bug, fixed:** the nominal transform of a cylinder–cylinder gap dropped
  the axial offset between the hole and pin origins. The pins are blocked axially,
  so the loops could not close.
- **Modelling, fixed in cell 9:** every pin was blocked in rotation about its axis in
  both plates, which also forbade any relative rotation of the plates about the pin
  axes. That rotation is only limited by the radial clearances. The pins are now
  blocked in the lower plate only, as in `model4`.

The nominal assembly is now feasible (the equalities are consistent: rank 24 = 24).
The stale outputs of cells 13–18 were cleared. The surrogate and imprecise-probability
cells were not run (torch).
