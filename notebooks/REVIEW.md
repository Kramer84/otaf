# Review of the reference notebooks

Checks of the notebooks that serve as references for the library (October 2026), against
the revised thesis manuscript (Ch. 4, *Application of the Proposed Methodology*, and the
4-pin appendix). The thesis figures were produced by the scripts in `scripts/`
(`train_neural_networks.py`, `imprecise_bounds_analysis.py`, `analyze_results.py`; results in
`scripts/Summary_*.md`), which take the distributions and credal sets from
`otaf.example_models`; the notebooks are the step-by-step versions. For each notebook: what was checked, how, what was changed, and what is
left to the author. Outputs of cells whose code changed were cleared; the former outputs
are in the git history (commit `7314be8`).

## Shared conventions (thesis, Ch. 4)

- Centred, independent normal defects; capability $C_{q,p} = 1$; one tolerance per example,
  chosen so that $P_f = 5\,\%$ when every component is at its largest standard deviation
  ($\lambda = 1$, the "max sigma" point, also used to train the surrogates).
- Normalized parametrization: $\sigma_i = \lambda_i\,\sigma_{i,\max}$, $\lambda \in [0, 1]^n$,
  where $\sigma_{i,\max}$ is the standard deviation component $i$ may have alone at the
  capability target. The credal set is implicit: the capability constraint of each feature.
- Optimizations at constant capability: the capability constraint holds on every feature.
  The thesis relaxes it into a half-space whose side depends on the direction of the
  optimization ($\geq 0$ when minimizing $P_f$, $\leq 0$ when maximizing).
- Failure thresholds $\alpha_k \in \{0, 0.025, 0.05, 0.075\}$.

Every random draw can now be seeded: `otaf.common.set_seed(seed)` seeds NumPy, OpenTURNS and
PyTorch (`np.random.seed` alone does not seed OpenTURNS samples), and the functions that
sample take a `seed` argument (`NeuralRegressorNetwork`, `UniformSurfaceSampler`, the Sobol
experiment, `generate_correlated_samples`). The notebooks use them.

## Ex1: `Ex1_Pocket/`

**The models are correct** (sign conventions checked by hand; the automatic `model1` matches
the analytical solution, `tests/test_physical_invariants.py`). The thesis values are used:
$L_1 = 99.8$, $L_2 = 100$, $h = 10$, $t = 0.31$; at $\lambda = 1$, $P_f \approx 5\,\%$.

- `Ex1_2D_4dof_Analytical_Ref`: the "max sigma" point ($\lambda = 1$) is outside the credal
  set by construction; it is the calibration point of the tolerance, as in the thesis.
- `Ex1_2D_4dof_Opti_Manual`, fixed:
  - failed solves (`NaN`) were replaced by the worst slack of the batch, so they could count
    as passes; they now count as failures and are reported;
  - the $\lambda$ sample was drawn twice and unseeded; it is drawn once, seeded;
  - `ot.ComposedDistribution` (removed from OpenTURNS) → `ot.JointDistribution`;
    `otaf.uncertainty.find_best_worst_quantile` → `otaf.sampling.find_best_worst_quantile`.
- `Ex1_2D_4dof_Opti_Auto_Surrogate`, fixed: renamed functions of `model1`
  (`get_distribution_params`, `get_scaled_credal_set_constraints_function`) and
  `otaf.common.scaling`; seeds (training sample, surrogate, $P_f$ sample).

## Ex2 (16 components)

### `AnalyticalModels/Analytical_2D_Reference_Assembly_16DOF.ipynb`: rebuilt

The former notebook used the deprecated $(\lambda, 1-\lambda)$ parametrization, other
parameters ($t = 0.1$, $C_m = 0.3$) and one height for all faces. It is rebuilt with the
thesis geometry ($h_M = 10$, $h_F = 10.2$, $L_1 = 30$, $L_2 = 70$, $l_M = 10$, $l_F = 10.2$),
$t = 0.16$, the normalized $\lambda$ and the thesis credal set. Each face rotates about its own
centre and the clearances are taken at the ends of the overlap of the faces, as in OTAF.

The 7 equations leave one degree of freedom (the male part sliding along $x$), so the slack
has a closed form: no linear program per sample, $2 \times 10^5$ samples in 0.3 s. Checks:

| | |
|---|---|
| closed form vs linear program | max $|\Delta s| = 10^{-16}$ |
| vs `model2` (6000 samples each, $\lambda = 1$) | $P_f$ 4.88 % vs 4.72 %, KS $p = 0.39$ |
| $P_f$ at $\lambda = 1$ ($2 \times 10^5$ samples) | 4.81 %: $t = 0.16$ satisfies the 5 % rule |

Exploration of the credal set (8 angles: share of each face's variance given to the
translation), $\alpha_k = 0$, bounds re-estimated on an independent $10^6$ sample:

| | thesis (surrogate + GLD) | analytical reference |
|---|---|---|
| $\underline{P_f}$ | $2.54 \times 10^{-4}$ | $3.55 \times 10^{-4}$, pure translation |
| $\overline{P_f}$ | $1.16 \times 10^{-3}$ | $1.25 \times 10^{-3}$, mixed allocation (pure rotation: $1.16 \times 10^{-3}$) |

The thesis result (`scripts/Summary_model2_16_dof.md`: `model2` defaults, $t = 0.16$,
half-space constraints) is reproduced in location (minimum at pure translation, maximum at a
mixed allocation) and within 8 % for the upper bound; the lower bound differs by 40 % (GLD tail
fit against empirical fraction). `scripts/reference_checks/analytical_16dof_vs_model2.py` runs the
same comparison without Jupyter.

### `OptimizationBasedModels/2D_Model_16DOF_Auto_GLD*.ipynb`: ported

Both notebooks used the MI-SDOF statistical constraint classes removed in `2587e56`
(`MiSdofToleranceZones`, `FeatureLevelStatisticalConstraint`, ...), $t = 0.1\sqrt{2}$
($P_f = 2.0\,\%$ at $\lambda = 1$, not 5 %) and, without the surrogate, $\lambda$ scaling the
variances. Now:

- distribution from `model2.get_distribution_params(tol=0.16)` (5.0 % at $\lambda = 1$);
- credal set from `model2.get_scaled_credal_set_constraints_function`, relaxed into the
  thesis half-spaces, both optimizations starting on its boundary;
- $\lambda$ scales the standard deviations;
- seeds (MLP notebook).

Not run here (the surrogate needs torch; the direct model is too slow for this PC). The
stored results came from the former constraint, so they were cleared; they were not the thesis
figures, which come from `scripts/imprecise_bounds_analysis.py`.

## Ex3 (`model3`, 2 pins)

Re-implementation of the example of Dumas (2014) with its own parametrization (measured
points). Kept hand-coded, as decided.

## Ex4 (4 pins)

### `OptimizationBasedModels/ManualImplementation4Pins2plates.ipynb`

**The model is correct**: its equations are those of the thesis appendix (pins blocked
axially and in rotation in the lower plate) and of `model4`. The automatic pipeline builds
the same assembly (`model4_auto`) and gives the same slack sample by sample
(`tests/test_cylinder_interfaces.py`).

Plate rotation. The capability constraints `cons0`/`cons1` evaluate each plate at its corner,
50 mm from its centre along $x$ and $y$, so an optimization never leaves the credal set.
What was inconsistent was the normalization: $\sigma_{\theta,\max}$ was
$t / (6 C_{q,p}) / 37.5$ instead of the thesis' $t / (6 C_{q,p}) / 50$
($1/x_{\max}$, eq. *max_std_expressions*). A plate rotation alone could then only reach
$\lambda = 0.75$, and the $\lambda = 1$ point (tolerance calibration, surrogate training) had
plate rotations 1.33 times the thesis maximum. Fixed in the notebook and in
`model4.get_distribution_params`. Effect (baseline `plate-capability` vs `phase2-cylinders`):
`model4` $P_f$ 6.05 % → 5.40 %, `model4_x1.15` 16.35 % → 14.90 %; every other case unchanged
bit for bit. No effect on the 32-dimensional study (plates perfect).

The thesis 50-dimensional run (`scripts/Summary_model4_50_dof.md`) used the former
normalization. Its $P_f$ bounds stay valid: the credal set is the same set of standard
deviations in both normalizations, and both $[0, 1]$ boxes contain it. Only the reported
$\lambda$ of the plate rotations (the first two of each plate) are in the former unit:
$\lambda_{\text{thesis}} = \tfrac{4}{3}\,\lambda_{\text{former}}$ (e.g. 0.5049 → 0.673). A rerun of the
scripts now also trains the surrogate on a slightly different sample (plate rotations at
$\lambda = 1$ are 25 % smaller).

Also fixed: seeds (`np.random.seed` did not seed the OpenTURNS samples; the surrogate now
takes `seed`); the binary-classifier cells, whose module was removed in `2587e56`, were
deleted. The stored outputs (the thesis 32-dimensional run) were kept: they predate
the plate normalization and the seeds, which do not change the 32-dimensional model but do
change the 50-dimensional training sample of its surrogate.

### `OptimizationBasedModels/3D_Model_32DOF_Auto_GLD_EX2.ipynb`: ported

The automatic version of the 32-dimensional study. Besides the two fixes of Phase 2
(cylinder gaps, pins blocked in the lower plate only), it is aligned with the thesis:
$t = 0.21$ (the value of `model4` and of the thesis scripts), distribution and credal set from `model4_auto` (new
`eval_credal_set_constraints`: worst standard deviation at the hole ends,
$z = \pm h_{plate}/2$, as `model4`), half-space constraints, $\lambda$ scaling the standard
deviations, the four $\alpha_k$. The MI-SDOF cells and a basinhopping draft were removed.
Checked on a reduced sample (600 points): symbols identical to `model4_auto`, boundary
start point $\lambda = 1/\sqrt{2}$, $P_f(\lambda = 1) \approx 3\,\%$.

## Ex4 notebooks and the thesis scripts

The Ex4 notebooks now use the scripts' values ($t = 0.21$, half-space constraints, start on
the credal-set boundary).
