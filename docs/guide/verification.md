# Verification report

This page compares OTAF with reference values it did not produce: published
results for the same mechanism (literature), the results of the thesis the library
was written for, and stored results of earlier versions of the code. Every number
in the tables below comes from `scripts/verification/run_verification.py`. The
script uses the scipy solver only (no surrogate, no GPU) and takes about ten
minutes on a laptop. The raw results are in `docs/guide/verification_results.json`.

## How to read the tables

- **Pf** is the probability of failure of assembly, estimated by Monte Carlo as
  the fraction of samples whose optimal slack is negative (`s < 0`). The 95 %
  interval of OTAF's estimate is the Wilson score interval.
- **z** is the difference to the reference in standard errors: the binomial
  error of an estimate of the reference value with OTAF's sample size, plus the
  published Monte Carlo error of the reference when there is one (Dumas). As a
  rule of thumb, |z| < 2 is agreement and |z| > 3 is a discrepancy.
- **Settings** are given for every case: random seed, sample size, number of
  sides of the polygon that replaces each circular clearance
  (`circle_resolution`, Nd in Dumas' notation) and linearization strategy
  (inscribed, mean or circumscribed polygon, see {doc}`theory`).
- The 2D models (model1, model2) have plane contacts only and no linearization.

## 1. Literature: Dumas (2014)

`model3` re-implements the mechanism of Appendix C of Dumas' PhD thesis
(*appui plan et deux goupilles*: a cover on a base, located by two pins), with
his parametrization by measured points and his three parameter sets (table C.2:
diameter standard deviations 0.06, 0.03 and 0.02 mm). Dumas estimated the
assembly failure probability by Monte Carlo for each set, number of sides Nd
and linearization strategy (tables C.3, 3.3 and C.4). OTAF is run on the same
model with its own samples.

What agrees:

- **Parameter set 1** (Pf ≈ 6 %): with 20 000 samples OTAF gives 5.95 %
  [5.6 %, 6.3 %] against 6.11 % (Dumas, 30 000 samples, ±0.54 %); z = −0.5.
- **Effect of the polygon.** All set-1 rows with 4000 samples use the same sample,
  so their ratio to the Nd = 60 row cancels most of the sampling noise and
  isolates the effect of the linearization. OTAF reproduces Dumas' ratios to
  within 3 % (for example 1.356 against 1.368 for the inscribed square, 0.887
  against 0.885 for the circumscribed one). It also reproduces the ordering
  inscribed > mean > circumscribed and the convergence at large Nd.
- The ordering holds sample by sample, not only on average. When the hole is
  larger than the pin, a larger polygon can only enlarge the feasible set; when
  the pin is larger, every polygon fails. So the inscribed polygon is
  conservative for assembly, as Dumas observed
  (`test_model3_polygon_strategies_give_nested_failures`).

What does not agree, or is not checked:

- **Parameter set 2** (Dumas: Pf = 7.96 × 10⁻⁴): OTAF finds 6 failures in
  20 000 samples, 3.0 × 10⁻⁴ [1.4 × 10⁻⁴, 6.5 × 10⁻⁴]. The published value lies
  outside OTAF's 95 % interval (z = −2.5). Six failures are not enough to
  conclude, and set 1 agrees. Still, a difference that shows only in the tail
  would point to a detail of the model (for example a lever arm in the
  measured-point mapping) rather than to the solver. It is open: settle it with
  at least 2 × 10⁵ samples (about 15 minutes on one core).
- **Parameter set 3** (Pf ≈ 4 × 10⁻⁵) needs about 10⁷ samples and is not rerun.

## 2. Thesis results

The thesis bounds the probability of failure over a credal set of standard
deviations (`scripts/Summary_*.md`). Those values were computed with a neural
surrogate of the slack and a generalized lambda distribution (GLD) fitted to
its output (`scripts/imprecise_bounds_analysis.py`, `scripts/analyze_results.py`).
Here the variance allocation of each bound is recovered from the optimization
results (`OptimizationResults_*.csv`), with the same filtering as
`analyze_results.py`; the recovered allocation reproduces the summary value to
four digits. The true model is then solved at that allocation with the scipy
solver and seed 0. A difference therefore combines the error of the surrogate,
the error of the GLD tail fit and OTAF's Monte Carlo error. Bounds below
Pf = 10⁻³ cannot be checked with a few thousand samples and are listed as not
rerun.

- **Example 1 (model1, 4 DOF)**: the slack has a closed form, equal to OTAF's
  on every sample (`test_model1_slack_is_half_the_analytical_clearance`), so
  10⁶ samples are cheap.
  - The upper bound is reproduced to 5 %: 6.38 × 10⁻³ against 6.68 × 10⁻³.
    With 10⁶ samples even this small difference is significant (z = −3.7); it
    is the error of the surrogate and of the GLD fit, not of the solver.
  - **The lower bound is not.** At the allocation the optimizer returned as the
    minimum, the true Pf is 3.17 × 10⁻³ [3.06, 3.28] × 10⁻³ (closed form, 10⁶
    samples; the LP with 20 000 samples gives 3.6 × 10⁻³). The summary says
    1.65 × 10⁻³, so the surrogate + GLD estimate is low by a factor of about 1.9
    at this point. The true minimum over the credal set is at most
    3.17 × 10⁻³ but is not computed here.
- **Example 2 (model2, 16 DOF)**: the upper bound agrees (1.50 × 10⁻³ against
  1.16 × 10⁻³, z = +1.4). The lower bound, 2.5 × 10⁻⁴, is not rerun.
- **Example 3 (model3, 26 of 30 DOF)**: all four bounds agree (full and reduced
  runs, |z| ≤ 0.7).
- **Example 4 (model4, 50 DOF)**: the upper bound, 1.19 × 10⁻², is just
  outside OTAF's 95 % interval: 8.0 × 10⁻³ [5.7 × 10⁻³, 1.1 × 10⁻²] with 4000
  samples, z = −2.3. Check it with a larger sample before relying on it. The
  lower bounds (5.4 × 10⁻⁷, 0 and 2.0 × 10⁻⁵) are not rerun.

## 3. Regression baselines

`scripts/baselines/` stores the slack of every sample of 15 reference cases
(models 1–4 and `model4_auto`, with their thesis dispersion multipliers and
several face heights of model2), computed by earlier versions of the code. The
current code is compared with the latest stored baseline sample by sample.
Every case is reproduced exactly: no sample changes class and the slack does not
move (max |Δs| = 0). The numerical changes between versions are listed in
`CHANGELOG.md` and in the notes of each baseline.

## Tables

```{include} _verification_tables.md
```

## Tests behind this report

`tests/test_verification.py` keeps this page honest without rerunning the full
study:

- the tables above are regenerated from the saved results and must match the
  page;
- the reference values in the results are those of Dumas' tables and of
  `scripts/Summary_*.md`;
- every baseline case is reported as reproduced exactly, and the model1
  closed-form counts follow from the saved allocations;
- cheap reruns (marked `slow`): model3 against Dumas' set 1 with 2000 samples,
  the nesting of the three polygons, the upper thesis bounds of model3 and
  model4 with 1000 samples, and the first 200 samples of four baseline cases.

Rerun everything, for example after a change of the numerics:

```bash
python scripts/verification/run_verification.py              # all groups
python scripts/verification/run_verification.py --only dumas # one group
python scripts/verification/run_verification.py --render-only
```
