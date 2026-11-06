# Can a neural surrogate learn the true slack of Example 1?

Date: 2026-11-06. Code: `/home/kramer84/otaf`, branch `thesis-final-rerun`, uncommitted. **The thesis must not be changed from this work without the author's decision.**

## Answer

**Yes, on the credal set, and at the current network size.** A ReLU network with hidden layers 16-8-4 (257 parameters) learns s = (0.2 - (u4+u5) - 5|g4-g5|)/2, including the kink, pure translation and pure rotation.

| Slack RMSE (mm), median of seeds | Current recipe (3 seeds) | Recommended (5 seeds) |
|---|---|---|
| Over the credal set | 9.1e-4 | 9.2e-7 |
| Near the kink, 5\|g4-g5\| < 0.002 | 2.9e-3 | 2.8e-6 |
| Near the limit state, \|s\| < 0.02 | 3.8e-4 | 6.7e-7 |
| Pure translation (TT) | 3.4e-3 | 1.5e-6 |
| Pure rotation (RR) | 8.3e-4 | 3.0e-7 |
| Equal share (EE) | 9.4e-4 | 1.3e-6 |

That is 570x to 2800x lower, depending on the row. The recommended configuration is ReLU 16-8-4, trained on 200,000 points drawn from a mixture over the credal set, with MSE loss and 40,000 steps with cosine learning-rate decay. It uses no closed-form knowledge.

Three limits apply:

- **Close, not exact.** A hand-built 3-neuron ReLU network reaches RMSE 4e-8 mm, so the exact form is representable, but trained networks stay above it in most cases. Isolated errors remain: the largest error with s in [-0.02, 0.095] was 5.0e-4 mm over the 5 seeds, and at most 5.4e-4 of samples exceed 1e-4 mm at TT, RR, EE or over the credal set.
- **Valid on the credal set only.** With lambda = 1 for all variables and stds x1.35, which is off the credal set, RMSE rises to 2.0e-4 mm (median; worst seed 2.0e-3). In a box of +-4.5 sigma it is 2.4e-3 mm (worst seed 2.7e-2). See "Validity outside the credal set".
- **GELU does not learn the kink, whatever the budget tested.** At 1M points, near-kink RMSE stays about 2e-4 mm, around 100x the ReLU value at 1M (1.9e-6 mm). Moving GELU from 200k to 5M points did not improve it. With credal sampling its TT probability can still look right (+0.04% at 64-32-16) because training places points exactly on g4 = g5 = 0. The probability looks right while the slack function is still rounded.

## Scope change

The author narrowed the goal to the geometric question while the runs were in progress. The queue was stopped at 03:51 on 2026-11-06. See "Run status" for what was cut. No stage below is presented as complete unless it finished.

## Method

- **Labels:** the closed form. It was checked against the LP on 1500 samples in each of TT, RR, EE and five training domains; the largest difference was 1.1e-16 (`logs/check_lp.log`).
- **Test sample:** 5e6 standard normals from a fixed stream (seed 20261106), independent of all training seeds and never used for training or for choosing weights. "Overall" gives each sample its own allocation, uniform on [0,1]^2. Allocation-specific slack metrics use the first 2e6 points.
- **Selection:** weights and the single map's seed (lowest validation RMSE of the five) were chosen on validation MSE. The configurations themselves are compared on the test sample, which is the point of the study, so the recommendation has seen it. The gaps it rests on are far larger than the noise.
- **Common recipe, except the baseline:** AdamW, no weight decay, learning rate 2e-3 cosine to 1e-6, batch 4096, 40,000 steps, shuffled batches, PyTorch default initialisation, no weight clamp, 60/40 split.
- **Baseline:** retrained through `otaf.surrogate.NeuralRegressorNetwork` with the options of `train_neural_networks.py`: Xavier gain 0.02, weights clamped to [-1, 1], unshuffled batches, growing learning rate, stop at R2 >= 0.9998.
- **Credal mixture:** per feature, lambda_d = 1 with probability 0.25, 0 with probability 0.25, otherwise uniform on [0,1]. Stds are multiplied by 1.35.

**Checks:**

- With seed 420, the library retrain stopped at 879 epochs and gave the same TT, RR and EE probabilities on the test sample as `artifacts/model1_4_dof_surrogate.pth` (TT 4.468e-3), with slack errors equal to the printed precision. The baseline is therefore reproduced, not approximated.
- The hand-built 3-neuron network gives TT, RR and EE probabilities equal to the closed form on the same sample.

**Notes against the brief:**

- The multiplier in the Example 1 checkpoint is 1.35, not 1.16-1.25 (`X_std`/sigma = 1.349; about 15% of the training points fail).
- The R2 >= 0.9998 stop corresponds to an RMSE of about 8e-4 mm, the error level the baseline reaches.

## Results

Slack RMSE in mm. Each cell is the median, with the range over seeds in brackets. All regions except the last two columns of the second table are on the credal set.

### Slack error

| Configuration | Seeds | Overall | Near kink | Near s=0 | TT | RR | EE |
|---|---|---|---|---|---|---|---|
| Baseline recipe retrained: GELU 16-8-4, inflated x1.35, 200k | 3 | 9.1e-04 (8.6e-04 to 9.3e-04) | 2.9e-03 (2.5e-03 to 3.0e-03) | 3.8e-04 (3.6e-04 to 3.8e-04) | 3.4e-03 (3.1e-03 to 3.4e-03) | 8.3e-04 (7.4e-04 to 8.4e-04) | 9.4e-04 (9.0e-04 to 9.7e-04) |
| ReLU 16-8-4, inflated, 200k | 5 | 1.7e-06 (1.3e-06 to 2.0e-05) | 9.4e-06 (6.2e-06 to 8.0e-05) | 5.5e-07 (3.8e-07 to 9.5e-06) | 6.1e-05 (1.9e-05 to 6.7e-05) | 7.5e-07 (3.7e-07 to 1.1e-05) | 1.9e-06 (1.4e-06 to 2.3e-05) |
| **ReLU 16-8-4, credal mixture, 200k** | 5 | 9.2e-07 (8.6e-08 to 4.8e-06) | 2.8e-06 (4.3e-07 to 1.2e-05) | 6.7e-07 (7.6e-08 to 5.2e-06) | 1.5e-06 (1.4e-07 to 2.1e-05) | 3.0e-07 (1.2e-07 to 1.8e-06) | 1.3e-06 (1.1e-07 to 5.7e-06) |
| ReLU 16-8-4, credal mixture, 1M | 5 | 1.0e-06 (3.6e-07 to 3.2e-06) | 1.9e-06 (6.4e-07 to 4.3e-06) | 2.6e-06 (5.6e-07 to 6.6e-06) | 1.3e-06 (3.5e-07 to 1.6e-05) | 6.9e-07 (3.4e-07 to 2.2e-06) | 1.3e-06 (3.1e-07 to 3.7e-06) |
| GELU 16-8-4, credal mixture, 1M | 3 | 9.8e-05 (7.1e-05 to 1.5e-04) | 2.1e-04 (1.9e-04 to 2.2e-04) | 5.9e-05 (5.2e-05 to 8.1e-05) | 2.0e-04 (1.9e-04 to 3.4e-04) | 8.2e-05 (6.0e-05 to 1.3e-04) | 1.0e-04 (7.4e-05 to 1.6e-04) |
| GELU 64-32-16, inflated, 200k | 3 | 5.7e-05 (4.3e-05 to 1.1e-04) | 2.2e-04 (1.8e-04 to 4.5e-04) | 1.7e-05 (1.7e-05 to 2.9e-05) | 5.1e-04 (4.2e-04 to 8.1e-04) | 5.0e-05 (3.8e-05 to 9.7e-05) | 6.0e-05 (4.5e-05 to 1.1e-04) |
| GELU 64-32-16, credal mixture, 1M | 3 | 4.3e-05 (3.7e-05 to 4.7e-05) | 1.8e-04 (1.6e-04 to 1.8e-04) | 2.2e-05 (2.1e-05 to 2.3e-05) | 6.6e-05 (6.0e-05 to 7.2e-05) | 3.5e-05 (2.9e-05 to 3.7e-05) | 4.6e-05 (3.9e-05 to 5.0e-05) |

"Near kink" is 5|g4-g5| < 0.002 and "Near s=0" is |s| < 0.02, both over the credal set. The closed form evaluated in float32 through the same harness has RMSE 3.8e-9 mm; the hand-built ReLU network has 4e-8 mm.

### Coverage of the credal set, including near-zero translation and rotation

| Configuration | Seeds | Near pure translation, lambda_g = 0.01 | lambda_g = 0.1 | Near pure rotation, lambda_u = 0.01 | lambda_u = 0.1 | Mixed vertex T/R | Worst of 21x21 credal map | Allocation x1.35, lambda = 1 | Box +-4.5 sigma |
|---|---|---|---|---|---|---|---|---|---|
| Baseline recipe retrained: GELU 16-8-4, inflated x1.35, 200k | 3 | 3.1e-03 (2.8e-03 to 3.1e-03) | 1.8e-03 (1.6e-03 to 1.8e-03) | 8.3e-04 (7.4e-04 to 8.4e-04) | 8.3e-04 (7.4e-04 to 8.4e-04) | 9.4e-04 (9.0e-04 to 9.7e-04) | 3.4e-03 (3.1e-03 to 3.4e-03) | 8.1e-04 (8.1e-04 to 8.1e-04) | 2.6e-03 (1.9e-03 to 2.8e-03) |
| ReLU 16-8-4, inflated, 200k | 5 | 1.9e-05 (6.3e-06 to 9.5e-05) | 6.7e-06 (2.4e-06 to 5.8e-05) | 8.0e-07 (3.7e-07 to 1.1e-05) | 1.3e-06 (4.2e-07 to 1.1e-05) | 2.2e-06 (7.0e-07 to 2.2e-05) | 6.1e-05 (1.9e-05 to 6.8e-05) | 3.2e-05 (1.3e-05 to 1.4e-04) | 6.7e-04 (2.0e-04 to 8.7e-03) |
| **ReLU 16-8-4, credal mixture, 200k** | 5 | 1.6e-06 (1.5e-07 to 1.9e-05) | 2.2e-06 (1.5e-07 to 1.5e-05) | 3.0e-07 (1.2e-07 to 1.8e-06) | 3.3e-07 (1.2e-07 to 1.8e-06) | 6.9e-07 (1.2e-07 to 3.0e-06) | 2.1e-06 (4.6e-07 to 3.7e-05) | 2.0e-04 (9.5e-05 to 2.0e-03) | 2.4e-03 (7.8e-04 to 2.7e-02) |
| ReLU 16-8-4, credal mixture, 1M | 5 | 2.0e-06 (2.6e-07 to 1.5e-05) | 1.9e-06 (1.5e-07 to 1.1e-05) | 6.9e-07 (3.4e-07 to 2.2e-06) | 7.9e-07 (3.7e-07 to 2.7e-06) | 1.2e-06 (2.7e-07 to 3.6e-06) | 1.9e-06 (6.7e-07 to 2.8e-05) | 1.6e-04 (1.7e-05 to 1.8e-03) | 2.2e-03 (2.6e-04 to 2.6e-02) |
| GELU 16-8-4, credal mixture, 1M | 3 | 1.6e-04 (1.4e-04 to 2.0e-04) | 2.3e-04 (1.8e-04 to 3.3e-04) | 8.2e-05 (6.0e-05 to 1.3e-04) | 8.2e-05 (6.0e-05 to 1.3e-04) | 1.0e-04 (7.5e-05 to 1.6e-04) | 2.0e-04 (1.9e-04 to 3.4e-04) | 2.2e-04 (1.7e-04 to 2.3e-04) | 1.6e-03 (1.4e-03 to 2.2e-03) |
| GELU 64-32-16, inflated, 200k | 3 | 3.1e-04 (2.4e-04 to 5.8e-04) | 1.4e-04 (1.1e-04 to 2.7e-04) | 5.0e-05 (3.8e-05 to 9.7e-05) | 5.0e-05 (3.8e-05 to 9.7e-05) | 6.0e-05 (4.5e-05 to 1.1e-04) | 5.1e-04 (4.2e-04 to 8.1e-04) | 5.0e-05 (4.1e-05 to 8.7e-05) | 3.0e-04 (2.9e-04 to 3.5e-04) |
| GELU 64-32-16, credal mixture, 1M | 3 | 1.6e-04 (1.6e-04 to 1.6e-04) | 1.2e-04 (1.1e-04 to 1.3e-04) | 3.5e-05 (2.9e-05 to 3.7e-05) | 3.6e-05 (2.9e-05 to 3.7e-05) | 4.7e-05 (4.1e-05 to 4.9e-05) | 8.5e-05 (7.6e-05 to 9.2e-05) | 1.2e-04 (1.2e-04 to 1.3e-04) | 1.3e-03 (1.2e-03 to 1.3e-03) |

The recommended configuration has no weak allocation: at its median seed, the worst RMSE on a 21x21 map of the credal set (2e5 points per allocation) is 2.1e-6 mm. The current recipe's worst point is pure translation, at 3.4e-3 mm. `figures/kink_profile.png` shows the profile across the kink: the current surrogate is 4.5e-3 mm too low at g4 = g5 with u4 = u5 = 0.1.

### Failure probability by direct sampling, alpha = 0

Difference from sampling the closed form on the same 5e6 points, in percent, mean ± sd over seeds:

| Configuration | Seeds | TT | RR | EE |
|---|---|---|---|---|
| Baseline recipe retrained: GELU 16-8-4, inflated x1.35, 200k | 3 | +50.97 ± 5.84 | +1.37 ± 2.95 | +0.80 ± 0.61 |
| ReLU 16-8-4, inflated, 200k | 5 | +0.35 ± 0.57 | -0.00 ± 0.00 | -0.00 ± 0.00 |
| **ReLU 16-8-4, credal mixture, 200k** | 5 | +0.00 ± 0.00 | +0.00 ± 0.01 | -0.00 ± 0.01 |
| ReLU 16-8-4, credal mixture, 1M | 5 | +0.00 ± 0.00 | -0.01 ± 0.01 | -0.01 ± 0.01 |
| GELU 16-8-4, credal mixture, 1M | 3 | +0.81 ± 1.02 | -0.01 ± 0.14 | -0.05 ± 0.08 |
| GELU 64-32-16, inflated, 200k | 3 | +4.81 ± 1.91 | -0.05 ± 0.12 | +0.00 ± 0.03 |
| GELU 64-32-16, credal mixture, 1M | 3 | +0.04 ± 0.07 | +0.02 ± 0.03 | -0.01 ± 0.05 |

For the single model selected on validation (recommended configuration, seed 2):

| Allocation | Surrogate | Standard error | Closed form, same sample | Paired difference (SE) | Exact |
|---|---|---|---|---|---|
| TT | 3.0976e-3 | 2.5e-5 | 3.0976e-3 | 0 (0: no sign disagreement) | 3.0983e-3 |
| RR | 6.1126e-3 | 3.5e-5 | 6.1126e-3 | 0 (2.8e-7) | 6.1967e-3 |
| EE | 6.1040e-3 | 3.5e-5 | 6.1042e-3 | -2e-7 (2.0e-7) | 6.1871e-3 |

These differences are within sampling noise of zero, so they do not show a difference from the closed form. They show the surrogate and the closed form agree on the same points, and only to that resolution.

The test sample itself is low at RR and EE by about 1.3% against quadrature (2.4 standard errors); the closed form shows the same offset. It is a property of the sample, which was fixed before any result and was not re-drawn. Twelve independent 5e6 samples checked separately average 6.188e-3 at RR (SE 1.1e-5) and 3.089e-3 at TT (SE 8.5e-6). The RR mean is within 1 SE of exact; the TT mean is about 1.1 SE below it.

### Single 50x50 map

Recommended configuration, seed 2, 5e6 points, re-using the surface already computed. See `figures/pf_surfaces.png`, whose middle panel is the current checkpoint's existing surface, shown for reference only.

| alpha | Against quadrature: median | 5th pct | 95th pct | min | max | Largest difference from closed form, same sample |
|---|---|---|---|---|---|---|
| 0 | -1.00% | -1.48% | -0.40% | -1.89% | +0.53% | 0.014% |
| 0.01 | -0.73% | -1.00% | -0.32% | -1.35% | +0.13% | 0.009% |
| 0.025 | -0.23% | -0.44% | -0.06% | -0.59% | +0.03% | 0.005% |
| 0.05 | -0.09% | -0.18% | -0.01% | -0.26% | +0.05% | 0.002% |
| 0.075 | -0.07% | -0.11% | -0.03% | -0.15% | +0.02% | 0.001% |

The median offsets from quadrature match the closed form on the same sample at each alpha, so they are sampling offset and not surrogate error. At alpha = 0 the minimum is at TT. The plateau, defined as within 2% of the maximum, covers 92.0% of the map against 92.7% exact (overlap 0.99, also the level the closed form's own sample reaches).

### GLD on the best model

One seed (recommended, seed 2), TT, RR and EE, alpha = 0, 5e6 points, gldpy-gpu VSL by L-moments. The float32 fit (as in `plot_pf_2d_4dof_model.py`) and the float64 fit agree within 1e-7; float64 values are shown.

| Slack used | TT: direct / GLD | RR: direct / GLD | EE: direct / GLD |
|---|---|---|---|
| Closed form | 3.098e-3 / 2.898e-3 | 6.113e-3 / 4.404e-3 | 6.104e-3 / 5.904e-3 |
| Recommended surrogate | 3.098e-3 / 2.898e-3 | 6.113e-3 / 4.404e-3 | 6.104e-3 / 5.904e-3 |
| Current checkpoint | 4.468e-3 / 4.417e-3 | 5.988e-3 / 2.729e-3 | 6.152e-3 / 6.089e-3 |

**A better surrogate does not remove the pure rotation problem.** The GLD on the recommended surrogate equals the GLD on the exact slack and is 28% below direct sampling at RR, 6.4% below at TT and 3.3% below at EE. This error belongs to the GLD fit on this slack distribution. The better surrogate does move the RR value from 2.73e-3 to 4.40e-3, because the current checkpoint added its own error to the GLD's.

## What each factor did

The table below has one seed per candidate, so differences smaller than the seed spread in the first table are not evidence. Near TT the seed spread of ReLU 16-8-4 inflated is +0.35% ± 0.57%.

| Candidate (one seed) | Params | Overall | Near kink | Near s=0 | TT | RR | TT Pf % |
|---|---|---|---|---|---|---|---|
| A0_old_stop | 257 | 8.7e-04 | 2.6e-03 | 4.3e-04 | 3.2e-03 | 7.6e-04 | +52.89 |
| A1_old_nostop | 257 | 4.6e-04 | 1.7e-03 | 1.7e-04 | 2.1e-03 | 4.1e-04 | +25.55 |
| A2_old_nostop_noclamp | 257 | 4.1e-04 | 1.4e-03 | 1.6e-04 | 1.9e-03 | 3.6e-04 | +22.79 |
| A3_new_gelu16 | 257 | 1.6e-04 | 6.1e-04 | 5.4e-05 | 1.0e-03 | 1.4e-04 | +10.56 |
| B_gelu32 | 833 | 9.1e-05 | 3.5e-04 | 3.9e-05 | 7.0e-04 | 8.1e-05 | +6.48 |
| B_gelu64 | 2945 | 4.3e-05 | 1.8e-04 | 1.7e-05 | 4.2e-04 | 3.8e-05 | +3.35 |
| B_gelu64x4 | 12865 | 6.4e-05 | 2.4e-04 | 1.9e-05 | 5.3e-04 | 5.6e-05 | +4.78 |
| B_leaky16 | 257 | 1.8e-05 | 5.3e-05 | 7.0e-06 | 1.2e-04 | 4.4e-06 | +1.84 |
| B_leaky32 | 833 | 8.8e-06 | 2.2e-05 | 2.0e-06 | 4.1e-05 | 1.1e-05 | +0.25 |
| B_leaky64 | 2945 | 1.5e-05 | 4.8e-05 | 9.3e-06 | 1.1e-04 | 1.6e-05 | +0.67 |
| B_leaky64x4 | 12865 | 3.5e-05 | 8.2e-05 | 2.2e-05 | 1.6e-04 | 3.7e-05 | +1.31 |
| B_relu16 | 257 | 2.0e-05 | 8.0e-05 | 9.5e-06 | 6.7e-05 | 1.1e-05 | -0.01 |
| B_relu32 | 833 | 6.8e-06 | 1.6e-05 | 1.5e-06 | 2.5e-05 | 4.6e-06 | +0.32 |
| B_relu64 | 2945 | 1.4e-05 | 3.4e-05 | 1.0e-05 | 6.9e-05 | 1.1e-05 | -0.02 |
| B_relu64x4 | 12865 | 3.5e-05 | 9.1e-05 | 1.7e-05 | 1.4e-04 | 3.8e-05 | +0.49 |
| B_relu_1h32 | 193 | 7.2e-08 | 3.1e-07 | 4.1e-08 | 5.2e-06 | 6.1e-08 | +0.05 |
| B_relu_1h3 | 19 | 2.4e-02 | 3.9e-02 | 3.3e-02 | 3.9e-02 | 2.4e-02 | +1473.51 |
| B_relu_1h8 | 49 | 2.6e-08 | 1.5e-07 | 1.9e-08 | 3.0e-06 | 2.1e-08 | +0.03 |
| C_gelu64_1M | 2945 | 4.4e-05 | 1.8e-04 | 1.8e-05 | 4.3e-04 | 3.8e-05 | +3.49 |
| C_gelu64_5M | 2945 | 4.5e-05 | 1.9e-04 | 1.8e-05 | 4.4e-04 | 3.9e-05 | +3.55 |
| C_relu16_1M | 257 | 1.9e-05 | 8.6e-05 | 7.2e-06 | 8.0e-05 | 1.5e-05 | +1.09 |
| C_relu16_5M | 257 | 1.9e-05 | 9.0e-05 | 8.5e-06 | 1.1e-04 | 1.0e-05 | +0.30 |
| D_gelu64_credal+limit+kink | 2945 | 3.3e-05 | 7.8e-05 | 1.5e-05 | 1.5e-04 | 2.7e-05 | +0.12 |
| D_relu16_credal+kink | 257 | 1.2e-06 | 2.7e-06 | 1.1e-06 | 3.8e-06 | 9.2e-07 | -0.01 |
| D_relu16_credal+limit+kink | 257 | 2.0e-05 | 1.6e-05 | 8.4e-06 | 3.0e-05 | 8.6e-06 | -0.01 |
| D_relu16_credal+limit | 257 | 5.1e-06 | 1.3e-05 | 4.8e-06 | 1.0e-05 | 3.8e-06 | +0.04 |
| D_relu16_credal | 257 | 1.0e-06 | 1.9e-06 | 2.6e-06 | 1.3e-06 | 3.9e-07 | +0.01 |
| D_relu16_uniform | 257 | 8.4e-08 | 4.8e-07 | 4.5e-08 | 8.4e-06 | 7.6e-08 | +0.06 |
| E_UPPERBOUND_gelu16_absfeature | 273 | 4.7e-06 | 5.5e-06 | 4.8e-06 | 6.0e-06 | 4.3e-06 | -0.05 |
| E_gelu64_wmse | 2945 | 8.5e-05 | 3.2e-04 | 2.2e-05 | 6.4e-04 | 7.5e-05 | +5.04 |
| E_relu16_wmse | 257 | 4.3e-06 | 2.1e-05 | 9.1e-07 | 1.0e-04 | 2.6e-06 | +0.19 |

- **Training settings of the current recipe, GELU 16-8-4 (A0 to A3):** the R2 stop, weight clamp and short schedule together cost about 5x in overall slack RMSE. Removing them takes TT from +53% to +11% with no other change. This does not reach the true shape.
- **Activation:** the largest factor. At 257 parameters on the same data and recipe, ReLU gets 8x lower overall RMSE than GELU on seed 0 (B_relu16 against A3). Seed 0 is the worst of the five ReLU seeds; the ReLU median is about 90x lower. LeakyReLU behaved like ReLU within this one-seed screen.
- **Size:** no evidence that more parameters help ReLU. Overall RMSE was 6.8e-6 mm for 32-16-8, 1.4e-5 for 64-32-16 and 3.5e-5 for 64x4, on one seed each, against 1.3e-6 to 2.0e-5 mm across the five seeds of 16-8-4.
- **Training points:** no measurable effect from 200k to 5M, for ReLU or GELU. The limit is not data. This matters for larger models, where LP labels are not free.
- **Training domain:** credal mixture was the second largest factor. ReLU's TT RMSE falls from 6.1e-5 to 1.5e-6 mm and seed spread at TT falls from +0.35 ± 0.57% to 0.00%. Mass on g4 = g5 = 0 is what matters, and it uses no knowledge of where the kink is.
- **Enrichment:** near-limit-state and kink enrichment gave no improvement over plain credal in the one-seed screen. Kink enrichment uses the location of the kink, which is not available for larger models.
- **Weighted loss near s = 0:** one trial each. On ReLU, near-s=0 RMSE went from 9.5e-6 to 9.1e-7 mm against B_relu16, but that is within the unweighted seed range (3.8e-7 to 9.5e-6), and TT RMSE got slightly worse (6.7e-5 to 1.0e-4). GELU 64-32-16 got worse (overall 4.3e-5 to 8.5e-5, TT Pf +3.35% to +5.04%). Not recommended on this evidence.
- **Upper bound, closed-form feature |g4-g5| as input (not available for larger models):** GELU 16-8-4 reaches 4.7e-6 mm overall and 6.0e-6 mm at TT. This confirms the kink is what limits GELU. It is the only row using this feature.
- **Few-neuron solution:** training does not always find it. One hidden layer with 3 neurons, the exact minimum, failed on the single seed tried (2.4e-2 mm, +1474% at TT). With 8 neurons it reached 2.6e-8 mm on its single seed, the lowest overall RMSE of any trained network here, and 32 neurons reached 7.2e-8 mm. Both were on inflated data. The planned repeats of these were cut, so 8 neurons may not be robust. This is not a recommendation.

### Validity outside the credal set

A credal-trained network never sees all four variables at full spread, so it is not a global fit. Inflated training does better off the set, and a uniform box better still, on the one seed available:

| Configuration | x1.35, lambda = 1 | Box +-4.5 sigma | Seeds |
|---|---|---|---|
| ReLU 16-8-4, credal, 200k | 2.0e-4 (9.5e-5 to 2.0e-3) | 2.4e-3 (7.8e-4 to 2.7e-2) | 5 |
| ReLU 16-8-4, inflated, 200k | 3.2e-5 (1.3e-5 to 1.4e-4) | 6.7e-4 (2.0e-4 to 8.7e-3) | 5 |
| ReLU 16-8-4, uniform box, 1M | 4.7e-5 | 7.6e-7 | 1 |

The uniform box also has 8.4e-8 mm overall and 8.4e-6 mm at TT on the credal set. It looks strong in this 4-variable problem, but its repeats were cut, so one seed does not establish it, and box sampling is not expected to scale to 50 variables. Any use at lambda = 1 or wider requires a check first; a mixture of credal and inflated sampling was not tried.

Far-tail errors exist even on the credal set: seed 4 at TT has a 3.7e-2 mm error at s = +0.295, far from the limit state, on under 1e-5 of samples. Seeds 1 and 4 have high validation RMSE (8e-5 and 1.4e-4 mm) while their test RMSE on the credal set is low. This probably reflects errors in the inflated tails of the training domain, though that was inferred and not isolated. One RMSE is not a guarantee for every point.

## Speed and size

| Model | Parameters | s per 1e6 points, GPU | s per 1e6 points, 2 CPU threads |
|---|---|---|---|
| LP (scipy `milp`, 1 core) | n/a | n/a | 2361 (2.36 ms per sample, 2000 samples) |
| Current GELU 16-8-4 | 257 | 0.005 | 0.16 |
| Recommended ReLU 16-8-4 | 257 | 0.005 | 0.13 |

So the recommended surrogate on 2 CPU threads is about 18,000x faster than the LP on one core, at the same size as the current one. These are single measurements on a shared machine. Training took about 100 s on the GPU.

## Recommended configuration and commands

ReLU, hidden 16-8-4, credal mixture with stds x1.35, 200,000 points, 60/40 split, MSE, AdamW without weight decay, learning rate 2e-3 cosine to 1e-6, batch 4096, 40,000 steps, no R2 stop, no clamp. Select on validation MSE. Selected checkpoint: `ckpt/F_relu16_credal_200k_s2.pth`.

From `/home/kramer84/otaf/scripts` (existing results are skipped, so move `results/` and `ckpt/` aside to retrain):

```bash
P=/home/kramer84/otaf/.venv/bin/python
CAP="systemd-run --user --scope -q -p MemoryMax=12G"
$CAP $P ex1_surrogate_study/check_lp.py 1500
$CAP $P ex1_surrogate_study/cf_surfaces.py
$CAP $P ex1_surrogate_study/controls.py
$CAP $P ex1_surrogate_study/run.py --stage F5 --only F_relu16_credal_200k --seeds 0 1 2 3 4 --n-surf 5000000
$CAP $P ex1_surrogate_study/slack_metrics.py    # all saved checkpoints with a result, plus the two existing artifacts
$CAP $P ex1_surrogate_study/gld_check.py F_relu16_credal_200k_s2
$CAP $P ex1_surrogate_study/figures.py F_relu16_credal_200k_s2
```

Stages are defined in `configs.py`. `results/*.json` hold the full metrics per candidate, and `make_tables.py` regenerates the tables above. D_relu16_credal_s0 and F_relu16_credal_1M_s0 are the same configuration and gave the same slack figures, which is one check of repeatability on this machine; reproducibility on other hardware was not tested.

To use it with the existing code, a small library change is needed: `NeuralRegressorNetwork.from_checkpoint` hardcodes GELU and cannot load this checkpoint as it stands. `src/` was not edited; `ex1lib.load_ckpt` is used here.

## What generalises to the 16-, 30- and 50-variable models

None of this was tested on those models. These are expectations to check, not results.

- **Likely to carry over:** the R2 stop, weight clamp and short schedule limit accuracy regardless of model, and they are in the shared training code. Trying more LP labels first is unlikely to be the fix if the same data insensitivity holds.
- **Reason to expect ReLU to fit:** an LP's optimal value is piecewise linear in its right-hand side, so the slack should be piecewise linear wherever deviations enter only there. This was not checked for these models, and the MILP and cylinder cases need checking.
- **Unknown:** how much ReLU size is needed as the number of active-set pieces grows. 16-8-4 works here because this target has very few pieces. Sampling the credal set is also harder in high dimension, since per-feature vertices become rare under an independent mixture, and the classical allocations may need explicit mass.
- **Does not carry over:** the closed-form labels (larger models need LP or MILP labels), kink enrichment and the |g4-g5| feature.

## Run status

Seeds that exist:

| Stage | Content | Seeds | Status |
|---|---|---|---|
| Controls | closed form, hand-built ReLU, Nov. and May checkpoints | n/a | complete |
| B1 | baseline retrain through the library | 420, 1, 2 | complete |
| A, B, C, D, E | screening | 0 | complete, **one seed only** |
| F5 | ReLU 16-8-4: credal 200k, credal 1M, inflated 200k | 0-4 | complete |
| F3 | GELU 16-8-4 credal 1M, GELU 64-32-16 credal 1M, GELU 64-32-16 inflated 200k | 0-2 | complete |
| F3 | ReLU 16-8-4 uniform 1M | none | **cut**: seed 0 was stopped in training, no checkpoint; only the single-seed `D_relu16_uniform_s0` exists |
| G (one-hidden-layer repeats, 3/4/8 neurons, 5 seeds each) | whether training reliably finds the few-neuron solution | none | **cut, never started**; only single-seed `B_relu_1h3`, `1h8`, `1h32` exist |
| GLD | TT, RR, EE on one seed | 2 | complete at those points. The 50x50 GLD grid and alpha > 0 were dropped |

The coordinator's request calls the GLD test "stage G". In this study's `configs.py`, stage G is the one-hidden-layer repeats, which were cut. The GLD test was run.

**Failed runs:** the first launch of stages A and B crashed during the 11th candidate (`B_relu64x4`) with GPU out of memory, while other jobs were sharing the card. It was resumed with smaller evaluation batches and retries; that candidate was rerun and completed, and no finished result was lost. No job was killed by `MemoryMax`, and no candidate is marked failed.

**Not tested:** other optimisers, learning rates and step counts beyond those listed; a credal plus inflated training mixture; other enrichment widths; label noise; multi-seed repeats of any screening row.

## Files

`ex1lib.py`, `configs.py`, `run.py`, `controls.py`, `check_lp.py`, `cf_surfaces.py`, `slack_metrics.py`, `tail_check.py`, `gld_check.py`, `lp_timing.py`, `figures.py`, `summarize.py`, `aggregate.py`, `make_tables.py`, `queue*.sh`; `results/`, `ckpt/`, `figures/`, `logs/`.
