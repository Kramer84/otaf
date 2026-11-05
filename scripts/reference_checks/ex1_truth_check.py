"""Ex1: three independent evaluations of Pf at the classical allocations (alpha = 0), no surrogate and no GLD.

(a) quadrature of the closed form, (b) Monte Carlo on the notebook's corner-by-corner assembly function,
(c) the OTAF linear program on the same kind of sample. Also checks that Pf depends on the allocation only through
A = lambda_u4^2 + lambda_u5^2. Usage: python scripts/reference_checks/ex1_truth_check.py [n_mc] [n_lp]
"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, Path(__file__).parent.as_posix()); sys.path.insert(0, Path(__file__).parents[1].as_posix() + "/baselines")
import ex1_analytical_pf as E
import otaf
from baseline_cases import solve_slack

n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000
n_lp = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
rng = np.random.default_rng(2026)
K = {"TT": ((1, 0), (1, 0)), "RR": ((0, 1), (0, 1)), "EE": ((2**-.5, 2**-.5), (2**-.5, 2**-.5)), "TR": ((1, 0), (0, 1))}
tail = stats.norm.sf(E.C0 / (E.SU * 2**.5))
print(f"sigma_u={E.SU:.5f} sigma_g={E.SG:.5f}; closed form TT = Phi(-c/(sqrt2 s)) = {tail:.5e}; RR = 2 x TT = {2*tail:.5e}")

def notebook_gap(u4, g4, u5, g5):
    """Corner-by-corner geometry of notebooks/Ex1_Pocket/Ex1_2D_4dof_Analytical_Ref.ipynb."""
    X1 = np.stack([E.L1 + u5 - E.Y * g5, E.L1 + u5 + E.Y * g5])
    X2 = np.stack([E.L2 - u4 - E.Y * g4, E.L2 - u4 + E.Y * g4])
    return (X2 - X1).min(axis=0)

m = otaf.example_models.model1
soc = m.get_system_of_constraints_assembly_model()
dist, sym, ms, mu = m.get_distribution_params()
print("OTAF variables:", sym, "max std:", np.round(ms, 5))
print(f"{'case':4s} {'quadrature':>11s} {'MC notebook (n=%.0e)' % n_mc:>24s} {'LP slack<0 (n=%.0e)' % n_lp:>24s}")
for k, (l4, l5) in K.items():
    q = E.pf(l4, l5)
    f = 0
    for _ in range(n_mc // 1_000_000):
        z = rng.standard_normal((1_000_000, 4))
        f += (notebook_gap(z[:, 0] * E.SU * l4[0], z[:, 1] * E.SG * l4[1], z[:, 2] * E.SU * l5[0], z[:, 3] * E.SG * l5[1]) < 0).sum()
    pmc = f / n_mc
    lam = np.array([l4[0], l4[1], l5[0], l5[1]], float)
    otaf.common.set_seed(7)
    x = (np.array(dist.getSample(n_lp)) - mu) * lam + mu
    s, st = solve_slack(soc, x @ np.asarray(m.sample_multiplier).T)
    pl = np.mean(s[st == 0] < 0)
    print(f"{k:4s} {q:11.5e} {pmc:12.5e} +-{1.96*np.sqrt(pmc/n_mc):.1e} {pl:12.5e} +-{1.96*np.sqrt(max(pl,1e-9)/n_lp):.1e}", flush=True)
print("\nPf as a function of A = lambda_u4^2 + lambda_u5^2 only (quadrature, different allocations with the same A):")
for A in (0.0, 0.5, 1.0, 1.5, 2.0):
    v = []
    for u4 in np.linspace(max(0, A - 1)**.5, min(1, A)**.5, 4):
        u5 = max(A - u4**2, 0)**.5
        v.append(E.pf((u4, (1 - u4**2)**.5), (u5, (1 - u5**2)**.5)))
    print(f"  A={A:.1f}: " + " ".join(f"{x:.5e}" for x in v))
