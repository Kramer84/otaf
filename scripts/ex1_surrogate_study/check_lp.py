"""Closed form against the LP (scipy milp as in baselines/baseline_cases.py), at the allocations and in the training domains used here."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1] / "baselines"))
import ex1lib as L
from otaf.example_models import model1
from baseline_cases import solve_slack
_, _, ms, mu = model1.get_distribution_params()
print("model1 sigma_max", ms, "mu", mu, "ours", L.SIGMA); assert np.allclose(ms, L.SIGMA) and np.allclose(mu, 0)
soc = model1.get_system_of_constraints_assembly_model()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
rng = np.random.default_rng(99)
sets = {k: rng.standard_normal((N, 4)) * L.SIGMA * L.lam_vec(*v) for k, v in L.LAM.items()}
for dom in ["inflated", "credal", "uniform", "credal+limit", "credal+kink"]:
    sets[dom] = L.make_domain(dom, N, 99)
for k, x in sets.items():
    s, st = solve_slack(soc, x @ np.asarray(model1.sample_multiplier).T)
    ok = st == 0
    print(f"{k:14s} n={len(x)} solved={ok.sum()} max|cf-LP|={np.abs(L.closed_form(x) - s)[ok].max():.2e} fail share={np.mean(s[ok] < 0):.3f}", flush=True)
