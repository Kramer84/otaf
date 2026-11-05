"""Exact Pf (no surrogate, no GLD) of the classical allocations, alpha = 0.

ex2: closed-form slack on independent samples: T, R, E (1e7 samples) and the 40 lowest and 40 highest of the 6561 vertices
ranked by ex2_vertices_pf.py, re-evaluated on an independent 5e6 sample (removes the selection bias of the ranking).
ex4: linear program on fresh samples at pure rotation, for the 50-dim and the 32-dim models.
Usage: python scripts/reference_checks/classical_cases_exact.py ex2 | ex4 [n]
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, Path(__file__).parent.as_posix()); sys.path.insert(0, Path(__file__).parents[1].as_posix()); sys.path.insert(0, Path(__file__).parents[1].as_posix() + "/baselines")

def ex2():
    import analytical_16dof_vs_model2 as A
    K = {"T": (1.0, 0.0), "R": (0.0, 1.0), "E": (2**-.5, 2**-.5)}
    lam = lambda c: np.ravel([K[k] for k in c])
    def pf(c, n_batches, seed):
        f = sum(A.pf(lam(c), A.standard_sample(1_000_000, seed + b)) * 1_000_000 for b in range(n_batches))
        return f / (n_batches * 1_000_000)
    for h in "TRE":
        p = pf(h * 8, 10, 500)
        print(f"{h}: {p:.4e} +- {1.96*(p/1e7)**.5:.1e}", flush=True)
    r = json.load(open("ex2_vertices_pf_n200000_thr0.0.json"))["pf"]
    srt = sorted(r, key=r.get)
    for lab, cs, pick in (("lowest", srt[:40], min), ("highest", srt[-40:], max)):
        v = {c: pf(c, 5, 900) for c in cs}
        k = pick(v, key=v.get)
        print(f"{lab} of the 40 candidates on an independent 5e6 sample: {k} {v[k]:.4e} +- {1.96*(v[k]/5e6)**.5:.1e}; candidates span {min(v.values()):.4e} to {max(v.values()):.4e}", flush=True)

_soc = None
def chunk(a):
    global _soc
    import otaf
    from baseline_cases import solve_slack
    m = otaf.example_models.model4
    if _soc is None:
        _soc = m.get_system_of_constraints_assembly_model()
    dist, sym, ms, mu = m.get_distribution_params()
    lam, seed, n = a
    otaf.common.set_seed(seed)
    x = (np.array(dist.getSample(n)) - mu) * lam + mu
    s, st = solve_slack(_soc, x @ np.asarray(m.sample_multiplier).T)
    return int((s[st == 0] < 0).sum()), int((st == 0).sum())

def ex4(n):
    import otaf
    import classical_allocations_ex3_ex4 as C
    m = otaf.example_models.model4
    ms = m.get_distribution_params()[2]
    for lab, red in (("50 dim", False), ("32 dim", True)):
        lam = C.ex4_vectors(m, ms, red)["R"][0]
        with Pool(5) as p:
            res = p.map(chunk, [(lam, 4000 + k, 1000) for k in range(n // 1000)])
        f, t = map(sum, zip(*res)); q = f / t
        print(f"Ex4 {lab}, pure rotation, LP: Pf={q:.3e} +- {1.96*(q*(1-q)/t)**.5:.1e} ({f} failures / {t})", flush=True)

if __name__ == "__main__":
    ex2() if sys.argv[1] == "ex2" else ex4(int(sys.argv[2]) if len(sys.argv) > 2 else 40000)
