"""Pf of the classical allocations with the same pipeline as the bounds (surrogate + GLD, 1e5 samples, alpha = 0).

Ex1: the 3^2 combinations of T (pure translation), R (pure rotation), E (equal share) per feature.
Ex2: the 3^8 combinations (and the three homogeneous ones). Ex3: all lambda = 1. Ex4: see classical_allocations_ex3_ex4.py.
Usage: python scripts/reference_checks/classical_cases_pipeline.py ex1|ex2 [n_procs]
"""
import itertools, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, Path(__file__).parents[1].as_posix())
import otaf, imprecise_bounds_analysis as IB

KIND = {"T": (1.0, 0.0), "R": (0.0, 1.0), "E": (2 ** -.5, 2 ** -.5)}
CFG = {"ex1": ("model1", "model1_4_dof", 2), "ex2": ("model2", "model2_16_dof", 8)}
_S = {}

def setup(key):
    if key not in _S:
        name, f, n = CFG[key]
        model = getattr(otaf.example_models, name)
        sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(f"artifacts/{f}_surrogate.pth")
        dist, sym, ms, mu = model.get_distribution_params()
        otaf.common.set_seed(0)
        smp = np.array(dist.getSample(100_000))
        _S[key] = (IB.get_model_evaluator(smp, mu, sur), IB.make_gld("gpu"))
    return _S[key]

def work(args):
    key, combos = args
    ev, gld = setup(key)
    out = {}
    for c in combos:
        lam = np.ravel([KIND[k] for k in c])
        s = np.asarray(ev(lam)).ravel()
        _, pf = IB.fit_gld_pf(gld, s, 0.0)
        out[c] = (float(pf), float(np.mean(s < 0)))
    return out

if __name__ == "__main__":
    key = sys.argv[1]; npr = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n = CFG[key][2]
    combos = ["".join(c) for c in itertools.product("TRE", repeat=n)]
    chunks = [combos[i::npr * 4] for i in range(npr * 4)]
    with Pool(npr) as p:
        res = {}
        for r in p.map(work, [(key, c) for c in chunks]):
            res.update(r)
    json.dump(res, open(f"rerun_logs/classical_pipeline_{key}.json", "w"))
    srt = sorted(res.items(), key=lambda kv: kv[1][0])
    hom = {h: res[h * n] for h in "TRE"}
    print(key, "homogeneous (GLD, direct MC):", {k: (f"{v[0]:.3e}", f"{v[1]:.3e}") for k, v in hom.items()})
    print("lowest 3:", [(k, f"{v[0]:.3e}") for k, v in srt[:3]], " highest 3:", [(k, f"{v[0]:.3e}") for k, v in srt[-3:]])
    print("range over combos (GLD):", f"{srt[0][1][0]:.3e}", f"{srt[-1][1][0]:.3e}")
