"""R1.8 B for Ex3 and Ex4: Pf of classical precise choices with the same surrogate + GLD pipeline.

Ex3: only the full allocation (all lambda = 1, the maximum of the credal set).
Ex4 (50 dof) and Ex4 reduced (32 dof, hole rotations and translations only): per feature, pure
translation (T: u, v, w and sizes), pure rotation (R: beta, gamma) and equal share (E: every
component of the feature at the same lambda). Each feature is scaled so that its capability
constraint is active (bisection), pins keep lambda = 1 on their diameter.
Reports P(slack < alpha) for alpha in the sweep, GLD (as in the optimisation) and direct Monte Carlo.
Usage: python scripts/reference_checks/classical_allocations_ex3_ex4.py [mc_size]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, Path(__file__).parents[1].as_posix())
import otaf
import imprecise_bounds_analysis as IB

MC = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100_000
ALPHAS = [0.0, 0.025, 0.05, 0.075]
gld = None

def _gld():
    global gld
    gld = gld or IB.make_gld("gpu")
    return gld

FILES = {"model3": "model3_30_dof", "model4": "model4_50_dof"}

def evaluator(name):
    model = getattr(otaf.example_models, name)
    sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(f"artifacts/{FILES[name]}_surrogate.pth")
    dist, sym, ms, mu = model.get_distribution_params()
    otaf.common.set_seed(0)
    smp = np.array(dist.getSample(MC))
    return model, ms, IB.get_model_evaluator(smp, mu, sur)

def report(label, slack):
    row = []
    for a in ALPHAS:
        _, pf = IB.fit_gld_pf(_gld(), slack, a)
        row.append(f"{pf:.3e}/{np.mean(slack < a):.3e}")
    print(f"  {label:34s} " + "  ".join(row), flush=True)

def scale_feature(cons_i, comps, template, mask_idx, cons_fn, active=True):
    """Bisection on s in [0, 1] so that constraint cons_i of the vector built from template is 0."""
    def c(s):
        x = template.copy(); x[mask_idx] = s
        return cons_fn(x)[cons_i]
    if c(1.0) <= 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if c(mid) <= 0 else (lo, mid)
    return lo

def ex4_vectors(model, ms, reduced):
    cons = model.get_scaled_credal_set_constraints_function(ms)
    feats = []  # (constraint index, rotation idx, translation idx incl. size)
    feats += [(0, [0, 1], [2]), (1, [3, 4], [5])]
    # constraint ci (2..9) acts on the hole whose variables start at BASE[ci - 2], diameter at 36 + ci
    BASE = [6, 14, 22, 30, 10, 18, 26, 34]
    for ci in range(2, 10):
        b = BASE[ci - 2]
        feats.append((ci, [b, b + 1], [b + 2, b + 3] + ([] if reduced else [36 + ci])))
    out = {}
    for kind in "TRE":
        x = np.zeros(50)
        if not reduced:
            x[46:50] = 1.0
        for ci, rot, tr in feats:
            if reduced and ci < 2:
                continue
            idx = {"T": tr, "R": rot, "E": rot + tr}[kind]
            if kind == "E" and not reduced and ci >= 2:
                pass
            s = scale_feature(ci, None, x, idx, cons)
            x[idx] = s
        c = np.array(cons(x))
        out[kind] = (x, c)
    return out

if __name__ == "__main__":
    print("alpha:            " + "  ".join(f"{a:^21}" for a in ALPHAS) + "   (GLD / direct MC)")
    model, ms, ev = evaluator("model3")
    print("Ex3 (26 dof)")
    report("all lambda = 1 (maximum)", np.asarray(ev(np.ones(model.dim))).ravel())
    for name, reduced in (("model4", False), ("model4", True)):
        model, ms, ev = evaluator(name)
        vecs = ex4_vectors(model, ms, reduced)
        print("Ex4 reduced (32 dof)" if reduced else "Ex4 (50 dof)")
        for kind, lab in (("T", "pure translation (T)"), ("R", "pure rotation (R)"), ("E", "equal share (E)")):
            x, c = vecs[kind]
            report(f"{lab} [c in {c[:10].min():+.1e},{c[:10].max():+.1e}]", np.asarray(ev(x)).ravel())
