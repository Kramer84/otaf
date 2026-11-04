"""Ex2 (16 DOF): exact bounds of Pf over the credal set, no surrogate and no GLD.

Pf(phi) = P(slack < threshold) with the closed-form slack of analytical_16dof_vs_model2.py, on a
common standard-normal sample (so the objective is deterministic). phi in [0, pi/2]^8 gives the
quarter-circle point (lambda_u, lambda_g) = (cos phi, sin phi) of each of the 8 planar features.
Local searches (COBYQA) start from the best/worst vertices of ex2_vertices_pf.py; the optima are then
re-evaluated on an independent, larger sample.
Usage: python scripts/reference_checks/ex2_exact_bounds.py [threshold] [n_opt] [n_check]
"""
import json, sys, glob
import numpy as np
from scipy.optimize import minimize, Bounds
sys.path.insert(0, __import__("pathlib").Path(__file__).parent.as_posix())
import analytical_16dof_vs_model2 as A

thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
n_opt = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
n_chk = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
KIND = {"T": 0.0, "R": np.pi / 2, "E": np.pi / 4}
z = A.standard_sample(n_opt, seed=11)
f = lambda phi: A.pf(A.boundary_lambda(phi), z, thr)

vf = sorted(glob.glob(f"ex2_vertices_pf_n*_thr{thr}.json"))
verts = json.load(open(vf[-1]))["pf"] if vf else {}
srt = sorted(verts.items(), key=lambda kv: kv[1])
starts = {"min": [k for k, _ in srt[:3]] or ["TTTTTTTT"], "max": [k for k, _ in srt[-3:]] or ["EEEEEEEE"]}
zc = A.standard_sample(n_chk, seed=99)
for goal, sign in (("min", 1.0), ("max", -1.0)):
    best = None
    for s in starts[goal]:
        x0 = np.array([KIND[c] for c in s])
        r = minimize(lambda p: sign * f(p), x0, method="COBYQA", bounds=Bounds(0, np.pi / 2), options={"maxfev": 1500})
        print(f"{goal} from {s}: start {f(x0):.5f} -> {f(r.x):.5f}", flush=True)
        if best is None or sign * f(r.x) < sign * f(best):
            best = r.x
    pc = A.pf(A.boundary_lambda(best), zc, thr)
    print(f"== {goal}: Pf={f(best):.5e} (opt sample), {pc:.5e} (independent n={n_chk}); phi/pi*180 = {np.round(np.degrees(best), 1).tolist()}", flush=True)
