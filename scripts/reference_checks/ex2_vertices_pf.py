"""Ex2 (16 DOF), R1.8 B: Pf of the classical allocations with the exact closed-form slack.

Each of the 8 planar features takes one of three discrete allocations on the credal-set
boundary (lambda_u^2 + lambda_g^2 = 1): T = pure translation (1,0), R = pure rotation (0,1),
E = equal share (1/sqrt2, 1/sqrt2). All 3^8 = 6561 combinations are evaluated on the same
standard-normal sample (common random numbers). Reports the three homogeneous choices,
the extremes over the 6561 vertices, and the vertices closest to the interval ends.
Usage: python scripts/reference_checks/ex2_vertices_pf.py [n_samples] [threshold]
"""
import itertools, sys, time, json
import numpy as np
sys.path.insert(0, __import__("pathlib").Path(__file__).parent.as_posix())
import analytical_16dof_vs_model2 as A

KIND = {"T": (1.0, 0.0), "R": (0.0, 1.0), "E": (2 ** -.5, 2 ** -.5)}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
z = A.standard_sample(n, seed=11)
res = {}
t0 = time.time()
for combo in itertools.product("TRE", repeat=8):
    lam = np.ravel([KIND[k] for k in combo])
    res["".join(combo)] = float(A.pf(lam, z, thr))
print(f"{len(res)} vertices, n={n}, threshold={thr}, {time.time()-t0:.0f}s")
for h in ("TTTTTTTT", "RRRRRRRR", "EEEEEEEE"):
    print(f"homogeneous {h[0]}: Pf={res[h]:.4f}")
srt = sorted(res.items(), key=lambda kv: kv[1])
print("lowest 5:", [(k, round(v, 4)) for k, v in srt[:5]])
print("highest 5:", [(k, round(v, 4)) for k, v in srt[-5:]])
json.dump({"n": n, "threshold": thr, "pf": res}, open(f"ex2_vertices_pf_n{n}_thr{thr}.json", "w"))
