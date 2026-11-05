"""Ex4: does the maximisation of Pf (alpha = 0) find a higher value when started from the classical allocations?

Same objective, constraints and COBYQA settings as imprecise_bounds_analysis.py (surrogate + GLD, 1e5 samples,
half-space constraint c(x) <= 0), but x0 = pure rotation R / equal share E / warm start instead of the single warm
start. Usage: python scripts/reference_checks/ex4_multistart_check.py [reduced] [maxiter]
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
sys.path.insert(0, Path(__file__).parents[1].as_posix()); sys.path.insert(0, Path(__file__).parent.as_posix())
import otaf, imprecise_bounds_analysis as IB
import classical_allocations_ex3_ex4 as C

reduced = len(sys.argv) > 1 and sys.argv[1] == "reduced"
maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
TOL = 1e-3 if reduced else 1e-2  # feasibility tolerance of the analysis (strict-tol)
model = otaf.example_models.model4
sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint("artifacts/model4_50_dof_surrogate.pth")
dist, sym, ms, mu = model.get_distribution_params()
otaf.common.set_seed(0); smp = np.array(dist.getSample(100_000))
base = IB.get_model_evaluator(smp, mu, sur)
act = list(range(6, 38)) if reduced else list(range(50))
def full(xr):
    x = np.zeros(50); x[act] = xr; return x
ev = lambda xr: base(full(xr))
cons_full = model.get_scaled_credal_set_constraints_function(ms)
vecs = C.ex4_vectors(model, ms, reduced)
gld = IB.make_gld("gpu")
tracker = otaf.optimization.OptimizationTracker(bounds=Bounds(1e-9, 1.0), constraint_tolerance=1e-5, precision_decimals=8)
x_warm = IB.optimize_scaling_vector(lambda x: cons_full(full(x)), n_vars=len(act))
starts = {"warm": x_warm, "R": vecs["R"][0][act], "E": vecs["E"][0][act]}
for lab, x0 in starts.items():
    key = f"{lab}"
    con = NonlinearConstraint(lambda x: np.array(cons_full(full(x)))[:10 if not reduced else 10], -np.inf, 0.0, keep_feasible=False)
    f0 = IB.fit_gld_pf(gld, np.asarray(ev(x0)).ravel(), 0.0)[1]
    best = [f0 if np.array(cons_full(full(x0)))[:10].max() <= TOL else 0.0, x0]
    def obj(x):
        s = np.asarray(ev(x)).ravel(); _, pf = IB.fit_gld_pf(gld, s, 0.0)
        pf = pf if np.isfinite(pf) else float(np.mean(s < 0))
        feas = np.array(cons_full(full(x)))[:10].max() <= TOL and x.min() >= -1e-6 and x.max() <= 1 + 1e-6
        if feas and pf > best[0]: best[0], best[1] = pf, x.copy()
        return -np.log(1e-16 + pf)
    r = minimize(obj, x0, method="COBYQA", bounds=Bounds(1e-9, 1.0),
                 constraints=con, options={"maxiter": maxiter, "maxfev": maxiter, "feasibility_tol": 1e-5,
                                           "initial_tr_radius": np.sqrt(len(act)) / np.sqrt(2), "final_tr_radius": 0.001, "disp": False})
    cmax = np.array(cons_full(full(best[1]))).max()
    print(f"start {lab:5s}: Pf(GLD) {f0:.3e} -> best feasible (c<={TOL:g}) {best[0]:.3e}  (max constraint {cmax:+.1e}, {r.nfev} evals)", flush=True)
