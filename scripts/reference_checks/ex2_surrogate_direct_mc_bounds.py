"""Ex2: bounds of Pf over the credal set with the surrogate and direct Monte Carlo (no GLD).

Objective = fraction of a fixed (common random numbers) sample of 10^6 points whose surrogate slack is
below the threshold; phi in [0, pi/2]^8 gives (lambda_u, lambda_g) = (cos phi, sin phi) per feature.
COBYQA from the all-translation / equal-share / all-rotation allocations. Optima are re-evaluated on an
independent surrogate sample and with the exact closed-form slack. Usage: python ... [threshold]
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, Bounds
sys.path.insert(0, Path(__file__).parents[1].as_posix()); sys.path.insert(0, Path(__file__).parent.as_posix())
import otaf, imprecise_bounds_analysis as IB
import analytical_16dof_vs_model2 as A

thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
m = otaf.example_models.model2
dist, sym, ms, mu = m.get_distribution_params()
sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint("artifacts/model2_16_dof_surrogate.pth")
otaf.common.set_seed(1); s_opt = np.array(dist.getSample(1_000_000)); ev = IB.get_model_evaluator(s_opt, mu, sur)
otaf.common.set_seed(2); s_chk = np.array(dist.getSample(1_000_000)); ev_c = IB.get_model_evaluator(s_chk, mu, sur)
zx = A.standard_sample(2_000_000, 123)
lam = lambda phi: A.boundary_lambda(phi)
f = lambda phi: float(np.mean(np.asarray(ev(lam(phi))).ravel() < thr))
starts = {"T": np.zeros(8), "E": np.full(8, np.pi / 4), "R": np.full(8, np.pi / 2)}
for goal, sign, keys in (("min", 1.0, "TE"), ("max", -1.0, "ER")):
    best = None
    for k in keys:
        r = minimize(lambda p: sign * f(p), starts[k], method="COBYQA", bounds=Bounds(0, np.pi / 2), options={"maxfev": 400})
        print(f"{goal} from {k}: {f(starts[k]):.4e} -> {f(r.x):.4e}  ({r.nfev} evals)", flush=True)
        if best is None or sign * f(r.x) < sign * f(best): best = r.x
    pc = float(np.mean(np.asarray(ev_c(lam(best))).ravel() < thr))
    print(f"== {goal}: optimisation sample {f(best):.4e} | independent surrogate sample {pc:.4e} | exact {A.pf(lam(best), zx, thr):.4e}")
    print("   phi (deg):", np.round(np.degrees(best), 1).tolist(), flush=True)
