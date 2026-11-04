"""Validity of the neural surrogates over the credal set, against the exact MILP slack.

For each example and several allocations (all lambda = 1, all translation T, all rotation R, equal share E; for
Ex3 only lambda = 1) a fresh sample is drawn (seed 123, independent of the training and of the optimisation
samples) and the surrogate slack is compared with the linear-program slack: bias and RMSE of the slack, the same in
the band |s| < 0.02 around the limit state, the fraction of samples on the wrong side of s = 0 and, in the last
column, the Pf from the surrogate and from the linear program on the same sample.
Usage: python scripts/reference_checks/surrogate_validity.py [n_ex12] [n_ex3] [n_ex4]
"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, Path(__file__).parents[1].as_posix()); sys.path.insert(0, Path(__file__).parents[1].as_posix() + "/baselines")
import otaf
import imprecise_bounds_analysis as IB
import classical_allocations_ex3_ex4 as C
from baseline_cases import solve_slack

NS = [int(a) for a in sys.argv[1:4]] if len(sys.argv) > 3 else [20000, 10000, 4000]
FILES = {"model1": "model1_4_dof", "model2": "model2_16_dof", "model3": "model3_30_dof", "model4": "model4_50_dof"}
_soc = {}

def milp_chunk(args):
    name, x = args
    if name not in _soc:
        _soc[name] = getattr(otaf.example_models, name).get_system_of_constraints_assembly_model()
    return solve_slack(_soc[name], x)[0]

def allocations(name):
    if name in ("model1", "model2"):
        n = 2 if name == "model1" else 8
        return {k: np.ravel([v] * n) for k, v in
                {"T": (1.0, 0.0), "R": (0.0, 1.0), "E": (2 ** -.5, 2 ** -.5)}.items()}
    if name == "model3":
        return {"all lambda=1": np.ones(30)}
    model = otaf.example_models.model4
    ms = model.get_distribution_params()[2]
    v = C.ex4_vectors(model, ms, False)
    return {k: v[k][0] for k in "TRE"}

def main():
    print(f"{'example / allocation':26s} {'N':>6s} {'bias':>9s} {'RMSE':>8s} {'RMSE|s|<.02':>11s} {'bias|s|<.02':>11s} {'wrong sign':>10s} {'Pf sur':>9s} {'Pf LP':>9s}")
    for name, n in zip(["model1", "model2", "model3", "model4"], NS[:1] + NS[:1] + NS[1:2] + NS[2:3]):
        model = getattr(otaf.example_models, name)
        dist, sym, ms, mu = model.get_distribution_params()
        sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(f"artifacts/{FILES[name]}_surrogate.pth")
        otaf.common.set_seed(123)
        smp = np.array(dist.getSample(n))
        ev = IB.get_model_evaluator(smp, mu, sur)
        for lab, lam in allocations(name).items():
            x = (smp - mu) * lam + mu
            s_sur = np.asarray(ev(lam)).ravel()
            xm = x @ np.asarray(model.sample_multiplier).T
            chunks = np.array_split(xm, 28)
            with Pool(4) as p:
                s_lp = np.concatenate(p.map(milp_chunk, [(name, c) for c in chunks]))
            ok = np.isfinite(s_lp); e = (s_sur - s_lp)[ok]; b = np.abs(s_lp[ok]) < 0.02
            print(f"{name + ' ' + lab:26s} {ok.sum():6d} {e.mean():+9.2e} {np.sqrt((e**2).mean()):8.2e} {np.sqrt((e[b]**2).mean()):11.2e} "
                  f"{e[b].mean():+11.2e} {np.mean((s_sur[ok] < 0) != (s_lp[ok] < 0)):10.2e} {np.mean(s_sur[ok] < 0):9.2e} {np.mean(s_lp[ok] < 0):9.2e}", flush=True)

if __name__ == "__main__":
    main()
