"""Ex1: Pf over the allocation grid with the surrogate, large sample, GLD against direct sampling.

On one common sample, for each grid point (lambda_d1, lambda_d2) and each alpha, four estimates of P(s < alpha):
direct sampling and GLD fit on the surrogate slack, direct sampling and GLD fit on the closed-form slack
s = (0.2 - (u4+u5) - 5|g4-g5|) / 2. This separates the error of the GLD from the error of the neural surrogate.
Also stores, per grid point, the error of the surrogate against the closed form (bias, RMSE, the same in the band
|s| < 0.02, fraction of samples on the wrong side of s = 0). Results are checkpointed after every row.
Usage: python scripts/reference_checks/ex1_grid_large.py [--grid-res 50] [--n 5000000] [--out NAME]
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from gldpy_gpu import gpu as gld_gpu
import otaf
from otaf.example_models import model1

HERE = Path(__file__).resolve().parents[1]
ALPHAS = np.array([0.0, 0.01, 0.025, 0.05, 0.075])


def closed_form_slack(x):
    """x = centred (u4, g4, u5, g5). Slack s = gap / 2 of the 2D model (checked against the LP with --check-lp)."""
    return (0.2 - (x[:, 0] + x[:, 2]) - 5.0 * np.abs(x[:, 1] - x[:, 3])) / 2.0


def gld_pf(gld, data):
    """L-moment VSL fit in double precision, then the CDF at every alpha."""
    t = torch.as_tensor(data, dtype=torch.float64, device=gld.device)
    p = gld.fit_LMM(t)
    out = [float(gld.CDF_num(float(a), p, xtol=1e-9)) for a in ALPHAS]
    return np.clip(out, 0, 1), p.detach().cpu().numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-res", type=int, default=50)
    ap.add_argument("--n", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--surrogate", default=str(HERE / "artifacts/model1_4_dof_surrogate.pth"))
    ap.add_argument("--out", default="grid50_n5e6")
    ap.add_argument("--check-lp", type=int, default=0, help="compare closed form and LP slack on this many samples, then exit")
    a = ap.parse_args()
    torch.set_num_threads(2)

    dist, _, _, mu = model1.get_distribution_params()
    otaf.common.set_seed(a.seed)
    z = np.array(dist.getSample(a.n)) - mu
    if a.check_lp:
        sys.path.insert(0, (HERE / "baselines").as_posix())
        from baseline_cases import solve_slack
        soc = model1.get_system_of_constraints_assembly_model()
        for lam in ([1, 0, 1, 0], [0, 1, 0, 1], [.6, .8, .8, .6]):
            x = z[: a.check_lp] * np.array(lam) + mu
            s_lp, st = solve_slack(soc, x @ np.asarray(model1.sample_multiplier).T)
            print(lam, "max |closed form - LP| =", np.abs(closed_form_slack(z[: a.check_lp] * np.array(lam)) - s_lp)[st == 0].max())
        return

    sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(a.surrogate)
    gld = gld_gpu.GLD("VSL")
    scales = np.linspace(0, 1, a.grid_res)
    out = HERE / "ex1_grid_5M" / f"{a.out}.npz"
    shape = (a.grid_res, a.grid_res, len(ALPHAS))
    R = dict(gld_sur=np.full(shape, np.nan), mc_sur=np.full(shape, np.nan), gld_exact=np.full(shape, np.nan), mc_exact=np.full(shape, np.nan),
             bias=np.full(shape[:2], np.nan), rmse=np.full(shape[:2], np.nan), bias_band=np.full(shape[:2], np.nan),
             rmse_band=np.full(shape[:2], np.nan), wrong_sign=np.full(shape[:2], np.nan), par_sur=np.full(shape[:2] + (4,), np.nan))
    start = 0
    if out.exists():
        with np.load(out) as d:
            R = {k: d[k] for k in R}
        start = int(np.argmax(np.isnan(R["mc_sur"][:, 0, 0]))) if np.isnan(R["mc_sur"]).any() else a.grid_res
        print("resuming at row", start, flush=True)
    t0 = time.time()
    for i in range(start, a.grid_res):        # row i: lambda_d2 = scales[i]; column j: lambda_d1 = scales[j]
        for j in range(a.grid_res):
            u4, u5 = scales[j], scales[i]
            lam = np.array([u4, (1 - u4**2) ** .5, u5, (1 - u5**2) ** .5])
            zz = z * lam
            s_sur = np.squeeze(sur.evaluate_model_non_standard_space(zz + mu, batch_size=1_000_000).numpy()).astype(np.float64)
            s_ex = closed_form_slack(zz)
            R["mc_sur"][i, j] = [(s_sur < al).mean() for al in ALPHAS]
            R["mc_exact"][i, j] = [(s_ex < al).mean() for al in ALPHAS]
            R["gld_sur"][i, j], R["par_sur"][i, j] = gld_pf(gld, s_sur)
            R["gld_exact"][i, j], _ = gld_pf(gld, s_ex)
            e = s_sur - s_ex
            band = np.abs(s_ex) < 0.02
            R["bias"][i, j], R["rmse"][i, j] = e.mean(), (e**2).mean() ** .5
            R["bias_band"][i, j], R["rmse_band"][i, j] = e[band].mean(), (e[band] ** 2).mean() ** .5
            R["wrong_sign"][i, j] = ((s_sur < 0) != (s_ex < 0)).mean()
        np.savez(out, scales=scales, alphas=ALPHAS, n=a.n, seed=a.seed, surrogate=a.surrogate, **R)
        print(f"row {i+1}/{a.grid_res}  {(time.time()-t0)/(i+1-start):.1f} s/row", flush=True)


if __name__ == "__main__":
    main()
