"""Ex1: exact surface by quadrature at each alpha, error distributions of the large grid, and where the surrogate errs.

Writes ex1_grid_5M/exact_quadrature.npz, prints the numbers quoted in the thesis notes, and draws the figures
(comparison of estimators at alpha = 0, distribution of the errors).
Usage: python scripts/reference_checks/ex1_grid_large_analysis.py [may|nov]  (default nov: surrogates of 3 Nov 2026)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, Path(__file__).parent.as_posix())
import ex1_analytical_pf as E
import otaf
from otaf.example_models import model1

HERE = Path(__file__).resolve().parents[1] / "ex1_grid_5M"
TAG = sys.argv[1] if len(sys.argv) > 1 else "nov"
d = np.load(HERE / ("grid50_n5e6_may.npz" if TAG == "may" else "grid50_n5e6.npz"))
SUFFIX = "_may" if TAG == "may" else "_nov"
sc, al = d["scales"], d["alphas"]
qf = HERE / "exact_quadrature.npz"
if qf.exists():
    Q = np.load(qf)["pf"]
else:
    # Pf depends on the allocation only through (lambda_d1, lambda_d2); row i = lambda_d2, column j = lambda_d1
    Q = np.array([[[E.pf((u4, (1 - u4**2) ** .5), (u5, (1 - u5**2) ** .5), E.C0 - 2 * a) for u4 in sc] for u5 in sc] for a in al])
    np.savez(qf, pf=Q, scales=sc, alphas=al)

print("Exact surface (quadrature)")
for k, a in enumerate(al):
    g = Q[k]; i = np.unravel_index(g.argmax(), g.shape)
    flat = (g >= g.max() - 0.005 * (g.max() - g.min())).mean()
    print(f"alpha={a:<6} TT {g[-1,-1]:.4e} RR {g[0,0]:.4e} max {g.max():.4e} at l1={sc[i[1]]:.2f},l2={sc[i[0]]:.2f}  RR/max {g[0,0]/g.max():.4f}  share of grid within 0.5% of range below max {flat:.2f}")

print("\nRelative errors over the 2500 grid points (surrogate: " + TAG + "; n=5e6)")
pairs = {"GLD on surrogate vs sampling on surrogate": ("gld_sur", "mc_sur"), "GLD on closed form vs sampling on closed form": ("gld_exact", "mc_exact"),
         "sampling on surrogate vs sampling on closed form": ("mc_sur", "mc_exact"), "GLD on surrogate vs sampling on closed form": ("gld_sur", "mc_exact")}
for lab, (x, y) in pairs.items():
    e = d[x][:, :, 0] / d[y][:, :, 0] - 1
    print(f"{lab:50s} med {np.median(e):+.3f} p5 {np.percentile(e,5):+.3f} p95 {np.percentile(e,95):+.3f} min {e.min():+.3f} max {e.max():+.3f} share|e|>5% {np.mean(np.abs(e)>.05):.3f}")

# Where does the surrogate err: bias as a function of the distance to the kink |g4-g5| (all components active, lambda=1)
sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(str(HERE.parent / ("artifacts/old_may2026/model1_4_dof_surrogate.pth" if TAG == "may" else "artifacts/model1_4_dof_surrogate.pth")))
dist, _, ms, mu = model1.get_distribution_params()
otaf.common.set_seed(5)
z = np.array(dist.getSample(2_000_000)) - mu
s = np.squeeze(sur.evaluate_model_non_standard_space(z + mu, batch_size=1_000_000).numpy()).astype(float)
ex = (0.2 - z[:, 0] - z[:, 2] - 5 * np.abs(z[:, 1] - z[:, 3])) / 2
kink = 5 * np.abs(z[:, 1] - z[:, 3])
print("\nSurrogate error vs distance to the kink, x = 5|g4-g5| (mm), lambda = 1")
for lo, hi in [(0, .002), (.002, .005), (.005, .01), (.01, .02), (.02, .05), (.05, .1), (.1, .2), (.2, 1)]:
    m = (kink >= lo) & (kink < hi)
    print(f"  x in [{lo},{hi}): n={m.sum():8d} bias {(s-ex)[m].mean():+.2e}")

# Figures
plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True, constrained_layout=True)
panels = [("Exact (quadrature)", Q[0]), ("Sampling, closed-form slack", d["mc_exact"][:, :, 0]), ("Sampling, surrogate", d["mc_sur"][:, :, 0]), ("GLD, surrogate", d["gld_sur"][:, :, 0])]
for a, (t, g) in zip(ax, panels):
    im = a.pcolormesh(sc, sc, g, vmin=1.4e-3, vmax=6.5e-3, cmap="viridis", shading="auto")
    a.set_title(t); a.set_xlabel(r"$\lambda_{d,1}$"); a.set_aspect("equal")
ax[0].set_ylabel(r"$\lambda_{d,2}$")
fig.colorbar(im, ax=ax, label=r"$P_f$ at $\alpha=0$", shrink=.9)
fig.savefig(HERE / f"ex1_estimators_alpha0{SUFFIX}.png", dpi=250)

fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
bins = np.linspace(-.6, .2, 81)
for (lab, (x, y)), c in zip(list(pairs.items())[:3], ("#1f6fb2", "#d9822b", "#2f9e6f")):
    e = d[x][:, :, 0] / d[y][:, :, 0] - 1
    ax.hist(e.ravel(), bins=bins, histtype="step", lw=1.6, color=c, label=lab)
ax.set_yscale("log"); ax.set_xlabel(r"relative error on $P_f$ at $\alpha=0$"); ax.set_ylabel("grid points (of 2500)"); ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=.25)
fig.savefig(HERE / f"ex1_error_distribution_alpha0{SUFFIX}.png", dpi=250)
