"""Ex1: Pf(alpha = 0) over the allocation (lambda_u4, lambda_u5), 0 = pure rotation, 1 = pure translation.

Left: exact (quadrature of the closed form; Pf depends on the allocation only through A = lambda_u4^2 + lambda_u5^2, so a
1-D table is interpolated). Middle and right: surrogate + GLD grids used in the thesis (Nov. 2) and rerun with gldpy-gpu 0.4.0.
Usage: python scripts/reference_checks/ex1_pf_surface.py OUTPUT.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, Path(__file__).parent.as_posix())
import ex1_analytical_pf as E

out = sys.argv[1]
ART = Path(__file__).parents[1] / "artifacts"
As = np.linspace(0, 2, 161)
tab = []
for A in As:
    u4 = min(1.0, (A / 2) ** .5); u5 = min(1.0, max(A - u4**2, 0) ** .5)
    tab.append(E.pf((u4, (1 - u4**2) ** .5), (u5, (1 - u5**2) ** .5)))
tab = np.array(tab)
u = np.linspace(0, 1, 101); U4, U5 = np.meshgrid(u, u)
P = np.interp(U4**2 + U5**2, As, tab)
i = tab.argmax()
print(f"exact: min {tab.min():.4e} at A={As[tab.argmin()]:.3f}; max {tab.max():.4e} at A={As[i]:.3f}")
for a in (0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2):
    print(f"  A={a:<4} Pf={np.interp(a, As, tab):.4e}")

def grid(name):
    d = np.load(ART / name, allow_pickle=True)
    return d["U4_grid"], d["U5_grid"], d["PF"][list(np.round(d["alphas"], 6)).index(0.0)]

panels = [("Exact (analytical model)", U4, U5, P),
          ("Surrogate + GLD, grid behind the thesis figure", *grid("pf_model1_50x50_mc1M.npz")),
          ("Surrogate + GLD, rerun with gldpy-gpu 0.4.0", *grid("pf_model1_50x50_mc1M_rerun.npz"))]
vmax = max(p[3].max() for p in panels) * 1e3
fig = plt.figure(figsize=(15, 5.2), dpi=170, facecolor="white")
for k, (title, X, Y, Z) in enumerate(panels):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d", facecolor="white")
    ax.plot_surface(X, Y, Z * 1e3, cmap="Blues", vmin=-1.5, vmax=vmax, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=.97)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, vmax)
    ax.set_xlabel(r"$\lambda_{u,4}$  (0 rotation, 1 translation)", fontsize=8.5, labelpad=4)
    ax.set_ylabel(r"$\lambda_{u,5}$  (0 rotation, 1 translation)", fontsize=8.5, labelpad=4)
    ax.set_zlabel(r"$P_f \times 10^{3}$", fontsize=9)
    ax.tick_params(labelsize=8, pad=0); ax.view_init(elev=24, azim=-125)
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.set_pane_color((1, 1, 1, 1)); p._axinfo["grid"]["color"] = (.88, .88, .88, 1)
    ax.set_title(title, fontsize=10.5, color="#222222")
    for (i0, j0, lab) in ((0, 0, "RR"), (-1, -1, "TT")):
        v = Z[i0, j0] * 1e3
        ax.scatter([X[i0, j0]], [Y[i0, j0]], [v], s=26, color="#111111", depthshade=False, zorder=10)
        ax.text(X[i0, j0], Y[i0, j0], v + vmax * .05, f"{lab} {v:.2f}", fontsize=8.5, color="#111111", zorder=11)
fig.suptitle(r"Example 1: probability of failure at $\alpha=0$ over the allocation of the two features", fontsize=12.5, color="#222222", y=.99)
fig.tight_layout(rect=(0, 0, 1, .97))
fig.savefig(out, facecolor="white")
for t, X, Y, Z in panels:
    print(f"{t}: RR {Z[0,0]:.3e}  TT {Z[-1,-1]:.3e}  min {Z.min():.3e}  max {Z.max():.3e}")
