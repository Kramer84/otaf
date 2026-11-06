"""Figures of the report. Usage: python ex1_surrogate_study/figures.py BEST_TAG"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import ex1lib as L
best = sys.argv[1]; R = L.HERE / "results"; F = L.HERE / "figures"
torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(0.3)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C = {"base": "#c0392b", "best": "#2a78d6", "ex": "#333333"}
nets = {"Current surrogate (GELU 16-8-4)": (L.load_ckpt(L.SCRIPTS / "artifacts/model1_4_dof_surrogate.pth"), C["base"]), "Recommended (ReLU 16-8-4, credal mixture)": (L.load_ckpt(L.HERE / "ckpt" / f"{best}.pth"), C["best"])}
# 1. Slack across the kink at u4+u5 = 0.2 (where s = 0 on the ridge)
d = torch.linspace(-0.012, 0.012, 2001, device=L.DEV)
x = torch.stack([torch.full_like(d, .1), d / 2, torch.full_like(d, .1), -d / 2], 1)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
ax[0].plot(d.cpu(), L.closed_form(x).cpu(), color=C["ex"], lw=2.5, alpha=.35, label="Exact")
with torch.no_grad():
    for k, (n, c) in nets.items():
        ax[0].plot(d.cpu(), n(x).cpu(), color=c, lw=1.2, label=k)
        ax[1].plot(d.cpu(), (n(x) - L.closed_form(x)).cpu(), color=c, lw=1.2, label=k)
ax[0].set(xlabel="g4 - g5", ylabel="slack s (mm)", title="Slack across the kink, u4 = u5 = 0.1")
ax[1].set(xlabel="g4 - g5", ylabel="surrogate - exact (mm)", title="Error across the kink"); ax[1].axhline(0, color="#999", lw=.6)
fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=7.5, loc="outside lower center", ncols=3)
fig.savefig(F / "kink_profile.png", dpi=200); plt.close(fig)
# 2. Pf surfaces at alpha=0, shared scale
tb = L.TestBed()
Q = tb.exact[0]
Ps = [("Exact (quadrature)", Q), ("Current surrogate", np.load(R / "B0_nov_checkpoint_surface.npy")[0]), ("Recommended", np.load(R / f"{best}_surface.npy")[0])]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True, sharey=True)
for a, (t, P) in zip(ax, Ps):
    im = a.imshow(P * 1e3, origin="lower", extent=(0, 1, 0, 1), vmin=3, vmax=6.4, cmap="viridis"); a.set(title=t, xlabel=r"$\lambda_{d,1}$")
ax[0].set_ylabel(r"$\lambda_{d,2}$"); fig.colorbar(im, ax=ax, label=r"$P_f \times 10^3$, $\alpha=0$", shrink=.9)
fig.savefig(F / "pf_surfaces.png", dpi=200); plt.close(fig)
# 3. Bias versus distance to the kink at lambda = 1 (comparable with ex1_grid_large_analysis.py)
z = tb.z32[:2_000_000]
ex = L.closed_form(z); kink = 5 * torch.abs(z[:, 1] - z[:, 3])
print("bias vs x = 5|g4-g5|, lambda = 1 (not on the credal set): bin, n, then one bias per model")
with torch.no_grad():
    bias = {k: n(z) - ex for k, (n, c) in nets.items()}
for lo, hi in [(0, .002), (.002, .005), (.005, .01), (.01, .02), (.02, .05), (.05, .1), (.1, .2), (.2, 1)]:
    m = (kink >= lo) & (kink < hi)
    print(f"  [{lo},{hi}) n={int(m.sum()):8d} " + " ".join(f"{float(b[m].mean()):+.2e}" for b in bias.values()))
