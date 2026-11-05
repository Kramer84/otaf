"""Ex1: thesis-style Pf maps from the large grid, GLD and direct sampling, at every alpha.
Usage (from scripts/): python reference_checks/ex1_grid_large_images.py [may|nov]  (default nov: surrogates of 3 Nov 2026)
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())
import plot_pf_2d_4dof_model as P

HERE = Path(__file__).resolve().parents[1] / "ex1_grid_5M"
TAG = sys.argv[1] if len(sys.argv) > 1 else "nov"
d = np.load(HERE / ("grid50_n5e6_may.npz" if TAG == "may" else "grid50_n5e6.npz"))
sc = d["scales"]
U, V = np.meshgrid(sc, sc)
for key, name in (("gld_sur", "gld"), ("mc_sur", "sampling")):
    out = HERE / (f"thesis_style_{name}" if TAG == "may" else f"thesis_style_{name}_nov")
    out.mkdir(exist_ok=True)
    for k, a in enumerate(d["alphas"]):
        fig, ax = P.plot_single_pf_grid(U, V, d[key][:, :, k], alpha=float(a), norm_type="power", n_levels=200, highlight_pct=0.005,
                                        highlight_mode="value", base_cmap="viridis", log_min_floor=1e-5, save=True,
                                        save_path=str(out / f"Pf_Grid_alpha_{a}.png"), dpi=300, show=False, usetex=True, transparent=False)
        P.plt.close(fig)
print("images done")
