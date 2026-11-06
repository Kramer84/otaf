"""Ex1: thesis-style Pf maps by direct sampling on the selected ReLU surrogate, from the surface already computed by the study.
Usage (from scripts/): python reference_checks/ex1_relu_maps.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())
import plot_pf_2d_4dof_model as P

HERE = Path(__file__).resolve().parents[1]
S = np.load(HERE / "ex1_surrogate_study/results/F_relu16_credal_200k_s2_surface.npy")
sc = np.load(HERE / "ex1_grid_5M/grid50_n5e6.npz")["scales"]
U, V = np.meshgrid(sc, sc)
out = HERE / "ex1_grid_5M/thesis_style_sampling_relu"
out.mkdir(exist_ok=True)
for k, a in enumerate((0.0, 0.01, 0.025, 0.05, 0.075)):
    fig, ax = P.plot_single_pf_grid(U, V, S[k], alpha=a, norm_type="power", n_levels=200, highlight_pct=0.005, highlight_mode="value",
                                    base_cmap="viridis", log_min_floor=1e-5, save=True, save_path=str(out / f"Pf_Grid_alpha_{a}.png"),
                                    dpi=300, show=False, usetex=True, transparent=False)
    P.plt.close(fig)
print("maps done")
