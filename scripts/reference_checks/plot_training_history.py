"""Loss history of the ReLU surrogates, read from the checkpoints (Ex2-4) and the Ex1 study results.
Writes scripts/training_history/TrainingMLP{4,16,30,50}Dim.png and prints the last values.
Usage (from scripts/): python reference_checks/plot_training_history.py
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "training_history"
OUT.mkdir(exist_ok=True)
h4 = np.load(HERE / "ex1_surrogate_study/results/F_relu16_credal_200k_s2_hist.npy")
h4[:, 2] = 1 - h4[:, 2]  # the study stores the validation R2 of the normalised slack
hist = {4: h4}
for dim, name in ((16, "model2_16_dof"), (30, "model3_30_dof"), (50, "model4_50_dof")):
    ck = torch.load(HERE / f"artifacts/{name}_surrogate.pth", map_location="cpu", weights_only=False)
    hist[dim] = np.array(ck["history"])
plt.rcParams.update({"font.family": "serif", "font.size": 11})
for dim, h in sorted(hist.items()):
    print(f"{dim} variables: shape {h.shape}, first {h[0]}, last {h[-1]}, min val {h[:, 2].min():.3e}")
    fig, ax = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
    ax.plot(h[:, 0], h[:, 1], lw=1.2, color="#1f6fb2", label="Training")
    ax.plot(h[:, 0], h[:, 2], lw=1.2, color="#c8502d", label="Validation")
    ax.set_yscale("log"); ax.set_xlabel("Step"); ax.set_ylabel("MSE (normalised slack)")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4)); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    fig.savefig(OUT / f"TrainingMLP{dim}Dim.png", dpi=200)
    plt.close(fig)
