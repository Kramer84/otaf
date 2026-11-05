"""Loss history of the surrogates of 3 Nov 2026, read from the training logs (last batch of each epoch).
Writes scripts/training_history/TrainingMLP{4,16,30,50}Dim.png and prints the last value of each model.
Usage (from scripts/): python reference_checks/plot_training_history.py
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "training_history"
OUT.mkdir(exist_ok=True)
hist, cur = {}, None
for log in ("step2_model12.log", "step2_model3.log", "step2_model4.log"):
    text = (HERE / "rerun_logs" / log).read_text(errors="ignore").replace("\r", "\n")
    for line in text.split("\n"):
        m = re.search(r"\[(model\d+_(\d+)_dof)\]", line)
        if m:
            cur = int(m.group(2)); hist.setdefault(cur, {})
        m = re.search(r"Epoch (\d+):.*loss=([0-9.e-]+)", line)
        if m and cur is not None:
            hist[cur][int(m.group(1))] = float(m.group(2))
plt.rcParams.update({"font.family": "serif", "font.size": 11})
for dim, h in sorted(hist.items()):
    ep = sorted(h)
    print(f"{dim} variables: {ep[-1] + 1} epochs, last loss {h[ep[-1]]:.3e}, min {min(h.values()):.3e}")
    fig, ax = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
    ax.plot(ep, [h[e] for e in ep], lw=1.2, color="#1f6fb2")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); ax.grid(alpha=.25)
    fig.savefig(OUT / f"TrainingMLP{dim}Dim.png", dpi=200)
    plt.close(fig)
