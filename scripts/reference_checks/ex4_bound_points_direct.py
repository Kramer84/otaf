"""Ex4: direct sampling on the surrogate at the GLD bound points of the current Summary files, on 5e6 points.
Usage (from scripts/): python reference_checks/ex4_bound_points_direct.py
"""
import re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, Path(__file__).parents[1].as_posix())
import otaf, imprecise_bounds_analysis as IB

def matrix(text, tag):
    b = re.search(r"x\}_\{\\%s\} = \\begin\{bmatrix\}(.*?)\\end" % tag, text, re.S).group(1)
    return np.array(re.findall(r"\{(\d+\.\d*)\s*\}", b), float)

m = otaf.example_models.model4
sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint("artifacts/model4_50_dof_surrogate.pth")
dist, sym, ms, mu = m.get_distribution_params()
N, CH = 5_000_000, 500_000
for name, idx in (("Summary_model4_50_dof.md", np.arange(50)), ("Summary_model4_50_dof_reduced.md", np.arange(6, 38))):
    t = Path(name).read_text()
    for tag in ("min", "max"):
        v = matrix(t, tag); assert v.size == idx.size, v.size
        x = np.zeros(50); x[idx] = v
        otaf.common.set_seed(7); n = 0; mn = np.inf
        for _ in range(N // CH):
            s = np.asarray(IB.get_model_evaluator(np.array(dist.getSample(CH)), mu, sur)(x)).ravel()
            n += int((s < 0).sum()); mn = min(mn, s.min())
        print(f"{name} {tag}: failed {n} of {N} -> {n / N:.3e}; lowest slack {mn:.4f}", flush=True)
