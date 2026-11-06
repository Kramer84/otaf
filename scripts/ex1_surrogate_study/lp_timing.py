"""Time of the LP per sample, for comparison with the surrogate evaluation (one process)."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1] / "baselines"))
import ex1lib as L
from otaf.example_models import model1
from baseline_cases import solve_slack
soc = model1.get_system_of_constraints_assembly_model()
x = np.random.default_rng(1).standard_normal((2000, 4)) * L.SIGMA
t = time.time(); solve_slack(soc, x @ np.asarray(model1.sample_multiplier).T); d = time.time() - t
print(f"LP: {d/2000*1e3:.3f} ms per sample, {d/2000*1e6:.0f} s per 1e6 samples on one core")
