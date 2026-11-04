"""Ex3 against Dumas (2014), Tab. 3.7: plane + two pins, parameter sets 1-3.

Exact MILP slack (no surrogate) on Monte Carlo samples of model3 with the three
parameter sets of Dumas. Reports Pf = P(slack < 0) with a binomial 95% interval.
Usage: python scripts/reference_checks/dumas_model3_pf.py [n_samples] [param_set ...]
"""
import sys, time
from multiprocessing import Pool
import numpy as np
import openturns as ot
import otaf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "baselines"))
from baseline_cases import solve_slack  # noqa: E402

model = otaf.example_models.model3
SOC = None

def work(args):
    global SOC
    ps, seed, n = args
    if SOC is None:
        SOC = model.get_system_of_constraints_assembly_model()
    dist, *_ = model.get_distribution_params(param_set=ps)
    ot.RandomGenerator.SetSeed(seed)
    x = np.array(dist.getSample(n)) @ np.asarray(model.sample_multiplier).T
    s, st = solve_slack(SOC, x)
    return int((s[st == 0] < 0).sum()), int((st == 0).sum()), int((st != 0).sum())

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    sets = [int(a) for a in sys.argv[2:]] or [1, 2, 3]
    nproc, chunk = 4, 5000
    for ps in sets:
        t = time.time()
        jobs = [(ps, 1000 * ps + k, chunk) for k in range(n // chunk)]
        with Pool(nproc) as p:
            res = p.map(work, jobs)
        f = sum(r[0] for r in res); m = sum(r[1] for r in res); bad = sum(r[2] for r in res)
        pf = f / m; hw = 1.96 * np.sqrt(pf * (1 - pf) / m)
        print(f"param_set={ps}: Pf={pf:.3e} +- {hw:.1e} (95%), N={m}, solver failures={bad}, {time.time()-t:.0f}s", flush=True)
