"""Reference cases for measuring how much code changes move OTAF's results.

A baseline stores, for each case, the optimal slack ``s`` of every sample
(``s < 0`` means the assembly fails), so that two versions of the code can be
compared sample by sample: Pf, slack differences and reclassified samples.

Cases are small (a few thousand samples, scipy MILP only, no torch) so a
baseline can be produced on a laptop.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import openturns as ot

import otaf

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Case:
    name: str
    model: str  # attribute of otaf.example_models
    note: str = ""
    build_kwargs: dict = field(default_factory=dict)  # get_system_of_constraints_assembly_model
    dist_kwargs: dict = field(default_factory=dict)  # get_distribution_params
    multiplier: float = 1.0  # dispersion multiplier applied to every marginal
    size: int = 2000
    seed: int = 0


# Thesis dispersion multipliers per model (see tests/test_general_model_creation.py).
CASES = [
    Case("model1", "model1", "automatic pipeline, analytical reference exists"),
    Case("model1_x1.35", "model1", "automatic pipeline", multiplier=1.35),
    Case("model2", "model2", "automatic pipeline, female faces 10.2 vs male 10"),
    Case("model2_x1.21", "model2", "automatic pipeline", multiplier=1.21),
] + [
    Case(
        f"model2_hF{hF}_x1.21",
        "model2",
        note,
        build_kwargs={"hF": hF},
        dist_kwargs={"hF": hF},
        multiplier=1.21,
    )
    for hF, note in [
        (10.0, "female faces as high as male faces: facing points coincide (control)"),
        (10.24, "height difference just inside the 0.25 mm pairing radius"),
        (10.3, "height difference beyond the pairing radius: top contacts lost today"),
        (11.0, "height difference beyond the pairing radius: top contacts lost today"),
    ]
] + [
    Case("model3", "model3", "hand-coded matrices (control)"),
    Case("model3_x1.26", "model3", "hand-coded matrices (control)", multiplier=1.26),
    Case("model4", "model4", "hand-coded matrices (control)"),
    Case("model4_x1.15", "model4", "hand-coded matrices (control)", multiplier=1.15),
    Case("model4_auto", "model4_auto", "automatic pipeline, same assembly as model4 (plates perfect)"),
    Case("model4_auto_x1.15", "model4_auto", "automatic pipeline", multiplier=1.15),
    Case(
        "model4_auto_inscribed_x1.15",
        "model4_auto",
        "automatic pipeline, inscribed polygon (conservative)",
        build_kwargs={"strategy": "inscribed"},
        multiplier=1.15,
    ),
]


def _git_revision() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "src"], capture_output=True, text=True)
        return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def solve_slack(soc, deviations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal slack and ``scipy.optimize.milp`` status per sample.

    Same problem as ``otaf.uncertainty.milp_batch_sequential``. Kept separate so that
    the baselines do not depend on the library code they check: any non-optimal
    status (not only a missing solution) gives ``s = nan``.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    c, a_ub, b_ub, a_eq, b_eq, bounds = soc(deviations)
    bounds = Bounds(bounds[:, 0], bounds[:, 1], keep_feasible=False)
    slack = np.full(b_ub.shape[1], np.nan)
    status = np.zeros(b_ub.shape[1], dtype=int)
    for k in range(b_ub.shape[1]):
        res = milp(
            c=c,
            bounds=bounds,
            constraints=(
                LinearConstraint(a_ub, -np.inf, b_ub[:, k]),
                LinearConstraint(a_eq, b_eq[:, k], b_eq[:, k]),
            ),
            options={"disp": False, "presolve": True},
        )
        status[k] = res.status
        if res.status == 0:
            slack[k] = res.x[-1]
    return slack, status


def compute_case(case: Case, soc_cache: dict | None = None) -> dict:
    """Solve one case and return its slack per sample plus summary statistics."""
    model = getattr(otaf.example_models, case.model)
    key = (case.model, json.dumps(case.build_kwargs, sort_keys=True))
    if soc_cache is not None and key in soc_cache:
        soc = soc_cache[key]
    else:
        soc = model.get_system_of_constraints_assembly_model(**case.build_kwargs)
        if soc_cache is not None:
            soc_cache[key] = soc

    distribution, *_ = model.get_distribution_params(**case.dist_kwargs)
    if case.multiplier != 1.0:
        distribution = otaf.distribution.multiply_composed_distribution_standard_with_constants(
            distribution, [case.multiplier] * distribution.getDimension()
        )
    ot.RandomGenerator.SetSeed(case.seed)
    sample = np.array(distribution.getSample(case.size)) @ np.asarray(model.sample_multiplier).T

    start = time.perf_counter()
    slack, status = solve_slack(soc, sample)
    failed = status != 0
    return {
        "slack": slack,
        "pf": float(np.mean(slack[~failed] < 0)) if (~failed).any() else float("nan"),
        "n_solver_failures": int(failed.sum()),
        "solver_statuses": sorted({int(x) for x in status[failed]}),
        "n_interface": int(soc.nI),
        "n_compatibility": int(soc.nC),
        "solve_seconds": time.perf_counter() - start,
    }


def run(cases=CASES, verbose=True) -> dict:
    """Compute every case. Returns ``{case_name: result}``."""
    cache, results = {}, {}
    for case in cases:
        results[case.name] = compute_case(case, cache)
        if verbose:
            r = results[case.name]
            print(
                f"{case.name:22s} Pf={r['pf']:.4f}  nI={r['n_interface']:4d}  "
                f"failures={r['n_solver_failures']} {r['solver_statuses'] or ''} ({r['solve_seconds']:.1f}s)",
                flush=True,
            )
    return results


def save(results: dict, tag: str, cases=CASES) -> Path:
    """Write ``data/<tag>.npz`` (slacks) and ``data/<tag>.json`` (metadata and summaries)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / f"{tag}.npz", **{name: r["slack"] for name, r in results.items()})
    meta = {
        "tag": tag,
        "git_revision": _git_revision(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": {
            c.name: {**asdict(c), **{k: v for k, v in results[c.name].items() if k != "slack"}}
            for c in cases
            if c.name in results
        },
    }
    path = DATA_DIR / f"{tag}.json"
    path.write_text(json.dumps(meta, indent=2))
    return path


def load(tag: str) -> tuple[dict, dict]:
    """Return ``(metadata, {case_name: slack})`` for a saved baseline."""
    meta = json.loads((DATA_DIR / f"{tag}.json").read_text())
    with np.load(DATA_DIR / f"{tag}.npz") as data:
        slacks = {name: data[name] for name in data.files}
    return meta, slacks


def compare(reference: tuple[dict, dict], current: dict):
    """Per-case divergence between a saved baseline and freshly computed results."""
    import pandas as pd

    meta, slacks = reference
    rows = []
    for name, ref_slack in slacks.items():
        if name not in current:
            continue
        new = current[name]
        s = new["slack"]
        rows.append(
            {
                "case": name,
                "Pf_ref": meta["cases"][name]["pf"],
                "Pf_now": new["pf"],
                "dPf": new["pf"] - meta["cases"][name]["pf"],
                "failures_ref": meta["cases"][name]["n_solver_failures"],
                "failures_now": new["n_solver_failures"],
                "reclassified": int(np.sum((ref_slack < 0) != (s < 0))),
                "max_abs_ds": float(np.nanmax(np.abs(s - ref_slack))) if np.isfinite(s - ref_slack).any() else float("nan"),
                "nI_ref": meta["cases"][name]["n_interface"],
                "nI_now": new["n_interface"],
                "note": meta["cases"][name]["note"],
            }
        )
    return pd.DataFrame(rows).set_index("case")
