"""Verification of OTAF against published and stored reference values.

Three groups of cases, each comparing a reference value with a fresh OTAF run:

- ``dumas``: ``model3`` against the assembly failure probabilities published by
  Dumas (2014, PhD thesis, tables 3.3, C.3 and C.4) for the same mechanism.
- ``thesis``: the bounds of the failure probability in ``scripts/Summary_*.md``
  (surrogate + GLD fit, used in the thesis), recomputed with the scipy solver at
  the same variance allocations.
- ``baselines``: the reference cases of ``scripts/baselines`` against the latest
  stored baseline, sample by sample.

``run`` computes everything (a few minutes, scipy only), ``render_tables``
writes the tables of ``docs/guide/verification.md`` from the saved results.
Usage: ``python scripts/verification/run_verification.py``.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import openturns as ot

import otaf

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
RESULTS_PATH = ROOT / "docs" / "guide" / "verification_results.json"
TABLES_PATH = ROOT / "docs" / "guide" / "_verification_tables.md"
BASELINE_TAG = "plate-capability"
SEED = 0
Z_95 = 1.959964

# --- Dumas (2014) ---------------------------------------------------------------
# Assembly failure probability Pda of the "appui plan et deux goupilles" mechanism
# (Appendix C, the mechanism of model3), by parameter set, number of polygon sides Nd
# and linearization strategy. Each set was computed once with N_MC samples; "ci95" is
# the published half-width of the 95 % interval of the Monte Carlo estimate.
DUMAS_2014 = {
    1: {
        "table": "C.3",
        "n_mc": 30_000,
        "ci95": 0.54e-2,
        "pf": {
            4: {"inscribed": 8.37e-2, "mean": 6.42e-2, "circumscribed": 5.41e-2},
            8: {"inscribed": 6.47e-2, "mean": 6.17e-2, "circumscribed": 5.92e-2},
            60: {"inscribed": 6.12e-2, "mean": 6.11e-2, "circumscribed": 6.11e-2},
        },
    },
    2: {
        "table": "3.3",
        "n_mc": 3_000_000,
        "ci95": 0.65e-4,
        "pf": {60: {"inscribed": 8.05e-4, "mean": 8.01e-4, "circumscribed": 7.96e-4}},
    },
    3: {
        "table": "C.4",
        "n_mc": 50_000_000,
        "ci95": 0.36e-5,
        "pf": {60: {"inscribed": 4.19e-5, "mean": 4.16e-5, "circumscribed": 4.13e-5}},
    },
}

# OTAF reruns of the Dumas cases: (parameter set, Nd, strategy, sample size).
# Set 3 (Pf ~ 4e-5) would need ~1e7 samples and is not rerun.
DUMAS_RUNS = [
    (1, nd, strategy, 4_000)
    for nd in (4, 8, 60)
    for strategy in ("inscribed", "mean", "circumscribed")
] + [(1, 60, "circumscribed", 20_000), (2, 60, "circumscribed", 20_000)]

# --- Thesis summaries --------------------------------------------------------------
# scripts/Summary_<name>.md, written by scripts/analyze_results.py from the
# optimization results OptimizationResults_<name>.csv. Reduced runs optimize a subset
# of the variance scalings; the others stay at zero.
THESIS_SUMMARIES = {
    "model1_4_dof": {"model": "model1", "active": None},
    "model2_16_dof": {"model": "model2", "active": None},
    "model3_30_dof": {"model": "model3", "active": None},
    "model3_30_dof_reduced": {"model": "model3", "active": list(range(26))},
    "model4_50_dof": {"model": "model4", "active": None},
    "model4_50_dof_reduced": {"model": "model4", "active": list(range(6, 38))},
}
# Sample size of the scipy rerun per model; None: the reference is too small to be
# checked with a few thousand samples.
THESIS_RERUN_SIZE = {"model1": 20_000, "model2": 20_000, "model3": 4_000, "model4": 4_000}
MIN_CHECKABLE_PF = 1e-3
# model1 has a closed-form slack (tests/test_physical_invariants.py), cheap enough
# for a large sample at the low-probability bounds.
MODEL1_CLOSED_FORM_SIZE = 1_000_000


# --- Statistics --------------------------------------------------------------------


def wilson_interval(failures: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95 % Wilson score interval of a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = failures / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def z_score(pf: float, n: int, reference: float, reference_se: float = 0.0) -> float:
    """Difference to the reference in standard errors (binomial error at the reference)."""
    se = np.sqrt(reference * (1 - reference) / n + reference_se**2)
    return float((pf - reference) / se) if se > 0 else float("nan")


def _pf_record(slack: np.ndarray, status: np.ndarray | None = None) -> dict:
    solved = np.isfinite(slack) if status is None else (status == 0)
    n = int(solved.sum())
    failures = int((slack[solved] < 0).sum())
    low, high = wilson_interval(failures, n)
    return {
        "pf": failures / n if n else float("nan"),
        "failures": failures,
        "n": n,
        "solver_failures": int((~solved).sum()),
        "ci95": [low, high],
    }


# --- Models and sampling -------------------------------------------------------------


def build_settings(model_name: str, **build_kwargs) -> dict:
    """Linearization settings of a model build (defaults of the model function, overridden)."""
    model = getattr(otaf.example_models, model_name)
    params = inspect.signature(model.get_system_of_constraints_assembly_model).parameters
    settings = {}
    for key in ("Nd", "CIRCLE_RESOLUTION", "circle_resolution", "strategy"):
        if key in params:
            value = build_kwargs.get(key, params[key].default)
            settings["circle_resolution" if key != "strategy" else "linearization"] = (
                str(getattr(value, "value", value)).lower()
            )
    return settings


def build_model(model_name: str, **build_kwargs):
    """SOC model of an example model, without the construction warnings."""
    logging.disable(logging.WARNING)
    try:
        return getattr(otaf.example_models, model_name).get_system_of_constraints_assembly_model(**build_kwargs)
    finally:
        logging.disable(logging.NOTSET)


def solve(soc, deviations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal slack and solver status per sample, through the library's batch solver."""
    result, status = otaf.uncertainty.compute_gap_optimizations_on_sample_batch(
        soc, deviations, n_cpu=1, batch_size=10_000, dtype="float64", progress_bar=False, return_status=True
    )
    return result[:, -1], np.asarray(status)


def scaled_sample(model_name: str, scaling: np.ndarray | None, size: int, seed: int = SEED, **dist_kwargs) -> np.ndarray:
    """Defect sample in the solver's variables, standard deviations scaled by ``scaling``.

    Same transformation as the thesis scripts: ``(x - mu) * scaling + mu`` on the
    model's distribution, then the model's ``sample_multiplier``.
    """
    model = getattr(otaf.example_models, model_name)
    distribution, _, _, mu = model.get_distribution_params(**dist_kwargs)
    ot.RandomGenerator.SetSeed(seed)
    sample = np.array(distribution.getSample(size))
    if scaling is not None:
        sample = (sample - mu) * scaling + mu
    return sample @ np.asarray(model.sample_multiplier).T


def model1_closed_form_slack(deviations: np.ndarray, X1: float = 99.8, X2: float = 100.0, X3: float = 10.0) -> np.ndarray:
    """Slack of model1 in closed form: half the smallest clearance at the two corners.

    Equal to OTAF's slack to 1e-7 on every sample (tests/test_physical_invariants.py).
    """
    u4, gamma4, u5, gamma5 = np.asarray(deviations).T
    y = X3 / 2
    clearance = np.minimum(
        (X2 - u4 - y * gamma4) - (X1 + u5 - y * gamma5),
        (X2 - u4 + y * gamma4) - (X1 + u5 + y * gamma5),
    )
    return clearance / 2


# --- Dumas (2014) cases ----------------------------------------------------------------


def dumas_case(param_set: int, nd: int, strategy: str, size: int, seed: int = SEED, soc_cache: dict | None = None) -> dict:
    """Assembly Pf of model3 for one Dumas (2014) parameter set, Nd and strategy."""
    ref = DUMAS_2014[param_set]
    reference = ref["pf"][nd][strategy]
    key = (nd, strategy)
    if soc_cache is not None and key in soc_cache:
        soc = soc_cache[key]
    else:
        soc = build_model("model3", Nd=nd, strategy=otaf.geometry.LinearizationStrategy(strategy))
        if soc_cache is not None:
            soc_cache[key] = soc
    start = time.perf_counter()
    slack, status = solve(soc, scaled_sample("model3", None, size, seed, param_set=param_set))
    rec = _pf_record(slack, status)
    reference_se = ref["ci95"] / Z_95
    return {
        "group": "dumas",
        "param_set": param_set,
        "table": ref["table"],
        "reference": reference,
        "reference_n": ref["n_mc"],
        "reference_ci95": ref["ci95"],
        **rec,
        "difference": rec["pf"] - reference,
        "z": z_score(rec["pf"], rec["n"], reference, reference_se),
        "settings": {"seed": seed, "size": size, "circle_resolution": str(nd), "linearization": strategy},
        "seconds": time.perf_counter() - start,
    }


# --- Thesis summaries ------------------------------------------------------------------


def parse_summary(name: str) -> dict:
    """Bounds and number of analyzed points written in ``scripts/Summary_<name>.md``."""
    text = (SCRIPTS / f"Summary_{name}.md").read_text()
    number = r"([0-9.]+e[+-]\d+)"
    return {
        "min": float(re.search(rf"Global Minimum Pf : {number}", text).group(1)),
        "max": float(re.search(rf"Global Maximum Pf : {number}", text).group(1)),
        "n_points": int(re.search(r"Valid Boundary Points Analyzed: (\d+)", text).group(1)),
    }


def recover_thesis_allocations(name: str) -> dict:
    """Variance scalings behind the min and max of a summary, from the optimization CSV.

    Repeats ``scripts/analyze_results.py``: keep the points on the boundary of the
    credal set (``strict_tol``, found from the number of points the summary
    analyzed), evaluate the fitted GLD at slack 0 and take the extremes. Raises if
    the recovered values differ from the summary.
    """
    sys.path.insert(0, str(SCRIPTS))
    import analyze_results
    from gldpy import GLD

    summary = parse_summary(name)
    csv = SCRIPTS / f"OptimizationResults_{name}.csv"
    for strict_tol in (5e-4, 1e-3, 5e-3, 1e-2):
        df = analyze_results.load_and_clean_data(csv, strict_tol=strict_tol)
        if len(df) == summary["n_points"]:
            break
    else:
        raise ValueError(f"{name}: no strict_tol gives the {summary['n_points']} points of the summary")

    gld = GLD("VSL")
    best = {"min": (np.inf, None), "max": (-np.inf, None)}
    for alloc in analyze_results.get_all_valid_allocations(df):
        try:
            pf = float(gld.CDF_num(0.0, list(alloc["gld_params"]), xtol=1e-6)[0])
        except Exception:  # same as analyze_results: allocations whose GLD fails are skipped
            continue
        if pf < best["min"][0]:
            best["min"] = (pf, alloc)
        if pf > best["max"][0]:
            best["max"] = (pf, alloc)

    row_of = {id(p): i for i, p in zip(df.index, df["point"], strict=True)}
    out = {"strict_tol": strict_tol, "n_points": len(df)}
    for bound in ("min", "max"):
        pf, alloc = best[bound]
        if not np.isclose(pf, summary[bound], rtol=1e-3, atol=1e-12):
            raise ValueError(f"{name} {bound}: recovered {pf:.4e}, summary says {summary[bound]:.4e}")
        out[bound] = {
            "reference": pf,
            "scaling": [float(x) for x in alloc["point"]],
            "optimized_at_slack": float(alloc["optimized_at_slack"]),
            "max_abs_constraint": float(alloc["constraint_val_max"]),
            "csv_row": int(row_of.get(id(alloc["point"]), -1)),
        }
    return out


def thesis_case(name: str, bound: str, allocation: dict, seed: int = SEED, soc_cache: dict | None = None) -> dict:
    """Pf of the true model at one bound's allocation of a thesis summary."""
    spec = THESIS_SUMMARIES[name]
    model_name = spec["model"]
    model = getattr(otaf.example_models, model_name)
    scaling = np.zeros(model.dim)
    point = np.asarray(allocation["scaling"])
    if spec["active"] is None:
        scaling[:] = point
    else:
        scaling[spec["active"]] = point
    reference = allocation["reference"]
    record = {
        "group": "thesis",
        "summary": name,
        "model": model_name,
        "bound": bound,
        "reference": reference,
        "scaling": allocation["scaling"],
        "max_abs_constraint": allocation["max_abs_constraint"],
        "optimized_at_slack": allocation["optimized_at_slack"],
        "settings": {"seed": seed, **build_settings(model_name)},
    }
    size = THESIS_RERUN_SIZE[model_name]
    if reference < MIN_CHECKABLE_PF:
        record["checked"] = False
        record["reason"] = f"Pf < {MIN_CHECKABLE_PF:g}: needs far more than a few thousand samples"
    else:
        record["checked"] = True
    if record["checked"] or model_name == "model1":
        # model1 is cheap: rerun every bound with the LP even when underpowered.
        if soc_cache is not None and model_name in soc_cache:
            soc = soc_cache[model_name]
        else:
            soc = build_model(model_name)
            if soc_cache is not None:
                soc_cache[model_name] = soc
        start = time.perf_counter()
        slack, status = solve(soc, scaled_sample(model_name, scaling, size, seed))
        rec = _pf_record(slack, status)
        record.update(rec)
        record["difference"] = rec["pf"] - reference
        record["z"] = z_score(rec["pf"], rec["n"], reference)
        record["settings"]["size"] = size
        record["seconds"] = time.perf_counter() - start
    if model_name == "model1":
        s = model1_closed_form_slack(scaled_sample(model_name, scaling, MODEL1_CLOSED_FORM_SIZE, seed))
        rec = _pf_record(s)
        record["closed_form"] = {**rec, "difference": rec["pf"] - reference, "z": z_score(rec["pf"], rec["n"], reference)}
    return record


# --- Baselines -------------------------------------------------------------------------


def baseline_cases(tag: str = BASELINE_TAG) -> list[dict]:
    """Recompute every baseline case and compare it with the stored baseline ``tag``."""
    sys.path.insert(0, str(SCRIPTS / "baselines"))
    import baseline_cases as bc

    reference = bc.load(tag)
    current = bc.run(verbose=False)
    table = bc.compare(reference, current)
    meta = reference[0]["cases"]
    rows = []
    for case in bc.CASES:
        if case.name not in table.index:
            continue
        r = table.loc[case.name]
        settings = {"seed": case.seed, "size": case.size, "multiplier": case.multiplier}
        settings.update(build_settings(case.model, **case.build_kwargs))
        if case.build_kwargs:
            settings["build"] = case.build_kwargs
        rows.append(
            {
                "group": "baselines",
                "case": case.name,
                "tag": tag,
                "reference": float(r["Pf_ref"]),
                "pf": float(r["Pf_now"]),
                "difference": float(r["dPf"]),
                "reclassified": int(r["reclassified"]),
                "max_abs_ds": float(r["max_abs_ds"]),
                "solver_failures": int(r["failures_now"]),
                "reference_solver_failures": int(meta[case.name]["n_solver_failures"]),
                "settings": settings,
            }
        )
    return rows


# --- Run, save, render -----------------------------------------------------------------


def run(groups=("dumas", "thesis", "baselines"), verbose: bool = True) -> dict:
    """Compute the given groups of cases."""
    # Provenance per group, so that groups can be recomputed separately (--only).
    provenance = {"otaf_version": otaf.__version__, "git_revision": _git_revision(), "created": time.strftime("%Y-%m-%d")}
    results = {"provenance": {group: provenance for group in groups}, "baseline_tag": BASELINE_TAG}
    say = print if verbose else (lambda *a, **k: None)
    if "dumas" in groups:
        cache, rows = {}, []
        for param_set, nd, strategy, size in DUMAS_RUNS:
            rows.append(dumas_case(param_set, nd, strategy, size, soc_cache=cache))
            r = rows[-1]
            say(f"dumas set {param_set} Nd={nd:2d} {strategy:13s} N={size}: {r['pf']:.4e} vs {r['reference']:.4e} (z={r['z']:+.2f}, {r['seconds']:.0f}s)", flush=True)
        results["dumas"] = rows
    if "thesis" in groups:
        cache, rows = {}, []
        for name in THESIS_SUMMARIES:
            allocations = recover_thesis_allocations(name)
            for bound in ("min", "max"):
                rows.append(thesis_case(name, bound, allocations[bound], soc_cache=cache))
                r = rows[-1]
                say(f"thesis {name} {bound}: ref {r['reference']:.4e} otaf {r.get('pf', float('nan')):.4e}", flush=True)
        results["thesis"] = rows
    if "baselines" in groups:
        results["baselines"] = baseline_cases()
        say(f"baselines: {len(results['baselines'])} cases", flush=True)
    return results


def _git_revision() -> str:
    import subprocess

    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src"], capture_output=True, text=True)
        return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def save(results: dict, path: Path = RESULTS_PATH) -> None:
    """Write the results as JSON."""
    path.write_text(json.dumps(results, indent=1) + "\n")


def load(path: Path = RESULTS_PATH) -> dict:
    """Read the saved results."""
    return json.loads(path.read_text())


def _e(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return "0" if x == 0 else f"{x:.{digits - 1}e}"


def _pct_ci(r: dict) -> str:
    return f"{_e(r['pf'])} [{_e(r['ci95'][0], 2)}, {_e(r['ci95'][1], 2)}]"


def _signed(x: float) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:+.2e}"


def render_tables(results: dict) -> str:
    """Markdown tables of the verification page (included by docs/guide/verification.md)."""
    out = [
        "<!-- Generated by scripts/verification/run_verification.py from verification_results.json; do not edit. -->",
        "",
    ]

    def computed(group):
        p = results["provenance"][group]
        return [f"Computed on {p['created']} with OTAF {p['otaf_version']} (revision `{p['git_revision']}`).", ""]

    if "dumas" in results:
        out += [
            "(verification-dumas-table)=",
            "### Table 1: model3 against Dumas (2014)",
            "",
            *computed("dumas"),
            "| Set | Nd | Linearization | Dumas (2014) | OTAF Pf [95 % CI] | Difference | z | "
            "Relative to Nd = 60, Dumas / OTAF | N (seed) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in results["dumas"]:
            s = r["settings"]
            # Same sample, same strategy, Nd = 60: the ratio isolates the effect of the polygon.
            at_60 = next(
                (q for q in results["dumas"] if q["param_set"] == r["param_set"] and q["settings"]["size"] == s["size"]
                 and q["settings"]["linearization"] == s["linearization"] and q["settings"]["circle_resolution"] == "60"),
                None,
            )
            ratio = "–"
            if at_60 is not None and s["circle_resolution"] != "60" and at_60["pf"] > 0:
                published_60 = DUMAS_2014[r["param_set"]]["pf"][60][s["linearization"]]
                ratio = f"{r['reference'] / published_60:.3f} / {r['pf'] / at_60['pf']:.3f}"
            out.append(
                f"| {r['param_set']} | {s['circle_resolution']} | {s['linearization']} | "
                f"{_e(r['reference'])} (table {r['table']}, N = {r['reference_n']:.0e}) | {_pct_ci(r)} | "
                f"{_signed(r['difference'])} | {r['z']:+.1f} | {ratio} | {s['size']} ({s['seed']}) |"
            )
        out.append("")
    if "thesis" in results:
        out += [
            "(verification-thesis-table)=",
            "### Table 2: thesis bounds (surrogate + GLD) against the scipy solver",
            "",
            *computed("thesis"),
            "| Summary | Bound | Reference | OTAF Pf [95 % CI] | Difference | z | N (seed) | Linearization |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in results["thesis"]:
            s = r["settings"]
            lin = f"{s['linearization']}, {s['circle_resolution']} sides" if "linearization" in s else "none (planes only)"
            if "pf" in r:
                otaf_col = _pct_ci(r) + ("" if r["checked"] else " (low power)")
                diff, z, n = _signed(r["difference"]), f"{r['z']:+.1f}", f"{s['size']} ({s['seed']})"
            else:
                otaf_col, diff, z, n = "not rerun: " + r["reason"], "–", "–", "–"
            out.append(f"| `{r['summary']}` | {r['bound']} | {_e(r['reference'])} | {otaf_col} | {diff} | {z} | {n} | {lin} |")
            if "closed_form" in r:
                c = r["closed_form"]
                out.append(
                    f"| `{r['summary']}` | {r['bound']}, closed form | {_e(r['reference'])} | {_pct_ci(c)} | "
                    f"{_signed(c['difference'])} | {c['z']:+.1f} | {c['n']:.0e} ({s['seed']}) | none (planes only) |"
                )
        out.append("")
    if "baselines" in results:
        out += [
            "(verification-baseline-table)=",
            f"### Table 3: baseline cases against `{results['baseline_tag']}`",
            "",
            *computed("baselines"),
            "| Case | Pf baseline | Pf now | Reclassified samples | max \\|Δs\\| | Solver failures | N (seed) | Settings |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in results["baselines"]:
            s = dict(r["settings"])
            n = f"{s.pop('size')} ({s.pop('seed')})"
            settings = ", ".join(f"{k}={v}" for k, v in s.items() if not (k == "multiplier" and v == 1.0)) or "defaults"
            out.append(
                f"| `{r['case']}` | {r['reference']:.4f} | {r['pf']:.4f} | {r['reclassified']} | "
                f"{_e(r['max_abs_ds'], 2)} | {r['solver_failures']} | {n} | {settings} |"
            )
        out.append("")
    return "\n".join(out)
