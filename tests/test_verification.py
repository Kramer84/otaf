"""Checks behind the verification report (``docs/guide/verification.md``).

The report's numbers come from ``scripts/verification/run_verification.py``
(minutes of scipy solves), saved in ``docs/guide/verification_results.json``.
These tests keep the page, the saved results and their reference sources in step,
and rerun the cheap cases at reduced size.
"""

import sys
from pathlib import Path

import numpy as np
import openturns as ot
import pytest

import otaf
from otaf.geometry import LinearizationStrategy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "verification"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

import baseline_cases  # noqa: E402
import verification_cases as vc  # noqa: E402

RESULTS = vc.load()


# --- The page, the saved results and the references agree -----------------------------


def test_tables_are_generated_from_the_saved_results():
    # Regenerate with: python scripts/verification/run_verification.py --render-only
    assert vc.TABLES_PATH.read_text() == vc.render_tables(RESULTS)


def test_report_covers_every_case():
    assert [(r["param_set"], int(r["settings"]["circle_resolution"]), r["settings"]["linearization"], r["settings"]["size"])
            for r in RESULTS["dumas"]] == vc.DUMAS_RUNS
    assert [(r["summary"], r["bound"]) for r in RESULTS["thesis"]] == [
        (name, bound) for name in vc.THESIS_SUMMARIES for bound in ("min", "max")
    ]
    assert [r["case"] for r in RESULTS["baselines"]] == [c.name for c in baseline_cases.CASES]


def test_dumas_references_are_the_published_values():
    for r in RESULTS["dumas"]:
        published = vc.DUMAS_2014[r["param_set"]]
        assert r["reference"] == published["pf"][int(r["settings"]["circle_resolution"])][r["settings"]["linearization"]]
        assert r["reference_n"] == published["n_mc"]


def test_thesis_references_are_the_summary_values():
    for r in RESULTS["thesis"]:
        assert r["reference"] == pytest.approx(vc.parse_summary(r["summary"])[r["bound"]], rel=1e-3, abs=1e-12)


def test_baselines_are_reproduced_exactly():
    # The report must not hide a change of results: every baseline sample keeps its slack.
    for r in RESULTS["baselines"]:
        assert r["reclassified"] == 0, r["case"]
        assert r["max_abs_ds"] <= 1e-9, r["case"]
        assert r["solver_failures"] == r["reference_solver_failures"], r["case"]


def test_model1_closed_form_values_are_reproduced():
    # numpy only (no solver): the saved counts must follow from the saved scalings.
    for r in RESULTS["thesis"]:
        if "closed_form" not in r:
            continue
        sample = vc.scaled_sample("model1", np.asarray(r["scaling"]), r["closed_form"]["n"], r["settings"]["seed"])
        assert int((vc.model1_closed_form_slack(sample) < 0).sum()) == r["closed_form"]["failures"]


# --- Cheap reruns ------------------------------------------------------------------


@pytest.mark.slow
def test_model3_matches_dumas_parameter_set_1():
    r = vc.dumas_case(1, 60, "circumscribed", size=2000)
    assert r["solver_failures"] == 0
    assert abs(r["z"]) < 3, r


@pytest.mark.slow
def test_model3_polygon_strategies_give_nested_failures():
    # The polygon factor scales the radial clearance. When the hole is larger than
    # the pin, a larger factor (inscribed < mean < circumscribed) only enlarges the
    # feasible set, so the slack can only grow; when the pin is larger, the slack is
    # negative for every factor. Failures are therefore nested sample by sample, as
    # Dumas (2014) observed on the probabilities (inscribed is conservative).
    deviations = vc.scaled_sample("model3", None, 300, seed=5)
    slack = {
        s: vc.solve(vc.build_model("model3", Nd=8, strategy=s), deviations)[0]
        for s in (LinearizationStrategy.INSCRIBED, LinearizationStrategy.MEAN, LinearizationStrategy.CIRCUMSCRIBED)
    }
    inscribed, mean, circumscribed = slack.values()
    # Columns 22-25: hole and pin diameters (d_1b, d_3b, d_1c, d_4c).
    positive = (deviations[:, 22] > deviations[:, 23]) & (deviations[:, 24] > deviations[:, 25])
    assert positive.any() and (inscribed < 0).sum() > (circumscribed < 0).sum()
    np.testing.assert_array_less(inscribed[positive], mean[positive] + 1e-9)
    np.testing.assert_array_less(mean[positive], circumscribed[positive] + 1e-9)
    assert not ((circumscribed < 0) & (mean >= 0)).any()
    assert not ((mean < 0) & (inscribed >= 0)).any()


@pytest.mark.slow
@pytest.mark.parametrize("summary", ["model3_30_dof", "model4_50_dof"])
def test_thesis_max_bound_is_reproduced(summary):
    saved = next(r for r in RESULTS["thesis"] if r["summary"] == summary and r["bound"] == "max")
    model = getattr(otaf.example_models, saved["model"])
    deviations = vc.scaled_sample(saved["model"], np.asarray(saved["scaling"]), 1000)
    slack, status = vc.solve(vc.build_model(saved["model"]), deviations)
    assert (status == 0).all()
    pf = np.mean(slack < 0)
    assert abs(vc.z_score(pf, len(slack), saved["reference"])) < 3, (pf, saved["reference"], model.__name__)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["model1", "model2_x1.21", "model3", "model4_auto_x1.15"])
def test_baseline_samples_are_reproduced(name):
    # First 200 samples of a baseline case (same sample as the stored one).
    case = next(c for c in baseline_cases.CASES if c.name == name)
    _, stored = baseline_cases.load(vc.BASELINE_TAG)
    model = getattr(otaf.example_models, case.model)
    distribution, *_ = model.get_distribution_params(**case.dist_kwargs)
    if case.multiplier != 1.0:
        distribution = otaf.distribution.multiply_composed_distribution_standard_with_constants(
            distribution, [case.multiplier] * distribution.getDimension()
        )
    ot.RandomGenerator.SetSeed(case.seed)
    sample = np.array(distribution.getSample(case.size)) @ np.asarray(model.sample_multiplier).T
    slack, _ = vc.solve(vc.build_model(case.model, **case.build_kwargs), sample[:200])
    np.testing.assert_allclose(slack, stored[name][:200], atol=1e-9, rtol=0)
