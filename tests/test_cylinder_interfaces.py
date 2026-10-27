"""Cylinder-cylinder interfaces built by the automatic pipeline.

The reference is the hand-coded ``model4`` (two plates joined by four pins):
``model4_auto`` describes the same assembly as an input dictionary, and both
must give the same slack sample by sample.
"""

import logging
from pathlib import Path

import numpy as np
import openturns as ot
import pytest
import sympy as sp
from scipy.optimize import Bounds, LinearConstraint, milp

import otaf
from otaf.geometry import LinearizationStrategy, circle_linearization_factor

from matrix_utils import MATRIX_NAMES

auto = otaf.example_models.model4_auto
CLEARANCE = (20.0 - 19.8) / 2


def _slack(soc, deviations):
    """Optimal slack per sample (status-aware, as in scripts/baselines)."""
    c, a_ub, b_ub, a_eq, b_eq, bounds = soc(np.atleast_2d(deviations))
    slack = []
    for k in range(b_ub.shape[1]):
        res = milp(
            c=c,
            bounds=Bounds(bounds[:, 0], bounds[:, 1]),
            constraints=(LinearConstraint(a_ub, -np.inf, b_ub[:, k]), LinearConstraint(a_eq, b_eq[:, k], b_eq[:, k])),
        )
        assert res.status == 0, res.message
        slack.append(res.x[-1])
    return np.array(slack)


def _build(**kwargs):
    logging.disable(logging.WARNING)
    try:
        return auto.get_system_of_constraints_assembly_model(**kwargs)
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def one_pin():
    return _build(circle_resolution=16, NX=1, NY=1)


@pytest.fixture(scope="module")
def four_pins():
    return _build(circle_resolution=64)


def _set(soc, values):
    x = np.zeros(soc.nD)
    names = [str(s) for s in soc.deviation_symbols]
    for name, value in values.items():
        x[names.index(name)] = value
    return x


# --- Linearization ----------------------------------------------------------------------


def test_linearization_factors():
    assert circle_linearization_factor(LinearizationStrategy.CIRCUMSCRIBED, 8) == 1.0
    assert circle_linearization_factor(LinearizationStrategy.INSCRIBED, 8) == pytest.approx(np.cos(np.pi / 8))
    assert circle_linearization_factor("mean", 8) == pytest.approx((1 + np.cos(np.pi / 8)) / 2)
    with pytest.raises(ValueError):
        circle_linearization_factor(LinearizationStrategy.INSCRIBED, 2)


def test_example_models_share_the_library_strategy():
    # A separate enum per module made model4 silently fall back to CIRCUMSCRIBED.
    assert otaf.example_models.model3.LinearizationStrategy is LinearizationStrategy
    assert otaf.example_models.model4.LinearizationStrategy is LinearizationStrategy


@pytest.mark.parametrize("strategy", list(LinearizationStrategy))
def test_nominal_slack_is_the_linearized_radial_clearance(strategy):
    # Also guards the axial nominal offset between the hole and pin origins: when it
    # was dropped, the loops could not close and the nominal assembly was infeasible.
    soc = _build(circle_resolution=16, strategy=strategy, NX=1, NY=1)
    expected = circle_linearization_factor(strategy, 16) * CLEARANCE
    # Coefficients are rounded to 8 decimals when the matrices are generated.
    assert _slack(soc, np.zeros(soc.nD))[0] == pytest.approx(expected, abs=1e-8)


# --- Diameter deviations ----------------------------------------------------------------


def test_diameter_variables_follow_the_feature_indices(one_pin):
    names = [str(s) for s in one_pin.deviation_symbols]
    # The holes share their feature index with their position/orientation deviations;
    # the pin has no deviation matrix and gets the next free index.
    assert names == [
        "v_d_0", "w_d_0", "beta_d_0", "gamma_d_0", "d_d_0",
        "v_d_1", "w_d_1", "beta_d_1", "gamma_d_1", "d_d_1",
        "d_d_4",
    ]


def test_precomputed_deviation_symbols_match_the_built_model(four_pins):
    _, symbols, std, _ = auto.get_distribution_params()
    assert symbols == four_pins.deviation_symbols
    assert len(std) == auto.dim == four_pins.nD


@pytest.mark.parametrize(
    "deviations, expected",
    [
        ({"d_d_0": 0.04, "d_d_1": 0.04}, CLEARANCE + 0.02),  # both holes larger
        ({"d_d_4": 0.04}, CLEARANCE - 0.02),  # pin larger
        ({"d_d_0": 0.04}, CLEARANCE),  # the upper hole still limits
        ({"d_d_0": -0.06}, CLEARANCE - 0.03),
    ],
)
def test_diameter_deviations_change_the_clearance(one_pin, deviations, expected):
    assert _slack(one_pin, _set(one_pin, deviations))[0] == pytest.approx(expected, abs=1e-12)


def test_without_diameter_deviation_there_are_no_diameter_variables():
    soc = _build(circle_resolution=8, NX=1, NY=1, diameter_deviation=False)
    assert not [s for s in soc.deviation_symbols if str(s).startswith("d_")]


def test_diameter_deviation_is_rejected_on_planes():
    data = auto.get_assembly_data(NX=1, NY=1)
    data["PARTS"]["0"]["a"]["DIAMETER_DEVIATION"] = True
    with pytest.raises(ValueError, match="DIAMETER_DEVIATION"):
        otaf.AssemblyDataProcessor(data)


# --- Differential test against the hand-coded model4 ------------------------------------


def _model4_from_snapshot():
    snap = np.load(Path(__file__).parent / "data" / "matrix_snapshots" / "model4.npz")
    soc = otaf.SystemOfConstraintsAssemblyModel(matrices=[snap[k] for k in MATRIX_NAMES])
    soc.deviation_symbols = [sp.Symbol(str(x)) for x in snap["deviation_symbols"]]
    soc.gap_symbols = [sp.Symbol(str(x)) for x in snap["gap_symbols"]]
    return soc


def _model4_to_auto(soc_auto):
    """Matrix M such that ``x_auto = M @ (x_model4 - nominal)`` (plates perfect).

    model4 (hand-coded, origin at the lower-left pin): holes 1d, 2d, 1e, ... with
    columns (beta, gamma, v, w) = rotations about and translations along global
    (x, y), at ``6 + 4 i``; hole diameters at 38..45, pin diameters at 46..49.
    Its plate 1 is above z = 0 (model4_auto part "1"), plate 2 below (part "0").
    model4_auto: per hole, (v, w, beta, gamma) along / about the local y and z
    axes, which are global (y, x) below and (y, -x) above.
    """
    names = [str(s) for s in soc_auto.deviation_symbols]
    col = {n: i for i, n in enumerate(names)}
    # Hole feature indices of model4_auto (see test_diameter_variables_follow_the_feature_indices).
    auto_index = {("0", "b"): 0, ("1", "b"): 1, ("0", "c"): 4, ("1", "c"): 5,
                  ("0", "d"): 6, ("1", "d"): 7, ("0", "e"): 8, ("1", "e"): 9}
    auto_pin_index = {"b": 10, "c": 11, "d": 12, "e": 13}
    # model4 pin label of each model4_auto pin (same axis).
    m4_pin = {"b": "g", "c": "d", "d": "f", "e": "e"}
    local_yz_in_global_xy = {"0": np.array([[0.0, 1.0], [1.0, 0.0]]), "1": np.array([[0.0, 1.0], [-1.0, 0.0]])}
    M = np.zeros((len(names), 50))
    for s, h in m4_pin.items():
        for part, plate in (("0", 2), ("1", 1)):
            R = local_yz_in_global_xy[part]
            k = auto_index[(part, s)]
            c4 = 6 + 4 * (2 * "defg".index(h) + plate - 1)
            for local, (gx, gy) in (((f"v_d_{k}", f"w_d_{k}"), (c4 + 2, c4 + 3)), ((f"beta_d_{k}", f"gamma_d_{k}"), (c4, c4 + 1))):
                rows = [col[local[0]], col[local[1]]]
                M[rows, gx] = R[:, 0]
                M[rows, gy] = R[:, 1]
            M[col[f"d_d_{k}"], 38 + 4 * (plate - 1) + "defg".index(h)] = 1.0
        M[col[f"d_d_{auto_pin_index[s]}"], 46 + "defg".index(h)] = 1.0
    return M


def test_automatic_pins_match_hand_coded_model4(four_pins):
    model4 = _model4_from_snapshot()
    dist, _, _, mu = otaf.example_models.model4.get_distribution_params()
    ot.RandomGenerator.SetSeed(1)
    x4 = np.array(dist.getSample(60)) * 1.3  # wider than nominal dispersion: some failures
    x4[:, 38:] += mu[38:] * (1 - 1.3)
    x4[:, :6] = 0.0  # perfect plates, as in model4_auto
    nominal = np.r_[np.zeros(38), mu[38:]]
    s4 = _slack(model4, x4)
    s_auto = _slack(four_pins, (x4 - nominal) @ _model4_to_auto(four_pins).T)
    assert (s4 < 0).any() and (s4 > 0).any()
    np.testing.assert_allclose(s_auto, s4, atol=1e-8)


# --- Credal set (capability constraints of the thesis) ------------------------------------


def test_credal_constraints_are_zero_on_the_capability_boundary():
    _, symbols, max_std, _ = auto.get_distribution_params(tol=0.25)
    names = [str(s) for s in symbols]
    kw = {"tol": 0.25}
    # Each component alone at its maximal standard deviation is on the boundary.
    for i, name in enumerate(names):
        x = np.zeros(len(names))
        x[i] = 1.0
        values = auto.eval_scaled_credal_set_constraints(x, max_std, **kw)
        assert values.max() == pytest.approx(0.0, abs=1e-12), name
    # Translation and rotation sharing the variance in both directions: still on it.
    x = np.where([n.split("_")[0] in ("v", "w", "beta", "gamma") for n in names], np.sqrt(0.5), 0.0)
    np.testing.assert_allclose(auto.eval_scaled_credal_set_constraints(x, max_std, **kw)[:8], 0.0, atol=1e-12)
    # All components at their maximum: sqrt(3) times the target for holes (d, v, gamma).
    values = auto.get_scaled_credal_set_constraints_function(max_std, **kw)(np.ones(len(names)))
    np.testing.assert_allclose(values, [np.sqrt(3) - 1] * 8 + [0.0] * 4)
