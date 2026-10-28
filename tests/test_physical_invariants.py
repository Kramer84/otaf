"""Physical and mathematical invariants of the example models.

Sample sizes stay small (a few hundred points) so the whole module runs on a
laptop. ``s`` is the slack variable (last gap variable): the assembly fails
when ``s < 0``.
"""

import numpy as np
import openturns as ot
import pytest

import otaf

MODELS = [
    "model1",
    "model2",
    "model3",
    pytest.param("model4", marks=pytest.mark.slow),
]


def solve_slack(soc, deviations):
    """Optimal slack ``s`` for each row of ``deviations`` (solver variable order)."""
    result = otaf.uncertainty.compute_gap_optimizations_on_sample_batch(
        soc, np.atleast_2d(deviations), n_cpu=1, batch_size=10_000, dtype="float64"
    )
    return result[:, -1]


def sample_deviations(name, size, seed=0):
    """Sample a model's defect distribution and map it to the solver's variables.

    Returns ``(nominal, sample)``, both expressed in the solver's variable order.
    """
    model = getattr(otaf.example_models, name)
    distribution, _, _, mean = model.get_distribution_params()
    ot.RandomGenerator.SetSeed(seed)
    sample = np.array(distribution.getSample(size))
    multiplier = np.asarray(model.sample_multiplier)
    return mean @ multiplier.T, sample @ multiplier.T


@pytest.mark.parametrize("name", ["model1", "model2", "model3", pytest.param("model4", marks=pytest.mark.slow)])
def test_distribution_labels_match_solver_variables(name, soc_model):
    # compute_gap_optimizations_* assume the sample columns, after sample_multiplier,
    # follow soc.deviation_symbols. model3 samples measured points and maps them to the
    # solver variables (sample_multiplier_labels names both sides).
    model = getattr(otaf.example_models, name)
    if name == "model3":
        _, labels, _ = model.sample_multiplier_labels()
    else:
        _, labels, _, _ = model.get_distribution_params()
    assert [str(x) for x in labels] == [str(x) for x in soc_model(name).deviation_symbols]


@pytest.mark.parametrize("name", MODELS)
def test_nominal_assembly_is_feasible_with_positive_margin(name, soc_model):
    nominal, _ = sample_deviations(name, size=1)
    assert solve_slack(soc_model(name), nominal)[0] > 0


def test_model1_slack_is_half_the_analytical_clearance(soc_model):
    # Analytical reference: notebooks/Ex1_Pocket/Ex1_2D_4dof_Analytical_Ref.ipynb.
    # The minimum clearance j between part and pocket is reached at one of the two
    # corners (y = ±h/2). The SOC slack centers the part, hence s = j / 2.
    L1, L2, y_max = 99.8, 100.0, 5.0
    _, sample = sample_deviations("model1", size=500, seed=888)
    u4, gamma4, u5, gamma5 = sample.T
    clearance = np.minimum(
        (L2 - u4 - y_max * gamma4) - (L1 + u5 - y_max * gamma5),
        (L2 - u4 + y_max * gamma4) - (L1 + u5 + y_max * gamma5),
    )
    assert (clearance < 0).any(), "sample should contain failures to be meaningful"

    slack = solve_slack(soc_model("model1"), sample)

    np.testing.assert_allclose(slack, clearance / 2, atol=1e-7)


@pytest.mark.parametrize("name", MODELS)
def test_failures_persist_when_deviations_grow(name, soc_model):
    # The optimal slack of an LP is concave in its right-hand side, which is affine in
    # the deviations. Along x(k) = nominal + k * (x - nominal), with s(0) > 0, a sample
    # that fails at k therefore also fails at every k' > k. In tolerancing terms:
    # widening the dispersion never repairs an assembly (common random numbers).
    soc = soc_model(name)
    nominal, sample = sample_deviations(name, size=200, seed=1)
    deviation = sample - nominal

    failed_before = np.zeros(len(sample), dtype=bool)
    for k in (1.0, 1.5, 2.0, 3.0):
        failed = solve_slack(soc, nominal + k * deviation) < 0
        assert not (failed_before & ~failed).any(), f"a failure disappeared at k={k}"
        failed_before = failed
    assert failed_before.any(), "largest dispersion should produce failures"


def _model2_slack_keeping_gap_direction(monkeypatch, keep, hM, hF, size=500):
    """Optimal slack of model2 when the P1b/P2b gap is kept in direction ``keep``."""
    from otaf._assembly_modeling._interface_constraints import interfaceLoopHandling as ilh

    original = ilh.InterfaceLoopHandling.extract_unique_gap_matrices_from_expanded_loops
    both = {"GP1bB0P2bB0", "GP2bB0P1bB0"}

    def forced(self):
        return (set(original(self)) - both) | {keep}

    monkeypatch.setattr(ilh.InterfaceLoopHandling, "extract_unique_gap_matrices_from_expanded_loops", forced)
    model = otaf.example_models.model2
    soc = model.get_system_of_constraints_assembly_model(hM=hM, hF=hF)
    distribution, *_ = model.get_distribution_params(hM=hM, hF=hF)
    distribution = otaf.distribution.multiply_composed_distribution_standard_with_constants(
        distribution, [1.21] * model.dim
    )
    ot.RandomGenerator.SetSeed(0)
    return solve_slack(soc, np.array(distribution.getSample(size)))


# model2 uses the gap between surfaces P1b and P2b in both directions in its compatibility
# loops and keeps only one. Each direction pivots the gap at a different origin, so the raw
# coefficients may differ, but the physical answer (the optimal slack) must not. Unequal
# face heights used to break this (proximity pairing of points and a trailing JCor inverse
# in the interface loops shifting the lever arms by the offset between the origins).


@pytest.mark.parametrize("hF", [10.0, 10.2, 10.3, 11.0])
def test_model2_gap_direction_is_irrelevant(monkeypatch, hF):
    forward = _model2_slack_keeping_gap_direction(monkeypatch, "GP1bB0P2bB0", hM=10.0, hF=hF)
    backward = _model2_slack_keeping_gap_direction(monkeypatch, "GP2bB0P1bB0", hM=10.0, hF=hF)
    np.testing.assert_allclose(forward, backward, atol=1e-9)


def test_model2_contact_at_surface_origin_is_kept():
    # Move the origins of P1b and P2b to the bottom of the faces (swap B0 and B1). The
    # reference pair of the compatibility loops is then itself the bottom contact, and must
    # still produce its contact constraint: same number of contacts, same nominal margin.
    model = otaf.example_models.model2

    def build(data):
        sda = model.get_assembly_data_processor_object(data)
        clh = model.get_compatibility_loop_handling_object(sda)
        ilh = model.get_interface_loop_handling_object(sda, clh)
        soc = otaf.SystemOfConstraintsAssemblyModel(clh.get_compatibility_expressions(), ilh.get_interface_expressions())
        soc.embedOptimizationVariable()
        return soc

    default = build(model.get_assembly_data())
    moved_data = model.get_assembly_data()
    for part in ("1", "2"):
        points = moved_data["PARTS"][part]["b"]["POINTS"]
        points["B0"], points["B1"] = points["B1"], points["B0"]
    moved = build(moved_data)

    assert moved.nI == default.nI
    np.testing.assert_allclose(
        solve_slack(moved, np.zeros((1, moved.nD))), solve_slack(default, np.zeros((1, default.nD))), atol=1e-12
    )


@pytest.mark.parametrize("name", MODELS)
def test_lp_and_milp_give_the_same_slack(name, soc_model):
    # The problem has no integer variables: scipy's milp (HiGHS branch and bound, used by
    # the library) must agree with an LP solved by another algorithm (interior point).
    from scipy.optimize import linprog

    soc = soc_model(name)
    _, sample = sample_deviations(name, size=100, seed=2)
    c, a_ub, b_ub, a_eq, b_eq, bounds = soc(sample)
    lp = np.array([
        linprog(c, A_ub=a_ub, b_ub=b_ub[:, k], A_eq=a_eq, b_eq=b_eq[:, k], bounds=bounds, method="highs-ipm").x[-1]
        for k in range(len(sample))
    ])
    np.testing.assert_allclose(solve_slack(soc, sample), lp, atol=1e-6)


def _relabel(data, parts, surfaces):
    """Copy of an input dictionary with parts and surfaces renamed (points follow their
    surface) and the compatibility loops listed in reverse order."""
    import re

    def label(match):
        part, surface, point = match.group(1), match.group(2), match.group(3) or ""
        new_surface = surfaces.get(surface, surface)
        return f"P{parts.get(part, part)}{new_surface}" + (new_surface.upper() + point[1:] if point else "")

    rename = lambda text: re.sub(r"P(\d+)([a-z]+)([A-Z]+\d+)?", label, text)  # noqa: E731
    out = {key: value for key, value in data.items() if key not in ("PARTS", "LOOPS")}
    out["LOOPS"] = {
        kind: {f"L{i}": rename(loop) for i, loop in enumerate(reversed(loops.values()))}
        for kind, loops in data["LOOPS"].items()
    }
    out["PARTS"] = {}
    for part, part_surfaces in data["PARTS"].items():
        new_part = out["PARTS"].setdefault(parts.get(part, part), {})
        for surface, surface_data in part_surfaces.items():
            new_surface = surfaces.get(surface, surface)
            new = dict(surface_data)
            if "POINTS" in new:
                new["POINTS"] = {new_surface.upper() + p[1:]: xyz for p, xyz in new["POINTS"].items()}
            if "INTERACTIONS" in new:
                new["INTERACTIONS"] = [rename(i) for i in new["INTERACTIONS"]]
            new_part[new_surface] = new
    return out


def _build_with_deviation_map(model, data):
    """SOC model of ``data`` and the deviation-matrix index of each surface (``{"D1b": 5, ...}``)."""
    adp = model.get_assembly_data_processor_object(data)
    clh = model.get_compatibility_loop_handling_object(adp)
    ilh = model.get_interface_loop_handling_object(adp, clh)
    soc = otaf.SystemOfConstraintsAssemblyModel(clh.get_compatibility_expressions(), ilh.get_interface_expressions())
    soc.embedOptimizationVariable()
    return soc, {name: matrices[0].ID for name, matrices in clh._deviation_matrix_map.items()}


@pytest.mark.parametrize("name", ["model1", "model2"])
def test_slack_does_not_depend_on_labels_or_loop_order(name):
    # Labels and the order of the loops only name and arrange things: swapping the two
    # parts, permuting the surface letters and reversing the loop list (which renumbers
    # the variables) must give the same slack once the deviation variables are matched
    # by the surface they belong to.
    model = getattr(otaf.example_models, name)
    data = model.get_assembly_data()
    surfaces = sorted({s for part in data["PARTS"].values() for s in part})
    permuted = dict(zip(surfaces, surfaces[1:] + surfaces[:1], strict=True))
    swapped = {"1": "2", "2": "1"}
    soc, index = _build_with_deviation_map(model, data)
    soc_r, index_r = _build_with_deviation_map(model, _relabel(data, swapped, permuted))

    # "D1b" (part 1, surface b) is "D2c" after relabelling, etc.
    new_index = {i: index_r[f"D{swapped[dev[1:-1]]}{permuted[dev[-1]]}"] for dev, i in index.items()}
    columns = {}
    for k, symbol in enumerate(soc.deviation_symbols):
        component, _, i = str(symbol).rpartition("_d_")
        columns[f"{component}_d_{new_index[int(i)]}"] = k
    order = [columns[str(symbol)] for symbol in soc_r.deviation_symbols]

    assert any(old != new for old, new in new_index.items()), "relabelling should renumber the variables"

    _, sample = sample_deviations(name, size=200, seed=3)
    np.testing.assert_allclose(solve_slack(soc_r, sample[:, order]), solve_slack(soc, sample), atol=1e-9)
