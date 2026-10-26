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


@pytest.mark.parametrize("name", ["model1", "model2", pytest.param("model4", marks=pytest.mark.slow)])
def test_distribution_labels_match_solver_variables(name, soc_model):
    # compute_gap_optimizations_* assume the sample columns follow soc.deviation_symbols.
    # model3 samples a reduced parameter space mapped by position (sample_multiplier),
    # so its labels cannot be checked by name.
    _, labels, _, _ = getattr(otaf.example_models, name).get_distribution_params()
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


@pytest.mark.xfail(strict=True, reason="open bug: the two directions of a gap give different interface constraints")
def test_model2_gap_direction_does_not_change_the_model(monkeypatch):
    # model2 uses the gap between surfaces P1b and P2b in both directions in its
    # compatibility loops; only one direction is kept. Physically both describe the
    # same contact, so keeping either one must give the same constraint system.
    from matrix_utils import assert_same_constraint_system, system_from_model
    from otaf._assembly_modeling._interface_constraints import interfaceLoopHandling as ilh

    original = ilh.InterfaceLoopHandling.extract_unique_gap_matrices_from_expanded_loops
    forward, backward = "GP1bB0P2bB0", "GP2bB0P1bB0"

    def build(keep):
        def forced(self):
            gaps = set(original(self)) - {forward, backward}
            return gaps | {keep}

        monkeypatch.setattr(ilh.InterfaceLoopHandling, "extract_unique_gap_matrices_from_expanded_loops", forced)
        return system_from_model(otaf.example_models.model2.get_system_of_constraints_assembly_model())

    assert_same_constraint_system(build(forward), build(backward))
