"""A failed solve in a batch is reported per sample instead of crashing the batch."""

import numpy as np
import pytest

import otaf
from otaf.exceptions import SolverFailureWarning
from otaf.uncertainty import compute_gap_optimizations_on_sample_batch, milp_batch_sequential


@pytest.fixture(scope="module")
def soc():
    return otaf.example_models.model1.get_system_of_constraints_assembly_model()


def _deviations(soc):
    # Sample 1 moves a wall by 100 mm: the loops cannot close with gaps bounded to +-3 mm.
    deviations = np.zeros((3, soc.nD))
    deviations[1, :] = 100.0
    return deviations


def test_batch_reports_the_failed_sample(soc):
    with pytest.warns(SolverFailureWarning, match="1 of 3 samples.*infeasible"):
        x, status = compute_gap_optimizations_on_sample_batch(
            soc, _deviations(soc), dtype="float64", return_status=True
        )
    assert status.tolist() == [0, 2, 0]
    assert np.isnan(x[1]).all()
    assert x[[0, 2], -1] == pytest.approx([0.1, 0.1], abs=1e-7)


def test_failed_sample_in_a_parallel_batch(soc):
    deviations = np.repeat(_deviations(soc), 2, axis=0)  # 6 samples in batches of 2
    with pytest.warns(SolverFailureWarning, match="2 of 6 samples"):
        x, status = compute_gap_optimizations_on_sample_batch(
            soc, deviations, batch_size=2, n_cpu=1, dtype="float64", return_status=True
        )
    assert status.tolist() == [0, 0, 2, 2, 0, 0]
    assert np.isnan(x[2:4, -1]).all() and not np.isnan(x[[0, 1, 4, 5], -1]).any()


def test_milp_batch_sequential_status(soc):
    c, a_ub, b_ub, a_eq, b_eq, bounds = soc(_deviations(soc))
    with pytest.warns(SolverFailureWarning):
        x = milp_batch_sequential(c, bounds, a_ub, b_ub, a_eq, b_eq)
    assert x.shape == (3, len(c))
    assert np.isnan(x[1]).all()


def test_no_warning_without_failures(soc, recwarn):
    x, status = compute_gap_optimizations_on_sample_batch(
        soc, np.zeros((2, soc.nD)), dtype="float64", return_status=True
    )
    assert (status == 0).all()
    assert not [w for w in recwarn if issubclass(w.category, SolverFailureWarning)]
