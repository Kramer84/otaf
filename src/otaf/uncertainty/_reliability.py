from __future__ import annotations

__author__ = "Kramer84"
__all__ = [
    "compute_failure_probability_subset_sampling",
    "compute_failure_probability_FORM",
    "compute_failure_probability_NAIS",
    "compute_failure_probability_SUBSET",
    "compute_gap_optimizations_on_sample",
    "milp_batch_sequential",
    "compute_gap_optimizations_on_sample_batch",
    "MILP_STATUS_MESSAGES",
]

import logging
import warnings

import numpy as np
import openturns as ot
from beartype import beartype
from beartype.typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from joblib import Parallel, cpu_count, delayed
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp

from otaf.common import get_tqdm_range
from otaf.exceptions import SolverFailureWarning

logger = logging.getLogger(__name__)

MILP_STATUS_MESSAGES = {
    0: "optimal solution found",
    1: "iteration or time limit reached",
    2: "problem is infeasible",
    3: "problem is unbounded",
    4: "other solver failure",
}
"""Status codes of ``scipy.optimize.milp`` (``OptimizeResult.status``)."""

if hasattr(ot, "JointDistribution"):
    JointDistribution = ot.JointDistribution
else:
    JointDistribution = ot.ComposedDistribution
if TYPE_CHECKING:
    from otaf import SystemOfConstraintsAssemblyModel


@beartype
def compute_failure_probability_FORM(
    ot_function: ot.Function,
    composed_distribution: JointDistribution,
    threshold: float = 0.0,
    start_point: Optional[ot.Point] = None,
    verbose: bool = False,
    solver: Optional[ot.OptimizationAlgorithm] = None,
) -> tuple[float, ot.FORMResult]:
    """Compute failure probability using FORM.

    Calculate the failure probability for an event defined by a
    threshold violation by finding the design point in the standard
    normal space and applying a first-order approximation.

    Parameters
    ----------
    ot_function : ot.Function
        The performance function of the system.
    composed_distribution : JointDistribution
        The joint probability distribution of the input random
        variables.
    threshold : float, optional
        The limit state threshold. Default is 0.0.
    start_point : ot.Point, optional
        Initial starting point for the optimization algorithm in
        physical space. Default is None.
    verbose : bool, optional
        If True, log reliability diagnostics at INFO level on the
        ``otaf.uncertainty._reliability`` logger. Default is False.
    solver : ot.OptimizationAlgorithm, optional
        The optimization algorithm to find the design point. Uses
        ``ot.Cobyla()`` with strict convergence tolerances if None.

    Returns
    -------
    tuple[float, ot.FORMResult]
        The calculated failure probability and the full FORM result
        object.
    """
    comp_rand_vect = ot.CompositeRandomVector(
        ot_function, ot.RandomVector(composed_distribution)
    )
    event = ot.ThresholdEvent(comp_rand_vect, ot.Less(), threshold)
    if not solver:
        solver = ot.Cobyla()
        solver.setMaximumIterationNumber(10000)
        solver.setMaximumAbsoluteError(0.001)
        solver.setMaximumRelativeError(0.001)
        solver.setMaximumResidualError(0.001)
        solver.setMaximumConstraintError(0.001)
    algoFORM = ot.FORM(solver, event, start_point)
    algoFORM.run()
    result = algoFORM.getResult()
    pf = result.getEventProbability()
    if verbose:
        logger.info("Design point in physical space : %s", result.getPhysicalSpaceDesignPoint())
        logger.info("Design point in standard space : %s", result.getStandardSpaceDesignPoint())
        logger.info("Hasofer index : %s", result.getHasoferReliabilityIndex())
        logger.info("Probability of failure (FORM) Pf = %.16f", pf)
    return (pf, result)


def compute_failure_probability_NAIS(
    ot_python_function: ot.PythonFunction,
    distribution: JointDistribution,
    threshold: float = 0.0,
    quantile_level: float = 0.001,
    verbose: bool = False,
) -> tuple[float, ot.SimulationResult]:
    """Compute failure probability using the NAIS algorithm.

    Parameters
    ----------
    ot_python_function : ot.PythonFunction
        The performance function ``g(X)``.
    distribution : JointDistribution
        The input random vector distribution.
    threshold : float, optional
        The threshold value. Default is 0.0.
    quantile_level : float, optional
        The quantile level for the NAIS algorithm. Default is 0.001.
    verbose : bool, optional
        If True, log the estimate and its confidence interval at INFO
        level on the ``otaf.uncertainty._reliability`` logger. Default is
        False.

    Returns
    -------
    proba : float
        The estimated failure probability.
    result : ot.SimulationResult
        Additional NAIS algorithm results object.
    """
    output_random_vector = ot.CompositeRandomVector(
        ot_python_function, ot.RandomVector(distribution)
    )
    failure_event = ot.ThresholdEvent(output_random_vector, ot.LessOrEqual(), threshold)
    algo = ot.NAIS(failure_event, quantile_level)
    algo.run()
    result = algo.getResult()
    proba = result.getProbabilityEstimate()
    if verbose:
        logger.info("Proba NAIS = %s", proba)
        logger.info("Current coefficient of variation = %s", result.getCoefficientOfVariation())
        length_95 = result.getConfidenceLength()
        logger.info("Confidence length (0.95) = %s", length_95)
        logger.info(
            "Confidence interval (0.95) = %s",
            [proba - length_95 / 2, proba + length_95 / 2],
        )
    return (proba, result)


def compute_failure_probability_SUBSET(
    ot_python_function: ot.PythonFunction,
    distribution: JointDistribution,
    threshold: float = 0.0,
    verbose: bool = False,
    proposal_range=2,
    target_probability=0.1,
) -> tuple[float, ot.SimulationResult, Any]:
    """Compute failure probability using subset sampling.

    Parameters
    ----------
    ot_python_function : ot.PythonFunction
        The performance function ``g(X)``.
    distribution : JointDistribution
        The input random vector distribution.
    threshold : float, optional
        The threshold value. Default is 0.0.
    verbose : bool, optional
        If True, log the estimate and its confidence interval at INFO
        level on the ``otaf.uncertainty._reliability`` logger. Default is
        False.
    proposal_range : float, optional
        The proposal range for the Markov chain Monte Carlo steps.
        Default is 2.0.
    target_probability : float, optional
        The target probability for each conditional step. Default
        is 0.1.

    Returns
    -------
    proba : float
        The estimated failure probability.
    result : ot.SimulationResult
        Additional subset sampling algorithm results object.
    algo : ot.SubsetSampling
        The subset sampling algorithm instance used for the
        simulation.
    """
    output_random_vector = ot.CompositeRandomVector(
        ot_python_function, ot.RandomVector(distribution)
    )
    failure_event = ot.ThresholdEvent(output_random_vector, ot.LessOrEqual(), threshold)
    algo = ot.SubsetSampling(failure_event, proposal_range, target_probability)
    algo.setKeepSample(True)
    algo.run()
    result = algo.getResult()
    proba = result.getProbabilityEstimate()
    if verbose:
        logger.info("Proba Subset = %s", proba)
        logger.info("Current coefficient of variation = %s", result.getCoefficientOfVariation())
        length_95 = result.getConfidenceLength()
        logger.info("Confidence length (0.95) = %s", length_95)
        logger.info(
            "Confidence interval (0.95) = %s",
            [proba - length_95 / 2, proba + length_95 / 2],
        )
    return (proba, result, algo)


@beartype
def compute_failure_probability_subset_sampling(
    constraint_matrix_generator: Callable,
    defect_distribition_vector: ot.RandomVector,
    C: Optional[np.ndarray] = None,
    bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    n_cpu: int = 1,
) -> ot.simulation.ProbabilitySimulationResult:
    """Calculate failure probability for a sample of defects.

    Parameters
    ----------
    constraint_matrix_generator : Callable
        Class or callable for fixing deviations and defining
        constraints.
    defect_distribition_vector : ot.RandomVector
        The random vector representing the defect distribution.
    C : np.ndarray, optional
        Coefficient matrix for the linear objective function.
        Default is None.
    bounds : list of list of float or np.ndarray, optional
        Bounds for gap variables. Default is None.
    n_cpu : int, optional
        Number of CPUs to use for parallel execution. Default is 1.

    Returns
    -------
    ot.SimulationResult
        The simulation result object containing the estimated failure
        probability and simulation diagnostics.
    """

    def defect_func(X):
        """Small function to work as an intermediary."""
        optimizations = compute_gap_optimizations_on_sample(
            constraint_matrix_generator, X, C, bounds, n_cpu
        )
        opt_var_values = ot.Point(
            np.array([opt.fun for opt in optimizations], dtype=float)
        )
        return opt_var_values

    defect_func_ot = ot.PythonFunction(
        defect_distribition_vector.getDimension(), 1, defect_func
    )
    Y = ot.CompositeRandomVector(defect_func_ot, defect_distribition_vector)
    failureEvent = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.0)
    algo = ot.SubsetSampling(failureEvent)
    algo.setKeepSample(True)
    algo.run()
    return algo.getResult()


def compute_gap_optimizations_on_sample(
    constraint_matrix_generator: SystemOfConstraintsAssemblyModel,
    deviation_array: np.ndarray,
    C: Optional[np.ndarray] = None,
    bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    n_cpu: int = 1,
    progress_bar: Optional[bool] = False,
) -> List[OptimizeResult]:
    """Compute gap optimizations for a set of samples using MILP.

    Solve a sequence of Mixed-Integer Linear Programming (MILP)
    problems derived from a system of constraints to determine the
    optimal gap for each sample.

    Parameters
    ----------
    constraint_matrix_generator : SystemOfConstraintsAssemblyModel
        Generator object that produces the constraint matrices and
        bounds.
    deviation_array : np.ndarray
        Array representing deviations to be processed.
    C : np.ndarray, optional
        Coefficient matrix for the linear objective function. Default
        is None.
    bounds : list of list of float or np.ndarray, optional
        Bounds for the optimization variables. Default is None.
    n_cpu : int, optional
        Number of CPUs to use for parallel processing. Default is 1.
    progress_bar : bool, optional
        Whether to display a progress bar. Default is False.

    Returns
    -------
    list of OptimizeResult
        A list of optimization result objects for each sample in the
        input array.
    """
    c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(
        deviation_array, bounds=bounds, C=C
    )
    if progress_bar:
        _range = get_tqdm_range()
    else:
        _range = range
    optimizations = []
    if 0 <= n_cpu <= 1:
        optimizations = [
            milp(
                c=c,
                bounds=Bounds(bounds[:, 0], bounds[:, 1], keep_feasible=False),
                constraints=(
                    LinearConstraint(a_ub, -np.inf, b_ub[:, k]),
                    LinearConstraint(a_eq, b_eq[:, k], b_eq[:, k]),
                ),
                options={"disp": False, "presolve": True},
            )
            for k in _range(b_ub.shape[1])
        ]
    elif n_cpu < 0 or n_cpu > 1:
        optimizations = Parallel(n_jobs=n_cpu)(
            (
                delayed(milp)(
                    c=c,
                    bounds=Bounds(bounds[:, 0], bounds[:, 1], keep_feasible=False),
                    constraints=(
                        LinearConstraint(a_ub, -np.inf, b_ub[:, k]),
                        LinearConstraint(a_eq, b_eq[:, k], b_eq[:, k]),
                    ),
                    options={"disp": False, "presolve": True},
                )
                for k in _range(b_ub.shape[1])
            )
        )
    return optimizations


def _milp_batch(
    c: np.ndarray,
    bounds: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
) -> tuple:
    """Solve the batch; return ``(x, status)``, rows of NaN where milp gave no solution."""
    bounds = Bounds(bounds[:, 0], bounds[:, 1], keep_feasible=False)
    n_samples = b_ub.shape[1]
    x = np.full((n_samples, len(c)), np.nan)
    status = np.empty(n_samples, dtype=int)
    for k in range(n_samples):
        result = milp(
            c=c,
            bounds=bounds,
            constraints=(
                LinearConstraint(a_ub, -np.inf, b_ub[:, k]),
                LinearConstraint(a_eq, b_eq[:, k], b_eq[:, k]),
            ),
            options={"disp": False, "presolve": True},
        )
        status[k] = result.status
        if result.x is not None:
            x[k] = result.x
    return x, status


def _warn_solver_failures(x: np.ndarray, status: np.ndarray) -> None:
    failed = np.isnan(x).all(axis=1) if x.size else np.zeros(len(status), dtype=bool)
    if failed.any():
        codes, counts = np.unique(status[failed], return_counts=True)
        detail = ", ".join(
            f"{n} x status {code} ({MILP_STATUS_MESSAGES.get(int(code), 'unknown')})"
            for code, n in zip(codes, counts)
        )
        warnings.warn(
            f"The optimisation failed for {int(failed.sum())} of {len(status)} samples ({detail}); "
            "their rows are NaN. Pass return_status=True to get the status of every sample.",
            SolverFailureWarning,
            stacklevel=3,
        )


def milp_batch_sequential(
    c: np.ndarray,
    bounds: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    return_status: bool = False,
) -> Union[np.ndarray, tuple]:
    """Optimize a batch of linear programming problems sequentially.

    Solve a sequence of Mixed-Integer Linear Programming (MILP)
    problems sharing common objective coefficients and constraint
    matrices, but varying constraint bounds.

    Parameters
    ----------
    c : np.ndarray
        Coefficients of the linear objective function to be
        minimized.
    bounds : np.ndarray
        An (n, 2) array defining the lower and upper bounds of
        variables.
    a_ub : np.ndarray
        2D array for the upper-bound inequality constraints.
    b_ub : np.ndarray
        2D array of upper-bound values for each inequality
        constraint per problem.
    a_eq : np.ndarray
        2D array for the equality constraints.
    b_eq : np.ndarray
        2D array of equality constraint values per problem.
    return_status : bool, optional
        Also return the ``scipy.optimize.milp`` status of every problem
        (default False).

    Returns
    -------
    x : np.ndarray
        2D array of optimized decision variables for each problem in
        the batch. A row is NaN when the solver returned no solution
        (e.g. infeasible problem).
    status : np.ndarray of int
        Only if `return_status`: the status of each problem, see
        `MILP_STATUS_MESSAGES` (0 is optimal).

    Warns
    -----
    otaf.exceptions.SolverFailureWarning
        If some problems have no solution.

    Notes
    -----
    This function solves MILP problems using
    ``scipy.optimize.milp``. Solver options are set to
    ``disp=False`` and ``presolve=True`` for efficiency. Problems
    share `c`, `a_ub`, and `a_eq`, while `b_ub` and `b_eq` vary per
    batch element. A failed problem no longer aborts the batch.
    """
    x, status = _milp_batch(c, bounds, a_ub, b_ub, a_eq, b_eq)
    _warn_solver_failures(x, status)
    return (x, status) if return_status else x


def compute_gap_optimizations_on_sample_batch(
    constraint_matrix_generator: SystemOfConstraintsAssemblyModel,
    deviation_array: np.ndarray,
    C: Optional[np.ndarray] = None,
    bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    n_cpu: int = 1,
    batch_size: int = 1000,
    progress_bar: bool = False,
    verbose: int = 0,
    dtype: str = "float32",
    return_status: bool = False,
) -> Union[np.ndarray, tuple]:
    """Compute gap optimizations using batch processing and MILP.

    Perform parallel batch optimization for a system of
    constraints, grouping samples into batches to improve
    computational throughput and reduce parallelization overhead.

    Parameters
    ----------
    constraint_matrix_generator : SystemOfConstraintsAssemblyModel
        Generator object that produces constraint matrices.
    deviation_array : np.ndarray
        Array of deviations to process.
    C : np.ndarray, optional
        Coefficient matrix for the linear objective function. Default
        is None.
    bounds : list of list of float or np.ndarray, optional
        Bounds for the optimization variables. Default is None.
    n_cpu : int, optional
        Number of CPUs for parallel processing. Negative values are
        relative to total available CPUs. Default is 1.
    batch_size : int, optional
        Number of points per parallel batch. Default is 1000.
    progress_bar : bool, optional
        Whether to display a progress bar. Default is False.
    verbose : int, optional
        Verbosity level for debugging. Default is 0.
    dtype : str, optional
        Data type for the resulting optimized decision variable array.
        Default is ``'float32'``.
    return_status : bool, optional
        Also return the ``scipy.optimize.milp`` status of every sample
        (default False).

    Returns
    -------
    x : np.ndarray
        Optimized decision variables for all samples stacked into a
        single array. A row is NaN when the solver returned no solution
        for that sample; the slack ``x[:, -1]`` is then NaN too, so count
        failures with ``np.isnan`` before computing a failure probability.
    status : np.ndarray of int
        Only if `return_status`: the status of each sample, see
        `MILP_STATUS_MESSAGES` (0 is optimal).

    Warns
    -----
    otaf.exceptions.SolverFailureWarning
        If some samples have no solution.
    """
    c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(
        deviation_array, bounds=bounds, C=C
    )
    N_cpu_avail = cpu_count()
    N_cpu_reque = n_cpu if n_cpu >= 0 else N_cpu_avail + n_cpu
    N_points = b_ub.shape[1]
    if N_points > batch_size:
        if N_points // batch_size < N_cpu_reque:
            N_cpu_reque = N_points // batch_size
        if progress_bar:
            _range = get_tqdm_range()(0, N_points, batch_size, unit_scale=batch_size)
        else:
            _range = range(0, N_points, batch_size)
        batches = Parallel(n_jobs=N_cpu_reque)(
            (
                delayed(_milp_batch)(
                    c,
                    bounds,
                    a_ub,
                    b_ub[:, i : i + batch_size],
                    a_eq,
                    b_eq[:, i : i + batch_size],
                )
                for i in _range
            )
        )
        x = np.vstack([batch[0] for batch in batches], dtype=dtype)
        status = np.concatenate([batch[1] for batch in batches])
    else:
        x, status = _milp_batch(c, bounds, a_ub, b_ub, a_eq, b_eq)
        x = np.array(x, dtype=dtype)
    _warn_solver_failures(x, status)
    return (x, status) if return_status else x
