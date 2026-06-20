from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "compute_failure_probability_subset_sampling",
    "compute_failure_probability_FORM",
    "compute_failure_probability_NAIS",
    "compute_failure_probability_SUBSET",
    "compute_gap_optimizations_on_sample",
    "milp_batch_sequential",
    "compute_gap_optimizations_on_sample_batch",
]

import numbers

from time import time

import numpy as np

from scipy.optimize import milp, OptimizeResult, LinearConstraint, Bounds

import openturns as ot

from joblib import Parallel, delayed, cpu_count

from beartype import beartype
from beartype.typing import List, Union, Callable, Optional, Any

from otaf.common import get_tqdm_range

# Robust check for version compatibility
if hasattr(ot, 'JointDistribution'):
    # New versions (v1.24+)
    JointDistribution = ot.JointDistribution
else:
    # Older versions
    JointDistribution = ot.ComposedDistribution

@beartype
def compute_failure_probability_FORM(
    otFunc: ot.Function,
    composed_distribution: JointDistribution,
    threshold: float = 0.0,
    start_point: Optional[ot.Point] = None,
    verbose: bool = False,
    solver: Optional[ot.OptimizationAlgorithm] = None
) -> tuple[float, ot.FORMResult]:
    """Compute the probability of failure using the First-Order Reliability Method (FORM).

    Calculate the failure probability for an event defined by a threshold violation
    by finding the design point in the standard normal space and applying a
    first-order approximation.

    Parameters
    ----------
    otFunc : ot.Function
        The performance function of the system.
    composed_distribution : JointDistribution
        The joint probability distribution of the input random variables.
    threshold : float, optional
        The limit state threshold, by default 0.0.
    start_point : ot.Point, optional
        Initial starting point for the optimization algorithm in physical space.
    verbose : bool, optional
        If True, print reliability diagnostics, by default False.
    solver : ot.OptimizationAlgorithm, optional
        The optimization algorithm to find the design point. Uses COBYLA 
        with strict convergence tolerances if None.

    Returns
    -------
    tuple[float, ot.FORMResult]
        The calculated failure probability and the full FORM result object.
    """
    comp_rand_vect = ot.CompositeRandomVector(otFunc, ot.RandomVector(composed_distribution))
    event = ot.ThresholdEvent(comp_rand_vect, ot.Less(), threshold)
    if not solver:
        solver = ot.Cobyla()
        solver.setMaximumIterationNumber(10000)
        solver.setMaximumAbsoluteError(1.0e-3)
        solver.setMaximumRelativeError(1.0e-3)
        solver.setMaximumResidualError(1.0e-3)
        solver.setMaximumConstraintError(1.0e-3)
    algoFORM = ot.FORM(solver, event, start_point)
    algoFORM.run()
    result = algoFORM.getResult()
    pf = result.getEventProbability()
    if verbose:
        print(f"Design point in physical space : {result.getPhysicalSpaceDesignPoint()}")
        print(f"Design point in standard space : {result.getStandardSpaceDesignPoint()}")
        print(f"Hasofer index : {result.getHasoferReliabilityIndex()}")
        print(f"Probability of failure (FORM) Pf = {pf:.16f}")
    return pf, result


def compute_failure_probability_NAIS(
    func: ot.PythonFunction,
    distribution: JointDistribution,
    threshold: float = 0.0,
    quantile_level: float = 0.001,
    verbose: bool = False,
) -> tuple[float, ot.SimulationResult]:
    """Compute the failure probability using the NAIS algorithm.

    Parameters
    ----------
    - func: The function g(X)
    - distribution: The input random vector distribution
    - threshold: The threshold value (default=0.0)
    - verbose: Print additional information (default=False)

    Returns
    -------
    - proba: The estimated failure probability
    - result: Additional NAIS algorithm results (see docstring)
    """
    # Create the output random vector Y = g(X)
    output_random_vector = ot.CompositeRandomVector(func, ot.RandomVector(distribution))
    # Create the event { Y = g(X) <= threshold }
    failure_event = ot.ThresholdEvent(output_random_vector, ot.LessOrEqual(), threshold)

    # Set up the NAIS algorithm & run
    algo = ot.NAIS(failure_event, quantile_level)
    algo.run()
    # Retrieve results
    result = algo.getResult()
    proba = result.getProbabilityEstimate()
    if verbose:
        print("Proba NAIS =", proba)
        print("Current coefficient of variation =", result.getCoefficientOfVariation())
        length_95 = result.getConfidenceLength()
        print("Confidence length (0.95) =", length_95)
        print("Confidence interval (0.95) =", [proba - length_95 / 2, proba + length_95 / 2])
    return proba, result


def compute_failure_probability_SUBSET(
    func: ot.PythonFunction,
    distribution: JointDistribution,
    threshold: float = 0.0,
    verbose: bool = False,
    proposalRange=2,
    targetProbability=0.1,
) -> tuple[float, ot.SimulationResult, Any]:
    """
    Compute the failure probability using the NAIS algorithm.

    Parameters
    ----------
    - func: The function g(X)
    - distribution: The input random vector distribution
    - threshold: The threshold value (default=0.0)
    - verbose: Print additional information (default=False)

    Returns
    -------
    - proba: The estimated failure probability
    - result: Additional NAIS algorithm results (see docstring)
    """
    # Create the output random vector Y = g(X)
    output_random_vector = ot.CompositeRandomVector(func, ot.RandomVector(distribution))
    # Create the event { Y = g(X) <= threshold }
    failure_event = ot.ThresholdEvent(output_random_vector, ot.LessOrEqual(), threshold)

    # Set up the NAIS algorithm & run
    algo = ot.SubsetSampling(failure_event, proposalRange, targetProbability)
    algo.setKeepSample(True)
    algo.run()
    # Retrieve results
    result = algo.getResult()
    proba = result.getProbabilityEstimate()
    if verbose:
        print("Proba Subset =", proba)
        print("Current coefficient of variation =", result.getCoefficientOfVariation())
        length_95 = result.getConfidenceLength()
        print("Confidence length (0.95) =", length_95)
        print("Confidence interval (0.95) =", [proba - length_95 / 2, proba + length_95 / 2])
    return proba, result, algo


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
        constraint_matrix_generator (Callable): Class for fixing deviations and defining constraints.
        deviation_array (np.ndarray): Array of random deviations for each variable.
        C (np.ndarray, optional): Coefficient matrix for the linear objective function. Defaults to None.
        bounds (List, optional): Bounds for gap variables. Defaults to None.
        method (str, optional): Algorithm for the optimization problem. Defaults to 'highs'.
        n_cpu (int, optional): Number of CPUs to use for parallel execution. Defaults to 1.

    Returns
    -------
        float: Failure probability for the fixed deviations.

    Raises
    ------
        TypeError: If constraint_matrix_generator is not callable.
    """

    def defect_func(X):
        """Small function to work as an intermediary."""
        optimizations = compute_gap_optimizations_on_sample(
            constraint_matrix_generator, X, C, bounds, n_cpu
        )
        opt_var_values = ot.Point(np.array([opt.fun for opt in optimizations], dtype=float))
        return opt_var_values

    defect_func_ot = ot.PythonFunction(defect_distribition_vector.getDimension(), 1, defect_func)

    Y = ot.CompositeRandomVector(defect_func_ot, defect_distribition_vector)
    failureEvent = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.0)
    algo = ot.SubsetSampling(failureEvent)
    algo.setKeepSample(True)
    algo.run()
    return algo.getResult()


def compute_gap_optimizations_on_sample(
    constraint_matrix_generator: "SystemOfConstraintsAssemblyModel",
    deviation_array: np.ndarray,
    C: Optional[np.ndarray] = None,
    bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    n_cpu: int = 1,
    progress_bar: Optional[bool] = False,
) -> List[OptimizeResult]:
    """Compute gap optimizations for a set of samples using MILP.

    Solve a sequence of Mixed-Integer Linear Programming (MILP) problems derived 
    from a system of constraints to determine the optimal gap for each sample.

    Parameters
    ----------
    constraint_matrix_generator : SystemOfConstraintsAssemblyModel
        Generator object that produces the constraint matrices and bounds.
    deviation_array : np.ndarray
        Array representing deviations to be processed.
    C : np.ndarray, optional
        Coefficient matrix for the linear objective function.
    bounds : Union[List[List[float]], np.ndarray], optional
        Bounds for the optimization variables.
    n_cpu : int, optional
        Number of CPUs to use for parallel processing, by default 1.
    progress_bar : bool, optional
        Whether to display a progress bar, by default False.
    verbose : int, optional
        Verbosity level for debugging, by default 0.

    Returns
    -------
    List[OptimizeResult]
        A list of optimization result objects for each sample in the input array.
    """
    c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(
        deviation_array, bounds=bounds, C=C
    )

    # Linprog is being depreciated on scipys side, so we switch to milp

    if progress_bar:
        _range = get_tqdm_range()
    else:
        _range = range

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
    return optimizations


def milp_batch_sequential(
    c: np.ndarray,
    bounds: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray
) -> np.ndarray:
    """Optimize a batch of linear problems iteratively using MILP.

    Solve a sequence of Mixed-Integer Linear Programming (MILP) problems 
    sharing common objective coefficients and constraint matrices, but 
    varying constraint bounds.

    Parameters
    ----------
    c : np.ndarray
        Coefficients of the linear objective function to be minimized.
    bounds : np.ndarray
        An (n, 2) array defining the lower and upper bounds of variables.
    a_ub : np.ndarray
        2D array for the upper-bound inequality constraints.
    b_ub : np.ndarray
        2D array of upper-bound values for each inequality constraint per problem.
    a_eq : np.ndarray
        2D array for the equality constraints.
    b_eq : np.ndarray
        2D array of equality constraint values per problem.

    Returns
    -------
    np.ndarray
        2D array of optimized decision variables for each problem in the batch.

    Notes
    -----
    - This function solves MILP problems using `scipy.optimize.milp`.
    - Solver options are set to `disp=False` and `presolve=True` for efficiency.
    - Problems share `c`, `a_ub`, and `a_eq`, while `b_ub` and `b_eq` vary per batch element.
    """
    bounds = Bounds(bounds[:, 0], bounds[:, 1], keep_feasible=False)
    optimizations = [
        milp(
            c=c,
            bounds=bounds,
            constraints=(
                LinearConstraint(a_ub, -np.inf, b_ub[:, k]),
                LinearConstraint(a_eq, b_eq[:, k], b_eq[:, k]),
            ),
            options={"disp": False, "presolve": True},
        ).x
        for k in range(b_ub.shape[1])
    ]
    return np.array(optimizations)


def compute_gap_optimizations_on_sample_batch(
    constraint_matrix_generator: "SystemOfConstraintsAssemblyModel",
    deviation_array: np.ndarray,
    C: Optional[np.ndarray] = None,
    bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    n_cpu: int = 1,
    batch_size: int = 1000,
    progress_bar: bool = False,
    verbose: int = 0,
    dtype: str = "float32"
) -> np.ndarray:
    """Compute gap optimizations on a sample using batch processing and MILP.

    Perform parallel batch optimization for a system of constraints, grouping 
    samples into batches to improve computational throughput and reduce parallelization 
    overhead.

    Parameters
    ----------
    constraint_matrix_generator : SystemOfConstraintsAssemblyModel
        Generator object that produces constraint matrices.
    deviation_array : np.ndarray
        Array of deviations to process.
    C : np.ndarray, optional
        Coefficient matrix for the linear objective function.
    bounds : Union[List[List[float]], np.ndarray], optional
        Bounds for the optimization variables.
    n_cpu : int, optional
        Number of CPUs for parallel processing. Negative values are relative 
        to total available CPUs, by default 1.
    batch_size : int, optional
        Number of points per parallel batch, by default 1000.
    progress_bar : bool, optional
        Whether to display a progress bar, by default False.
    verbose : int, optional
        Verbosity level for debugging, by default 0.
    dtype : str, optional
        Data type for the resulting optimized decision variable array, by default "float32".

    Returns
    -------
    np.ndarray
        Optimized decision variables for all samples stacked into a single array.
    """
    c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(
        deviation_array, bounds=bounds, C=C
    )

    N_cpu_avail = cpu_count()
    N_cpu_reque = n_cpu if n_cpu >= 0 else N_cpu_avail + n_cpu

    N_points = b_ub.shape[1]

    if N_points > batch_size:
        if (
            N_points // batch_size < N_cpu_reque
        ):  # We reduce the amount of workers to reduce overhead. (hopefully)
            N_cpu_reque = N_points // batch_size

        if progress_bar:
            _range = get_tqdm_range()(0, N_points, batch_size, unit_scale=batch_size)
        else:
            _range = range(0, N_points, batch_size)

        # Use joblib Parallel to handle batches in parallel
        optimizations = Parallel(n_jobs=N_cpu_reque)(
            delayed(milp_batch_sequential)(
                c, bounds, a_ub, b_ub[:, i : i + batch_size], a_eq, b_eq[:, i : i + batch_size]
            )
            for i in _range
        )
        return np.vstack(optimizations, dtype=dtype)
    else:
        optimizations = milp_batch_sequential(c, bounds, a_ub, b_ub, a_eq, b_eq)
        return np.array(optimizations, dtype=dtype)
