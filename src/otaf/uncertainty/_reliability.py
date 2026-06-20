from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "compute_failure_probability_subset_sampling",
    "compute_failure_probability_FORM",
    "compute_failure_probability_NAIS",
    "compute_failure_probability_SUBSET",
    "compute_gap_optimizations_on_sample",
    "compute_failure_probability_basic",
    "compute_adaptive_failure_probability",
    "compute_gap_optimizations_on_sample_w_steps",
    "milp_batch_sequential",
    "compute_gap_optimizations_on_sample_batch",
]

import numbers

from time import time

import numpy as np

from scipy.optimize import linprog, milp, OptimizeResult, LinearConstraint, Bounds

import openturns as ot

from joblib import Parallel, delayed, cpu_count

from beartype import beartype
from beartype.typing import List, Union, Callable, Optional

from otaf.common import get_tqdm_range

# Robust check for version compatibility
if hasattr(ot, 'JointDistribution'):
    # New versions (v1.24+)
    JointDistribution = ot.JointDistribution
else:
    # Older versions
    JointDistribution = ot.ComposedDistribution

def compute_failure_probability_FORM(
    otFunc, composed_distribution, threshold=0.0, start_point=None, verbose=False, solver=None
):
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
        print("Design point in physical space : ", result.getPhysicalSpaceDesignPoint())
        print("Design point in standard space : ", result.getStandardSpaceDesignPoint())
        print("Hasofer index : ", result.getHasoferReliabilityIndex())
        print("Probability of failure (FORM) Pf = {:.16f}".format(result.getEventProbability()))
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

    Args:
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
    """Compute gap optimizations on a sample using a constraint matrix generator.

    Args:
        constraint_matrix_generator (otaf.SystemOfConstraintsAssemblyModel): Generator for constraint matrices.
        deviation_array (np.ndarray): Deviation array.
        C (np.ndarray, optional): Coefficient matrix for the linear objective function.
        bounds (Union[list[list[float]], np.ndarray], optional): Bounds for the variables.
        n_cpu (int, optional): Number of CPUs to use for parallel processing.
        progress_bar (bool, optional): Whether to show a progress bar.
        verbose (int, optional): Level of verbosity for debugging.

    Returns
    -------
        list: List of optimization results.
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


def milp_batch_sequential(c, bounds, a_ub, b_ub, a_eq, b_eq):
    """
    Optimize a batch of linear optimization problems iteratively.

    Parameters
    ----------
    c : array_like
        Coefficients of the linear objective function to be minimized.
    bounds : array_like
        An (n, 2) array defining the lower and upper bounds of variables.
    a_ub : array_like
        2-D array for the upper-bound inequality constraints.
    b_ub : array_like
        2-D array of upper-bound values for each inequality constraint per problem.
    a_eq : array_like
        2-D array for the equality constraints.
    b_eq : array_like
        2-D array of equality constraint values per problem.

    Returns
    -------
    np.ndarray
        2-D array of optimized decision variables for each problem in the batch.

    Notes
    -----
    Solves a batch of mixed-integer linear programming (MILP) problems iteratively
    using the `milp` solver from `scipy.optimize`. Each problem shares `c`, `a_ub`,
    and `a_eq`, but has unique `b_ub` and `b_eq` vectors. Solver options disable
    display (`disp: False`) and enable presolve (`presolve: True`).
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
    batch_size=1000,
    progress_bar: Optional[bool] = False,
    verbose=0,
    dtype="float32",
) -> List[OptimizeResult]:
    """Compute gap optimizations on a sample using a constraint matrix generator with batching.

    Args:
        constraint_matrix_generator (otaf.SystemOfConstraintsAssemblyModel): Generator for constraint matrices.
        deviation_array (np.ndarray): Deviation array.
        C (np.ndarray, optional): Coefficient matrix for the linear objective function.
        bounds (Union[list[list[float]], np.ndarray], optional): Bounds for the variables.
        n_cpu (int, optional): Number of CPUs to use for parallel processing.
        progress_bar (bool, optional): Whether to show a progress bar.
        verbose (int, optional): Level of verbosity for debugging.

    Returns
    -------
        list: List of optimization results.
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


@beartype
def compute_failure_probability_basic(
    constraint_matrix_generator: Callable,
    deviation_array: np.ndarray,
    C: Optional[np.ndarray] = None,
    bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    n_cpu: int = 1,
) -> float:
    """Calculate failure probability for a sample of defects.

    Args:
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
    c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(
        deviation_array, bounds=bounds, C=C
    )
    n = b_ub.shape[1]
    if n_cpu <= 1:
        optimizations = np.array(
            [
                linprog(
                    c=c,
                    A_ub=a_ub,
                    b_ub=b_ub[:, k],
                    A_eq=a_eq,
                    b_eq=b_eq[:, k],
                    bounds=bounds,
                    method=method,
                )
                for k in range(n)
            ],
            dtype=int,
        )
    elif n_cpu < 0 or n_cpu > 1:
        optimizations = Parallel(n_jobs=n_cpu)(
            delayed(linprog)(
                c,
                A_ub=a_ub,
                b_ub=b_ub[:, k],
                A_eq=a_eq,
                b_eq=b_eq[:, k],
                bounds=bounds,
                method=method,
            )
            for k in range(n)
        )
        successes = np.array([int(opti.success) for opti in optimizations], int)
    print(f"Experiment size: {n} | N_success: {sum(successes)}")
    return 1 - sum(successes) / n


@beartype
def compute_adaptive_failure_probability(
    constraint_matrix_generator: Callable,
    deviation_array: np.ndarray,
    C: Optional[np.ndarray] = None,
    bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    epsilon_pf: float = 0.001,
    epsilon_std: float = 0.005,
    verbose: int = 0,
    **kwargs,
) -> float:
    """Calculate the failure probability for a sample of defects.

    Args:
        deviation_array (np.ndarray): Array representing the range of random deviations for each variable.
        constraint_matrix_generator (Callable): Class responsible for fixing deviations and defining constraints.
        bounds (List, optional): Bounds for the gap variables. Defaults to None.
        C (np.ndarray, optional): Coefficient matrix for the linear objective function. Defaults to None.
        method (str, optional): The algorithm to use in the optimization problem. Defaults to 'highs'.
        epsilon_pf (float, optional): Convergence threshold for the ratio of change in failure probability. Defaults to 0.001.
        epsilon_std (float, optional): Convergence threshold for the standard deviation of failure probabilities. Defaults to 0.005.
        verbose (int, optional): Verbosity level for printing progress. Defaults to 0.

    Returns
    -------
        float: Failure probability for the fixed deviations.

    Raises
    ------
        TypeError: If constraint_matrix_generator is not callable.
    """
    start_time = time()
    c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(
        deviation_array, bounds=bounds, C=C
    )
    successes_count = 0  # Initialize sum of successes
    successes = []
    failure_probs = []

    for k in range(b_ub.shape[1]):
        success = int(
            milp(
                c=c,
                bounds=Bounds(bounds[:, 0], bounds[:, 1], keep_feasible=False),
                constraints=(
                    LinearConstraint(a_ub, -np.inf, b_ub[:, k]),
                    LinearConstraint(a_eq, b_eq[:, k], b_eq[:, k]),
                ),
                options={"disp": False, "presolve": True},
            ).success
        )

        successes.append(success)
        successes_count += success
        failure_probs.append(float(1 - successes_count / len(successes)))
        if (
            k >= 100
            and (
                abs(failure_probs[-1] - (1 - (sum(successes) - 1) / (len(successes) + 1)))
                <= epsilon_pf
            )
            and (np.std(failure_probs[k:]) <= epsilon_std)
            and (sum(successes) <= k + 1)
            and (sum(successes) > 1)
        ):
            if verbose > 0:
                print(
                    " | ".join(
                        f"Experiment size : {k + 1}",
                        f"N_success : {sum(successes)}",
                        f"Pf : {round(failure_probs[-1], 5)}",
                        f"eps std : {np.std(failure_probs).round(5)}",
                        f"eps Pf : {round(abs(failure_probs[-1] - (1 - (sum(successes)-1)/(len(successes)+1))),5)}",
                        f"Elapsed time: {time() - start_time:.6f} s.",
                    )
                )
            return failure_probs[-1]
        if k > 200 and failure_probs[-1] > 0.999:
            return failure_probs[-1]
        if k == b_ub.shape[1] - 1:
            return failure_probs[-1]
    return failure_probs[-1]


@beartype
def linprog_w_steps(
    c,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    method="revised simplex",
    callback=None,
    options=None,
    x0=None,
):
    # List to store intermediate solutions
    intermediate_solutions = []

    # Define a callback function to collect intermediate solutions
    def internal_callback(res):
        intermediate_solutions.append(res.x.copy())
        if callback:
            callback(res)

    # Perform the linear programming optimization
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
        callback=internal_callback,
        options=options,
        x0=x0,
    )

    # Return the array of all solutions for each step
    return intermediate_solutions, res


@beartype
def compute_gap_optimizations_on_sample_w_steps(
    constraint_matrix_generator: Callable,
    deviation_array: np.ndarray,
    C: Optional[np.ndarray] = None,
    bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    n_cpu: int = 1,
    verbose: int = 0,
    progress_bar: bool = False,
) -> List:
    """Calculate failure probability for a sample of defects.

    Args:
        constraint_matrix_generator (Callable): Class for fixing deviations and defining constraints.
        deviation_array (np.ndarray): Array of random deviations for each variable.
        C (np.ndarray, optional): Coefficient matrix for the linear objective function. Defaults to None.
        bounds (List, optional): Bounds for gap variables. Defaults to None.
        n_cpu (int, optional): Number of CPUs to use for parallel execution. Defaults to 1.
        verbose (int, optional): Verbosity level. Defaults to 0.
        progress_bar (bool, optional): Show progress bar. Defaults to False.

    Returns
    -------
        List: List of optimization results including intermediate steps.

    Raises
    ------
        TypeError: If constraint_matrix_generator is not callable.
    """
    c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(
        deviation_array, bounds=bounds, C=C
    )

    if verbose > 0:
        print("Bounds:", bounds)

    if progress_bar:
        _range = get_tqdm_range()(b_ub.shape[1])
    else:
        _range = range(b_ub.shape[1])

    def optimize_single(k):
        return linprog_w_steps(
            c=c,
            A_ub=a_ub,
            b_ub=b_ub[:, k],
            A_eq=a_eq,
            b_eq=b_eq[:, k],
            bounds=bounds,
            method="revised simplex",
        )

    if n_cpu <= 1:
        optimizations = [optimize_single(k) for k in _range]
    else:
        optimizations = Parallel(n_jobs=n_cpu)(delayed(optimize_single)(k) for k in _range)

    return optimizations
