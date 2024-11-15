# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "normal_score_mu",
    "normal_score_sigma",
    "intermediate_lambda_function",
    "intermediate_lambda_function_derivative",
    "monte_carlo_non_compliancy_rate_gradient",
    "monte_carlo_non_compliancy_rate_gradient_start_space",
    "lambda_sample_composition",
    "monte_carlo_non_compliancy_rate_at_threshold_w_gradient",
    "monte_carlo_non_compliancy_rate_w_gradient",
    "sample_non_compliancy_at_threshold",
    "sample_non_compliancy_rate",
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

import os
import logging
import numbers
import re

from time import time

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.optimize import linprog, milp, OptimizeResult, LinearConstraint, Bounds

import openturns as ot
import trimesh as tr
from functools import partial, lru_cache

from joblib import Parallel, delayed, cpu_count

from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional

import otaf


#######################################################################################
########### SCORE FUNCTIONS / GRADIENTS


@beartype
def intermediate_lambda_function(lambdas: np.ndarray) -> np.ndarray:
    """Ensure sum(lambdas**2) == 1."""
    return np.sqrt(lambdas)


@beartype
def intermediate_lambda_function_derivative(lambdas: np.ndarray) -> np.ndarray:
    """Derivative of the lambda function."""
    return 1 / (2 * np.sqrt(lambdas))


@beartype
def normal_score_mu(
    sample: np.ndarray, mean: Union[np.ndarray, float], std: Union[np.ndarray, float]
) -> np.ndarray:
    """Gradient of log Gaussian w.r.t. mean."""
    return (1 / std) * ((sample - mean) / std)


@beartype
def normal_score_sigma(
    sample: np.ndarray, mean: Union[np.ndarray, float], std: Union[np.ndarray, float]
) -> np.ndarray:
    """Gradient of log Gaussian w.r.t. standard deviation."""
    return (1 / std) * (np.square((sample - mean) / std) - 1)


@beartype
def monte_carlo_non_compliancy_rate_gradient(
    input_samples: np.ndarray,
    output_non_compliancy_sample: np.ndarray,
    input_distribution_mean: Union[np.ndarray, float],
    input_distribution_standard: Union[np.ndarray, float],
    ref_standards: Union[np.ndarray, list],
) -> np.ndarray:
    """Gradient of non-compliancy rate w.r.t. standard deviations."""
    gradients = np.multiply(
        output_non_compliancy_sample[:, np.newaxis],
        normal_score_sigma(input_samples, input_distribution_mean, input_distribution_standard),
    )
    gradients = np.multiply(gradients, np.asarray(ref_standards))
    return gradients.mean(axis=0)


@beartype
def monte_carlo_non_compliancy_rate_gradient_start_space(
    lambdas: np.ndarray, gradient_composed: np.ndarray
) -> np.ndarray:
    """Gradient in the space of lambdas."""
    return intermediate_lambda_function_derivative(lambdas) * gradient_composed


@beartype
def lambda_sample_composition(lambdas: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Compose lambdas with samples."""
    return sample * lambdas[np.newaxis, :]


@beartype
def monte_carlo_non_compliancy_rate(non_compliancy_monte_carlo_sample: np.ndarray) -> numbers.Real:
    """Compute non-compliancy rate."""
    return non_compliancy_monte_carlo_sample.mean()


def monte_carlo_non_compliancy_rate_at_threshold_w_gradient(
    lambdas, compliancy_threshold, sample, means, standards, model, model_is_bool=False
):
    """Compute non-compliancy rate and its gradient at a given lambda."""
    lambdas_sqrt = intermediate_lambda_function(lambdas)
    standards_composed = np.multiply(lambdas_sqrt, standards)
    means_composed = np.multiply(lambdas_sqrt, means)
    samples_composed = lambda_sample_composition(lambdas_sqrt, sample)
    model_results = model(samples_composed)
    if not model_is_bool:
        model_non_compliancy_sample = sample_non_compliancy_at_threshold(
            model_results, compliancy_threshold
        )
    else:
        model_non_compliancy_sample = model_results
    model_non_compliancy_gradient = monte_carlo_non_compliancy_rate_gradient(
        samples_composed, model_non_compliancy_sample, means_composed, standards_composed, standards
    )
    model_non_compliancy_gradient_start_space = (
        monte_carlo_non_compliancy_rate_gradient_start_space(lambdas, model_non_compliancy_gradient)
    )
    model_non_compliancy_rate = monte_carlo_non_compliancy_rate(model_non_compliancy_sample)
    return model_non_compliancy_rate, model_non_compliancy_gradient_start_space


def monte_carlo_non_compliancy_rate_w_gradient(
    compliancy_threshold, sample, means, standards, model, model_is_bool=False
):
    """Partial function for computing non-compliancy rate and gradient."""
    return partial(
        monte_carlo_non_compliancy_rate_at_threshold_w_gradient,
        compliancy_threshold=compliancy_threshold,
        sample=sample,
        means=means,
        standards=standards,
        model=model,
        model_is_bool=model_is_bool,
    )


#######################################################################################
########### MODEL POST-PROCESSING


def sample_non_compliancy_at_threshold(
    model_results: Optional[np.ndarray] = None,
    compliancy_threshold: float = 0.0,
    optimizations: Optional[List[OptimizeResult]] = None,
    optimization_variable: bool = False,
) -> np.ndarray:
    """Calculate non-compliancy at a specified threshold level.

    Args:
        model_results (np.ndarray, optional): Array of gap values for each sample. If provided, non-compliancy
            is determined based on these values.
        compliancy_threshold (float, optional): Threshold value for determining compliancy. Defaults to 0.0.
        optimizations (List[OptimizeResult], optional): List of optimization results. Used if `model_results` is
            not provided. Defaults to None.
        optimization_variable (bool, optional): Flag indicating whether the optimizations represent a variable to
            be checked for compliancy. Defaults to False.

    Returns:
        np.ndarray: Array indicating non-compliancy at each sample or optimization result. Each element is 1 if the
            sample/optimization is non-compliant (below threshold), otherwise 0.
    """
    if model_results is not None:
        return np.where(model_results <= compliancy_threshold, 1, 0)

    elif optimization_variable is True:
        opt_var_values = np.array(
            [opt.fun for opt in optimizations]
        )  # if s variable all the others are 0 so c@y=s
        return np.where(opt_var_values <= compliancy_threshold, 1, 0)

    elif optimizations:
        successes = np.array([int(opt.success) for opt in optimizations], dtype=int)
        return 1 - successes  # Inevrting to get non compliancy rate

    else:
        raise ValueError("Arguments missing")


@beartype
def sample_non_compliancy_rate(non_compliancy_indicator_sample: np.ndarray):
    """Returns the mean of an indicator sample. The sample is supposed to be representative of the
    distribution and only is valid in monte carlo simulations.
    """
    return non_compliancy_indicator_sample.mean()


#######################################################################################
########### MODEL EVALUATIONS / FAILURE PROBABILITIES


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
    distribution: ot.ComposedDistribution,
    threshold: float = 0.0,
    quantile_level: float = 0.001,
    verbose: bool = False,
) -> tuple[float, ot.SimulationResult]:
    """
    Compute the failure probability using the NAIS algorithm.

    Parameters:
    - func: The function g(X)
    - distribution: The input random vector distribution
    - threshold: The threshold value (default=0.0)
    - verbose: Print additional information (default=False)

    Returns:
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
    distribution: ot.ComposedDistribution,
    threshold: float = 0.0,
    verbose: bool = False,
    proposalRange=2,
    targetProbability=0.1,
) -> tuple[float, ot.SimulationResult]:
    """
    Compute the failure probability using the NAIS algorithm.

    Parameters:
    - func: The function g(X)
    - distribution: The input random vector distribution
    - threshold: The threshold value (default=0.0)
    - verbose: Print additional information (default=False)

    Returns:
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

    Returns:
        float: Failure probability for the fixed deviations.

    Raises:
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
    constraint_matrix_generator: otaf.ToleranceAnalysisMatrixPreparer,
    deviation_array: np.ndarray,
    C: Optional[np.ndarray] = None,
    bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    n_cpu: int = 1,
    progress_bar: Optional[bool] = False,
) -> List[OptimizeResult]:
    """Compute gap optimizations on a sample using a constraint matrix generator.

    Args:
        constraint_matrix_generator (otaf.ToleranceAnalysisMatrixPreparer): Generator for constraint matrices.
        deviation_array (np.ndarray): Deviation array.
        C (np.ndarray, optional): Coefficient matrix for the linear objective function.
        bounds (Union[list[list[float]], np.ndarray], optional): Bounds for the variables.
        n_cpu (int, optional): Number of CPUs to use for parallel processing.
        progress_bar (bool, optional): Whether to show a progress bar.
        verbose (int, optional): Level of verbosity for debugging.

    Returns:
        list: List of optimization results.
    """
    c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(
        deviation_array, bounds=bounds, C=C
    )

    # Linprog is being depreciated on scipys side, so we switch to milp

    if progress_bar:
        _range = otaf.common.get_tqdm_range()
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
    constraint_matrix_generator: otaf.ToleranceAnalysisMatrixPreparer,
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
        constraint_matrix_generator (otaf.ToleranceAnalysisMatrixPreparer): Generator for constraint matrices.
        deviation_array (np.ndarray): Deviation array.
        C (np.ndarray, optional): Coefficient matrix for the linear objective function.
        bounds (Union[list[list[float]], np.ndarray], optional): Bounds for the variables.
        n_cpu (int, optional): Number of CPUs to use for parallel processing.
        progress_bar (bool, optional): Whether to show a progress bar.
        verbose (int, optional): Level of verbosity for debugging.

    Returns:
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
            _range = otaf.common.get_tqdm_range()(0, N_points, batch_size, unit_scale=batch_size)
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

    Returns:
        float: Failure probability for the fixed deviations.

    Raises:
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

    Returns:
        float: Failure probability for the fixed deviations.

    Raises:
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

    Returns:
        List: List of optimization results including intermediate steps.

    Raises:
        TypeError: If constraint_matrix_generator is not callable.
    """
    c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(
        deviation_array, bounds=bounds, C=C
    )

    if verbose > 0:
        print("Bounds:", bounds)

    if progress_bar:
        _range = otaf.common.get_tqdm_range()(b_ub.shape[1])
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
