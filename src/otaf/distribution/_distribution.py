# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "get_composed_normal_defect_distribution",
    "multiply_composed_distribution_with_constant",
    "multiply_composed_distribution_standard_with_constants",
    "get_means_standards_composed_distribution",
    "generate_correlated_samples",
    "compute_sup_inf_distributions",
    "get_prob_below_threshold",
]

import copy
import logging
import numpy as np
import sympy as sp
import openturns as ot
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional


@beartype
def get_composed_normal_defect_distribution(
    defect_names: List[Union[str, sp.Symbol]],
    mu_list: Optional[List[Union[int, float]]] = [],
    sigma_list: Optional[List[Union[int, float]]] = [],
    mu_dict: Optional[Dict[str, Union[int, float]]] = {},
    sigma_dict: Optional[Dict[str, Union[int, float]]] = {},
) -> ot.ComposedDistribution:
    """Create a composed distribution of defects based on their names and associated standard deviations.

    Args:
        defect_names (list): A list of defect variable names (symbols).
        mu_list (list, optional): List of means for each defect. If not provided, defaults to 0.0 for all defects.
        sigma_list (list, optional): List of standard deviations for each defect. If not provided, defaults to 1.0 for all defects.
        mu_dict (dict, optional): Dictionary mapping defect names to their mean values.
        sigma_dict (dict, optional): Dictionary mapping defect names to their standard deviation values.

    Returns:
        ot.ComposedDistribution: A composed distribution object.

    Notes:
        - The defect names are expected to have specific prefixes to identify their type:
            - ``u_`` for translation along the x-axis
            - ``v_`` for translation along the y-axis
            - ``w_`` for translation along the z-axis
            - ``alpha_`` for rotation around the x-axis
            - ``beta_`` for rotation around the y-axis
            - ``gamma_`` for rotation around the z-axis
        - This has to be reflected in the mu_dict/sigma_dict if provided, e.g., ``sigma_dict = {'u':1.0}``.
    """
    ND = len(defect_names)
    distributions = []

    if [mu_list, sigma_list, mu_dict, sigma_dict] == [None] * 4:
        logging.warning(
            "No argument for composed distribution have been passed, returning centered unit standard distribution"
        )
        for defect_name in defect_names:
            dist = ot.Normal(0.0, 1.0)
            dist.setDescription([str(defect_name)])
            distributions.append(dist)
        composed_distribution = ot.ComposedDistribution(distributions)
        logging.debug(f"Composed unit distribution created: {composed_distribution}")

    elif mu_list or sigma_list:
        logging.info("Using mu or sigma list of parameters.")
        if mu_list is None:
            mu_list = [0.0] * ND
        if sigma_list is None:
            sigma_list = [1.0] * ND
        if not (len(mu_list) == len(sigma_list) == ND):
            raise ValueError(
                f"The passed list of means and standard deviations have the wrong number of elements. N_means: {len(mu_list)} | N_stds: {len(sigma_list)} | N_defects: {ND} "
            )
        for i, defect_name in enumerate(defect_names):
            dist = ot.Normal(mu_list[i], sigma_list[i])
            dist.setDescription([str(defect_name)])
            distributions.append(dist)
        composed_distribution = ot.ComposedDistribution(distributions)
        logging.debug(f"Composed distribution created: {composed_distribution}")

    elif mu_dict or sigma_dict:
        logging.info("Using mu or sigma dictionary of parameters.")
        mu_keys = list(mu_dict.keys())
        sigma_keys = list(sigma_dict.keys())
        mu, sigma = 0, 1
        for i, defect_name in enumerate(defect_names):
            if any([key in str(defect_name) for key in mu_keys]):
                mu = mu_dict[mu_keys[[key in str(defect_name) for key in mu_keys].index(True)]]
            else:
                logging.info(
                    f"No mean value for defect {str(defect_name)} in mu_dict, setting mean to 0"
                )

            if any([key in str(defect_name) for key in sigma_keys]):
                sigma = sigma_dict[
                    sigma_keys[[key in str(defect_name) for key in sigma_keys].index(True)]
                ]
            else:
                logging.warning(
                    f"No standard deviation value for defect {str(defect_name)} in sigma_dict, setting standard deviation to 1"
                )

            dist = ot.Normal(mu, sigma)
            dist.setDescription([str(defect_name)])
            distributions.append(dist)
        composed_distribution = ot.ComposedDistribution(distributions)
        logging.debug(f"Composed distribution created: {composed_distribution}")
    return composed_distribution


def multiply_composed_distribution_with_constant(composed_distribution, constant):
    composed_distribution = copy.copy(composed_distribution)
    parameters = composed_distribution.getParameter()
    parameters = np.array(parameters) * constant
    composed_distribution.setParameter(parameters)
    return composed_distribution


def multiply_composed_distribution_standard_with_constants(composed_distribution, constants):
    ## Assumes all distiributions are normals (mean/std)
    composed_distribution = copy.copy(composed_distribution)
    parameters = composed_distribution.getParameter()
    for i in range(len(constants)):
        parameters[2 * i + 1] *= constants[i]
    composed_distribution.setParameter(parameters)
    return composed_distribution


def get_means_standards_composed_distribution(composed_distribution):
    """
    Extracts the means and standard deviations from the composed distribution.
    Assumes all distributions are normal (mean/std).

    Parameters:
        composed_distribution: The composed distribution of normal distributions.

    Returns:
        means: A list of the means of the distributions.
        stds: A list of the standard deviations of the distributions.
    """
    # Get the parameters of the composed distribution (assumes mean/std for normal distributions)
    parameters = composed_distribution.getParameter()

    means = []
    stds = []

    # Iterate over the parameters and extract means and standard deviations
    for i in range(len(parameters) // 2):  # Assuming every distribution has a mean and std
        means.append(parameters[2 * i])  # The mean is at index 2*i
        stds.append(parameters[2 * i + 1])  # The std is at index 2*i + 1

    return means, stds


def generate_correlated_samples(mu1=0, mu2=0, sigma1=1, sigma2=1, corr=0, N=1):
    """Generate bivariate correlated samples."""
    R = ot.CorrelationMatrix(2, [1, corr, corr, 1])
    sample = np.array(ot.Normal([mu1, mu2], [sigma1, sigma2], R).getSample(N))
    return sample


def compute_sup_inf_distributions(distributions, x_min=-10, x_max=10, n_points=int(1e4)):
    """
    Compute the supremum (sup) and infimum (inf) CDFs for a list of distributions.

    This function evaluates the Cumulative Distribution Functions (CDFs) at `n_points`
    evenly spaced points between `x_min` and `x_max` for a given list of distributions.
    It then computes the pointwise supremum and infimum of the CDFs across the
    distributions at each of these points.

    Parameters:
    ----------
    distributions : list
        A list of objects where each object has a `computeCDF(x)` method to evaluate
        the CDF at a point x.
    x_min : float
        The lower bound of the x values over which the CDFs are evaluated.
    x_max : float
        The upper bound of the x values over which the CDFs are evaluated.
    n_points : int, optional
        The number of points at which the CDFs are evaluated between `x_min` and `x_max`.
        Default is 10,000.

    Returns:
    -------
    sup_data : np.ndarray
        A 2D array with shape (n_points, 2), where the first column is the x values
        and the second column contains the pointwise supremum of the CDFs at each x.
    inf_data : np.ndarray
        A 2D array with shape (n_points, 2), where the first column is the x values
        and the second column contains the pointwise infimum of the CDFs at each x.
    """
    x_arr = np.linspace(x_min, x_max, n_points)
    sup_points = np.zeros(n_points)
    inf_points = np.zeros(n_points)

    for i, x in enumerate(x_arr):
        cdfs = [d.computeCDF(x) for d in distributions]
        sup_points[i] = max(cdfs)
        inf_points[i] = min(cdfs)

    sup_data = np.column_stack([x_arr, sup_points])
    inf_data = np.column_stack([x_arr, inf_points])

    return sup_data, inf_data


def get_prob_below_threshold(data_inf_sup, threshold=0):
    """
    Get the probability of the gap statistic being below a specified threshold.

    This function finds the element in the array `data_inf_sup` where the
    absolute value of the difference between the first column and the threshold
    is the smallest, and then returns the corresponding value from the second column.

    Parameters:
    data_inf_sup : numpy array
        Array where the first column contains gap values and the second column contains probabilities.
    threshold : float, optional
        The threshold to check against (default is 0).

    Returns:
    float
        The probability corresponding to the gap closest to the threshold.
    """
    # Find the index where the gap is closest to the threshold
    min_index = np.argmin(np.abs(data_inf_sup[:, 0] - threshold))

    # Return the corresponding probability in the second column
    return data_inf_sup[min_index, 1]
