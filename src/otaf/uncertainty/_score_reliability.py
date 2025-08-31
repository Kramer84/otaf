from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "normal_score_mu",
    "normal_score_sigma",
    "intermediate_lambda_function",
    "intermediate_lambda_function_derivate",
    "monte_carlo_non_compliancy_rate_gradient",
    "monte_carlo_non_compliancy_rate_gradient_start_space",
    "lambda_sample_composition",
    "monte_carlo_non_compliancy_rate_at_threshold_w_gradient",
    "monte_carlo_non_compliancy_rate_w_gradient",
    "sample_non_compliancy_at_threshold",
]

import os
import logging
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
def normal_score_mu(
    sample: np.ndarray, mean: Union[np.ndarray, float], standard: Union[np.ndarray, float]
):
    """Calculate the gradient/score of the logarithm of the Gaussian distribution with respect to the mean.

    Args:
        sample (np.ndarray): The observed sample of shape NxM.
        mean (float or np.ndarray): The mean of the Gaussian distribution. If an array, it should be of shape M.
        standard (float or np.ndarray): The standard deviation of the Gaussian distribution. If an array, it should be of shape M.

    Returns:
        np.ndarray: The gradient of the logarithm of the Gaussian distribution with respect to the mean at each point.
        Array of shape NxM.
    """
    return (1 / standard) * ((sample - mean) / standard)


@beartype
def normal_score_sigma(
    sample: np.ndarray, mean: Union[np.ndarray, float], standard: Union[np.ndarray, float]
):
    """Calculate the gradient/score of the logarithm of the Gaussian distribution with respect to the standard.

    Args:
        sample (np.ndarray): The observed sample of shape NxM.
        mean (float or np.ndarray): The mean of the Gaussian distribution. If an array, it should be of shape M.
        standard (float or np.ndarray): The standard deviation of the Gaussian distribution. If an array, it should be of shape M.

    Returns:
        np.ndarray: The gradient of the logarithm of the Gaussian distribution with respect to the standard at each point.
        Array of shape NxM.
    """
    return (1 / standard) * (np.square((sample - mean) / standard) - 1)


def intermediate_lambda_function(lambdas: np.ndarray):
    """Intermediate function acting on the lambdas to respect the constraint on the total standard deviation such
    that the sum of lambdas**2 == 1
    """
    return np.sqrt(lambdas)


def intermediate_lambda_function_derivate(lambdas: np.ndarray):
    """Derivate of intermediate function acting on the lambdas to respect the constraint on the total standard deviation"""
    return 1 / (2 * np.sqrt(lambdas))


def monte_carlo_non_compliancy_rate_gradient(
    input_samples,
    output_non_compliancy_sample,
    input_distribution_mean,
    input_distribution_standard,
    ref_standards,
):
    """Calculate the gradient of the non-compliancy rate with respect to the modified distributions.

    This function computes the gradient of the non-compliancy rate with respect to the standard deviations of the
    composed distribution, based on Monte Carlo samples.

    Args:
        input_samples (np.ndarray): Array of reference samples for the composed distribution.
        output_non_compliancy_sample (np.ndarray): Array indicating non-compliancy of the result for each point of the input samples.
        input_distribution_mean (np.ndarray): Array of mean values for the composed distribution. (multiplied by lambda)
        input_distribution_standard (np.ndarray): Array of standard deviations for the composed distribution. (multiplied by lambda)
        ref_standards (np.ndarray): Array of reference standard deviations. (before multiplication with lambda.)

    Returns:
        np.ndarray: Array containing the gradient of the non-compliancy rate with respect to the standard deviations
            of the composed distribution. The gradient is computed as the mean gradient over all Monte Carlo samples.
    """
    gradients = np.multiply(
        output_non_compliancy_sample[:, np.newaxis],
        normal_score_sigma(input_samples, input_distribution_mean, input_distribution_standard),
    )
    gradients = np.multiply(gradients, ref_standards)
    return gradients.mean(axis=0)


def monte_carlo_non_compliancy_rate_gradient_start_space(
    lambdas: np.ndarray, gradient_composed: np.ndarray
):
    """Expressing the gradient in the space of the lambdas using classic function composition."""
    return intermediate_lambda_function_derivate(lambdas) * gradient_composed


@beartype
def lambda_sample_composition(lambdas: np.ndarray, sample: np.ndarray):
    """Compose a point of lambda values of dimension M with a sample of size N x M"""
    return sample * lambdas[np.newaxis, :]


def monte_carlo_non_compliancy_rate(non_compliancy_monte_carlo_sample):
    return non_compliancy_monte_carlo_sample.mean()


def monte_carlo_non_compliancy_rate_at_threshold_w_gradient(
    lambdas, compliancy_threshold, sample, means, standards, model, model_is_bool=False
):
    """Function to get the non compliancy probability with respect to the distribution allocation
    parameters (lambda), and the gradient with respect to these lambda values at that point.
    Serves to feed the optimization algorithm to find the sup/inf non compliancy probabilities
    """
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
