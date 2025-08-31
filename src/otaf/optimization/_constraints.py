from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "lambda_constraint_dict_from_composed_distribution",
    "bounds_from_composed_distribution",
    "scaling",
    "create_constraint_checker",
]

import itertools
import logging
import copy
import re
import numpy as np
import sympy as sp
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
import openturns as ot
from functools import partial, lru_cache
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional


def _fun(x, feature_idx, feature_indices):
    res = sum([x[i] for i, idx in enumerate(feature_indices) if feature_idx == idx]) - 1
    return res


def lambda_constraint_dict_from_composed_distribution(
    composed_distribution, tol=1e-12, keep_feasible=True
):
    """Condition lambda parameters to sum to 1 for each feature.

    Args:
        sample (ot.Sample): Lambda parameters with each feature ending in an integer sequence.

    Returns:
        ot.Sample: Conditioned lambda parameters.

    Raises:
        ValueError: If the description is missing or invalid.
    """
    constraint = tuple()

    if not composed_distribution.getDescription():
        raise ValueError("Description missing. Must be provided.")

    deviation_symbols = composed_distribution.getDescription()

    feature_indices = [
        int(re.search(r"\d+$", symbol).group())
        for symbol in deviation_symbols
        if re.search(r"\d+$", symbol)
    ]

    if not feature_indices:
        raise ValueError("Invalid description. Each feature should end with integers.")

    unique_features_ids = np.unique(feature_indices).tolist()

    n_symb = len(deviation_symbols)
    n_feat = len(unique_features_ids)
    constraint_matrix = np.zeros((n_feat, n_symb), dtype="float64")

    for i, feature_idx in enumerate(unique_features_ids):
        constraint += (
            {
                "type": "eq",
                "fun": partial(_fun, feature_idx=feature_idx, feature_indices=feature_indices),
            },
        )
        for j, idx in enumerate(feature_indices):
            if feature_idx == idx:
                constraint_matrix[i, j] = 1.0
    lb = np.ones((n_feat,)) - tol
    ub = np.ones((n_feat,)) + tol

    linearConstraint = LinearConstraint(constraint_matrix, lb, ub, keep_feasible)

    return constraint, linearConstraint


def bounds_from_composed_distribution(composed_distribution, tol=1e-24):
    dim = composed_distribution.getDimension()
    return Bounds(lb=[tol] * dim, ub=[1 - tol] * dim)


def scaling(scale_factor):
    """
    Decorator to scale the output of a function by a specified scaling factor.

    Parameters
    ----------
    scale_factor : float
        The factor by which to scale the output of the wrapped function.

    Returns
    -------
    function
        A decorator that applies the scaling to the output of the wrapped function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                # Scale both objective and Jacobian if it's a tuple (result and Jacobian)
                return tuple(r * scale_factor for r in result)
            else:
                # Scale only the result
                return result * scale_factor

        return wrapper

    return decorator


def create_constraint_checker(constraint, tolerance=1e-6):
    """
    Creates a function to check if a vector x satisfies the given constraint (linear or nonlinear)
    within a specified tolerance.

    Parameters
    ----------
    constraint : LinearConstraint or NonlinearConstraint
        The constraint object containing A, lb, and ub for linear, or fun, lb, and ub for nonlinear.
    tolerance : float, optional
        The tolerance within which the constraint should be satisfied (default is 1e-6).

    Returns
    -------
    function
        A function that takes a vector x and returns a boolean indicating if the constraint
        is satisfied within the tolerance, and the residuals as a dictionary.
    """

    def constraint_checker(x):
        """
        Checks if the vector x satisfies the constraint within the specified tolerance.

        Parameters
        ----------
        x : array_like
            The vector of independent variables to check.

        Returns
        -------
        bool
            True if the constraint is satisfied within the tolerance, False otherwise.
        dict
            A dictionary containing the lower and upper residuals.
        """
        if isinstance(constraint, LinearConstraint):
            # For linear constraints, use the residual method
            sl, sb = constraint.residual(x)
            lower_residuals_check = np.all(sl >= -tolerance)
            upper_residuals_check = np.all(sb >= -tolerance)
        elif isinstance(constraint, NonlinearConstraint):
            # For nonlinear constraints, evaluate the function directly
            fx = constraint.fun(x)
            lower_residuals = fx - constraint.lb
            upper_residuals = constraint.ub - fx
            lower_residuals_check = np.all(lower_residuals >= -tolerance)
            upper_residuals_check = np.all(upper_residuals >= -tolerance)
        else:
            raise ValueError("Unsupported constraint type.")

        # Check if both lower and upper residuals are within tolerance
        constraint_satisfied = lower_residuals_check and upper_residuals_check

        # Return the result and the residuals for further analysis if needed
        residuals = {
            "lower_residuals": lower_residuals if 'lower_residuals' in locals() else sl,
            "upper_residuals": upper_residuals if 'upper_residuals' in locals() else sb
        }
        return constraint_satisfied, residuals

    return constraint_checker
