from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "FeatureLevelStatisticalConstraint",
    "ComposedAssemblyLevelStatisticalConstraint",
    "NormalizedAssemblyLevelConstraint"
]

import numpy as np
import sympy as sp

import openturns as ot

from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional, Sequence, Any

from otaf.sampling import scale_sample_with_params, generate_and_transform_sequence

@beartype
class FeatureLevelStatisticalConstraint:
    """
    Class representing a statistical constraint at the feature level for the
    distribution parameters of the DOFs. Two main constraint types are supported:
    - Probability-based
    - Variance-based

    The distribution is from openturns. Constraint satisfaction is estimated
    via Monte Carlo.

    ...
    """
    def __init__(
        self,
        modal_interval_func: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
        mif_args: Tuple = (),
        n_dof: int = 0,
        n_sample: int = 100000,
        target: str = 'std', #or prob
        target_val: float = 0.1,
        isNormal: bool = False,
        normalizeOutput: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        modal_interval_func : Callable
            Function that takes a sample plus additional args and returns
            a tuple (valid, distance).
        mif_args : Tuple
            Extra arguments passed to modal_interval_func.
        n_dof : int
            Number of degrees of freedom (if relevant).
        n_sample : int
            Number of samples for Monte Carlo.
        """
        self.modal_interval_func = modal_interval_func
        self.arguments = mif_args
        self.n_dof = n_dof
        self.n_sample = n_sample

        # Either the target probability of failure or target standard deviation
        self.target = target
        self.target_val = target_val

        self.dist: Optional[ot.Distribution] = None
        self.dist_params = None

        self.sample: Optional[np.ndarray] = None
        self.res: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self.isNormal = isNormal
        self.normalSample = None #We can define a standard normal sample that will be used to generate samples using mean/std operations from it. Would make calculations faster and less biased

        self.normalizeOutput = normalizeOutput

        self.__initialize_obj()

    def __call__(
        self,
        param: Union[list, np.ndarray, ot.Point],
        ) -> float:
        """Gets called with a set of params and returns the error from the target
        """
        if self.isNormal :
            self.dist_params = param
            sample = scale_sample_with_params(self.normalSample, param)
            self.compute_on_sample(sample)
        else :
            self.set_distribution_parameter(param)
            self.generate_sample()
            self.compute_on_sample()

        if self.target == 'std':
            return (self.res[1].std() - self.target_val) / (self.target_val if self.normalizeOutput else 1.0)
        elif self.target == 'prob':
            return ((1 - self.res[0].mean()) - self.target_val) / (self.target_val if self.normalizeOutput else 1.0)
        else:
            raise ValueError("Target must be std or prob")

    def __initialize_obj(self):

        if self.isNormal :
            # Later we'll use some better generator
            # sample = np.random.normal(size=(self.n_sample, self.n_dof))
            dist = ot.ComposedDistribution([ot.Normal()]*self.n_dof)
            sample = generate_and_transform_sequence(self.n_dof, self.n_sample, dist)
            self.normalSample = sample

    def set_distribution(self, dist: ot.Distribution) -> None:
        """
        Assign an OpenTurns distribution object.

        Parameters
        ----------
        dist : ot.Distribution
            The distribution to be used in sample generation and computations.
        """
        self.dist = dist
        self.dist_params = dist.getParameter()

    def set_distribution_parameter(self, param: Union[list, np.ndarray, ot.Point]) -> None:
        """
        Set parameters for the stored distribution.

        Parameters
        ----------
        param : Union[list, np.ndarray, ot.Point]
            Parameter values for the distribution.
        """
        if self.dist is None:
            raise ValueError("Distribution must be set before its parameters.")
        self.dist_params = param
        self.dist.setParameter(param)

    def check_initialized(self) -> None:
        """
        Check if critical members are initialized, otherwise raise errors.
        """
        if self.dist is None and not self.isNormal:
            raise ValueError("Distribution is not set.")
        if self.modal_interval_func is None:
            raise ValueError("modal_interval_func is not set.")

    def generate_sample(self, size: Optional[int] = None) -> None:
        """
        Generate a random sample from the distribution.

        Parameters
        ----------
        size : int, optional
            Number of samples to generate. If None, defaults to self.n_sample.
        """
        self.check_initialized()
        actual_size = size or self.n_sample
        ot_sample = self.dist.getSample(actual_size)
        self.sample = np.array(ot_sample)

    def compute_on_sample(self, sample: Optional[np.ndarray] = None) -> None:
        """
        Compute the modal interval function on a sample and store the result.

        Parameters
        ----------
        sample : np.ndarray, optional
            The sample to evaluate. If None, uses self.sample.
        """
        self.check_initialized()
        if sample is None:
            if self.sample is None:
                raise ValueError("No sample provided or stored. "
                                 "Use generate_sample() or pass a sample.")
            sample = self.sample

        valid, distance = self.modal_interval_func(sample, *self.arguments)
        self.res = (valid, distance)

    def get_standard_deviation(self) -> float:
        """
        Returns the standard deviation of the distance from the last computation.

        Returns
        -------
        float
            Standard deviation of the 'distance' array.
        """
        if self.res is None:
            raise ValueError("No result stored. Call compute_on_sample() first.")
        return self.res[1].std()

    def get_failure_prob(self) -> float:
        """
        Returns the failure probability (1 - valid) from the last computation.

        Returns
        -------
        float
            Estimated failure probability, i.e., (1 - valid) / n_samples.
        """
        if self.res is None:
            raise ValueError("No result stored. Call compute_on_sample() first.")
        # self.res[0] is presumably the fraction (or array) of valid points
        # If self.res[0] is an array of 0/1 flags:
        #   failure_prob = (number of 0's) / n_sample
        # So the expression might be:
        valid_array = self.res[0]
        return (1 - valid_array.mean())  # average of (1 - valid_flags)


class ComposedAssemblyLevelStatisticalConstraint:
    """
    Combines feature-level constraints into a single constraint function
    for the entire assembly.

    Parameters
    ----------
    feature_constraint_list : Sequence[FeatureLevelStatisticalConstraint]
        A list of feature-level statistical constraint objects.

    Attributes
    ----------
    feature_constraint_list : Sequence[FeatureLevelStatisticalConstraint]
        The provided list of feature-level constraints.
    n_feature : int
        Number of features (length of feature_constraint_list).
    """

    def __init__(self, feature_constraint_list: Sequence[FeatureLevelStatisticalConstraint]):
        if not isinstance(feature_constraint_list, Sequence) or not feature_constraint_list:
            raise ValueError("feature_constraint_list must be a non-empty sequence.")

        self.feature_constraint_list = feature_constraint_list
        self.n_feature = len(self.feature_constraint_list)

    def __call__(self, composed_param: List[List[float]]) -> np.ndarray:
        """
        Evaluate the constraints for the given parameter values.

        Parameters
        ----------
        composed_param : List[List[float]]
            A list of parameter lists corresponding to each feature constraint.

        Returns
        -------
        np.ndarray
            Array of constraint evaluations for each feature.

        Raises
        ------
        AssertionError
            If the length of composed_param does not match the number of features.
        """
        if len(composed_param) != self.n_feature:
            raise ValueError(
                f"Expected {self.n_feature} parameter sets, but got {len(composed_param)}."
            )

        err_l = []
        for i, fc in enumerate(self.feature_constraint_list):
            err_l.append(fc(composed_param[i]))

        return np.array(err_l)


class NormalizedAssemblyLevelConstraint:
    """
    Handles normalized inputs for assembly constraints, using pre-defined
    parameter value bounds and assuming normal distributions.

    Parameters
    ----------
    composed_statistical_constraint : ComposedAssemblyLevelStatisticalConstraint
        The composed constraint object operating on original parameter values.
    param_val_bounds : List[List[Tuple[float, float]]]
        Bounds for each parameter of the feature distributions. The structure
        is a list of bounds per feature, with each bound as a tuple (min, max).

    Attributes
    ----------
    composed_statistical_constraint : ComposedAssemblyLevelStatisticalConstraint
        The provided composed statistical constraint.
    param_val_bounds : List[List[Tuple[float, float]]]
        The provided parameter value bounds.
    """
    def __init__(
        self,
        composed_statistical_constraint: ComposedAssemblyLevelStatisticalConstraint,
        param_val_bounds):
        self.composed_statistical_constraint = composed_statistical_constraint
        self.param_val_bounds = param_val_bounds

    def __call__(self, composed_param_normalized: List[List[float]]) -> np.ndarray:
        """
        Evaluate the normalized constraints.

        Parameters
        ----------
        composed_param_normalized : List[List[float]]
            Normalized parameter values (0 to 1) for each feature.

        Returns
        -------
        np.ndarray
            Array of constraint evaluations for the normalized parameters.
        """
        composed_param = []

        for i, feature_param in enumerate(composed_param_normalized):
            capped_params = []
            for j, norm_param in enumerate(feature_param):
                original_param = (
                    (self.param_val_bounds[i][j][1] - self.param_val_bounds[i][j][0]) * norm_param
                    + self.param_val_bounds[i][j][0]
                )
                capped_params.append(original_param)

            composed_param.append(capped_params)

        return self.composed_statistical_constraint(composed_param)
