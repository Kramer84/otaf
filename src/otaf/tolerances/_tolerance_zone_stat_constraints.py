# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "FeatureLevelStatisticalConstraint",
]

import numpy as np
import sympy as sp

import openturns as ot

from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional

import otaf


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
            sample = otaf.sampling.scale_sample_with_params(self.normalSample, param)
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
            self.normalSample = np.random.normal(size=(self.n_sample, self.n_dof))

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
    """Class to combine feature level constraints into a big
    constraint function for the whole assembly from which we can then
    construct the non linear constraint for scipy
    """
