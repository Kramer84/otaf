# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "MiSdofToleranceZones",
]

import numpy as np
import sympy as sp
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional

import otaf


class FeatureLevelStatisticalConstraint:
    """Class representing a statistical constraint at the feature level for the distribution
    parameters of the DOFs.

    There are 2 types of constraints, in probability or in variance.
    These will be estimated using monte carlo
    """
    def __init__(self, tol_zone, **feature_kwargs):
        self.tol_zone = tol_zone
        self.arguments = feature_kwargs

    def set_distribution(f=None, )
