# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = []

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

from joblib import Parallel, delayed

from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional

import otaf

from dataclasses import dataclass

import pyomo.environ as pe
from typing import Union
from miplearn.io import read_pkl_gz
from miplearn.solvers.pyomo import PyomoModel

import numpy as np
import pyomo.environ as pe
from miplearn.solvers.pyomo import PyomoModel
from miplearn.io import read_pkl_gz, write_pkl_gz
from miplearn.collectors.basic import BasicCollector
from miplearn.components.primal.actions import SetWarmStart
from miplearn.components.primal.mem import MemorizingPrimalComponent, MergeTopSolutions
from miplearn.extractors.fields import H5FieldsExtractor
from miplearn.solvers.learning import LearningSolver
from sklearn.neighbors import KNeighborsClassifier


########################################

########################################

#### NOT IMPORTED IN THE MODULE (__init__.py)

########################################

########################################

## From chat GPT, has to be testec!!!


class MLUnitCommitmentOptimizer:
    def __init__(self, n_neighbors=25):
        self.collector = BasicCollector()
        self.train_data = None
        self.test_data = None
        self.solver_ml = LearningSolver(
            components=[
                MemorizingPrimalComponent(
                    clf=KNeighborsClassifier(n_neighbors=n_neighbors),
                    extractor=H5FieldsExtractor(instance_fields=["static_constr_rhs"]),
                    constructor=MergeTopSolutions(25, [0.0, 1.0]),
                    action=SetWarmStart(),
                )
            ]
        )

    def build_uc_model(self, c, a_ub, b_ub, a_eq, b_eq, bounds) -> PyomoModel:
        model = pe.ConcreteModel()
        n = len(c)
        model.x = pe.Var(range(n), domain=pe.Binary)
        model.obj = pe.Objective(expr=sum(c[i] * model.x[i] for i in range(n)))

        model.constraints = pe.ConstraintList()
        for i in range(a_ub.shape[0]):
            model.constraints.add(sum(a_ub[i, j] * model.x[j] for j in range(n)) <= b_ub[i])

        for i in range(a_eq.shape[0]):
            model.constraints.add(sum(a_eq[i, j] * model.x[j] for j in range(n)) == b_eq[i])

        model.bounds = pe.ConstraintList()
        for i in range(bounds.shape[0]):
            model.bounds.add(bounds[i, 0] <= model.x[i] <= bounds[i, 1])

        return PyomoModel(model, "cbc")

    def fit(self, train_data):
        self.train_data = train_data
        self.collector.collect(train_data, self.build_uc_model, n_jobs=4)
        self.solver_ml.fit(train_data)

    def optimize(self, c, a_ub, b_ub, a_eq, b_eq, bounds):
        model = self.build_uc_model(c, a_ub, b_ub, a_eq, b_eq, bounds)
        self.solver_ml.optimize(model)
        return model

    def reoptimize(self, test_data):
        self.test_data = test_data
        self.solver_ml.optimize(test_data[0], self.build_uc_model)


# Example usage
def constraint_matrix_generator(deviation_array, bounds=None, C=None):
    # Placeholder function: Replace with actual implementation
    n = len(deviation_array)
    c = np.random.rand(n)
    a_ub = np.random.rand(n, n)
    b_ub = np.random.rand(n, n)
    a_eq = np.random.rand(n, n)
    b_eq = np.random.rand(n, n)
    bounds = np.random.rand(n, 2)
    return c, a_ub, b_ub, a_eq, b_eq, bounds


# Generate constraint matrices
c, a_ub, b_ub, a_eq, b_eq, bounds = constraint_matrix_generator(np.random.rand(10))

# Create optimizer instance
optimizer = MLUnitCommitmentOptimizer()

# Generate training and testing data
train_data = write_pkl_gz(
    [constraint_matrix_generator(np.random.rand(10)) for _ in range(450)], "uc/train"
)
test_data = write_pkl_gz(
    [constraint_matrix_generator(np.random.rand(10)) for _ in range(50)], "uc/test"
)

# Fit optimizer with training data
optimizer.fit(train_data)

# Optimize using the learned model
optimized_model = optimizer.optimize(c, a_ub, b_ub, a_eq, b_eq, bounds)
print("Optimized Objective Value:", optimized_model.inner.obj())

# Re-optimize with test data
optimizer.reoptimize(test_data)
