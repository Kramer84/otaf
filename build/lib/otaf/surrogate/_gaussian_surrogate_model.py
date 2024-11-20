# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "GaussianSurrogateModel",
]

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize


class GaussianSurrogateModel:
    def __init__(self, dimension, bounds, constraints=None, min_val=None, max_val=None):
        """
        Initialize the ConstrainedSurrogateModel.

        :param dimension: Dimensionality of the input space.
        :param bounds: Bounds for the input space as a list of tuples [(lower_bound, upper_bound)] for each dimension.
        :param constraints: A list of constraint dictionaries, in the form used by scipy.optimize (optional).
        :param min_val: Optional minimum value for the objective function.
        :param max_val: Optional maximum value for the objective function.
        """
        self.dimension = dimension
        self.bounds = bounds
        self.constraints = constraints if constraints else []
        self.min_val = min_val
        self.max_val = max_val

        # Initialize empty datasets for training
        self.X_train = np.empty((0, dimension))
        self.y_train = np.empty((0, 1))

        # Gaussian Process Regressor with RBF kernel
        kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    def fit_model(self):
        """Fit the Gaussian Process model to the current dataset."""
        if self.X_train.shape[0] > 0:  # Only fit if there are training points
            self.gp.fit(self.X_train, self.y_train)

    def predict(self, X):
        """
        Predict the objective function at new points.

        :param X: New points to predict on.
        :return: Mean prediction and standard deviation (uncertainty).
        """
        if self.X_train.shape[0] == 0:
            raise ValueError("Model has no training data.")

        y_pred, sigma = self.gp.predict(X, return_std=True)

        # Apply constraints if min_val or max_val are provided
        if self.min_val is not None:
            y_pred = np.maximum(y_pred, self.min_val)
        if self.max_val is not None:
            y_pred = np.minimum(y_pred, self.max_val)

        return y_pred, sigma

    def add_point(self, X_new, y_new):
        """
        Add a new point to the dataset and refit the model.

        :param X_new: New input point (1D array of length equal to the number of dimensions).
        :param y_new: Corresponding function value at the new point.
        """
        self.X_train = np.vstack((self.X_train, X_new.reshape(1, -1)))
        self.y_train = np.vstack((self.y_train, [[y_new]]))
        self.fit_model()

    def optimize_acquisition(self, acquisition_function):
        """
        Optimize the acquisition function to find the next point, subject to the constraints.

        :param acquisition_function: The acquisition function to maximize.
        :return: The next point to sample.
        """

        def neg_acquisition(x):
            return -acquisition_function(np.array(x).reshape(1, -1))[0]

        res = minimize(
            neg_acquisition,
            x0=np.random.uniform(low=[b[0] for b in self.bounds], high=[b[1] for b in self.bounds]),
            bounds=self.bounds,
            constraints=self.constraints,
            method="SLSQP",
        )

        return res.x

    def acquisition_function(self, X):
        """
        Simple acquisition function: Upper Confidence Bound (UCB).

        :param X: Points to evaluate the acquisition function on.
        :return: Acquisition function values.
        """
        y_pred, sigma = self.predict(X)
        # Upper Confidence Bound (UCB) acquisition function
        return y_pred + 1.96 * sigma

    def find_global_minima(self):
        """
        Find the global minimum in the constrained space.

        :return: The input point that minimizes the surrogate model's prediction.
        """
        res = minimize(
            lambda x: self.predict(np.array(x).reshape(1, -1))[0],
            x0=np.random.uniform(low=[b[0] for b in self.bounds], high=[b[1] for b in self.bounds]),
            bounds=self.bounds,
            constraints=self.constraints,
            method="SLSQP",
        )

        return res.x
