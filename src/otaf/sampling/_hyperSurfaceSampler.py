from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "LagrangeConstraintSolver",
    "UniformSurfaceSampler"
]

import numpy as np
from scipy.optimize import minimize

class LagrangeConstraintSolver:
    """
    Class to solve the problem of finding the subspace in Lambda
    where g(X) = 0, subject to the bounds for lambda_i.

    Attributes:
        dim (int): Dimensionality of the space (number of lambda_i variables).
        g (callable): Function g(X) that represents the constraint g(X) = 0.
        bounds (list of tuples): List of (min, max) tuples for each lambda_i.
        starting_points (np.ndarray): Initial points for the optimization, one per dimension.
    """
    def __init__(self, dim, g, bounds, starting_points):
        """
        Initialize the solver with dimensionality, constraint function,
        bounds for each dimension, and initial points for optimization.

        Args:
            dim (int): Dimensionality of the space.
            g (callable): Function g(X) representing the constraint g(X) = 0.
            bounds (list of tuples): List of (lambda_i_min, lambda_i_max) bounds for each lambda_i.
            starting_points (np.ndarray): Starting points for optimization.
        """
        self.dim = dim
        self.g = g
        self.bounds = bounds
        self.starting_points = starting_points

    def objective_function(self, X):
        """
        The objective function to minimize. In this case, we're simply trying to
        minimize the function g(X) to reach g(X) = 0.

        Args:
            X (np.ndarray): Array of lambda_i values.

        Returns:
            float: The value of g(X).
        """
        return np.abs(self.g(X))

    def solve(self):
        """
        Solves the optimization problem using SLSQP method to find the subspace
        where g(X) = 0.

        Returns:
            res (OptimizeResult): The result of the optimization, including
                                  the point X where g(X) = 0.
        """
        # Equality constraint for g(X) = 0
        constraints = {'type': 'eq', 'fun': self.g}

        # Perform optimization with SLSQP method
        res = minimize(self.objective_function, self.starting_points, method='SLSQP',
                       bounds=self.bounds, constraints=[constraints], tol=1e-9)

        return res



class UniformSurfaceSampler:
    def __init__(self, dim, g, bounds, starting_points, step_size=0.1):
        """
        Initialize the sampler with dimensionality, constraint function,
        bounds for each dimension, and initial points for sampling.

        Args:
            dim (int): Dimensionality of the space.
            g (callable): Function g(X) representing the constraint g(X) = 0.
            bounds (list of tuples): List of (lambda_i_min, lambda_i_max) bounds for each lambda_i.
            starting_points (np.ndarray): Starting points for sampling.
            step_size (float): Step size for generating new points on the surface.
        """
        self.dim = dim
        self.g = g
        self.bounds = bounds
        self.starting_points = starting_points
        self.step_size = step_size

    def objective_function(self, X):
        """Objective function to minimize, ensuring g(X) = 0."""
        return np.abs(self.g(X))

    def solve(self, x0):
        """
        Solves the optimization problem to find a new point close to x0 that still satisfies g(X) = 0.

        Args:
            x0 (np.ndarray): Initial guess for optimization.

        Returns:
            res (OptimizeResult): Resulting point on the surface.
        """
        constraints = {'type': 'eq', 'fun': self.g}
        res = minimize(self.objective_function, x0, method='SLSQP', bounds=self.bounds, constraints=[constraints])
        return res.x if res.success else None

    def sample_surface(self, num_samples):
        """
        Generate a uniform grid of points on the surface where g(X) = 0.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of sampled points that satisfy g(X) = 0.
        """
        sampled_points = []

        # Add initial points that are already on the surface
        for point in self.starting_points:
            if np.abs(self.g(point)) < 1e-6:  # Ensure point satisfies g(X) = 0
                sampled_points.append(point)

        # Generate additional points by stepping in random directions
        while len(sampled_points) < num_samples:
            # Randomly select a point from the existing sampled points
            x0 = sampled_points[np.random.randint(len(sampled_points))]

            # Create a random direction for the step
            random_direction = np.random.randn(self.dim)
            random_direction /= np.linalg.norm(random_direction)  # Normalize the direction

            # Take a step in the random direction and project back to the surface
            new_point = x0 + self.step_size * random_direction

            # Ensure the new point respects the bounds
            new_point = np.clip(new_point, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

            # Project the point back onto the surface using constrained optimization
            surface_point = self.solve(new_point)

            if surface_point is not None:
                sampled_points.append(surface_point)

        return np.array(sampled_points)

