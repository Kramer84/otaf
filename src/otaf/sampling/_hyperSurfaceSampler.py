from __future__ import annotations

__author__ = "Kramer84"
__all__ = ["LagrangeConstraintSolver", "UniformSurfaceSampler"]
from collections.abc import Callable
from typing import Optional, Union

import numpy as np
from beartype import beartype
from scipy.optimize import OptimizeResult, minimize


class LagrangeConstraintSolver:
    """
    Solve for the subspace where the constraint function equals zero.

    Parameters
    ----------
    dim : int
        Dimensionality of the space (number of variables).
    g : callable
        Constraint function ``g(X)`` such that ``g(X) = 0``.
    bounds : list of tuple
        List of (min, max) bounds for each variable.
    starting_points : ndarray
        Initial points for the optimization algorithm.

    Attributes
    ----------
    dim : int
        Dimensionality of the space (number of variables).
    g : callable
        Constraint function ``g(X)`` such that ``g(X) = 0``.
    bounds : list of tuple
        List of (min, max) bounds for each variable.
    starting_points : ndarray
        Initial points for the optimization algorithm.
    """

    @beartype
    def __init__(
        self,
        dim: int,
        g: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        starting_points: np.ndarray,
    ) -> None:
        self.dim = dim
        self.g = g
        self.bounds = bounds
        self.starting_points = starting_points

    @beartype
    def objective_function(self, X: np.ndarray) -> float:
        """
        Calculate the objective value for the constraint minimization.

        Parameters
        ----------
        X : ndarray
            Current vector of variables.

        Returns
        -------
        float
            Absolute value of the constraint function evaluated at `X`.
        """
        return np.abs(self.g(X))

    @beartype
    def solve(self) -> OptimizeResult:
        """
        Solve the optimization problem using the SLSQP method.

        Finds the subspace satisfying ``g(X) = 0`` subject to defined 
        variable bounds.

        Returns
        -------
        OptimizeResult
            The result object of the optimization process.
        """
        constraints = {"type": "eq", "fun": self.g}
        res = minimize(
            self.objective_function,
            self.starting_points,
            method="SLSQP",
            bounds=self.bounds,
            constraints=[constraints],
            tol=1e-09,
        )
        return res


class UniformSurfaceSampler:
    """
    Generate a uniform distribution of points on a surface where g(X) = 0.

    Parameters
    ----------
    dim : int
        Dimensionality of the space.
    g : callable
        Constraint function ``g(X)`` such that ``g(X) = 0``.
    bounds : list of tuple
        List of (min, max) bounds for each variable.
    starting_points : ndarray
        Initial points already satisfying the constraint.
    step_size : float, optional
        Step size for generating new points on the surface. Default is 0.1.

    Attributes
    ----------
    dim : int
        Dimensionality of the space.
    g : callable
        Constraint function ``g(X)`` such that ``g(X) = 0``.
    bounds : list of tuple
        List of (min, max) bounds for each variable.
    starting_points : ndarray
        Initial points already satisfying the constraint.
    step_size : float
        Magnitude of the random step taken during sampling.
    """

    @beartype
    def __init__(
        self,
        dim: int,
        g: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        starting_points: np.ndarray,
        step_size: float = 0.1,
    ) -> None:
        self.dim = dim
        self.g = g
        self.bounds = bounds
        self.starting_points = starting_points
        self.step_size = step_size

    @beartype
    def objective_function(self, X: np.ndarray) -> float:
        """
        Calculate the absolute value of the constraint function g(X).

        Parameters
        ----------
        X : ndarray
            Current vector of variables.

        Returns
        -------
        float
            The absolute constraint violation.
        """
        return np.abs(self.g(X))

    @beartype
    def solve(self, x0: np.ndarray) -> Optional[np.ndarray]:
        """
        Project an arbitrary point onto the surface g(X) = 0.

        Uses the SLSQP method to find the nearest point satisfying the 
        constraint.

        Parameters
        ----------
        x0 : ndarray
            Initial guess point for optimization.

        Returns
        -------
        ndarray or None
            The projected point on the surface, or ``None`` if 
            optimization failed.
        """
        constraints = {"type": "eq", "fun": self.g}
        res = minimize(
            self.objective_function,
            x0,
            method="SLSQP",
            bounds=self.bounds,
            constraints=[constraints],
        )
        return res.x if res.success else None

    @beartype
    def sample_surface(self, num_samples: int) -> np.ndarray:
        """
        Generate a uniform distribution of points on the surface.

        Parameters
        ----------
        num_samples : int
            The number of points to generate.

        Returns
        -------
        ndarray
            Array of points satisfying ``g(X) = 0``.
        """
        sampled_points = []
        for point in self.starting_points:
            if np.abs(self.g(point)) < 1e-06:
                sampled_points.append(point)
        while len(sampled_points) < num_samples:
            x0 = sampled_points[np.random.randint(len(sampled_points))]
            random_direction = np.random.randn(self.dim)
            random_direction /= np.linalg.norm(random_direction)
            new_point = x0 + self.step_size * random_direction
            new_point = np.clip(
                new_point, [b[0] for b in self.bounds], [b[1] for b in self.bounds]
            )
            surface_point = self.solve(new_point)
            if surface_point is not None:
                sampled_points.append(surface_point)
        return np.array(sampled_points)
