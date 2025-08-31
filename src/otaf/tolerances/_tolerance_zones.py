from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "MiSdofToleranceZones",
    "bound_distance",
    "points_within_bounds",
]

import numpy as np
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional



class MiSdofToleranceZones:
    """
    Class to define and validate tolerance zones for 2D points.

    Methods
    -------
    circular_zone(t, X)
        Validates if points lie within a circular tolerance zone of radius t/2.
    two_concentric_circles(t, X)
        Validates if points lie within a circular tolerance zone of radius t/2
        and respects additional constraints.

    Note
    ----
    From : 10.1007/s00170-010-2568-8
    """

    @staticmethod
    def circular_zone(X, t=0):
        """
        Validate if points lie within a circular tolerance zone.

        Parameters
        ----------
        t : float
            Diameter of the circular tolerance zone.
        X : np.ndarray, shape (N, 2)
            Array of N 2D points (u, v) representing defects.

        Returns
        -------
        valid : np.ndarray, shape (N,)
            Binary array where 1 indicates the point lies within the zone, and 0 otherwise.
        dist_func : np.ndarray, shape (N,)
            Array of distances of the points from the origin.
        """
        bounds = np.array([[-t/2, -t/2], [t/2, t/2]])
        dist_func = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        dist_valid = points_within_bounds(dist_func, [[0], [t/2]])
        bounds_valid = points_within_bounds(X, bounds)
        valid = dist_valid * bounds_valid
        return valid, dist_func

    @staticmethod
    def two_concentric_circles(X, t=0):
        """
        Validate if points lie within a two-concentric-circle tolerance zone.

        Parameters
        ----------
        t : float
            Diameter of the outer circular tolerance zone.
        X : np.ndarray, shape (N, 2)
            Array of N 2D points (u, v) representing defects.

        Returns
        -------
        valid : np.ndarray, shape (N,)
            Binary array where 1 indicates the point lies within the zone, and 0 otherwise.
        dist_func : np.ndarray, shape (N,)
            Array of distances of the points from the origin.
        """
        bounds = np.array([[-t/2, -t/2], [t/2, t/2]])
        dist_func = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        dist_valid = points_within_bounds(dist_func, [[0], [t/2]])
        bounds_valid = points_within_bounds(X, bounds)
        valid = dist_valid * bounds_valid
        return valid, dist_func

    @staticmethod
    def two_parallel_straight_lines(X, t=0, L=0):
        """
        Validate if points lie between two parallel straight lines.

        Parameters
        ----------
        t : float
            Width between the lines.
        L : float
            Scale factor for the slope of the lines.
        X : np.ndarray, shape (N, 2)
            Array of N 2D points (u, gamma).

        Returns
        -------
        valid : np.ndarray, shape (N,)
            Binary array where 1 indicates the point lies within the zone, and 0 otherwise.
        dist_func : np.ndarray, shape (N,)
            Distance function for each point.
        """
        bounds = np.array([[-t/2, -t/L], [t/2, t/L]])
        dist_func = np.abs(X[:, 0]) + np.abs((X[:, 1]*L)/2) # Here is false in the paper MI-SDOF
        dist_valid = points_within_bounds(dist_func, [[0], [t/2]])
        bounds_valid = points_within_bounds(X, bounds)
        valid = dist_valid * bounds_valid
        return valid, dist_func

    @staticmethod
    def two_parallel_curves(X, t=0):
        raise NotImplementedError

    @staticmethod
    def spherical_zone(X, t=0):
        """
        Validate if points lie within a spherical zone.

        Parameters
        ----------
        t : float
            Diameter of the sphere.
        X : np.ndarray, shape (N, 3)
            Array of N 3D points (u, v, w).

        Returns
        -------
        valid : np.ndarray, shape (N,)
            Binary array where 1 indicates the point lies within the zone, and 0 otherwise.
        dist_func : np.ndarray, shape (N,)
            Distance of each point from the origin.
        """
        bounds = np.array([[-t/2, -t/2, -t/2], [t/2, t/2, t/2]])
        dist_func = np.sqrt(X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2)
        dist_valid = points_within_bounds(dist_func, [[0], [t/2]])
        bounds_valid = points_within_bounds(X, bounds)
        valid = dist_valid * bounds_valid
        return valid, dist_func

    @staticmethod
    def cylindrical_zone(X, t=0, L=0):
        """
        Validate if points lie within a cylindrical zone.

        Parameters
        ----------
        t : float
            Diameter of the cylinder.
        L : float
            Scale factor for the cylinder's axis.
        X : np.ndarray, shape (N, 4)
            Array of N 3D points (u, v, alpha, beta).

        Returns
        -------
        valid : np.ndarray, shape (N,)
            Binary array where 1 indicates the point lies within the zone, and 0 otherwise.
        dist_func : np.ndarray, shape (N,)
            Distance function for each point.
        """
        bounds = np.array([[-t/2, -t/2, -t/L, -t/L], [t/2, t/2, t/L, t/L]])
        dist_func = np.sqrt((X[:, 0] + X[:, 2]*(L/2))**2 + (X[:, 1] + X[:, 3]*(L/2))**2)
        dist_valid = points_within_bounds(dist_func, [[0], [t/2]])
        bounds_valid = points_within_bounds(X, bounds)
        valid = dist_valid * bounds_valid
        return valid, dist_func

    @staticmethod
    def two_coaxial_cylinders(X, t=0, L=0):
        """
        Validate if points lie within two coaxial cylinders.

        Parameters
        ----------
        t : float
            Diameter of the outer cylinder.
        L : float
            Scale factor for the cylinder's axis.
        X : np.ndarray, shape (N, 4)
            Array of N 3D points (u, v, alpha, beta).

        Returns
        -------
        valid : np.ndarray, shape (N,)
            Binary array where 1 indicates the point lies within the zone, and 0 otherwise.
        dist_func : np.ndarray, shape (N,)
            Distance function for each point.
        """
        bounds = np.array([[-t/2, -t/2, -t/L, -t/L], [t/2, t/2, t/L, t/L]])
        dist_func = np.sqrt((X[:, 0] + X[:, 2]*(L/2))**2 + (X[:, 1] + X[:, 3]*(L/2))**2)
        dist_valid = points_within_bounds(dist_func, [[0], [t/2]])
        bounds_valid = points_within_bounds(X, bounds)
        valid = dist_valid * bounds_valid
        return valid, dist_func

    @staticmethod
    def parallelepiped_zone(X, t1=0, t2=0, L=0):
        """
        Validate if points lie within a parallelepiped zone.

        Parameters
        ----------
        t1 : float
            Width along one dimension.
        t2 : float
            Width along another dimension.
        L : float
            Scale factor for the parallelepiped's axis.
        X : np.ndarray, shape (N, 4)
            Array of N 3D points (u, v, alpha, beta).

        Returns
        -------
        valid : np.ndarray, shape (N,)
            Binary array where 1 indicates the point lies within the zone, and 0 otherwise.
        dist_func : tuple of np.ndarray
            Distance functions for each dimension (dist_func_1, dist_func_2).
        """
        bounds = np.array([[-t1/2, -t2/2, -t1/L, -t2/L], [t1/2, t2/2, t1/L, t2/L]])
        dist_func_1 = np.abs(X[:, 0]) + np.abs(X[:, 2]*(L/2))
        dist_func_2 = np.abs(X[:, 1]) + np.abs(X[:, 3]*(L/2))
        dist_valid_1 = points_within_bounds(dist_func_1, [[0], [t1/2]])
        dist_valid_2 = points_within_bounds(dist_func_2, [[0], [t2/2]])
        bounds_valid = points_within_bounds(X, bounds)
        valid = dist_valid_1 * dist_valid_2 * bounds_valid
        return valid, (dist_func_1, dist_func_2)

    @staticmethod
    def two_parallel_planes(X, t=0, Lx=0, Ly=0):
        """
        Validate if points lie between two parallel planes.

        Parameters
        ----------
        t : float
            Distance between the planes.
        Lx : float
            Scale factor along the x-axis.
        Ly : float
            Scale factor along the y-axis.
        X : np.ndarray, shape (N, 3)
            Array of N 3D points (w, alpha, beta).

        Returns
        -------
        valid : np.ndarray, shape (N,)
            Binary array where 1 indicates the point lies within the zone, and 0 otherwise.
        dist_func : np.ndarray, shape (N,)
            Distance function for each point.
        """
        bounds = np.array([[-t/2, -t/Ly, -t/Lx], [t/2, t/Ly, t/Lx]])
        dist_func = np.abs(X[:, 0]) + np.abs(X[:, 1]*(Ly/2)) + np.abs(X[:, 2]*(Lx/2))
        dist_valid = points_within_bounds(dist_func, [[0], [t/2]])
        bounds_valid = points_within_bounds(X, bounds)
        valid = dist_valid * bounds_valid
        return valid, dist_func

    @staticmethod
    def two_parallel_surfaces(X, t=0):
        #3D Tolerance Zone
        raise NotImplementedError


def bound_distance(val, lb, ub):
    """Return signed distance of val to bounds (negative if out of bounds)."""
    return min(val - lb, ub - val)

def points_within_bounds(sample, bounds):
    """
    Check if each point in the sample respects the bounds.

    Parameters
    ----------
    sample : np.ndarray, shape (N, M)
        Array of N points, each with M dimensions.
    bounds : np.ndarray, shape (2, M)
        Array of bounds for each dimension, with each row as [lower_bound, upper_bound].

    Returns
    -------
    np.ndarray, shape (N,)
        Boolean array where 1 means the point respects the bounds, and 0 otherwise.
    """
    sample = np.atleast_2d(sample.T).T
    bounds = np.asarray(bounds)
    lb, ub = bounds[0, :], bounds[1, :]
    return np.all((sample >= lb) & (sample <= ub), axis=1).astype(int)
