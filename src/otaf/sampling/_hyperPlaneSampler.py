from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "project_samples_to_hyperplane",
]

import numpy as np


def project_samples_to_hyperplane(a, b, samples, bounds=None):
    """
    Project (N-1)-dimensional samples onto the N-dimensional hyperplane defined by the equation:

    a_0 * x_0 + a_1 * x_1 + ... + a_{N-1} * x_{N-1} = b

    Parameters:
    a (list or np.ndarray): Coefficients of the hyperplane equation.
    b (float): Constant term in the hyperplane equation.
    samples (np.ndarray): Samples in (N-1)-dimensional space.
    bounds (list of tuples): Bounds for each dimension in the format [(min, max), ..., (min, max)].

    Returns:
    np.ndarray: Projected samples in N-dimensional space that satisfy the hyperplane equation.
    """
    a = np.array(a)
    N = len(a)
    projected_samples = []

    for sample in samples:
        # Solve for x0 in the equation a_0 x_0 + a_1 x_1 + ... + a_N-1 x_N-1 = b
        x_rest_sum = np.dot(a[1:], sample)
        x0 = (b - x_rest_sum) / a[0]
        projected_sample = [x0] + list(sample)

        # Check if the projected point satisfies the bounds, if bounds are provided
        if bounds:
            if all(bounds[i][0] <= projected_sample[i] <= bounds[i][1] for i in range(N)):
                projected_samples.append(projected_sample)
        else:
            projected_samples.append(projected_sample)

    return np.array(projected_samples)


# Example usage in 2D (to be used in the next steps with matplotlib)
