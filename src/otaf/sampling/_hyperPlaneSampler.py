from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "project_samples_to_hyperplane",
]

import numpy as np
from beartype import beartype
from typing import Optional, Union

@beartype
def project_samples_to_hyperplane(
    a: Union[list[float], np.ndarray],
    b: float,
    samples: np.ndarray,
    bounds: Optional[list[tuple[float, float]]] = None
) -> np.ndarray:
    """Project (N-1)-dimensional samples onto an N-dimensional hyperplane.

    Solve the hyperplane equation $a_0 x_0 + a_1 x_1 + ... + a_{N-1} x_{N-1} = b$
    to find the coordinate $x_0$, then reconstruct the full N-dimensional point.

    Parameters
    ----------
    a : Union[list[float], np.ndarray]
        Coefficients of the hyperplane equation.
    b : float
        Constant term in the hyperplane equation.
    samples : np.ndarray
        Samples in (N-1)-dimensional space with shape (M, N-1).
    bounds : list[tuple[float, float]], optional
        Bounds for each dimension in the format [(min, max), ...].
        Only samples whose projections fall within these bounds are returned.

    Returns
    -------
    np.ndarray
        Projected samples in N-dimensional space that satisfy the hyperplane equation.
    """
    a = np.array(a)
    N = len(a)
    projected_samples = []
    for sample in samples:
        x_rest_sum = np.dot(a[1:], sample)
        x0 = (b - x_rest_sum) / a[0]
        projected_sample = np.array([x0, *sample])
        if bounds is not None:
            if all(bounds[i][0] <= projected_sample[i] <= bounds[i][1] for i in range(N)):
                projected_samples.append(projected_sample)
        else:
            projected_samples.append(projected_sample)
    return np.array(projected_samples)
