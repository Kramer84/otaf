from __future__ import annotations

__author__ = "Kramer84"
__all__ = [
    "sigma_delta_circular_feature",
    "sigma_delta_3D_plane",
    "sigma_delta_cylindrical_feature",
]

from typing import Union
import numpy as np

def sigma_delta_circular_feature(
    theta: Union[float, np.ndarray],
    sr: float,
    su: float,
    sv: float,
) -> Union[float, np.ndarray]:
    """Calculate defect standard deviation for a circular feature.

    Parameters
    ----------
    theta : float or np.ndarray
        Angle or array of angles along the circular feature in
        radians.
    sr : float
        Standard deviation of the radial defect.
    su : float
        Standard deviation of the U-axis translational defect.
    sv : float
        Standard deviation of the V-axis translational defect.

    Returns
    -------
    float or np.ndarray
        The calculated standard deviation. If `theta` is an array,
        the output shape matches the shape of `theta`.
    """
    return np.sqrt(sr**2 + np.cos(theta) ** 2 * su**2 + np.sin(theta) ** 2 * sv**2)


def sigma_delta_3D_plane(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    sw: float,
    sa: float,
    sb: float,
) -> Union[float, np.ndarray]:
    """Calculate defect standard deviation for a 3D plane feature.

    Parameters
    ----------
    x : float or np.ndarray
        X-coordinate or array of coordinates on the plane.
    y : float or np.ndarray
        Y-coordinate or array of coordinates on the plane.
    sw : float
        Standard deviation of the translational defect normal to the
        plane.
    sa : float
        Standard deviation of the angular defect variation around the
        first axis.
    sb : float
        Standard deviation of the angular defect variation around the
        second axis.

    Returns
    -------
    float or np.ndarray
        The calculated standard deviation. If `x` or `y` are arrays,
        the output shape matches their broadcasted shape.
    """
    return np.sqrt(sw**2 + y**2 * sa**2 + x**2 * sb**2)


def sigma_delta_cylindrical_feature(
    z: Union[float, np.ndarray],
    theta: Union[float, np.ndarray],
    sr: float,
    su: float,
    salpha: float,
    sv: float,
    sbeta: float,
) -> Union[float, np.ndarray]:
    """Calculate defect standard deviation for a cylindrical feature.

    Parameters
    ----------
    z : float or np.ndarray
        Axial position or array of positions along the cylinder
        height.
    theta : float or np.ndarray
        Angular position or array of positions along the
        circumference in radians.
    sr : float
        Standard deviation of the radial defect.
    su : float
        Standard deviation of the axial translation defect.
    salpha : float
        Standard deviation of the angular variation defect around
        the alpha axis.
    sv : float
        Standard deviation of the transverse translation defect.
    sbeta : float
        Standard deviation of the angular variation defect around
        the beta axis.

    Returns
    -------
    float or np.ndarray
        The calculated standard deviation. If `z` or `theta` are
        arrays, the output shape matches their broadcasted shape.
    """
    return np.sqrt(
        sr**2
        + (su**2 + (sbeta * z) ** 2) * np.cos(theta) ** 2
        + (sv**2 + (salpha * z) ** 2) * np.sin(theta) ** 2
    )
