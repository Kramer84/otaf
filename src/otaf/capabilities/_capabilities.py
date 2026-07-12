from __future__ import annotations

__author__ = "Kramer84"
__all__ = ["process_capability"]


def process_capability(
    usl: float, lsl: float, mean: float, std_dev: float
) -> dict[str, float]:
    """Calculate the process capability indices Cp, CPU, CPL, Cpk, and k.

    Parameters
    ----------
    usl : float
        Upper Specification Limit.
    lsl : float
        Lower Specification Limit.
    mean : float
        Mean of the process ($\\bar{x}$).
    std_dev : float
        Standard deviation of the process ($\\sigma$).

    Returns
    -------
    dict of {str : float}
        A dictionary containing the calculated capability metrics:
        - 'Cp' : Process potential index.
        - 'CPU' : Upper process capability index.
        - 'CPL' : Lower process capability index.
        - 'Cpk' : Minimum process capability index.
        - 'k' : Standardized deviation from the specification center.

    Notes
    -----
    The capability indices are computed using the following formulas:

    .. math::

       C_p = \\frac{usl - lsl}{6\\sigma}

       CPU = \\frac{usl - \\mu}{3\\sigma}

       CPL = \\frac{\\mu - lsl}{3\\sigma}

       C_{pk} = \\min(CPU, CPL)

       k = \\frac{|\\mu - m|}{\\frac{usl - lsl}{2}}

    Where $m = \\frac{usl + lsl}{2}$ is the specification midpoint.

    Raises
    ------
    ZeroDivisionError
        If `std_dev` is 0, resulting in a division by zero during indices calculation.
    ValueError
        If `usl` is less than `lsl`, resulting in an invalid specification range,
        or if `std_dev` is negative.
    """
    if std_dev <= 0:
        if std_dev == 0:
            raise ZeroDivisionError("Standard deviation cannot be zero for capability calculations.")
        raise ValueError("Standard deviation must be a positive number.")
    if usl < lsl:
        raise ValueError("Invalid specification range: Upper Specification Limit (usl) must be greater than or equal to Lower Specification Limit (lsl).")
    Cp = (usl - lsl) / (6 * std_dev)
    CPU = (usl - mean) / (3 * std_dev)
    CPL = (mean - lsl) / (3 * std_dev)
    Cpk = min(CPU, CPL)
    m = (usl + lsl) / 2
    k = abs(mean - m) / ((usl - lsl) / 2)
    return {"Cp": Cp, "CPU": CPU, "CPL": CPL, "Cpk": Cpk, "k": k}
