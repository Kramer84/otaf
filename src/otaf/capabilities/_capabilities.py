from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "process_capability",
]

def process_capability(USL, LSL, mean, std_dev):
    """
    Calculates the process capability indices Cp, CPU, CPL, Cpk, and k.

    Parameters:
    USL (float): Upper Specification Limit
    LSL (float): Lower Specification Limit
    mean (float): Mean of the process (x̄)
    std_dev (float): Standard deviation of the process (σ)

    Returns:
    dict: Dictionary containing Cp, CPU, CPL, Cpk, and k.
    """
    # Calculate Cp (Process potential)
    Cp = (USL - LSL) / (6 * std_dev)

    # Calculate CPU (Upper capability)
    CPU = (USL - mean) / (3 * std_dev)

    # Calculate CPL (Lower capability)
    CPL = (mean - LSL) / (3 * std_dev)

    # Calculate Cpk (Process capability)
    Cpk = min(CPU, CPL)

    # Calculate midpoint (m)
    m = (USL + LSL) / 2

    # Calculate k (Deviation from center)
    k = abs(mean - m) / ((USL - LSL) / 2)

    return {
        'Cp': Cp,
        'CPU': CPU,
        'CPL': CPL,
        'Cpk': Cpk,
        'k': k
    }
