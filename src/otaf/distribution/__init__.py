"""Distribution analysis and manipulation tools for the OTAF project."""

from __future__ import annotations

__author__ = "Kramer84"
from ._distribution import (
    compute_sup_inf_distributions,
    generate_correlated_samples,
    get_composed_normal_defect_distribution,
    get_means_standards_composed_distribution,
    get_prob_below_threshold,
    multiply_composed_distribution_standard_with_constants,
    multiply_composed_distribution_with_constant,
)

__all__ = [
    "get_composed_normal_defect_distribution",
    "multiply_composed_distribution_with_constant",
    "multiply_composed_distribution_standard_with_constants",
    "get_means_standards_composed_distribution",
    "generate_correlated_samples",
    "compute_sup_inf_distributions",
    "get_prob_below_threshold",
]
