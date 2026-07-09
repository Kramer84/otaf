"""Tools for generating experimental designs, low-discrepancy sequences, and surface sampling."""

from __future__ import annotations

__author__ = "Kramer84"
from ._hyperPlaneSampler import project_samples_to_hyperplane
from ._hyperSurfaceSampler import LagrangeConstraintSolver, UniformSurfaceSampler
from ._sampling import (
    calculate_sample_in_deviation_domain,
    compose_defects_with_lambdas,
    condition_lambda_sample,
    condition_sample_array,
    find_best_worst_quantile,
    generate_and_transform_sequence,
    generate_imprecise_probabilistic_samples,
    generate_lhs_experiment,
    generate_random_permutations_with_sampling,
    generate_scaled_permutations,
    scale_sample_with_params,
    validate_and_extract_indices,
)

__all__ = [
    "condition_lambda_sample",
    "validate_and_extract_indices",
    "condition_sample_array",
    "find_best_worst_quantile",
    "generate_lhs_experiment",
    "generate_random_permutations_with_sampling",
    "generate_scaled_permutations",
    "generate_imprecise_probabilistic_samples",
    "generate_and_transform_sequence",
    "compose_defects_with_lambdas",
    "scale_sample_with_params",
    "calculate_sample_in_deviation_domain",
    "project_samples_to_hyperplane",
    "LagrangeConstraintSolver",
    "UniformSurfaceSampler",
]
