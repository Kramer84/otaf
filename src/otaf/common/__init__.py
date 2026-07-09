"""Common utility functions and helpers for OTAF."""

from __future__ import annotations

__author__ = "Kramer84"
from collections import defaultdict

from ._common import (
    alphabet_generator,
    arrays_close_enough,
    bidirectional_string_to_array_conversion,
    extract_expressions_with_variables,
    get_SE3_base,
    get_SE3_matrices_from_indices,
    get_symbol_coef_map,
    get_symbols_in_expressions,
    get_tqdm_range,
    inverse_mstring,
    is_running_in_notebook,
    merge_with_checks,
    parse_matrix_string,
    scaling,
    threshold_for_percentile_positive_values_below,
    validate_dict_keys,
)


def tree() -> defaultdict:
    """
    Create a recursive defaultdict that automatically creates nested dictionaries.

    Returns
    -------
    defaultdict
        A defaultdict that creates nested dictionaries on-the-fly.
    """
    return defaultdict(tree)


__all__ = [
    "alphabet_generator",
    "arrays_close_enough",
    "bidirectional_string_to_array_conversion",
    "extract_expressions_with_variables",
    "get_SE3_base",
    "get_SE3_matrices_from_indices",
    "get_symbol_coef_map",
    "get_symbols_in_expressions",
    "get_tqdm_range",
    "inverse_mstring",
    "is_running_in_notebook",
    "merge_with_checks",
    "parse_matrix_string",
    "scaling",
    "threshold_for_percentile_positive_values_below",
    "tree",
    "validate_dict_keys",
]
