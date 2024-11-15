# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "inverse_mstring",
    "merge_with_checks",
    "parse_matrix_string",
    "get_symbols_in_expressions",
    "get_relevant_expressions",
    "round_floats_in_expression",
    "get_symbol_coef_map",
    "get_SE3_base",
    "get_SE3_matrices_from_indices",
    "validate_dict_keys",
    "is_running_in_notebook",
    "get_tqdm_range",
    "alphabet_generator",
    "threshold_for_percentile_positive_values_below",
]

import re
import itertools
import string

import sympy as sp
import numpy as np

from beartype import beartype
from beartype.typing import List, Union, Dict, Tuple

from tqdm import trange
from tqdm.notebook import trange as trange_notebook

import otaf


@beartype
def inverse_mstring(matrix_str: str) -> str:
    """
    Generate the inverse of a matrix string representation.

    This function processes a matrix string representation (either a Transformation matrix
    or a Gap matrix) and returns its inverse string representation. The inverse is computed
    by swapping certain elements in the input string based on predefined patterns.

    Parameters
    ----------
    matrix_str : str
        The string representation of the matrix. Must follow the expected
        format for either a Transformation matrix or a Gap matrix.

    Returns
    -------
    str
        The inverse string representation of the matrix. Returns an empty string
        if `matrix_str` is empty.

    Raises
    ------
    ValueError
        If the matrix string does not match the expected format for a Transformation
        or Gap matrix, or if the matrix type is unsupported.

    Examples
    --------
    >>> inverse_mstring("TP123S1P456S2")
    'TP123S2P456S1'

    >>> inverse_mstring("GP1S1P1P2S2P2")
    'GP2S2P2P1S1P1'

    >>> inverse_mstring("")
    ''
    """
    if matrix_str:
        matrix_type = matrix_str[0]

        if matrix_type == "T":
            match = otaf.constants.T_MATRIX_PATTERN.fullmatch(matrix_str)
            if not match:
                raise ValueError(f"Invalid format for Transformation matrix: {matrix_str}")
            part, surface1, point1, surface2, point2 = match.groups()
            return f"TP{part}{surface2}{point2}{surface1}{point1}"

        elif matrix_type == "G":
            match = otaf.constants.G_MATRIX_MSTRING_PATTERN.fullmatch(matrix_str)
            if not match:
                raise ValueError(f"Invalid format for Gap matrix: {matrix_str}")
            part1, surface1, point1, part2, surface2, point2 = match.groups()
            return f"GP{part2}{surface2}{point2}P{part1}{surface1}{point1}"

        else:
            raise ValueError(f"Unsupported matrix type: {matrix_type}")
    else:
        return ""


def merge_with_checks(existing_points, new_points):
    """
    Merge two dictionaries with value consistency checks.

    This function merges `new_points` into `existing_points`. If a key in `new_points` already
    exists in `existing_points`, the function checks whether their corresponding values are
    numerically close. If not, a `ValueError` is raised. The `existing_points` dictionary is
    modified in place.

    Parameters
    ----------
    existing_points : dict
        The original dictionary of points to merge into. This dictionary is updated in place.
        Keys are typically strings, and values are numeric arrays or similar types.

    new_points : dict
        The new points to merge. Keys should match those in `existing_points` if they overlap.

    Returns
    -------
    dict
        The updated `existing_points` dictionary after merging.

    Raises
    ------
    ValueError
        If a key exists in both dictionaries and the corresponding values are not numerically close.

    Examples
    --------
    >>> existing_points = {'A': [1.0, 2.0], 'B': [3.0, 4.0]}
    >>> new_points = {'B': [3.0, 4.0], 'C': [5.0, 6.0]}
    >>> merge_with_checks(existing_points, new_points)
    {'A': [1.0, 2.0], 'B': [3.0, 4.0], 'C': [5.0, 6.0]}

    >>> new_points = {'B': [3.1, 4.1]}
    >>> merge_with_checks(existing_points, new_points)
    ValueError: Conflict for key B: existing value [3.0, 4.0], new value [3.1, 4.1]
    """
    for key, value in new_points.items():
        if key in existing_points:
            if not np.allclose(existing_points[key], value):
                raise ValueError(
                    f"Conflict for key {key}: existing value {existing_points[key]}, new value {value}"
                )
        else:
            existing_points[key] = value
    return existing_points


@beartype
def parse_matrix_string(matrix_str: str) -> Dict[str, Union[str, bool]]:
    """
    Parse a matrix string into its components.

    This function processes a string representation of a matrix, identifying its type
    (Transformation, Deviation, or Gap) based on the first character. It extracts and returns
    the relevant components in a structured dictionary format. The function also handles
    specific rules and patterns associated with each type.

    Parameters
    ----------
    matrix_str : str
        The string representation of the matrix to be parsed.

    Returns
    -------
    Dict[str, Union[str, bool]]
        A dictionary containing the parsed components of the matrix. The keys depend on the
        type of matrix but typically include:
        - 'type': The matrix type ('T', 'D', or 'G').
        - 'part', 'surface1', 'point1', 'surface2', 'point2': Components specific to
          Transformation or Gap matrices.
        - 'inverse': Boolean indicating if the matrix is an inverse (for Deviation and Gap types).
        - 'mstring': The reconstructed matrix string based on parsed components.

    Raises
    ------
    ValueError
        If the matrix string does not conform to the expected pattern for its type or
        if an unsupported matrix type is encountered.

    Examples
    --------
    >>> parse_matrix_string('TP1kK0aA0')
    {
        'type': 'T', 'part': '1', 'surface1': 'k', 'point1': 'K0',
        'surface2': 'a', 'point2': 'A0', 'mstring': 'TP1kK0aA0'
    }

    >>> parse_matrix_string('D1k')
    {
        'type': 'D', 'inverse': False, 'part': '1', 'surface': 'k',
        'point': 'K0', 'mstring': 'D1k'
    }

    >>> parse_matrix_string('GP1kK0P2aA0')
    {
        'type': 'G', 'inverse': False, 'part1': '1', 'surface1': 'k', 'point1': 'K0',
        'part2': '2', 'surface2': 'a', 'point2': 'A0',
        'mstring': 'GP1kK0P2aA0'
    }
    """
    result = {"type": matrix_str[0]}

    if result["type"] not in ["D", "T", "G"]:
        raise ValueError(f"Wrong type for matrix string {matrix_str}")

    if result["type"] == "T":
        match = otaf.constants.T_MATRIX_PATTERN.fullmatch(matrix_str)
        if not match:
            raise ValueError(f"Invalid format for Transformation matrix: {matrix_str}")
        if match:
            part, surface1, point1, surface2, point2 = match.groups()
        result.update(
            {
                "part": part,
                "surface1": surface1,
                "point1": point1,
                "surface2": surface2,
                "point2": point2,
                "mstring": f"TP{part}{surface1}{point1}{surface2}{point2}",
            }
        )

    elif result["type"] == "D":
        match1 = otaf.constants.D_MATRIX_PATTERN1.fullmatch(matrix_str)
        match2 = otaf.constants.D_MATRIX_PATTERN2.fullmatch(matrix_str)
        match3 = otaf.constants.D_MATRIX_PATTERN3.fullmatch(matrix_str)
        if not any([match1, match2, match3]):
            raise ValueError(f"Invalid format for Deviation matrix: {matrix_str}")
        if match1:
            is_inverse, part, surface = match1.groups()
            point = surface[0].upper() + "0"
        elif match2:
            is_inverse, part, surface, point = match2.groups()
        elif match3:
            is_inverse, part, surface, partbis, surfacebis = match3.groups()
            assert part == partbis and surface == surfacebis, f"Check definition of {matrix_str}"
            point = surface[0].upper() + "0"
        result.update(
            {
                "inverse": bool(is_inverse),
                "part": part,
                "surface": surface,
                "point": point,
                "mstring": f"D{part}{surface}",
            }
        )

    elif result["type"] == "G":
        match1 = otaf.constants.G_MATRIX_PATTERN1.fullmatch(matrix_str)
        match2 = otaf.constants.G_MATRIX_PATTERN2.fullmatch(matrix_str)
        if not any([match1, match2]):
            raise ValueError(f"Invalid format for Gap matrix: {matrix_str}")
        if match1:
            (is_inverse, part1, surface1, point1, part2, surface2, point2) = match1.groups()
        elif match2:
            is_inverse, part1, surface1, part2, surface2 = match2.groups()
            point1, point2 = surface1[0].upper() + "0", surface2[0].upper() + "0"
        result.update(
            {
                "inverse": bool(is_inverse),
                "part1": part1,
                "surface1": surface1,
                "point1": point1,
                "part2": part2,
                "surface2": surface2,
                "point2": point2,
                "mstring": f"GP{part1}{surface1}{point1}P{part2}{surface2}{point2}",
            }
        )
    return result


@beartype
def get_symbols_in_expressions(expr_list: List[sp.Expr]) -> Tuple[List[sp.Symbol], List[sp.Symbol]]:
    """
    Extract and sort unique symbols from a list of sympy expressions.

    This function processes a list of sympy expressions, extracts the free symbols, and
    categorizes them into deviation symbols and gap symbols based on their naming patterns.
    The symbols in each category are sorted according to a predefined order.

    Parameters
    ----------
    expr_list : List[sp.Expr]
        A list of sympy expressions from which symbols will be extracted.

    Returns
    -------
    Tuple[List[sp.Symbol], List[sp.Symbol]]
        - The first list contains sorted deviation symbols (symbols with "_d_" in their names).
        - The second list contains sorted gap symbols (symbols with "_g_" in their names).

    Raises
    ------
    ValueError
        If a symbol's name does not end with a number or cannot be categorized based on the
        predefined sort order.

    Notes
    -----
    - Sorting prioritizes symbols with a specific prefix order (`"u_", "v_", "w_", "alpha_",
      "beta_", "gamma_"`) followed by numerical suffixes.
    - The function assumes symbols are named consistently with the expected patterns.

    Examples
    --------
    >>> from sympy import symbols, Eq
    >>> u_d_1, v_g_2, w_d_3 = symbols("u_d_1 v_g_2 w_d_3")
    >>> expr_list = [u_d_1 + v_g_2, w_d_3]
    >>> get_symbols_in_expressions(expr_list)
    ([u_d_1, w_d_3], [v_g_2])
    """

    sort_order = ["u_", "v_", "w_", "alpha_", "beta_", "gamma_"]
    sort_key = lambda x: (
        int(re.search(r"(\d+)$", x.name).group(1)),
        [x.name.startswith(prefix) for prefix in sort_order].index(True),
    )

    free_symbols = {symb for expr in expr_list for symb in expr.free_symbols}

    deviation_symbols = sorted(
        (symb for symb in free_symbols if "_d_" in str(symb)),
        key=sort_key,  # otaf.sort_key_by_prefix_and_number
    )
    gap_symbols = sorted(
        (symb for symb in free_symbols if "_g_" in str(symb)),
        key=sort_key,  # otaf.sort_key_by_prefix_and_number
    )

    return deviation_symbols, gap_symbols


@beartype
def get_relevant_expressions(m: sp.MatrixBase) -> List[sp.Expr]:
    """Return all the relevant expressions needed for the compatibility equations.

    Filters out expressions without free variables.

    Args:
        m (Matrix): Sympy matrix representing a matrix.

    Returns:
        List[sp.Expr]: List of relevant expressions.
    """
    indices = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    return [m[i, j] for i, j in indices if len(m[i, j].free_symbols) > 0]


@beartype
def round_floats_in_expression(ex: sp.Expr, rnd: int = 6) -> sp.Expr:
    """Round floats in a sympy expression to the specified decimal places.

    Args:
        ex (sp.Expr): sympy expression to round.
        rnd (int, optional): Number of decimal places to round to. Defaults to 6.

    Returns:
        sp.Expr: Rounded sympy expression.
    """
    substitutions = {
        a: round(a, int(rnd)) for a in sp.preorder_traversal(ex) if isinstance(a, float)
    }
    return ex.subs(substitutions)


@beartype
def get_symbol_coef_map(expr: sp.Expr, rnd: int = 8) -> Dict[str, float]:
    """Extract coefficients of each variable in a first-order polynomial.

    Args:
        expr (sp.Expr): Polynomial expression.
        rnd (int): Rounding digits.

    Returns:
        Dict[str, float]: {variable: coefficient}. Includes constant as 'CONST'.
    """

    coef_map = dict(expr.as_coefficients_dict())
    if 1 in coef_map:
        coef_map["CONST"] = coef_map[1]
        del coef_map[1]
    else:
        coef_map["CONST"] = 0.0
    coef_map = {str(k): round(float(v), rnd) for k, v in coef_map.items()}
    return coef_map


def get_SE3_base(index: str) -> sp.MatrixBase:
    """Get SE(3) base matrix by index.

    Args:
        index (str): Index representing SE(3) base matrix.

    Returns:
        Matrix: SE(3) base matrix.

    Raises:
        KeyError: If index is not found in BASIS_DICT.
    """
    try:
        return sp.Matrix(otaf.constants.BASIS_DICT[index]["MATRIX"])
    except KeyError:
        raise KeyError(f"Index '{index}' not found in BASIS_DICT.")


def get_SE3_matrices_from_indices(
    se3_indices: List[Union[str, int]], multiplier: Union[float, int] = 1.0
) -> List[sp.MatrixBase]:
    """Get SE(3) matrices from indices.

    Args:
        se3_indices (List[str]): List of indices representing SE(3) base matrices.
        multiplier (Union[float, int], optional): Multiplier for the matrices. Defaults to 1.

    Returns:
        List[Matrix]: List of SE(3) matrices.
    """
    return [multiplier * get_SE3_base(str(i)) for i in se3_indices]


def validate_dict_keys(
    dictionary: dict, keys: List[str], dictionary_name: str, value_checks: dict = None
) -> None:
    """Validate keys in a dictionary and optionally check their values.

    Args:
        dictionary (dict): Dictionary to check.
        keys (List[str]): List of keys to check in the dictionary.
        dictionary_name (str): Name of the dictionary (for error messages).
        value_checks (dict, optional): Optional dictionary where keys are the dictionary keys to check.

    Raises:
        MissingKeyError: If a required key is missing.
        ValueError: If a value check fails.
    """
    for key in keys:
        if key not in dictionary:
            raise otaf.exceptions.MissingKeyError(key, dictionary_name)

        if value_checks and key in value_checks:
            validator = value_checks[key]
            if not validator(dictionary[key]):
                raise ValueError(
                    f"Value for key '{key}' in '{dictionary_name}' did not pass validation."
                )


def is_running_in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_tqdm_range():
    return trange_notebook if is_running_in_notebook() else trange


def alphabet_generator():
    n = 1  # Start with single-letter combinations
    # Infinite loop to keep generating sequences
    while True:
        # Generate all combinations for the current length `n`
        for s in itertools.product(string.ascii_lowercase, repeat=n):
            # Join tuple into a string and yield
            yield "".join(s)
        # Increment to generate sequences of increased length
        n += 1


def threshold_for_percentile_positive_values_below(
    arr: Union[list, np.ndarray], percentile: float
) -> Optional[float]:
    """
    Find the threshold value such that the given percentage of positive values
    in the array are below this threshold.

    Parameters:
    ----------
    arr : array-like
        Input array of floats. Should have at least one positive value.
    percentile : float
        The percentage (between 0 and 100) of positive values to consider.

    Returns:
    -------
    float or None
        The threshold value such that the given percentage of positive values are below it.
        Returns None if there are no positive values.

    Raises:
    ------
    ValueError
        If the percentile is not between 0 and 100, or if the input is empty.

    Examples:
    --------
    >>> threshold_for_percentile_positive_values_below([-1, 2, 3, 4, 5], 50)
    3.0
    >>> threshold_for_percentile_positive_values_below([-1, -2, -3], 50)
    None
    """
    # Convert input to a one-dimensional numpy array
    arr = np.squeeze(np.asarray(arr))

    if arr.ndim != 1:
        raise ValueError("Input must be one-dimensional or squeezable to one-dimensional.")
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100.")
    if arr.size == 0:
        raise ValueError("Input array is empty.")

    # Filter positive values
    positive_values = arr[arr > 0]

    if positive_values.size == 0:
        return None  # No positive values in the array

    # Use numpy percentile for precise computation
    threshold_value = np.percentile(positive_values, percentile)

    return threshold_value
