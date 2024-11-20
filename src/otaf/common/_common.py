# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "inverse_mstring",
    "merge_with_checks",
    "parse_matrix_string",
    "get_symbols_in_expressions",
    "extract_expressions_with_variables",
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
from beartype.typing import List, Union, Dict, Tuple, Optional

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
    {'type': 'T', 'part': '1', 'surface1': 'k', 'point1': 'K0',
     'surface2': 'a', 'point2': 'A0', 'mstring': 'TP1kK0aA0'}

    >>> parse_matrix_string('D1k')
    {'type': 'D', 'inverse': False, 'part': '1', 'surface': 'k',
     'point': 'K0', 'mstring': 'D1k'}

    >>> parse_matrix_string('GP1kK0P2aA0')
    {'type': 'G', 'inverse': False, 'part1': '1', 'surface1': 'k', 'point1': 'K0',
     'part2': '2', 'surface2': 'a', 'point2': 'A0', 'mstring': 'GP1kK0P2aA0'}
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
    try :
        deviation_symbols = sorted(
            (symb for symb in free_symbols if "_d_" in str(symb)),
            key=sort_key,  # otaf.sort_key_by_prefix_and_number
        )
        gap_symbols = sorted(
            (symb for symb in free_symbols if "_g_" in str(symb)),
            key=sort_key,  # otaf.sort_key_by_prefix_and_number
        )
    except ValueError :
        raise ValueError("Symbols must follow naming guidelines")

    return deviation_symbols, gap_symbols


@beartype
def extract_expressions_with_variables(m: sp.MatrixBase) -> List[sp.Expr]:
    """Extract matrix expressions containing free variables.

    This function filters specific off-diagonal and last-column elements of a given symbolic matrix
    and returns only those expressions that contain free variables (i.e., symbols).

    Parameters
    ----------
    m : sympy.MatrixBase
        Sympy matrix from which expressions are to be extracted.

    Returns
    -------
    List[sympy.Expr]
        A list of expressions from the matrix that contain free variables.

    Notes
    -----
    The indices for the filtered elements include:
    - Off-diagonal terms: (0, 1), (0, 2), (1, 2)
    - Last-column terms: (0, 3), (1, 3), (2, 3)
    """
    indices = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    return [m[i, j] for i, j in indices if len(m[i, j].free_symbols) > 0]


@beartype
def round_floats_in_expression(ex: sp.Expr, rnd: int = 6) -> sp.Expr:
    """
    Round floating-point numbers in a SymPy expression to a specified number of decimal places.

    Parameters
    ----------
    ex : sympy.Expr
        SymPy expression containing floating-point numbers to be rounded.
    rnd : int, optional
        Number of decimal places to round to (default is 6).

    Returns
    -------
    sympy.Expr
        A new SymPy expression with all floating-point numbers rounded to the specified decimal places.
    """
    substitutions = {
        a: round(a, int(rnd)) for a in sp.preorder_traversal(ex) if isinstance(a, float)
    }
    return ex.subs(substitutions)


@beartype
def get_symbol_coef_map(expr: sp.Expr, rnd: int = 8) -> Dict[str, float]:
    """
    Extract coefficients of each variable in a first-order polynomial.

    Parameters
    ----------
    expr : sympy.Expr
        The polynomial expression to extract coefficients from.
    rnd : int, optional
        Number of decimal places to round the coefficients to (default is 8).

    Returns
    -------
    Dict[str, float]
        A dictionary mapping variable names (as strings) to their coefficients (as floats).
        Includes the constant term, represented by the key 'CONST'.

    Notes
    -----
    - The function assumes the input is a first-order polynomial.
    - If no constant term exists in the expression, 'CONST' is set to 0.0.
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
    """
    Retrieve the SE(3) base matrix corresponding to a given index.

    Parameters
    ----------
    index : str
        The index representing the SE(3) base matrix.

    Returns
    -------
    sympy.MatrixBase
        The SE(3) base matrix associated with the given index.

    Raises
    ------
    KeyError
        If the specified index is not found in `BASIS_DICT`.

    Notes
    -----
    The SE(3) base matrix is retrieved from the predefined `BASIS_DICT` in the `otaf.constants` module.
    """
    try:
        return sp.Matrix(otaf.constants.BASIS_DICT[index]["MATRIX"])
    except KeyError:
        raise KeyError(f"Index '{index}' not found in BASIS_DICT.")


def get_SE3_matrices_from_indices(
    se3_indices: List[Union[str, int]], multiplier: Union[float, int] = 1.0
) -> List[sp.MatrixBase]:
    """
    Retrieve SE(3) matrices corresponding to a list of indices, optionally scaled by a multiplier.

    Parameters
    ----------
    se3_indices : List[Union[str, int]]
        List of indices representing SE(3) base matrices.
    multiplier : Union[float, int], optional
        A scalar multiplier applied to each matrix (default is 1.0).

    Returns
    -------
    List[sympy.MatrixBase]
        A list of SE(3) base matrices, each scaled by the specified multiplier.

    Notes
    -----
    The indices are converted to strings and used to retrieve the corresponding SE(3) base matrices
    from the predefined `BASIS_DICT` in the `otaf.constants` module.
    """
    return [multiplier * get_SE3_base(str(i)) for i in se3_indices]


def validate_dict_keys(
    dictionary: dict, keys: List[str], dictionary_name: str, value_checks: dict = None
) -> None:
    """
    Validate the presence of specific keys in a dictionary and optionally check their values.

    Parameters
    ----------
    dictionary : dict
        The dictionary to validate.
    keys : List[str]
        A list of keys that must be present in the dictionary.
    dictionary_name : str
        The name of the dictionary (used in error messages for clarity).
    value_checks : dict, optional
        A dictionary where keys are the dictionary keys to validate and values are functions
        that take the dictionary's value as input and return a boolean indicating whether the value is valid.
        If None, value validation is skipped (default is None).

    Raises
    ------
    otaf.exceptions.MissingKeyError
        If a required key is missing from the dictionary.
    ValueError
        If a value associated with a key fails the validation function provided in `value_checks`.

    Notes
    -----
    - `value_checks` is an optional mechanism to enforce additional constraints on dictionary values.
    - This function is designed for use cases where strict key and value validation is required.
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
    """
    Check if the current code is running in a Jupyter notebook environment.

    This function detects the type of IPython shell to determine whether the code is
    executed in a Jupyter notebook, a terminal, or a standard Python interpreter.

    Returns
    -------
    bool
        True if the code is running in a Jupyter notebook or qtconsole, False otherwise.

    Notes
    -----
    - If the IPython shell cannot be detected, it is assumed the code is running
      in a standard Python interpreter.
    - This function relies on the `get_ipython()` function, which is available
      only in IPython environments.
    """
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
    """
    Retrieve the appropriate tqdm range function based on the execution environment.

    Returns
    -------
    function
        `trange_notebook` if running in a Jupyter notebook, otherwise `trange`.

    Notes
    -----
    - This function checks the execution environment using `is_running_in_notebook`.
    - `trange_notebook` is used for better rendering in notebook environments.
    """
    return trange_notebook if is_running_in_notebook() else trange


def alphabet_generator():
    """
    Generate infinite sequences of alphabetic strings, starting with single letters.

    This generator produces all possible combinations of lowercase alphabetic characters
    ('a' to 'z'), incrementing the sequence length after exhausting combinations of the current length.

    Yields
    ------
    str
        The next alphabetic string in the sequence.

    Notes
    -----
    - The generator starts with single-letter combinations ('a', 'b', ..., 'z') and continues
      with multi-letter combinations (e.g., 'aa', 'ab', ..., 'zz', 'aaa', etc.).
    - It produces strings indefinitely, making it suitable for cases where infinite sequences are needed.
    """
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
    Compute the threshold value such that the given percentage of positive values
    in the input array are below this threshold.

    Parameters
    ----------
    arr : array-like
        Input array of floats. Must contain at least one positive value for meaningful output.
    percentile : float
        The percentage (between 0 and 100) of positive values to consider.

    Returns
    -------
    float or None
        The threshold value such that the specified percentage of positive values are below it.
        Returns None if there are no positive values in the input array.

    Raises
    ------
    ValueError
        If the percentile is not between 0 and 100, if the input array is empty,
        or if the input cannot be reduced to a one-dimensional array.

    Notes
    -----
    - This function operates on the positive values in the input array, ignoring non-positive values.
    - The input array is squeezed to handle cases where the input is a higher-dimensional array that can
      be reduced to one dimension.

    Examples
    --------
    # Compute the 50th percentile threshold for positive values
    >>> threshold_for_percentile_positive_values_below([-1, 2, 3, 4, 5], 50)
    3.0

    # Handle arrays with no positive values
    >>> threshold_for_percentile_positive_values_below([-1, -2, -3], 50)
    None
    """
    arr = np.squeeze(np.asarray(arr))

    if arr.ndim != 1:
        raise ValueError("Input must be one-dimensional or squeezable to one-dimensional.")
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100.")
    if arr.size == 0:
        raise ValueError("Input array is empty.")

    positive_values = arr[arr > 0]
    if positive_values.size == 0:
        return None  # No positive values in the array

    threshold_value = np.percentile(positive_values, percentile)
    return threshold_value
