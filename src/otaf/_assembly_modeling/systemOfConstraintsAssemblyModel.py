from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = ["SystemOfConstraintsAssemblyModel"]


import re
import logging

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import sympy as sp

from beartype import beartype
from beartype.typing import List, Tuple, Union, Optional

from otaf.common import get_symbol_coef_map, get_symbols_in_expressions


@beartype
class SystemOfConstraintsAssemblyModel:
    """
    Prepare matrices for tolerance analysis involving deviations and gaps.

    This class processes compatibility and interface equations to generate a matrix representation
    suitable for linear programming solvers like `scipy.optimize.linprog`.

    Attributes
    ----------
    deviation_symbols : list
        List of deviation variables.
    gap_symbols : list
        List of gap variables.
    A_eq_Def : numpy.ndarray
        Coefficient matrix for deviation variables in compatibility equations.
    A_eq_Gap : numpy.ndarray
        Coefficient matrix for gap variables in compatibility equations.
    K_eq : numpy.ndarray
        Constants in compatibility equations.
    A_ub_Def : numpy.ndarray
        Coefficient matrix for deviation variables in interface equations.
    A_ub_Gap : numpy.ndarray
        Coefficient matrix for gap variables in interface equations.
    K_ub : numpy.ndarray
        Constants in interface equations.
    nD : int
        Number of deviation variables.
    nG : int
        Number of gap variables.
    nC : int
        Number of compatibility equations.
    nI : int
        Number of interface equations.

    Methods
    -------
    __init__(compatibility_eqs, interface_eqs, verbose=0)
        Initialize the matrix preparer with compatibility and interface equations.
    __call__(deviation_array, bounds=None, C=None)
        Generate input matrices and bounds for linear programming optimization.

    Parameters
    ----------
    compatibility_eqs : list of sympy.Expr
        List of compatibility equations (equality constraints).
    interface_eqs : list of sympy.Expr
        List of interface equations (inequality constraints).
    verbose : int, optional
        Verbosity level for logging (default is 0).
    """
    def __init__(
        self, compatibility_eqs: List[sp.Expr], interface_eqs: List[sp.Expr], verbose: int = 0
    ) -> None:
        """
        Initialize the SystemOfConstraintsAssemblyModel.

        Processes the provided compatibility and interface equations to extract variables
        and prepare the matrix representation for optimization.

        Parameters
        ----------
        compatibility_eqs : list of sympy.Expr
            List of compatibility equations (equality constraints).
        interface_eqs : list of sympy.Expr
            List of interface equations (inequality constraints).
        verbose : int, optional
            Verbosity level for logging (default is 0).

        Raises
        ------
        ValueError
            If the equations are improperly formatted or the variables cannot be extracted.

        Notes
        -----
        - The number of compatibility and interface equations are determined during initialization.
        - Variables are extracted and classified as deviation or gap variables.
        """
        logging.info(
            f"[Type: SystemOfConstraintsAssemblyModel] Initializing SystemOfConstraintsAssemblyModel with {len(compatibility_eqs)} compatibility equations and {len(interface_eqs)} interface equations."
        )
        self.verbose = verbose

        self.compatibility_eqs = compatibility_eqs
        self.interface_eqs = interface_eqs

        self.nC = len(self.compatibility_eqs)
        self.nI = len(self.interface_eqs)
        self.deviation_symbols, self.gap_symbols = self.extractFreeGapAndDeviationVariables()
        logging.info(
            f"[Type: SystemOfConstraintsAssemblyModel] Derived deviation symbols: {self.deviation_symbols}, gap symbols: {self.gap_symbols}"
        )

        self.nD = len(self.deviation_symbols)
        self.nG = len(self.gap_symbols)
        logging.info(
            f"[Type: SystemOfConstraintsAssemblyModel] Number of deviation symbols (nD): {self.nD}"
        )
        logging.info(
            f"[Type: SystemOfConstraintsAssemblyModel] Number of gap symbols (nG): {self.nG}"
        )

        (
            self.A_eq_Def,
            self.A_eq_Gap,
            self.K_eq,
            self.A_ub_Def,
            self.A_ub_Gap,
            self.K_ub,
        ) = self.generateConstraintMatrices()
        logging.info("[Type: SystemOfConstraintsAssemblyModel] Matrix representation obtained.")

    def __call__(
        self,
        deviation_array: Union[np.ndarray, Iterable],
        bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
        C: Optional[Union[np.ndarray, List[Union[int, float]]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate input matrices and bounds for linear programming optimization.

        This method prepares inputs for `scipy.optimize.linprog` using the deviation variables
        and optionally provided bounds and objective coefficients.

        Parameters
        ----------
        deviation_array : numpy.ndarray or Iterable
            Array of shape (nDOE, nD) representing deviation variables.
        bounds : list of list of float or numpy.ndarray, optional
            Bounds for gap variables (default is automatically determined).
        C : numpy.ndarray or list of float, optional
            Coefficients of the linear objective function to be minimized (default is inferred).

        Returns
        -------
        tuple
            Contains the following elements:
            - C : numpy.ndarray
                Coefficients of the linear objective function.
            - A_ub : numpy.ndarray
                Matrix representing inequality constraints.
            - B_ub : numpy.ndarray
                Right-hand side of inequality constraints.
            - A_eq : numpy.ndarray
                Matrix representing equality constraints.
            - B_eq : numpy.ndarray
                Right-hand side of equality constraints.
            - bounds : numpy.ndarray
                Variable bounds.

        Raises
        ------
        ValueError
            If the number of deviation variables in `deviation_array` does not match `self.deviation_symbols`.

        Notes
        -----
        - Deviation variables must be in the same order as `self.deviation_symbols`.
        - Gap variables must be in the same order as `self.gap_symbols`.
        - Default bounds are generated if `bounds` is not provided or improperly formatted.
        """
        logging.info("[Type: SystemOfConstraintsAssemblyModel] Invoking the __call__ method.")
        deviation_array = np.atleast_2d(deviation_array)  # If there is only one deviation array

        if self.verbose > 0:
            logging.info(
                "[Type: SystemOfConstraintsAssemblyModel] The variables must be in the same order as in self.deviation_symbols"
            )
        if deviation_array.shape[1] != len(self.deviation_symbols):
            logging.error(
                f"[Type: SystemOfConstraintsAssemblyModel] The number of deviation variables should be {len(self.deviation_symbols)}, but {deviation_array.shape[1]} were provided."
            )
            raise ValueError(
                f"[Type: SystemOfConstraintsAssemblyModel] The number of deviation variables should be {len(self.deviation_symbols)}, but {deviation_array.shape[1]} were provided."
            )

        # Check or define objective function
        if C is None or not isinstance(C, np.ndarray) or C.shape != (self.nG,):
            logging.info(
                "[Type: SystemOfConstraintsAssemblyModel] The C array representing the coefficients of the linear objective function to be minimized has not been (well) defined"
            )
            C = np.zeros((self.nG,))
            if sp.Symbol("s") not in self.gap_symbols:
                # Select the first gap variable driving a translation in the x direction as the objective to be minimized.
                idGVarXTrans = next(
                    i for i, symb in enumerate(self.gap_symbols) if str(symb).startswith("u_g")
                )
                C[idGVarXTrans] = 1
            else:
                if self.verbose > 1:
                    print("Using s variable for objective")
                C[self.gap_symbols.index(sp.Symbol("s"))] = (
                    -1.0
                )  # Cause we want to maximize s, but this is a minimization.  # Guess its that! But this is important.

        # Check or define the bounds for the variables.
        if bounds is None:
            bounds = get_gap_symbol_bounds(self.gap_symbols)
        else:
            if len(bounds) != self.nG:
                logging.warning("Check bounds parameter, falling back to usual behavior")
                bounds = get_gap_symbol_bounds(self.gap_symbols)

        A_ub = -1 * self.A_ub_Gap
        B_ub = np.add(
            np.matmul(self.A_ub_Def, deviation_array.T), np.expand_dims(self.K_ub, axis=1)
        )

        A_eq = self.A_eq_Gap
        B_eq = np.subtract(
            np.matmul(-1 * self.A_eq_Def, deviation_array.T), np.expand_dims(self.K_eq, axis=1)
        )
        # np.add(np.matmul(-1*self.A_eq_Def, deviation_array.T), np.expand_dims(-1*self.K_eq, axis=1))
        bounds = np.array(bounds)
        logging.info(
            "[Type: SystemOfConstraintsAssemblyModel] Returning matrices and bounds for scipy optimization."
        )
        return C, A_ub, B_ub, A_eq, B_eq, bounds

    def generateConstraintMatrices(
        self, rnd: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose equations into matrix representations for compatibility and interface constraints.

        This method converts the equations into matrix forms suitable for linear programming:
        - Compatibility equations (equality constraints) are represented as:
          `A_eq_Def * X + A_eq_Gap * Y + K_eq = 0`.
        - Interface equations (inequality constraints) are represented as:
          `A_ub_Def * X + A_ub_Gap * Y + K_ub >= 0`.

        Parameters
        ----------
        rnd : int, optional
            Number of decimal places to round the matrix elements (default is 9).

        Returns
        -------
        tuple
            A tuple containing the following matrices:
            - A_eq_Def : numpy.ndarray
                Coefficient matrix for deviation variables in compatibility equations.
            - A_eq_Gap : numpy.ndarray
                Coefficient matrix for gap variables in compatibility equations.
            - K_eq : numpy.ndarray
                Constants in compatibility equations.
            - A_ub_Def : numpy.ndarray
                Coefficient matrix for deviation variables in interface equations.
            - A_ub_Gap : numpy.ndarray
                Coefficient matrix for gap variables in interface equations.
            - K_ub : numpy.ndarray
                Constants in interface equations.

        Notes
        -----
        - The method iterates through each compatibility and interface equation to extract coefficients
          for deviation and gap variables.
        - Variables not explicitly included in the equations are assigned zero coefficients in the matrices.
        """
        logging.info(
            "[Type: SystemOfConstraintsAssemblyModel] Generating matrix representation for constraints."
        )
        A_eq_Def = np.zeros((self.nC, self.nD))
        A_eq_Gap = np.zeros((self.nC, self.nG))
        K_eq = np.zeros((self.nC))

        A_ub_Def = np.zeros((self.nI, self.nD))
        A_ub_Gap = np.zeros((self.nI, self.nG))
        K_ub = np.zeros((self.nI))

        symb_rank_map_def = {str(var): i for i, var in enumerate(self.deviation_symbols)}
        symb_rank_map_gap = {str(var): i for i, var in enumerate(self.gap_symbols)}

        for i in range(self.nC):
            symb_coef_map = get_symbol_coef_map(self.compatibility_eqs[i])
            for variable, coefficient in symb_coef_map.items():
                if variable in symb_rank_map_def:
                    A_eq_Def[i, symb_rank_map_def[variable]] = coefficient
                elif variable in symb_rank_map_gap:
                    A_eq_Gap[i, symb_rank_map_gap[variable]] = coefficient
                elif variable == "CONST":
                    K_eq[i] += coefficient

        for i in range(self.nI):
            symb_coef_map = get_symbol_coef_map(self.interface_eqs[i])
            for variable, coefficient in symb_coef_map.items():
                if variable in symb_rank_map_def:
                    A_ub_Def[i, symb_rank_map_def[variable]] = coefficient
                elif variable in symb_rank_map_gap:
                    A_ub_Gap[i, symb_rank_map_gap[variable]] = coefficient
                elif variable == "CONST":
                    K_ub[i] += coefficient
        logging.info("[Type: SystemOfConstraintsAssemblyModel] Matrix representation generated.")
        return (
            A_eq_Def.round(rnd),
            A_eq_Gap.round(rnd),
            K_eq.round(rnd),
            A_ub_Def.round(rnd),
            A_ub_Gap.round(rnd),
            K_ub.round(rnd),
        )

    def extractFreeGapAndDeviationVariables(self) -> Tuple[List[sp.Symbol], List[sp.Symbol]]:
        """
        Extract sets of deviation and gap variables present in compatibility equations.

        This method identifies the free variables used in the compatibility equations and
        verifies that all variables appearing in the interface equations are included.

        Returns
        -------
        tuple
            A tuple containing:
            - deviation_symbols : list of sympy.Symbol
                List of deviation variables present in the compatibility equations.
            - gap_symbols : list of sympy.Symbol
                List of gap variables present in the compatibility equations.

        Raises
        ------
        AssertionError
            If any variable in the interface equations is not included in the compatibility equations.

        Notes
        -----
        - Deviation and gap variables are extracted separately from both compatibility and interface equations.
        - This ensures consistency between the two sets of equations.
        """
        logging.info(
            "[Type: SystemOfConstraintsAssemblyModel] Retrieving free gap and deviation variables."
        )
        deviation_symbols, gap_symbols = get_symbols_in_expressions(
            self.compatibility_eqs
        )
        symb_dev_interf, symb_gap_interf = get_symbols_in_expressions(
            self.interface_eqs
        )
        if not set(symb_dev_interf).issubset(deviation_symbols) or not set(
            symb_gap_interf
        ).issubset(gap_symbols):
            logging.error(
                "[Type: SystemOfConstraintsAssemblyModel] Interface variables not included in the compatibility variables."
            )
            raise AssertionError("Interface variables not included in the compatibility variables.")
        logging.info(
            "[Type: SystemOfConstraintsAssemblyModel] Free gap and deviation variables retrieved successfully."
        )
        return deviation_symbols, gap_symbols

    def validateOptimizationResults(
        self, gap_array: np.ndarray, deviation_array: np.ndarray, rnd: int = 9
    ) -> Tuple[
        List[Union[float, int, sp.Float, sp.Integer]], List[Union[float, int, sp.Float, sp.Integer]]
    ]:
        """
        Validate optimization results using original equations.

        This method evaluates the original compatibility and interface equations with given values
        for the gap and deviation variables, returning the computed results for validation.

        Parameters
        ----------
        gap_array : numpy.ndarray
            Array of gap variables.
        deviation_array : numpy.ndarray
            Array of deviation variables.
        rnd : int, optional
            Number of decimal places to round the results (default is 9).

        Returns
        -------
        tuple
            - List[float]: Results of evaluating the compatibility equations.
            - List[float]: Results of evaluating the interface equations.

        Notes
        -----
        - The method substitutes the provided gap and deviation values into the original equations.
        - Compatibility results close to zero indicate a valid solution, while larger values suggest potential issues.
        - Interface results show the satisfaction level of inequality constraints.
        """
        logging.info(
            "[Type: SystemOfConstraintsAssemblyModel] Validating optimization using original equations."
        )
        gap_dict = dict(zip(map(str, self.gap_symbols), gap_array))
        deviation_dict = dict(zip(map(str, self.deviation_symbols), deviation_array))
        fixed_vars = {**gap_dict, **deviation_dict}
        compatibility_result = []
        for i, comp_expr in enumerate(self.compatibility_eqs):
            compatibility_result.append(round(comp_expr.evalf(subs=fixed_vars), rnd))
            if round(compatibility_result[i], rnd) > 0.01:
                logging.info(
                    f"[Type: SystemOfConstraintsAssemblyModel] Result for compatibility expression {i} is : {round(compatibility_result[i], rnd)} \t (others are close to 0)"
                )
        interface_results = []
        for i, inter_expr in enumerate(self.interface_eqs):
            interface_results.append(round(inter_expr.evalf(subs=fixed_vars), rnd))
            logging.info(
                f"[Type: SystemOfConstraintsAssemblyModel] Result for interface expression {i} is : {round(interface_results[i], rnd)}"
            )
            if self.verbose > 0:
                print(
                    f"[Type: SystemOfConstraintsAssemblyModel] Result for interface expression {i} is : {round(interface_results[i], rnd)}"
                )
        return compatibility_result, interface_results

    def embedOptimizationVariable(self):
        """
        Embed an auxiliary optimization variable for feasibility.

        This method adds an auxiliary variable, `s`, to the gap variables. The variable `s` ensures
        that a feasible solution can be found, even in cases where the optimization problem would
        otherwise have no solution. The sign of `s` indicates whether the parts can be assembled,
        and the variable can be used in meta-model construction.

        Notes
        -----
        - The variable `s` is appended to the list of gap variables (`self.gap_symbols`).
        - The `A_ub_Gap` and `A_eq_Gap` matrices are updated to include the new variable:
          - `A_ub_Gap` is augmented with a column of -1.
          - `A_eq_Gap` is augmented with a column of zeros.
        """
        self.gap_symbols.append(sp.Symbol("s"))
        self.nG += 1
        self.A_ub_Gap = np.hstack([self.A_ub_Gap, -1 * np.ones((self.nI, 1))])
        self.A_eq_Gap = np.hstack([self.A_eq_Gap, np.zeros((self.nC, 1))])

    def get_feature_indices_and_dimensions(self):
        """
        Extract unique feature indices (classes) and their corresponding sizes from deviation symbols.

        This method processes the `self.deviation_symbols` list to identify unique class indices
        based on the pattern `_d_X` (where `X` is the numeric class identifier). It counts the number
        of variables associated with each class index and returns two lists:
        - A sorted list of unique class indices.
        - A list of corresponding sizes, representing the number of variables per class.

        Returns
        -------
        tuple
            - list of int
                Sorted list of unique class indices.
            - list of int
                List of sizes, where each size corresponds to the number of variables for a class index.

        Notes
        -----
        - Each deviation symbol is assumed to contain the class identifier in the format `_d_X`.
        - The method uses a regular expression to extract the class identifier and counts occurrences for each class.
        """
        class_size_map = defaultdict(int)

        # Extract classes from the vector
        classes = []

        for element in self.deviation_symbols:
            # Use regular expression to extract the index after '_d_'
            match = re.search(r"_d_(\d+)", str(element))
            if match:
                class_index = int(match.group(1))  # Extract the class number
                classes.append(class_index)  # Append to class list
                class_size_map[class_index] += 1  # Increase count for the class

        # Unique classes
        unique_classes = sorted(list(set(classes)))

        # Create the sizes vector: since each class has 4 associated variables
        sizes = [class_size_map[c] for c in unique_classes]

        return unique_classes, sizes


@beartype
def get_gap_symbol_bounds(gap_symbols: List[sp.Symbol]) -> np.ndarray:
    """
    Get bounds for a list of gap symbols.

    This function assigns bounds to gap variables based on their naming convention. Bounds
    are determined in millimeters for translational variables and in radians for rotational variables.

    Parameters
    ----------
    gap_symbols : list of sympy.Symbol
        A list of gap symbols for which to determine bounds.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array of shape (N, 2), where each row contains the [min, max] bounds
        for a corresponding gap symbol.

    Notes
    -----
    - Translational gap symbols:
      - Variables starting with 'u_g', 'v_g', or 'w_g' have bounds of [-3, 3] millimeters.
    - Rotational gap symbols:
      - Variables starting with 'alpha_g', 'beta_g', or 'gamma_g' have bounds of [-π/4, π/4] radians.
    - The auxiliary variable `s` (if present) is unbounded and has bounds of [-∞, ∞].
    - If a symbol does not match any known prefix, a warning is logged.

    Examples
    --------
    >>> get_gap_symbol_bounds([sp.Symbol("u_g1"), sp.Symbol("alpha_g2")])
    array([[-3.        ,  3.        ],
           [-0.78539816,  0.78539816]])
    """
    logging.info(
        f"[Type: function:get_gap_symbol_bounds] Getting bounds for gap symbols: {gap_symbols}."
    )
    bounds = []
    for s in gap_symbols:
        str_s = str(s)
        if str_s.startswith("u_g"):
            bounds.append([-3, 3])  # mm
        elif str_s.startswith(("v_g", "w_g")):
            bounds.append([-3, 3])  # mm
        elif str_s.startswith(("alpha_g", "beta_g", "gamma_g")):
            bounds.append([-np.pi / 4, np.pi / 4])  # rad
        elif str_s == "s":
            bounds.append([-np.inf, np.inf])
        else:
            logging.warning(f"[Type: function:get_gap_symbol_bounds] Unrecognized symbol: {s}")
    logging.debug(f"[Type: function:get_gap_symbol_bounds] Extracted bounds: {bounds}")
    return np.array(bounds, dtype="float64")
