# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = ["ToleranceAnalysisMatrixPreparer", "get_gap_symbol_bounds"]

import re
import logging

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import sympy as sp
import openturns as ot

from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Any, Set, Optional

import otaf


@beartype
class ToleranceAnalysisMatrixPreparer:
    """
    Fixes deviations in equations to obtain a new set of equations with gap variables.

    Callable with compatibility and interface equations, it prepares inputs for scipy.linprog.solver.

    Attributes:
        deviation_symbols (list): List of deviation variables.
        gap_symbols (list): List of gap variables.
        A_eq_Def (numpy.array): Coefficient matrix for defect variables in compatibility equations.
        A_eq_Gap (numpy.array): Coefficient matrix for gap variables in compatibility equations.
        K_eq (numpy.array): Constants in compatibility equations.
        A_ub_Def (numpy.array): Coefficient matrix for defect variables in interface equations.
        A_ub_Gap (numpy.array): Coefficient matrix for gap variables in interface equations.
        K_ub (numpy.array): Constants in interface equations.
        nD (int): Number of defect variables.
        nG (int): Number of gap variables.
        nC (int): Number of compatibility equations.
        nI (int): Number of interface equations.
    """

    def __init__(
        self, compatibility_eqs: List[sp.Expr], interface_eqs: List[sp.Expr], verbose: int = 0
    ) -> None:
        """
        Initialize the ToleranceAnalysisMatrixPreparer.

        Parameters:
            compatibility_eqs (list): List of compatibility equations (equality constraints).
            interface_eqs (list): List of interface equations (inequality constraints).
            verbose (int, optional): Verbosity level (0 for no output, higher values for more verbose outputs).
        """
        logging.info(
            f"[Type: ToleranceAnalysisMatrixPreparer] Initializing ToleranceAnalysisMatrixPreparer with {len(compatibility_eqs)} compatibility equations and {len(interface_eqs)} interface equations."
        )
        self.verbose = verbose

        self.compatibility_eqs = compatibility_eqs
        self.interface_eqs = interface_eqs

        self.nC = len(self.compatibility_eqs)
        self.nI = len(self.interface_eqs)
        self.deviation_symbols, self.gap_symbols = self.extractFreeGapAndDeviationVariables()
        logging.info(
            f"[Type: ToleranceAnalysisMatrixPreparer] Derived deviation symbols: {self.deviation_symbols}, gap symbols: {self.gap_symbols}"
        )

        self.nD = len(self.deviation_symbols)
        self.nG = len(self.gap_symbols)
        logging.info(
            f"[Type: ToleranceAnalysisMatrixPreparer] Number of deviation symbols (nD): {self.nD}"
        )
        logging.info(
            f"[Type: ToleranceAnalysisMatrixPreparer] Number of gap symbols (nG): {self.nG}"
        )

        (
            self.A_eq_Def,
            self.A_eq_Gap,
            self.K_eq,
            self.A_ub_Def,
            self.A_ub_Gap,
            self.K_ub,
        ) = self.generateConstraintMatrices()
        logging.info("[Type: ToleranceAnalysisMatrixPreparer] Matrix representation obtained.")

    def __call__(
        self,
        deviation_array: Union[np.ndarray, Iterable],
        bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
        C: Optional[Union[np.ndarray, List[Union[int, float]]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate input matrices for scipy optimization based on deviation_array.

        Args:
            deviation_array (np.ndarray): Array of shape (nDOE, nD) representing deviation variables.
            bounds (list, optional): Bounds for gap variables. Defaults to None.
            C (np.ndarray, optional): Coefficients of the linear objective function to be minimized. Defaults to None.

        Returns:
            C (np.ndarray): Coefficients of the linear objective function.
            A_ub (np.ndarray): Matrix representing inequality constraints.
            B_ub (np.ndarray): Upper bounds of inequality constraints.
            A_eq (np.ndarray): Matrix representing equality constraints.
            B_eq (np.ndarray): Right-hand side of equality constraints.
            bounds (np.ndarray): Variable bounds.

        Notes:
            - Deviation variables should be in the same order as self.deviation_symbols.
            - Gap variables should be in the same order as self.gap_symbols.
            - Default bounds for variables are specified.
            - Bounds can be overridden with the 'bounds' argument.
        """
        logging.info("[Type: ToleranceAnalysisMatrixPreparer] Invoking the __call__ method.")
        deviation_array = np.atleast_2d(deviation_array)  # If there is only one deviation array

        if self.verbose > 0:
            logging.info(
                "[Type: ToleranceAnalysisMatrixPreparer] The variables must be in the same order as in self.deviation_symbols"
            )
        if deviation_array.shape[1] != len(self.deviation_symbols):
            logging.error(
                f"[Type: ToleranceAnalysisMatrixPreparer] The number of deviation variables should be {len(self.deviation_symbols)}, but {deviation_array.shape[1]} were provided."
            )
            raise ValueError(
                f"[Type: ToleranceAnalysisMatrixPreparer] The number of deviation variables should be {len(self.deviation_symbols)}, but {deviation_array.shape[1]} were provided."
            )

        # Check or define objective function
        if C is None or not isinstance(C, np.ndarray) or C.shape != (self.nG,):
            logging.info(
                "[Type: ToleranceAnalysisMatrixPreparer] The C array representing the coefficients of the linear objective function to be minimized has not been (well) defined"
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
            "[Type: ToleranceAnalysisMatrixPreparer] Returning matrices and bounds for scipy optimization."
        )
        return C, A_ub, B_ub, A_eq, B_eq, bounds

    def generateConstraintMatrices(
        self, rnd: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decompose equations into matrix representation for compatibility and interface constraints.

        This method converts the equations under the X*A + Y*B + K form into matrices:

        - Compatibility equations (equality constraints) become A_eq_Def*X + A_eq_Gap*Y + K_eq = 0.
        - Interface equations (inequality constraints) become A_ub_Def*X + A_ub_Gap*Y + K_ub >= 0.

        Returns:
            A tuple containing matrices A_eq_Def, A_eq_Gap, K_eq, A_ub_Def, A_ub_Gap, and K_ub.

        Parameters:
            rnd (int, optional): Number of decimal places to round the matrix elements (default is 6).
        """
        logging.info(
            "[Type: ToleranceAnalysisMatrixPreparer] Generating matrix representation for constraints."
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
            symb_coef_map = otaf.common.get_symbol_coef_map(self.compatibility_eqs[i])
            for variable, coefficient in symb_coef_map.items():
                if variable in symb_rank_map_def:
                    A_eq_Def[i, symb_rank_map_def[variable]] = coefficient
                elif variable in symb_rank_map_gap:
                    A_eq_Gap[i, symb_rank_map_gap[variable]] = coefficient
                elif variable == "CONST":
                    K_eq[i] += coefficient

        for i in range(self.nI):
            symb_coef_map = otaf.common.get_symbol_coef_map(self.interface_eqs[i])
            for variable, coefficient in symb_coef_map.items():
                if variable in symb_rank_map_def:
                    A_ub_Def[i, symb_rank_map_def[variable]] = coefficient
                elif variable in symb_rank_map_gap:
                    A_ub_Gap[i, symb_rank_map_gap[variable]] = coefficient
                elif variable == "CONST":
                    K_ub[i] += coefficient
        logging.info("[Type: ToleranceAnalysisMatrixPreparer] Matrix representation generated.")
        return (
            A_eq_Def.round(rnd),
            A_eq_Gap.round(rnd),
            K_eq.round(rnd),
            A_ub_Def.round(rnd),
            A_ub_Gap.round(rnd),
            K_ub.round(rnd),
        )

    def extractFreeGapAndDeviationVariables(self) -> Tuple[List[sp.Symbol], List[sp.Symbol]]:
        """Return the sets of deviation variables and gap variables that are present in the compatibility equations.

        It also checks if all interface variables are included in the compatibility variables.

        Parameters:
            None

        Returns:
            A tuple containing the sets of deviation variables and gap variables present in the compatibility equations.

        Raises:
            AssertionError: If any interface variable is not included in the compatibility variables.
        """
        logging.info(
            "[Type: ToleranceAnalysisMatrixPreparer] Retrieving free gap and deviation variables."
        )
        deviation_symbols, gap_symbols = otaf.common.get_symbols_in_expressions(
            self.compatibility_eqs
        )
        symb_dev_interf, symb_gap_interf = otaf.common.get_symbols_in_expressions(
            self.interface_eqs
        )
        if not set(symb_dev_interf).issubset(deviation_symbols) or not set(
            symb_gap_interf
        ).issubset(gap_symbols):
            logging.error(
                "[Type: ToleranceAnalysisMatrixPreparer] Interface variables not included in the compatibility variables."
            )
            raise AssertionError("Interface variables not included in the compatibility variables.")
        logging.info(
            "[Type: ToleranceAnalysisMatrixPreparer] Free gap and deviation variables retrieved successfully."
        )
        return deviation_symbols, gap_symbols

    def validateOptimizationResults(
        self, gap_array: np.ndarray, deviation_array: np.ndarray, rnd: int = 9
    ) -> Tuple[
        List[Union[float, int, sp.Float, sp.Integer]], List[Union[float, int, sp.Float, sp.Integer]]
    ]:
        """Method to use the original equations of the problem (before the transformation),
        with a certain choice of values for the plays, and returning the value for each equation.

        Args:
            gap_array (np.ndarray): Array of gap variables.
            deviation_array (np.ndarray): Array of deviation variables.
            rnd (int, optional): Number of decimal places to round the results (default is 4).

        Returns:
            List[float]: List of results for interface equations.

        Notes:
            This method validates optimization results using the original equations of the problem.
        """
        logging.info(
            "[Type: ToleranceAnalysisMatrixPreparer] Validating optimization using original equations."
        )
        gap_dict = dict(zip(map(str, self.gap_symbols), gap_array))
        deviation_dict = dict(zip(map(str, self.deviation_symbols), deviation_array))
        fixed_vars = {**gap_dict, **deviation_dict}
        compatibility_result = []
        for i, comp_expr in enumerate(self.compatibility_eqs):
            compatibility_result.append(round(comp_expr.evalf(subs=fixed_vars), rnd))
            if round(compatibility_result[i], rnd) > 0.01:
                logging.info(
                    f"[Type: ToleranceAnalysisMatrixPreparer] Result for compatibility expression {i} is : {round(compatibility_result[i], rnd)} \t (others are close to 0)"
                )
        interface_results = []
        for i, inter_expr in enumerate(self.interface_eqs):
            interface_results.append(round(inter_expr.evalf(subs=fixed_vars), rnd))
            logging.info(
                f"[Type: ToleranceAnalysisMatrixPreparer] Result for interface expression {i} is : {round(interface_results[i], rnd)}"
            )
            if self.verbose > 0:
                print(
                    f"[Type: ToleranceAnalysisMatrixPreparer] Result for interface expression {i} is : {round(interface_results[i], rnd)}"
                )
        return compatibility_result, interface_results

    def embedOptimizationVariable(self):
        """This function adds a variable to the gap variables called s. This variable ensures
        that a solution is found even when the optimization would have no solution without it.
        Its sign shows if the parts can be assembled or not. It is also on this variable that
        we can later construct meta-models etc.
        """
        self.gap_symbols.append(sp.Symbol("s"))
        self.nG += 1
        self.A_ub_Gap = np.hstack([self.A_ub_Gap, -1 * np.ones((self.nI, 1))])
        self.A_eq_Gap = np.hstack([self.A_eq_Gap, np.zeros((self.nC, 1))])

    def get_feature_indices_and_dimensions(self):
        """
        Extracts unique feature indices (classes) and their corresponding sizes from the deviation symbols.

        The method processes the `self.deviation_symbols` list, extracting the numeric index
        following the pattern '_d_X' (where X is the class identifier). It counts the occurrences
        of each class index and returns two lists:

        - A list of unique class indices in sorted order.
        - A list of sizes representing how many elements are associated with each class index.

        Returns:
            tuple:
                - A list of unique class indices (sorted).
                - A list of corresponding class sizes (counts of variables per class).
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
    """Get bounds for a list of gap symbols.

    Args:
        gap_symbols (List[sp.Symbol]): A list of gap symbols.

    Returns:
        List[List[float]]: A list of bounds for each gap symbol. Bounds are in mm (millimeters) for symbols starting with 'u_g',
              and in mm for symbols starting with 'v_g' or 'w_g'. Bounds are in radians for symbols starting with
              'alpha_g', 'beta_g', or 'gamma_g'.
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
