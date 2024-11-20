# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = ["MatrixPolynomeTaylorExpansion"]


import logging
import sympy as sp

from beartype import beartype
from beartype.typing import List, Union, Optional

import otaf


@beartype
class MatrixPolynomeTaylorExpansion:
    """
    Initialize the MatrixPolynomeTaylorExpansion class.

    Parameters
    ----------
    matrix_list : List[Union[otaf.DeviationMatrix, otaf.GapMatrix, otaf.TransformationMatrix, otaf.I4, otaf.J4]]
        List of matrices to be used in the first-order Taylor expansion.
    """
    def __init__(
        self,
        matrix_list: List[
            Union[otaf.DeviationMatrix, otaf.GapMatrix, otaf.TransformationMatrix, otaf.I4, otaf.J4]
        ],
    ):
        logging.info(
            "[Type: MatrixPolynomeTaylorExpansion] Initializing with provided matrix list."
        )
        self.matrix_list = matrix_list

    def construct_FO_matrix_expansion(self, tolerance: float = 1e-8) -> sp.MatrixBase:
        """
        Construct the first-order Taylor expansion of a matrix product.

        This method computes the expansion by iterating through the provided list of matrices,
        identifying constant terms and first-order variable-dependent terms.

        Parameters
        ----------
        tolerance : float, optional
            Tolerance value for symbolic simplification (default is 1e-8).

        Returns
        -------
        sp.MatrixBase
            A symbolic matrix representing the simplified first-order Taylor expansion.
        """
        logging.info(
            "[Type: MatrixPolynomeTaylorExpansion] Constructing first-order matrix expansion."
        )
        MATRIX_K0 = self._compute_constant_term()
        polynomial_terms = [MATRIX_K0]

        for i, mat in enumerate(self.matrix_list):
            if mat.TYPE in ("D", "G"):
                logging.debug(f"[Type: {mat.TYPE}] Computing first-order elements.")
                variables = mat.get_variables()
                matrices = mat.get_matrix_inverse() if mat.inverse else mat.get_matrix()
                new_terms = [
                    self._compute_first_order_element(i, mat.TYPE, variables[j] * matrices[j])
                    for j in range(mat.n_variables)
                ]
                polynomial_terms.extend(new_terms)

        polynomial_sum = sum(polynomial_terms, start=sp.zeros(4))
        logging.info(
            "[Type: MatrixPolynomeTaylorExpansion] Simplifying first-order matrix expansion."
        )
        return sp.nsimplify(polynomial_sum, rational=False, full=True, tolerance=tolerance)

    def _compute_constant_term(self) -> sp.MatrixBase:
        """
        Compute the constant term in the Taylor expansion.

        The constant term, often denoted as K0, is derived from the product of matrices
        that are categorized as constant types (e.g., T, J4, I4).

        Returns
        -------
        sp.MatrixBase
            The constant term matrix.
        """
        logging.info("[Type: MatrixPolynomeTaylorExpansion] Computing constant term (K0).")
        constant_matrices = []
        for mat in self.matrix_list:
            if mat.TYPE in ("T", "J4", "I4"):
                constant_matrices.append(mat.get_matrix())
        return sp.prod(constant_matrices)

    def _compute_first_order_element(
        self, mat_idx: int, typ: str, var_mat: sp.MatrixBase
    ) -> sp.MatrixBase:
        """
        Compute a first-order term in the Taylor expansion.

        This method calculates the first-order contribution of a specific matrix with
        blocked variables by combining the variable-dependent matrix with other constant
        matrices in the list.

        Parameters
        ----------
        mat_idx : int
            Index of the matrix in the list that contains blocked variables.
        typ : str
            The type of the matrix ('D' or 'G') indicating it contains blocked variables.
        var_mat : sp.MatrixBase
            The matrix representing the blocked variable's contribution.

        Returns
        -------
        sp.MatrixBase
            A symbolic matrix representing the first-order term of the expansion.

        Raises
        ------
        ValueError
            If the type is not 'D' or 'G', or the index does not correspond to a matrix of the expected type.
        """
        logging.debug(
            f"[Type: MatrixPolynomeTaylorExpansion] Computing first-order element for index: {mat_idx} and type: {typ}."
        )
        if typ not in ("D", "G") or self.matrix_list[mat_idx].TYPE not in ("D", "G"):
            raise ValueError("Wrong type for first-order element expansion matrix or wrong index")
        MAT_MUL_ELEMS = []
        for i, mat in enumerate(self.matrix_list):
            if i == mat_idx:
                MAT_MUL_ELEMS.append(var_mat)
            elif mat.TYPE in ("T", "I4", "J4"):
                MAT_MUL_ELEMS.append(mat.get_matrix())
        return sp.prod(MAT_MUL_ELEMS)
