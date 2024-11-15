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
    """Class to compute first-order Taylor expansion of matrices with symbolic variables.

    Attributes:
        matrix_list (List[Union[DeviationMatrix, GapMatrix, TransformationMatrix]]):
            List of matrices to be used in the expansion.
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
        """Expand the matrix multiplication to construct a first-order expansion.

        Initializes a matrix-polynomial-list (MPL) with constant elements and then,
        iterates through the matrix list to add corresponding elements to the MPL.

        Args:
            tolerance (float): Tolerance value for simplification.

        Returns:
            polynomial_sum (Matrix): Simplified first-order expansion.
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
        """Calculate the constant term in the Taylor expansion (often referred to as K0).

        Returns:
            Matrix: The constant term matrix.
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
        """Calculate the first-order element of matrix multiplication with blocked variables.

        Args:
            mat_idx (int): Index of the matrix with blocked variables.
            typ (str): Type of the blocked variable at the index (D or G).
            var_mat (Matrix): Elementar matrix multiplied with the variable.

        Returns:
            MAT_MUL_RESULT (Matrix): First-order element of matrix multiplication.
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
