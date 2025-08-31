from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = ["DeviationMatrix", "GapMatrix", "TransformationMatrix", "I4", "J4"]


import re
import logging
import numpy as np
import sympy as sp
from beartype import beartype
from beartype.typing import List, Union, Optional


from otaf.constants import BASIS_DICT
from otaf.common import get_SE3_matrices_from_indices, inverse_mstring
from otaf.geometry import is_affine_transformation_matrix


@beartype
class DeviationMatrix:
    """
    Represents a deviation matrix used in structural analysis.

    A deviation matrix defines deviations in translation and rotation for parts and assemblies.
    It can be compared to a small deviation torsor.

    Parameters
    ----------
    index : int, optional
        An index used to differentiate variable names. The same index can be used multiple times.
    translations : str, optional
        Degrees of freedom for translation ('x', 'y', 'z'). Example: "xyz" for all three degrees.
    rotations : str, optional
        Degrees of freedom for rotation ('x', 'y', 'z').
    inverse : bool, optional
        Indicates whether the matrix represents deviations from nominal to real (True) or vice versa (False).
        This affects the signs of the matrices and is necessary if a matrix is used in two directions.
    name : str, optional
        Name of the matrix for identification (default is an empty string).
    """
    def __init__(
        self,
        index: int = -1,
        translations: str = "xyz",
        rotations: str = "xyz",
        inverse: bool = False,
        name: str = "",
    ):
        """
        Initialize a DeviationMatrix instance.

        Parameters
        ----------
        index : int, optional
            An index used to differentiate variable names. The same index can be used multiple times (default is -1).
        translations : str, optional
            Degrees of freedom for translation ('x', 'y', 'z'). Example: "xyz" for all three degrees (default is "xyz").
        rotations : str, optional
            Degrees of freedom for rotation ('x', 'y', 'z') (default is "xyz").
        inverse : bool, optional
            Indicates whether the matrix represents deviations from nominal to real (True) or vice versa (False).
            This affects the signs of the matrices (default is False).
        name : str, optional
            Name of the matrix for identification (default is an empty string).

        Raises
        ------
        AssertionError
            If the `index` is not an integer.
        """
        logging.info(
            f"Initializing DeviationMatrix with index: {index}, translations: {translations}, rotations: {rotations}, inverse: {inverse}"
        )
        assert isinstance(index, int), "Wrong index, check class init parameters"
        self.ID = index
        self.TYPE = "D"
        self.name = name
        self._translations = translations
        self._rotations = rotations
        self.inverse = inverse
        self.SE3_basis_indices = []
        self.variables = []
        self._initialize_basis_variables()
        logging.debug(f"Initialized {len(self.variables)} variables for the deviation matrix.")
        self.n_variables = len(self.variables)

    def __repr__(self):
        """
        Generate a string representation of the DeviationMatrix.

        Returns
        -------
        str
            String representation of the DeviationMatrix instance.
        """
        logging.debug(f"Generating string representation for DeviationMatrix with index {self.ID}.")
        return f"DeviationMatrix(index={self.ID}, translations='{self._translations}', rotations='{self._rotations}', inverse={self.inverse}, name={self.name})"

    def _initialize_basis_variables(self):
        """
        Initialize the symbolic basis variables for the deviation matrix.

        This method sets up the variables based on the specified translations and rotations.
        It uses the SE(3) basis indices defined in `otaf.constants.BASIS_DICT`.

        Notes
        -----
        The variables are stored as symbolic `sympy.Symbol` instances and indexed by the provided `index`.
        """
        logging.info("Initializing basis variables.")
        self.SE3_basis_indices.clear()
        self.variables.clear()
        for i in range(1, 7):  # also sorts
            axis = BASIS_DICT[str(i)]["AXIS"]
            if i <= 3:
                if axis in self._translations:
                    self.SE3_basis_indices.append(i)
                    var_name = BASIS_DICT[str(i)]["VARIABLE_D"]
                    self.variables.append(
                        sp.Symbol(f"{var_name}_{self.ID}" if self.ID >= 0 else var_name)
                    )
            else:
                if axis in self._rotations:
                    self.SE3_basis_indices.append(i)
                    var_name = BASIS_DICT[str(i)]["VARIABLE_D"]
                    self.variables.append(
                        sp.Symbol(f"{var_name}_{self.ID}" if self.ID >= 0 else var_name)
                    )

    def get_inverse(self):
        """
        Generate the inverse of the current DeviationMatrix.

        The inverse matrix has the same degrees of freedom (translations and rotations)
        but with reversed directionality.

        Returns
        -------
        DeviationMatrix
            A new instance of `DeviationMatrix` representing the inverse of the current matrix.
        """
        logging.info(f"[Type: 'D', ID: {self.ID}] Generating inverse DeviationMatrix.")
        return DeviationMatrix(
            index=self.ID,
            translations=self._translations,
            rotations=self._rotations,
            inverse=(not self.inverse),
            name=self.name,
        )

    def get_variables(self) -> List[sp.Symbol]:
        """
        Retrieve the symbolic variables associated with the deviation matrix.

        These variables correspond to the degrees of freedom in translations and rotations,
        defined during initialization.

        Returns
        -------
        List[sympy.Symbol]
            A list of symbolic variables representing the degrees of freedom.
        """
        logging.debug(f"[Type: 'D', ID: {self.ID}] Retrieving variables.")
        return self.variables

    def get_matrix(self) -> List[sp.MatrixBase]:
        """
        Generate the SE(3) matrix for the deviation matrix.

        The SE(3) matrix is constructed using the indices of the degrees of freedom specified
        during initialization.

        Returns
        -------
        List[sympy.MatrixBase]
            A list of symbolic matrices corresponding to the SE(3) basis elements.
        """
        logging.info(f"[Type: 'D', ID: {self.ID}] Generating SE3 matrix.")
        return get_SE3_matrices_from_indices(self.SE3_basis_indices)

    def get_matrix_inverse(self) -> List[sp.MatrixBase]:
        """
        Generate the inverse SE(3) matrix for the deviation matrix.

        The inverse SE(3) matrix is constructed by reversing the directionality
        of the degrees of freedom (i.e., multiplying by -1).

        Returns
        -------
        List[sympy.MatrixBase]
            A list of symbolic matrices corresponding to the inverse SE(3) basis elements.
        """
        logging.info(f"[Type: 'D', ID: {self.ID}] Generating inverse SE3 matrix.")
        return get_SE3_matrices_from_indices(self.SE3_basis_indices, multiplier=-1)


@beartype
class GapMatrix:
    """
    Represents a gap matrix used in structural analysis.

    A gap matrix defines gaps or clearances in translation and rotation for parts and assemblies.

    Parameters
    ----------
    index : int, optional
        An index used to differentiate variable names. The same index can be used multiple times (default is -1).
    translations_blocked : str, optional
        Degrees of freedom for translation that are blocked ('x', 'y', 'z') (default is an empty string).
    rotations_blocked : str, optional
        Degrees of freedom for rotation that are blocked ('x', 'y', 'z') (default is an empty string).
    inverse : bool, optional
        Indicates whether the matrix represents gaps or clearances (True) or blocked degrees of freedom (False).
    name : str, optional
        Name of the matrix for identification (default is an empty string).
    """
    def __init__(
        self,
        index: int = -1,
        translations_blocked: str = "",
        rotations_blocked: str = "",
        inverse: bool = False,
        name: str = "",
    ):
        """
        Initialize a GapMatrix instance.

        Parameters
        ----------
        index : int, optional
            An index used to differentiate variable names. The same index can be used multiple times (default is -1).
        translations_blocked : str, optional
            Degrees of freedom for translation that are blocked ('x', 'y', 'z') (default is an empty string).
        rotations_blocked : str, optional
            Degrees of freedom for rotation that are blocked ('x', 'y', 'z') (default is an empty string).
        inverse : bool, optional
            Indicates whether the matrix represents gaps or clearances (True) or blocked degrees of freedom (False).
        name : str, optional
            Name of the matrix for identification (default is an empty string).

        Raises
        ------
        AssertionError
            If the `index` is not an integer.
        """
        logging.info(
            f"[Type: 'G', ID: {index}] Initializing GapMatrix with blocked translations: {translations_blocked}, blocked rotations: {rotations_blocked}, inverse: {inverse}"
        )
        assert isinstance(index, int), "Wrong index, check class init parameters"
        self.ID = index
        self.TYPE = "G"
        self.name = name
        self._translations_blocked = translations_blocked
        self._rotations_blocked = rotations_blocked
        self.SE3_basis_indices = []
        self.variables = []
        self.inverse = inverse
        self._initialize_basis_variables()
        self.n_variables = len(self.variables)
        logging.debug(
            f"[Type: 'G', ID: {self.ID}] Initialized {len(self.variables)} variables for the gap matrix."
        )

    def _initialize_basis_variables(self):
        """
        Initialize the symbolic basis variables for the gap matrix.

        This method sets up variables based on the degrees of freedom that are not blocked
        in translations and rotations, as defined during initialization.

        Notes
        -----
        The variables are stored as symbolic `sympy.Symbol` instances and indexed by the provided `index`.
        """
        self.SE3_basis_indices.clear()
        self.variables.clear()
        for i in range(1, 7):
            axis = BASIS_DICT[str(i)]["AXIS"]
            if i <= 3:
                if axis not in self._translations_blocked:
                    self.SE3_basis_indices.append(i)
                    var_name = BASIS_DICT[str(i)]["VARIABLE_G"]
                    self.variables.append(
                        sp.Symbol(f"{var_name}_{self.ID}" if self.ID >= 0 else var_name)
                    )
            else:
                if axis not in self._rotations_blocked:
                    self.SE3_basis_indices.append(i)
                    var_name = BASIS_DICT[str(i)]["VARIABLE_G"]
                    self.variables.append(
                        sp.Symbol(f"{var_name}_{self.ID}" if self.ID >= 0 else var_name)
                    )
        logging.debug(f"[Type: 'G', ID: {self.ID}] Added {len(self.variables)} basis variables.")

    def __repr__(self):
        """
        Generate a string representation of the GapMatrix.

        Returns
        -------
        str
            String representation of the GapMatrix instance.
        """
        return f"GapMatrix(index={self.ID}, translations_blocked='{self._translations_blocked}', rotations_blocked='{self._rotations_blocked}', inverse={self.inverse}, name={self.name})"

    def get_inverse(self):
        """
        Generate the inverse of the current GapMatrix.

        The inverse matrix has the same blocked degrees of freedom for translations and rotations
        but reverses the directionality (gaps versus clearances).

        Returns
        -------
        GapMatrix
            A new instance of `GapMatrix` representing the inverse of the current matrix.
        """
        logging.info(f"[Type: 'G', ID: {self.ID}] Generating inverse DeviationMatrix.")
        return GapMatrix(
            index=self.ID,
            translations_blocked=self._translations_blocked,
            rotations_blocked=self._rotations_blocked,
            inverse=(not self.inverse),
            name=inverse_mstring(self.name),
        )

    def get_variables(self) -> List[sp.Symbol]:
        """
        Retrieve the symbolic variables associated with the gap matrix.

        These variables correspond to the degrees of freedom in translations and rotations
        that are not blocked, as defined during initialization.

        Returns
        -------
        List[sympy.Symbol]
            A list of symbolic variables representing the degrees of freedom.
        """
        logging.debug(f"[Type: 'G', ID: {self.ID}] Retrieving variables.")
        return self.variables

    def get_matrix(self) -> List[sp.MatrixBase]:
        """
        Generate the SE(3) matrix for the gap matrix.

        The SE(3) matrix is constructed using the indices of the degrees of freedom
        that are not blocked, specified during initialization.

        Returns
        -------
        List[sympy.MatrixBase]
            A list of symbolic matrices corresponding to the SE(3) basis elements.
        """
        logging.info(f"[Type: 'G', ID: {self.ID}] Generating SE3 matrix.")
        return get_SE3_matrices_from_indices(self.SE3_basis_indices)

    def get_matrix_inverse(self) -> List[sp.MatrixBase]:
        """
        Generate the inverse SE(3) matrix for the gap matrix.

        The inverse SE(3) matrix is constructed by reversing the directionality of the degrees of freedom
        (i.e., multiplying by -1).

        Returns
        -------
        List[sympy.MatrixBase]
            A list of symbolic matrices corresponding to the inverse SE(3) basis elements.
        """
        logging.info(f"[Type: 'G', ID: {self.ID}] Generating inverse SE3 matrix.")
        return get_SE3_matrices_from_indices(self.SE3_basis_indices, multiplier=-1)


@beartype
class TransformationMatrix:
    """
    Class representing transformation matrices within the nominal geometry of a part.

    Transformation matrices define the transformation between two coordinate systems.

    Parameters
    ----------
    index : int, optional
        An index used to differentiate transformation matrices. The same index can be used multiple times (default is -1).
    initial : np.ndarray, optional
        The initial 4x4 matrix representing the initial coordinate system (default is the identity matrix).
    final : np.ndarray, optional
        The final 4x4 matrix representing the final coordinate system (default is the identity matrix).
    name : str, optional
        Name of the transformation matrix for identification (default is an empty string).
    matrix : Optional[Union[np.ndarray, sympy.MatrixBase]], optional
        The explicit transformation matrix. If provided, it overrides `initial` and `final` matrices (default is None).

    Raises
    ------
    AssertionError
        If the provided matrix is not a valid affine transformation matrix.
    """
    def __init__(
        self,
        index: int = -1,
        initial: np.ndarray = np.identity(4),
        final: np.ndarray = np.identity(4),
        name: str = "",
        matrix: Optional[Union[np.ndarray, sp.MatrixBase]] = None,
    ):
        """
        Initialize a TransformationMatrix instance.

        Parameters
        ----------
        index : int, optional
            An index used to differentiate transformation matrices. The same index can be used multiple times (default is -1).
        initial : np.ndarray, optional
            The initial 4x4 matrix representing the initial coordinate system (default is the identity matrix).
        final : np.ndarray, optional
            The final 4x4 matrix representing the final coordinate system (default is the identity matrix).
        name : str, optional
            Name of the transformation matrix for identification (default is an empty string).
        matrix : Optional[Union[np.ndarray, sympy.MatrixBase]], optional
            The explicit transformation matrix. If provided, it overrides `initial` and `final` matrices (default is None).

        Raises
        ------
        AssertionError
            If the provided matrix is not a valid affine transformation matrix.
        """
        logging.info(f"[Type: 'T', ID: {index}] Initializing TransformationMatrix.")
        assert isinstance(index, int), "Wrong index, check class init parameters"
        self.ID = index
        self.TYPE = "T"
        self.name = name
        self._initial = initial
        self._final = final
        if matrix is not None:
            self._matrix = np.array(matrix, dtype="float64")
            assert is_affine_transformation_matrix(
                self._matrix
            ), f"The matrix {self._matrix} is not a transformation matrix."
        else:
            self._matrix = None
        self._check_affine_validity_and_inverse()

    def __repr__(self):
        """
        Generate a string representation of the TransformationMatrix.

        Returns
        -------
        str
            String representation of the TransformationMatrix instance, including its index and matrix values.
        """
        return f"TransformationMatrix with index {self.ID} and values:\n{self.get_matrix().__repr__()} and name : {self.name}"

    def _check_affine_validity_and_inverse(self):
        """
        Validate the affine transformation matrix and compute its inverse.

        Ensures the provided or calculated transformation matrix is a valid affine transformation,
        and verifies the computed inverse matrix for consistency.

        Raises
        ------
        AssertionError
            If the matrix or its inverse is not a valid affine transformation matrix.
        """
        if self._matrix is not None and self._matrix.shape == (4, 4):
            matrix = self._matrix.copy()
            matrix_inverse = np.linalg.inv(matrix)
        else:
            matrix = self.get_change_of_basis_matrix(self._initial, self._final, True)
            matrix_inverse = self.get_change_of_basis_matrix(self._final, self._initial, True)
            matrix_inverse_2 = np.linalg.inv(matrix)
            assert np.allclose(
                matrix_inverse, matrix_inverse_2
            ), f"Transformation matrix {self.name} wrong."

        assert is_affine_transformation_matrix(
            matrix
        ) and is_affine_transformation_matrix(
            matrix_inverse
        ), f"Transformation matrix {self.name} wrong."

    def get_change_of_basis_matrix(
        self, initial: np.ndarray, final: np.ndarray, as_array: bool = False
    ) -> Union[sp.MatrixBase, np.ndarray]:
        """
        Calculate the change of basis matrix between two coordinate systems.

        The change of basis matrix is computed as the relative transformation from the
        initial coordinate system to the final coordinate system.

        Parameters
        ----------
        initial : np.ndarray
            The initial 4x4 matrix representing the initial coordinate system.
        final : np.ndarray
            The final 4x4 matrix representing the final coordinate system.
        as_array : bool, optional
            If True, returns the matrix as a NumPy array; otherwise, returns a sympy.Matrix (default is False).

        Returns
        -------
        Union[sympy.MatrixBase, np.ndarray]
            The change of basis matrix between the initial and final coordinate systems.
        """
        logging.debug(f"[Type: 'T', ID: {self.ID}] Calculating change of basis matrix.")

        T = np.identity(4)
        translation = final[:3, 3] - initial[:3, 3]
        translation = initial[:3, :3].T @ translation
        T[:3, 3] = translation

        initial_rotational_submatrix = initial[:3, :3]
        final_rotational_submatrix = final[:3, :3]
        relative_rotation = initial_rotational_submatrix.T @ final_rotational_submatrix
        T[:3, :3] = relative_rotation

        logging.debug(f"[Type: 'T', ID: {self.ID}] Change of basis matrix calculated.")
        if as_array:
            return T
        return sp.Matrix(T.tolist())

    def get_inverse(self):
        """
        Generate the inverse of the current TransformationMatrix.

        The inverse transformation matrix swaps the initial and final coordinate systems,
        or directly computes the inverse of the provided matrix.

        Returns
        -------
        TransformationMatrix
            A new instance of `TransformationMatrix` representing the inverse transformation.
        """
        if self._matrix is not None and self._matrix.shape == (4, 4):
            return TransformationMatrix(
                index=self.ID,
                matrix=self.get_matrix_inverse(),
                name=inverse_mstring(self.name),
            )
        else:
            return TransformationMatrix(
                index=self.ID,
                initial=self._final,
                final=self._initial,
                name=inverse_mstring(self.name),
            )

    def get_matrix(self) -> sp.MatrixBase:
        """
        Retrieve the transformation matrix.

        If the transformation matrix was explicitly provided during initialization, it is returned.
        Otherwise, it is computed as the change of basis matrix between the initial and final coordinate systems.

        Returns
        -------
        sympy.MatrixBase
            The transformation matrix as a symbolic sympy.Matrix.
        """
        logging.info(f"[Type: 'T', ID: {self.ID}] Generating transformation matrix.")
        if self._matrix is not None and self._matrix.shape == (4, 4):
            return sp.Matrix(self._matrix)
        else:
            return self.get_change_of_basis_matrix(self._initial, self._final)

    def get_matrix_inverse(self) -> sp.MatrixBase:
        """
        Retrieve the inverse of the transformation matrix.

        If the transformation matrix was explicitly provided during initialization, its inverse is computed.
        Otherwise, the inverse is calculated as the change of basis matrix between the final and initial coordinate systems.

        Returns
        -------
        sympy.MatrixBase
            The inverse transformation matrix as a symbolic sympy.Matrix.
        """
        logging.info(f"[Type: 'T', ID: {self.ID}] Generating inverse transformation matrix.")
        if self._matrix is not None and self._matrix.shape == (4, 4):
            return sp.Matrix(np.linalg.inv(self._matrix))
        else:
            return self.get_change_of_basis_matrix(self._final, self._initial)


class I4:
    """
    Class representing a 4x4 identity matrix.

    The identity matrix is commonly used in transformations as a neutral element, where no
    translation or rotation is applied.
    """
    def __init__(self):
        """
        Initialize a 4x4 identity matrix.

        Sets the matrix type to "I4".
        """
        self.TYPE = "I4"

    def get_matrix(self) -> sp.MatrixBase:
        """
        Generate a 4x4 identity matrix.

        Returns
        -------
        sympy.MatrixBase
            A symbolic representation of a 4x4 identity matrix.
        """
        logging.info("[Type: I4] Generating 4x4 identity matrix.")
        return sp.Matrix(np.identity(4).tolist())

    def get_matrix_inverse(self) -> sp.MatrixBase:
        """
        Generate the inverse of the 4x4 identity matrix.

        Since the inverse of an identity matrix is itself, this method returns the same matrix.

        Returns
        -------
        sympy.MatrixBase
            A symbolic representation of the 4x4 identity matrix.
        """
        logging.info("[Type: I4] Generating inverse 4x4 identity matrix.")
        return self.get_matrix()


class J4:
    """
    Class representing a 4x4 rotation matrix for a 180° rotation around the z-axis.

    This matrix is often used to represent a transformation involving a half-turn
    rotation in a Cartesian coordinate system.
    """
    def __init__(self):
        """
        Initialize a 4x4 rotation matrix representing a 180° rotation around the z-axis.

        Sets the matrix type to "J4".
        """
        self.TYPE = "J4"

    def get_matrix(self) -> sp.MatrixBase:
        """
        Generate a 4x4 rotation matrix representing a 180° rotation around the z-axis.

        The matrix is constructed as follows:
            - The x-axis and y-axis are negated.
            - The z-axis and translation remain unchanged.

        Returns
        -------
        sympy.MatrixBase
            A symbolic representation of a 4x4 rotation matrix for a 180° z-axis rotation.
        """
        logging.info("[Type: J4] Generating 4x4 rotation matrix for 180° z-axis rotation.")
        J4 = np.identity(4)
        J4[0, 0] *= -1
        J4[1, 1] *= -1
        return sp.Matrix(J4)

    def get_matrix_inverse(self) -> sp.MatrixBase:
        """
        Generate the inverse of the 4x4 rotation matrix.

        Since the inverse of a 180° rotation around the z-axis is the same rotation,
        this method returns the same matrix.

        Returns
        -------
        sympy.MatrixBase
            A symbolic representation of the 4x4 rotation matrix for a 180° z-axis rotation.
        """
        logging.info("[Type: J4] Generating inverse 4x4 rotation matrix.")
        return self.get_matrix()
