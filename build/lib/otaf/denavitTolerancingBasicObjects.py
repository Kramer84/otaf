# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = ["DeviationMatrix", "GapMatrix", "TransformationMatrix", "I4", "J4"]

import re
import logging
import numpy as np
import sympy as sp
from beartype import beartype
from beartype.typing import List, Union, Optional
import otaf


@beartype
class DeviationMatrix:
    """Represents a deviation matrix used in structural analysis.

    A deviation matrix defines deviations in translation and rotation for parts and assemblies.

    Arguments
    ---------
    index : int, optional
        An index used to differentiate variable names. The same index can be used multiple times.
    translations : str, optional
        Degrees of freedom for translation ('x', 'y', 'z'). Example: "xyz" for all three degrees.
    rotations : str, optional
        Degrees of freedom for rotation ('x', 'y', 'z').
    inverse : bool, optional
        Flag to indicate whether the matrix represents deviations from nominal to real (True) or vice versa (False).
        This flag affects the signs of the matrices and is necessary if a matrix is used in two directions.
    """

    def __init__(
        self,
        index: int = -1,
        translations: str = "xyz",
        rotations: str = "xyz",
        inverse: bool = False,
        name: str = "",
    ):
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
        logging.debug(f"Generating string representation for DeviationMatrix with index {self.ID}.")
        return f"DeviationMatrix(index={self.ID}, translations='{self._translations}', rotations='{self._rotations}', inverse={self.inverse}, name={self.name})"

    def _initialize_basis_variables(self):
        logging.info("Initializing basis variables.")
        self.SE3_basis_indices.clear()
        self.variables.clear()
        for i in range(1, 7):  # also sorts
            axis = otaf.constants.BASIS_DICT[str(i)]["AXIS"]
            if i <= 3:
                if axis in self._translations:
                    self.SE3_basis_indices.append(i)
                    var_name = otaf.constants.BASIS_DICT[str(i)]["VARIABLE_D"]
                    self.variables.append(
                        sp.Symbol(f"{var_name}_{self.ID}" if self.ID >= 0 else var_name)
                    )
            else:
                if axis in self._rotations:
                    self.SE3_basis_indices.append(i)
                    var_name = otaf.constants.BASIS_DICT[str(i)]["VARIABLE_D"]
                    self.variables.append(
                        sp.Symbol(f"{var_name}_{self.ID}" if self.ID >= 0 else var_name)
                    )

    def get_inverse(self):
        logging.info(f"[Type: 'D', ID: {self.ID}] Generating inverse DeviationMatrix.")
        return DeviationMatrix(
            index=self.ID,
            translations=self._translations,
            rotations=self._rotations,
            inverse=(not self.inverse),
            name=self.name,
        )

    def get_variables(self) -> List[sp.Symbol]:
        logging.debug(f"[Type: 'D', ID: {self.ID}] Retrieving variables.")
        return self.variables

    def get_matrix(self) -> List[sp.MatrixBase]:
        logging.info(f"[Type: 'D', ID: {self.ID}] Generating SE3 matrix.")
        return otaf.common.get_SE3_matrices_from_indices(self.SE3_basis_indices)

    def get_matrix_inverse(self) -> List[sp.MatrixBase]:
        logging.info(f"[Type: 'D', ID: {self.ID}] Generating inverse SE3 matrix.")
        return otaf.common.get_SE3_matrices_from_indices(self.SE3_basis_indices, multiplier=-1)


@beartype
class GapMatrix:
    """
    Represents a gap matrix used in structural analysis.

    A gap matrix defines gaps or clearances in translation and rotation for parts and assemblies.

    Arguments
    ---------
    index : int, optional
        An index used to differentiate variable names. The same index can be used multiple times.
    translations_blocked : str, optional
        Degrees of freedom for translation that are blocked ('x', 'y', 'z').
    rotations_blocked : str, optional
        Degrees of freedom for rotation that are blocked ('x', 'y', 'z').
    inverse : bool, optional
        Flag to indicate whether the matrix represents gaps or clearances (True) or blocked degrees of freedom (False).
    """

    def __init__(
        self,
        index: int = -1,
        translations_blocked: str = "",
        rotations_blocked: str = "",
        inverse: bool = False,
        name: str = "",
    ):
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
        self.SE3_basis_indices.clear()
        self.variables.clear()
        for i in range(1, 7):
            axis = otaf.constants.BASIS_DICT[str(i)]["AXIS"]
            if i <= 3:
                if axis not in self._translations_blocked:
                    self.SE3_basis_indices.append(i)
                    var_name = otaf.constants.BASIS_DICT[str(i)]["VARIABLE_G"]
                    self.variables.append(
                        sp.Symbol(f"{var_name}_{self.ID}" if self.ID >= 0 else var_name)
                    )
            else:
                if axis not in self._rotations_blocked:
                    self.SE3_basis_indices.append(i)
                    var_name = otaf.constants.BASIS_DICT[str(i)]["VARIABLE_G"]
                    self.variables.append(
                        sp.Symbol(f"{var_name}_{self.ID}" if self.ID >= 0 else var_name)
                    )
        logging.debug(f"[Type: 'G', ID: {self.ID}] Added {len(self.variables)} basis variables.")

    def __repr__(self):
        return f"GapMatrix(index={self.ID}, translations_blocked='{self._translations_blocked}', rotations_blocked='{self._rotations_blocked}', inverse={self.inverse}, name={self.name})"

    def get_inverse(self):
        logging.info(f"[Type: 'G', ID: {self.ID}] Generating inverse DeviationMatrix.")
        return GapMatrix(
            index=self.ID,
            translations_blocked=self._translations_blocked,
            rotations_blocked=self._rotations_blocked,
            inverse=(not self.inverse),
            name=otaf.common.inverse_mstring(self.name),
        )

    def get_variables(self) -> List[sp.Symbol]:
        logging.debug(f"[Type: 'G', ID: {self.ID}] Retrieving variables.")
        return self.variables

    def get_matrix(self) -> List[sp.MatrixBase]:
        logging.info(f"[Type: 'G', ID: {self.ID}] Generating SE3 matrix.")
        return otaf.common.get_SE3_matrices_from_indices(self.SE3_basis_indices)

    def get_matrix_inverse(self) -> List[sp.MatrixBase]:
        logging.info(f"[Type: 'G', ID: {self.ID}] Generating inverse SE3 matrix.")
        return otaf.common.get_SE3_matrices_from_indices(self.SE3_basis_indices, multiplier=-1)

    def get_j4_rotation_matrix(self):
        logging.info(f"[Type: 'G', ID: {self.ID}] Generating J4 rotation matrix.")
        return ROTATION_MATRIX_180_Z_AXIS()


@beartype
class TransformationMatrix:
    """Class representing transformation matrices within the nominal geometry of a part.

    Transformation matrices define the transformation between two coordinate systems.
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
        Initialize a transformation matrix.

        Arguments
        ---------
        index : int, optional
            An index used to differentiate transformation matrices. The same index can be used multiple times.
        initial : np.ndarray, optional
            The initial 4x4 matrix representing the initial coordinate system.
        final : np.ndarray, optional
            The final 4x4 matrix representing the final coordinate system.
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
            assert otaf.geometry.is_affine_transformation_matrix(
                self._matrix
            ), f"The matrix {self._matrix} is not a transformation matrix."
        else:
            self._matrix = None
        self._check_affine_validity_and_inverse()

    def __repr__(self):
        return f"TransformationMatrix with index {self.ID} and values:\n{self.get_matrix().__repr__()} and name : {self.name}"

    def _check_affine_validity_and_inverse(self):
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

        assert otaf.geometry.is_affine_transformation_matrix(
            matrix
        ) and otaf.geometry.is_affine_transformation_matrix(
            matrix_inverse
        ), f"Transformation matrix {self.name} wrong."

    def get_change_of_basis_matrix(
        self, initial: np.ndarray, final: np.ndarray, as_array: bool = False
    ) -> Union[sp.MatrixBase, np.ndarray]:
        """Calculate the change of basis matrix between initial and final coordinate systems.

        Args:
            initial (np.ndarray): The initial 4x4 matrix representing the initial coordinate system.
            final (np.ndarray): The final 4x4 matrix representing the final coordinate system.

        Returns:
            Matrix: The change of basis matrix.
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
        if self._matrix is not None and self._matrix.shape == (4, 4):
            return TransformationMatrix(
                index=self.ID,
                matrix=self.get_matrix_inverse(),
                name=otaf.common.inverse_mstring(self.name),
            )
        else:
            return TransformationMatrix(
                index=self.ID,
                initial=self._final,
                final=self._initial,
                name=otaf.common.inverse_mstring(self.name),
            )

    def get_matrix(self) -> sp.MatrixBase:
        logging.info(f"[Type: 'T', ID: {self.ID}] Generating transformation matrix.")
        if self._matrix is not None and self._matrix.shape == (4, 4):
            return sp.Matrix(self._matrix)
        else:
            return self.get_change_of_basis_matrix(self._initial, self._final)

    def get_matrix_inverse(self) -> sp.MatrixBase:
        logging.info(f"[Type: 'T', ID: {self.ID}] Generating inverse transformation matrix.")
        if self._matrix is not None and self._matrix.shape == (4, 4):
            return sp.Matrix(np.linalg.inv(self._matrix))
        else:
            return self.get_change_of_basis_matrix(self._final, self._initial)


class I4:
    def __init__(self):
        """Initialize a 4x4 identity matrix."""
        self.TYPE = "I4"

    def get_matrix(self) -> sp.MatrixBase:
        logging.info("[Type: I4] Generating 4x4 identity matrix.")
        """Return a 4x4 identity matrix."""
        return sp.Matrix(np.identity(4).tolist())

    def get_matrix_inverse(self) -> sp.MatrixBase:
        logging.info("[Type: I4] Generating inverse 4x4 identity matrix.")
        """Return the inverse of the 4x4 identity matrix (also an identity matrix)."""
        return self.get_matrix()


class J4:
    def __init__(self):
        """Initialize a 4x4 rotation matrix representing a 180° rotation around the z-axis."""
        self.TYPE = "J4"

    def get_matrix(self) -> sp.MatrixBase:
        """Return a 4x4 rotation matrix representing a 180° rotation around the z-axis."""
        logging.info("[Type: J4] Generating 4x4 rotation matrix for 180° z-axis rotation.")
        J4 = np.identity(4)
        J4[0, 0] *= -1
        J4[1, 1] *= -1
        return sp.Matrix(J4)

    def get_matrix_inverse(self) -> sp.MatrixBase:
        """Return the inverse of the 4x4 rotation matrix (same rotation in reverse direction)."""
        logging.info("[Type: J4] Generating inverse 4x4 rotation matrix.")
        return self.get_matrix()
