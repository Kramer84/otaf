# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = ["CompatibilityLoopHandling"]

import re
import logging

import numpy as np
import sympy as sp

from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Any, Set
from beartype.meta import TYPE_CHECKING

if TYPE_CHECKING:
    from otaf import DeviationMatrix, GapMatrix, TransformationMatrix, I4, J4, AssemblyDataProcessor
import otaf


@beartype
class CompatibilityLoopHandling:
    """
    Class for processing system data and constructing compatibility constraint matrices.

    This class processes system data to generate and manage compatibility loops, constructing the
    matrices required to define compatibility constraints in geometrical product specifications.
    It facilitates tolerance analysis by providing compatibility equations and handling matrix
    expansions for compatibility loops.

    Attributes
    ----------
    ADP : otaf.AssemblyDataProcessor
        Object containing processed system data, including expanded compatibility loops.
    _deviation_matrix_map : dict
        Cache mapping deviation matrix identifiers to their corresponding matrices.
    _gap_matrix_map : dict
        Cache mapping gap matrix identifiers to their corresponding matrices.
    _transformation_matrix_map : dict
        Cache mapping transformation matrix identifiers to their corresponding matrices.
    compatibility_loops_matrices : dict
        Dictionary mapping compatibility loop IDs to lists of corresponding matrices.
    compatibility_loops_FO_matrices : dict
        Dictionary mapping compatibility loop IDs to first-order expanded matrices.
    compatibility_expressions : list of sympy.Expr
        List of symbolic expressions representing compatibility constraints.

    Methods
    -------
    __init__(assemblyDataProcessor)
        Initialize the class with processed system data.
    get_gap_matrix_by_id(ID)
        Retrieve a gap matrix by its identifier.
    get_deviation_matrix_by_id(ID)
        Retrieve a deviation matrix by its identifier.
    get_compatibility_expression_from_FO_matrices()
        Generate symbolic expressions for compatibility constraints.
    generate_loop_id_to_matrix_list_dict()
        Create a dictionary mapping loop IDs to matrix lists for compatibility loops.
    generate_FO_loop_matrices()
        Generate first-order matrix expansions for compatibility loops.
    apply_FO_matrix_expansion_to_matrix_loop_list(compatibility_loop_matrix_list)
        Apply first-order matrix expansion to a list of compatibility loop matrices.
    generate_matrices_from_expanded_loop(expanded_loop_str)
        Generate matrices for a compatibility loop from its expanded string representation.
    generate_transformation_matrix(el_info)
        Generate a transformation matrix from loop element information.
    generate_deviation_matrix(el_info)
        Generate a deviation matrix based on loop element information.
    generate_gap_matrix(el_info)
        Generate a gap matrix based on loop element information.
    calculate_nominal_gap_transform(el_info, ID, nullify_x=True, nullify_y=True, nullify_z=True)
        Calculate the nominal transformation matrix between two surfaces in a gap, with options
        to nullify specific translation components.
    """
    def __init__(self, assemblyDataProcessor: "AssemblyDataProcessor"):
        """
        Initialize the `CompatibilityLoopHandling` object with system data.

        Parameters
        ----------
        assemblyDataProcessor : otaf.AssemblyDataProcessor
            Object containing processed system data, including expanded compatibility loops.

        Notes
        -----
        - Caches for deviation, gap, and transformation matrices are initialized.
        - Compatibility loop matrices and their first-order expansions are precomputed.
        """
        logging.info("Initializing CompatibilityLoopHandling object.")
        self.ADP = assemblyDataProcessor

        self._deviation_matrix_map = {}
        self._gap_matrix_map = {}
        self._transformation_matrix_map = {}

        self.compatibility_loops_matrices = self.generate_loop_id_to_matrix_list_dict()
        self.compatibility_loops_FO_matrices = self.generate_FO_loop_matrices()
        self.compatibility_expressions = self.get_compatibility_expression_from_FO_matrices()

    def get_gap_matrix_by_id(
        self, ID: Union[str, int]
    ) -> List[Union["GapMatrix", "TransformationMatrix"]]:

        """
        Retrieve a gap matrix by its identifier.

        Parameters
        ----------
        ID : str or int
            The identifier of the gap matrix.

        Returns
        -------
        list of otaf.GapMatrix or otaf.TransformationMatrix
            The corresponding gap matrix or transformation matrix.

        Raises
        ------
        KeyError
            If the specified ID does not exist in the gap matrix map.
        """
        id_map = {str(mat[0].ID): key for key, mat in self._gap_matrix_map.items()}
        return self._gap_matrix_map[id_map[str(ID)]]

    def get_deviation_matrix_by_id(self, ID: Union[str, int]) -> List[Union["DeviationMatrix"]]:
        """
        Retrieve a deviation matrix by its identifier.

        Parameters
        ----------
        ID : str or int
            The identifier of the deviation matrix.

        Returns
        -------
        list of otaf.DeviationMatrix
            The corresponding deviation matrix.

        Raises
        ------
        KeyError
            If the specified ID does not exist in the deviation matrix map.
        """
        id_map = {str(mat[0].ID): key for key, mat in self._deviation_matrix_map.items()}
        return self._deviation_matrix_map[id_map[str(ID)]]

    def get_compatibility_expression_from_FO_matrices(self) -> List[sp.Expr]:
        """
        Generate symbolic expressions representing compatibility constraints.

        Returns
        -------
        list of sympy.Expr
            A list of symbolic expressions for the compatibility constraints derived from
            the first-order expanded matrices.
        """
        logging.info("Generating list of equations representing the compatibility loop")
        return [
            x
            for expr_f in list(self.compatibility_loops_FO_matrices.values())
            for x in otaf.common.extract_expressions_with_variables(expr_f)
        ]

    def generate_loop_id_to_matrix_list_dict(
        self,
    ) -> Dict[str, List[Union["TransformationMatrix", "DeviationMatrix", "GapMatrix"]]]:
        """
        Create a dictionary mapping compatibility loop IDs to lists of matrices.

        Returns
        -------
        dict
            A dictionary where keys are compatibility loop IDs, and values are lists
            of matrices representing the corresponding compatibility loops.
        """
        logging.info("Generating loop ID to matrix list dictionary")
        return {
            key: self.generate_matrices_from_expanded_loop(item)
            for key, item in self.ADP.compatibility_loops_expanded.items()
        }

    def generate_FO_loop_matrices(self) -> Dict[str, sp.MatrixBase]:
        """
        Generate first-order expanded matrices for compatibility loops.

        Returns
        -------
        dict
            A dictionary mapping compatibility loop IDs to their first-order expanded matrices.
        """
        logging.info("Generating loop ID to first order expanded matrices dictionary")
        return {
            key: self.apply_FO_matrix_expansion_to_matrix_loop_list(item)
            for key, item in self.compatibility_loops_matrices.items()
        }

    def apply_FO_matrix_expansion_to_matrix_loop_list(
        self,
        compatibility_loop_matrix_list: List[
            Union["TransformationMatrix", "DeviationMatrix", "GapMatrix"]
        ],
    ) -> sp.MatrixBase:
        """
        Apply first-order matrix expansion to a list of compatibility loop matrices.

        Parameters
        ----------
        compatibility_loop_matrix_list : list of otaf.TransformationMatrix, otaf.DeviationMatrix, otaf.GapMatrix
            List of matrices representing a compatibility loop.

        Returns
        -------
        sympy.MatrixBase
            The first-order expansion of the compatibility loop matrices.
        """
        return otaf.FirstOrderMatrixExpansion(
            compatibility_loop_matrix_list
        ).compute_first_order_expansion()

    def generate_matrices_from_expanded_loop(
        self, expanded_loop_str: str
    ) -> List[Union["TransformationMatrix", "DeviationMatrix", "GapMatrix"]]:
        """
        Generate matrices for a compatibility loop from its expanded string representation.

        Parameters
        ----------
        expanded_loop_str : str
            The string representation of an expanded compatibility loop.

        Returns
        -------
        list of otaf.TransformationMatrix, otaf.DeviationMatrix, otaf.GapMatrix
            List of matrices corresponding to the expanded loop.

        Raises
        ------
        ValueError
            If the matrix type in the string is unknown.
        """
        matrix_list = []
        matrix_strings = list(map(str.strip, expanded_loop_str.split("->")))
        for mstring in matrix_strings:
            elinfo = otaf.common.parse_matrix_string(mstring)
            if elinfo["type"] == "T":
                t_mat = self.generate_transformation_matrix(elinfo)
                matrix_list.extend(t_mat)
            elif elinfo["type"] == "D":
                d_mat = self.generate_deviation_matrix(elinfo)
                matrix_list.extend(d_mat)
            elif elinfo["type"] == "G":
                g_mat = self.generate_gap_matrix(elinfo)
                matrix_list.extend(g_mat)
            else:
                exc = ValueError(f"Unknown matrix type {elinfo['type']}")
                logging.error(exc_info=exc)
                raise exc
        return matrix_list

    def generate_transformation_matrix(self, el_info: dict) -> List["TransformationMatrix"]:
        """
        Generate a transformation matrix from loop element information.

        Parameters
        ----------
        el_info : dict
            Dictionary containing loop element information, including part IDs, surfaces,
            points, and transformation details.

        Returns
        -------
        list of otaf.TransformationMatrix
            A list containing the generated transformation matrix.

        Notes
        -----
        - If the matrix already exists in the transformation matrix map, it is retrieved
          from the cache.
        - The inverse of the generated transformation matrix is also cached.
        """
        if el_info["mstring"] not in self._transformation_matrix_map:
            # The transformation matrix has to be generated.
            point_start = self.ADP["PARTS"][el_info["part"]][el_info["surface1"]]["POINTS"][
                el_info["point1"]
            ]
            frame_start = self.ADP["PARTS"][el_info["part"]][el_info["surface1"]]["FRAME"]
            point_end = self.ADP["PARTS"][el_info["part"]][el_info["surface2"]]["POINTS"][
                el_info["point2"]
            ]
            frame_end = self.ADP["PARTS"][el_info["part"]][el_info["surface2"]]["FRAME"]
            t_mat = otaf.TransformationMatrix(
                initial=otaf.geometry.tfrt(frame_start, point_start),
                final=otaf.geometry.tfrt(frame_end, point_end),
                name=el_info["mstring"],
            )
            self._transformation_matrix_map[el_info["mstring"]] = [t_mat]
            self._transformation_matrix_map[otaf.common.inverse_mstring(el_info["mstring"])] = [
                t_mat.get_inverse()
            ]
            return [t_mat]
        else:
            return self._transformation_matrix_map[el_info["mstring"]]

    def generate_deviation_matrix(self, el_info: dict) -> List["DeviationMatrix"]:
        """
        Generate a deviation matrix based on loop element information.

        This method creates a deviation matrix for a specified element in the compatibility loop,
        taking into account surface types, constraints, and global constraints.

        Parameters
        ----------
        el_info : dict
            Dictionary containing loop element information, including the matrix string (`mstring`),
            surface type, and relevant constraints.

        Returns
        -------
        list of otaf.DeviationMatrix
            A list containing the generated deviation matrix or its inverse, depending on the
            specified configuration.

        Notes
        -----
        - If the deviation matrix for the given element already exists, it retrieves it from the cache.
        - Surface constraints and global constraints influence the degrees of freedom
          (translations and rotations) of the generated deviation matrix.

        Raises
        ------
        KeyError
            If required surface or global constraints are missing in the input data.
        """
        if el_info["mstring"] not in self._deviation_matrix_map:
            # Deviation matrix has to be generated.
            d_mat_id_max = (
                max([v[0].ID for v in self._deviation_matrix_map.values()])
                if bool(self._deviation_matrix_map)
                else -1
            )
            index = d_mat_id_max + 1
            is_inverse = el_info["inverse"]
            constraints = self.ADP["PARTS"][el_info["part"]][el_info["surface"]].get(
                "CONSTRAINTS_D", ["NONE"]
            )
            surf_type = self.ADP["PARTS"][el_info["part"]][el_info["surface"]]["TYPE"]
            surf_type = "-".join(map(str.lower, [surf_type, *constraints]))
            translations = "".join(
                sorted(
                    set(otaf.constants.SURF_TYPE_TO_DEVIATION_DOF[surf_type]["translations"])
                    - set(
                        otaf.constants.GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF[
                            self.ADP["GLOBAL_CONSTRAINTS"]
                        ]["translations_2remove"]
                    )
                )
            )
            rotations = "".join(
                sorted(
                    set(otaf.constants.SURF_TYPE_TO_DEVIATION_DOF[surf_type]["rotations"])
                    - set(
                        otaf.constants.GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF[
                            self.ADP["GLOBAL_CONSTRAINTS"]
                        ]["rotations_2remove"]
                    )
                )
            )
            d_mat = otaf.DeviationMatrix(
                index=index,
                translations=translations,
                rotations=rotations,
                inverse=is_inverse,
                name=el_info["mstring"],
            )
            self._deviation_matrix_map[el_info["mstring"]] = [d_mat]
            return [d_mat]
        else:
            if self._deviation_matrix_map[el_info["mstring"]][0].inverse == el_info["inverse"]:
                return self._deviation_matrix_map[el_info["mstring"]]
            else:
                return [self._deviation_matrix_map[el_info["mstring"]][0].get_inverse()]

    def generate_gap_matrix(
        self, el_info: dict
    ) -> List[Union["GapMatrix", "TransformationMatrix"]]:
        """
        Generate a gap matrix based on loop element information.

        This method creates a gap matrix for the specified pair of surfaces in the compatibility loop,
        taking into account surface types, contact constraints, manual constraints, and global constraints.

        Parameters
        ----------
        el_info : dict
            Dictionary containing loop element information, including the matrix string (`mstring`),
            part and surface identifiers, and relevant constraints.

        Returns
        -------
        list of Union[otaf.GapMatrix, otaf.TransformationMatrix]
            A list containing the generated gap matrix, its nominal transform, or their inverses,
            depending on the configuration.

        Notes
        -----
        - If the gap matrix for the given element already exists, it retrieves it from the cache.
        - Contact constraints and manual constraints (translations and rotations) influence the
          blocked degrees of freedom in the generated gap matrix.
        - A nominal gap transform is calculated and returned alongside the gap matrix.

        Raises
        ------
        KeyError
            If required surface or contact constraints are missing in the input data.
        """
        if el_info["mstring"] not in self._gap_matrix_map:
            g_mat_id_max = (
                max([v[0].ID for v in self._gap_matrix_map.values()])
                if bool(self._gap_matrix_map)
                else -1
            )
            index = g_mat_id_max + 1
            constraints1 = self.ADP["PARTS"][el_info["part1"]][el_info["surface1"]].get(
                "CONSTRAINTS_G", ["FLOATING"]
            )
            constraints2 = self.ADP["PARTS"][el_info["part2"]][el_info["surface2"]].get(
                "CONSTRAINTS_G", ["FLOATING"]
            )
            rotations_blocked_manual_1 = self.ADP["PARTS"][el_info["part1"]][
                el_info["surface1"]
            ].get("BLOCK_ROTATIONS_G", "")
            rotations_blocked_manual_2 = self.ADP["PARTS"][el_info["part2"]][
                el_info["surface2"]
            ].get("BLOCK_ROTATIONS_G", "")
            translations_blocked_manual_1 = self.ADP["PARTS"][el_info["part1"]][
                el_info["surface1"]
            ].get("BLOCK_TRANSLATIONS_G", "")
            translations_blocked_manual_2 = self.ADP["PARTS"][el_info["part2"]][
                el_info["surface2"]
            ].get("BLOCK_TRANSLATIONS_G", "")
            surf_type1 = self.ADP["PARTS"][el_info["part1"]][el_info["surface1"]]["TYPE"]
            surf_type2 = self.ADP["PARTS"][el_info["part2"]][el_info["surface2"]]["TYPE"]

            gap_transform_kwargs = otaf.constants.GAP_TYPE_TO_NULLIFIED_NOMINAL_COMPONENTS[
                "-".join(map(str.lower, [surf_type1, surf_type2]))
            ]
            nominal_gap_transform = self.calculate_nominal_gap_transform(
                el_info, index, **gap_transform_kwargs
            )

            is_inverse = el_info["inverse"]
            gap_constraints = list(set((*constraints1, *constraints2)))
            contact_type = "-".join(map(str.lower, [surf_type1, surf_type2, *gap_constraints]))
            manual_constraints_translation = "".join(
                set(translations_blocked_manual_1 + translations_blocked_manual_2)
            )
            manual_constraints_rotation = "".join(
                set(rotations_blocked_manual_1 + rotations_blocked_manual_2)
            )
            translations_blocked = "".join(
                sorted(
                    set(
                        otaf.constants.CONTACT_TYPE_TO_GAP_DOF[contact_type]["translations_blocked"]
                        + otaf.constants.GLOBAL_CONSTRAINTS_TO_GAP_DOF[
                            self.ADP["GLOBAL_CONSTRAINTS"]
                        ]["translations_blocked"]
                        + manual_constraints_translation
                    )
                )
            )
            rotations_blocked = "".join(
                sorted(
                    set(
                        otaf.constants.CONTACT_TYPE_TO_GAP_DOF[contact_type]["rotations_blocked"]
                        + otaf.constants.GLOBAL_CONSTRAINTS_TO_GAP_DOF[
                            self.ADP["GLOBAL_CONSTRAINTS"]
                        ]["rotations_blocked"]
                        + manual_constraints_rotation
                    )
                )
            )
            g_mat = otaf.GapMatrix(
                index=index,
                translations_blocked=translations_blocked,
                rotations_blocked=rotations_blocked,
                inverse=is_inverse,
                name=el_info["mstring"],
            )
            self._gap_matrix_map[el_info["mstring"]] = [g_mat, nominal_gap_transform]
            self._gap_matrix_map[otaf.common.inverse_mstring(el_info["mstring"])] = [
                nominal_gap_transform.get_inverse(),
                g_mat.get_inverse(),
            ]
            if not is_inverse:
                return [g_mat, nominal_gap_transform]
            else:
                return [nominal_gap_transform.get_inverse(), g_mat.get_inverse()]
        else:
            return self._gap_matrix_map[el_info["mstring"]]

    def calculate_nominal_gap_transform(
        self,
        el_info: dict,
        ID: int,
        nullify_x: bool = True,
        nullify_y: bool = True,
        nullify_z: bool = True,
    ) -> "TransformationMatrix":
        """
        Calculate the nominal transformation matrix between two surfaces in a gap.

        This method computes the nominal transformation matrix between two surfaces in the
        compatibility loop, with options to nullify specific components of the translation vector.

        Parameters
        ----------
        el_info : dict
            Dictionary containing loop element information, including part and surface identifiers.
        ID : int
            Unique identifier for the transformation matrix.
        nullify_x : bool, optional
            Whether to nullify the x-component of the translation vector. Defaults to True.
        nullify_y : bool, optional
            Whether to nullify the y-component of the translation vector. Defaults to True.
        nullify_z : bool, optional
            Whether to nullify the z-component of the translation vector. Defaults to True.

        Returns
        -------
        otaf.TransformationMatrix
            The calculated nominal transformation matrix with specified components nullified.

        Notes
        -----
        - The method extracts point and frame information for the specified surfaces and uses it
          to compute the nominal transformation matrix.
        - Nullification of translation components is applied directly to the matrix.

        Raises
        ------
        KeyError
            If required surface or frame information is missing in the input data.
        """
        point_start = self.ADP["PARTS"][el_info["part1"]][el_info["surface1"]]["POINTS"][
            el_info["point1"]
        ]
        frame_start = self.ADP["PARTS"][el_info["part1"]][el_info["surface1"]]["FRAME"]

        point_end = self.ADP["PARTS"][el_info["part2"]][el_info["surface2"]]["POINTS"][
            el_info["point2"]
        ]
        frame_end = self.ADP["PARTS"][el_info["part2"]][el_info["surface2"]]["FRAME"]
        # Calculate the nominal transformation matrix between the two surfaces
        nominal_transform = otaf.TransformationMatrix(
            initial=otaf.geometry.tfrt(frame_start, point_start),
            final=otaf.geometry.tfrt(frame_end, point_end),
        )
        # Get the matrix and nullify the components of the translation vector
        nominal_matrix = np.array(nominal_transform.get_matrix(), dtype=float)
        # Nullify specific components of the translation vector based on the flags
        if nullify_x:
            nominal_matrix[0, -1] = 0.0
        if nullify_y:
            nominal_matrix[1, -1] = 0.0
        if nullify_z:
            nominal_matrix[2, -1] = 0.0
        return otaf.TransformationMatrix(index=ID, matrix=nominal_matrix)
