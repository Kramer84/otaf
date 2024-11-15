# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = ["CompatibilityLoopHandling"]

import re
import logging

import numpy as np
import sympy as sp

from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Any, Set

import otaf


@beartype
class CompatibilityLoopHandling:
    """Class for processing system data and constructing lists of matrix multiplications.

    This class reads a dictionary containing system data and generates lists of matrix
    multiplications. It internally stores references to each transform created and maps each
    transform to its index and direction.
    """

    def __init__(self, systemDataAugmented: otaf.SystemDataAugmented):
        """Initialize the MatrixListsFromSystemData instance with system data.

        Args:
            systemDataAugmented (Dict[str, Dict[str, Dict[str, Any]]]): A dictionary containing system data.

        Attributes:
            systemDataAugmented (Dict[str, Dict[str, Dict[str, Any]]]): The provided system data.
            _deviation_matrix_map (dict): A map for deviation matrices.
            _gap_matrix_map (dict): A map for gap matrices.
            _transformation_matrix_map (dict): A map for constant transformation matrices.
        """
        logging.info("Initializing CompatibilityLoopHandling object.")
        self.SDA = systemDataAugmented

        self._deviation_matrix_map = {}
        self._gap_matrix_map = {}
        self._transformation_matrix_map = {}

        self.compatibility_loops_matrices = self.generate_loop_id_to_matrix_list_dict()
        self.compatibility_loops_FO_matrices = self.generate_FO_loop_matrices()
        self.compatibility_expressions = self.get_compatibility_expression_from_FO_matrices()

    def get_gap_matrix_by_ID(
        self, ID: Union[str, int]
    ) -> List[Union[otaf.GapMatrix, otaf.TransformationMatrix]]:
        """Get a gap matrix by its ID."""
        id_map = {str(mat[0].ID): key for key, mat in self._gap_matrix_map.items()}
        return self._gap_matrix_map[id_map[str(ID)]]

    def get_def_matrix_by_ID(self, ID: Union[str, int]) -> List[Union[otaf.DeviationMatrix]]:
        """Get a deviation matrix by its ID."""
        id_map = {str(mat[0].ID): key for key, mat in self._deviation_matrix_map.items()}
        return self._deviation_matrix_map[id_map[str(ID)]]

    def get_compatibility_expression_from_FO_matrices(self) -> List[sp.Expr]:
        logging.info("Generating list of equations representing the compatibility loop")
        return [
            x
            for expr_f in list(self.compatibility_loops_FO_matrices.values())
            for x in otaf.common.get_relevant_expressions(expr_f)
        ]

    def generate_loop_id_to_matrix_list_dict(
        self,
    ) -> Dict[str, List[Union[otaf.TransformationMatrix, otaf.DeviationMatrix, otaf.GapMatrix]]]:
        logging.info("Generating loop ID to matrix list dictionary")
        return {
            key: self.generate_matrices_from_expanded_loop(item)
            for key, item in self.SDA.compatibility_loops_expanded.items()
        }

    def generate_FO_loop_matrices(self) -> Dict[str, sp.MatrixBase]:
        logging.info("Generating loop ID to first order expanded matrices dictionary")
        return {
            key: self.apply_FO_matrix_expansion_to_matrix_loop_list(item)
            for key, item in self.compatibility_loops_matrices.items()
        }

    def apply_FO_matrix_expansion_to_matrix_loop_list(
        self,
        compatibility_loop_matrix_list: List[
            Union[otaf.TransformationMatrix, otaf.DeviationMatrix, otaf.GapMatrix]
        ],
    ) -> sp.MatrixBase:
        return otaf.MatrixPolynomeTaylorExpansion(
            compatibility_loop_matrix_list
        ).construct_FO_matrix_expansion()

    def generate_matrices_from_expanded_loop(
        self, expanded_loop_str: str
    ) -> List[Union[otaf.TransformationMatrix, otaf.DeviationMatrix, otaf.GapMatrix]]:
        """Generate a list of matrices based on an expanded compatibility loop."""
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

    def generate_transformation_matrix(self, el_info: dict) -> List[otaf.TransformationMatrix]:
        if el_info["mstring"] not in self._transformation_matrix_map:
            # The transformation matrix has to be generated.
            point_start = self.SDA["PARTS"][el_info["part"]][el_info["surface1"]]["POINTS"][
                el_info["point1"]
            ]
            frame_start = self.SDA["PARTS"][el_info["part"]][el_info["surface1"]]["FRAME"]
            point_end = self.SDA["PARTS"][el_info["part"]][el_info["surface2"]]["POINTS"][
                el_info["point2"]
            ]
            frame_end = self.SDA["PARTS"][el_info["part"]][el_info["surface2"]]["FRAME"]
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

    def generate_deviation_matrix(self, el_info: dict) -> List[otaf.DeviationMatrix]:
        if el_info["mstring"] not in self._deviation_matrix_map:
            # Deviation matrix has to be generated.
            d_mat_id_max = (
                max([v[0].ID for v in self._deviation_matrix_map.values()])
                if bool(self._deviation_matrix_map)
                else -1
            )
            index = d_mat_id_max + 1
            is_inverse = el_info["inverse"]
            constraints = self.SDA["PARTS"][el_info["part"]][el_info["surface"]].get(
                "CONSTRAINTS_D", ["NONE"]
            )
            surf_type = self.SDA["PARTS"][el_info["part"]][el_info["surface"]]["TYPE"]
            surf_type = "-".join(map(str.lower, [surf_type, *constraints]))
            translations = "".join(
                sorted(
                    set(otaf.constants.SURF_TYPE_TO_DEVIATION_DOF[surf_type]["translations"])
                    - set(
                        otaf.constants.GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF[
                            self.SDA["GLOBAL_CONSTRAINTS"]
                        ]["translations_2remove"]
                    )
                )
            )
            rotations = "".join(
                sorted(
                    set(otaf.constants.SURF_TYPE_TO_DEVIATION_DOF[surf_type]["rotations"])
                    - set(
                        otaf.constants.GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF[
                            self.SDA["GLOBAL_CONSTRAINTS"]
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
    ) -> List[Union[otaf.GapMatrix, otaf.TransformationMatrix]]:
        if el_info["mstring"] not in self._gap_matrix_map:
            g_mat_id_max = (
                max([v[0].ID for v in self._gap_matrix_map.values()])
                if bool(self._gap_matrix_map)
                else -1
            )
            index = g_mat_id_max + 1
            constraints1 = self.SDA["PARTS"][el_info["part1"]][el_info["surface1"]].get(
                "CONSTRAINTS_G", ["FLOATING"]
            )
            constraints2 = self.SDA["PARTS"][el_info["part2"]][el_info["surface2"]].get(
                "CONSTRAINTS_G", ["FLOATING"]
            )
            rotations_blocked_manual_1 = self.SDA["PARTS"][el_info["part1"]][
                el_info["surface1"]
            ].get("BLOCK_ROTATIONS_G", "")
            rotations_blocked_manual_2 = self.SDA["PARTS"][el_info["part2"]][
                el_info["surface2"]
            ].get("BLOCK_ROTATIONS_G", "")
            translations_blocked_manual_1 = self.SDA["PARTS"][el_info["part1"]][
                el_info["surface1"]
            ].get("BLOCK_TRANSLATIONS_G", "")
            translations_blocked_manual_2 = self.SDA["PARTS"][el_info["part2"]][
                el_info["surface2"]
            ].get("BLOCK_TRANSLATIONS_G", "")
            surf_type1 = self.SDA["PARTS"][el_info["part1"]][el_info["surface1"]]["TYPE"]
            surf_type2 = self.SDA["PARTS"][el_info["part2"]][el_info["surface2"]]["TYPE"]

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
                            self.SDA["GLOBAL_CONSTRAINTS"]
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
                            self.SDA["GLOBAL_CONSTRAINTS"]
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
    ) -> otaf.TransformationMatrix:
        # Extracting the necessary information from el_info and system_data
        point_start = self.SDA["PARTS"][el_info["part1"]][el_info["surface1"]]["POINTS"][
            el_info["point1"]
        ]
        frame_start = self.SDA["PARTS"][el_info["part1"]][el_info["surface1"]]["FRAME"]

        point_end = self.SDA["PARTS"][el_info["part2"]][el_info["surface2"]]["POINTS"][
            el_info["point2"]
        ]
        frame_end = self.SDA["PARTS"][el_info["part2"]][el_info["surface2"]]["FRAME"]
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
