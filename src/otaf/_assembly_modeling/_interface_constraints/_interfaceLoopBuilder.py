from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"


import re
import logging

from copy import copy

import numpy as np
import sympy as sp

from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Any, Set

from ..assemblyDataProcessor import AssemblyDataProcessor
import otaf


@beartype
class InterfaceLoopBuilder:
    """
    Builds interface equations and loops for interacting parts in a mechanical system.

    Attributes
    ----------
    x_component_mask_matrix : sp.Matrix
        Mask matrix used for processing x-component of interface equations.
    SDA : Dict[str, Any]
        Augmented system data containing part and surface information.
    CLH : Any
        Compatibility loop handling object used for gap matrix mapping and transformations.
    filtered_gap_matrices : Dict[str, Dict[str, Any]]
        Filtered gap matrices for each part and surface.
    CIRCLE_RESOLUTION : int
        Resolution for approximating circular interactions.
    interface_expressions : Optional[List[Any]]
        Generated interface expressions after processing.
    """

    x_component_mask_matrix = sp.Matrix(
        np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    )

    def __init__(
        self,
        assemblyDataProcessor: AssemblyDataProcessor,
        compatibility_loop_handling: Any,
        filtered_gap_matrices: Dict[str, Dict[str, Any]],
        circle_resolution: int = 8,
        silence_prints=True
    ) -> None:
        """
        Initialize the InterfaceLoopBuilder.

        Parameters
        ----------
        assemblyDataProcessor : Dict[str, Any]
            Augmented system data containing part and surface information.
        compatibility_loop_handling : Any
            Compatibility loop handling object for gap matrix mapping and transformations.
        filtered_gap_matrices : Dict[str, Dict[str, Any]]
            Filtered gap matrices for parts and surfaces.
        circle_resolution : int, optional
            Resolution for approximating circular interactions (default is 8).
        """
        self.ADP = assemblyDataProcessor
        self.CLH = compatibility_loop_handling
        self.filtered_gap_matrices = filtered_gap_matrices
        self.CIRCLE_RESOLUTION = circle_resolution
        self.interface_expressions = None
        self.sp = bool(silence_prints)

    def _print(self, *args):
        if self.sp:
            pass
        else:
            print(*args)

    def matrix_list_taylor_expansion(self, matrix_loop_list: List[Any]) -> Any:
        """
        Compute the first-order Taylor expansion for a list of matrices.

        Parameters
        ----------
        matrix_loop_list : List[Any]
            List of matrices to compute the Taylor expansion.

        Returns
        -------
        Any
            Expanded matrices after first-order Taylor expansion.
        """
        return otaf.FirstOrderMatrixExpansion(matrix_loop_list).compute_first_order_expansion()

    ###################################################################################################

    def populate_interface_expressions(self) -> None:
        """
        Populate the interface expressions attribute with calculated interface equations.
        """
        self.interface_expressions = self.get_interface_equations_from_facing_parts()

    def get_interface_equations_from_facing_parts(self) -> List[Any]:
        """
        Generate interface loops from facing parts based on unique gap matrices.

        Returns
        -------
        List[Any]
            List of interface equations derived from part interactions.
        """
        interface_equations = []
        for idPart, surfData in self.filtered_gap_matrices.items():
            for idSurf, pointData in surfData.items():
                # If len(pointData["USED"]) > 1 that could also mean that the cylinder is traversing multiple parts
                # So a new implementation would be needed then.
                surfType = self.ADP["PARTS"][idPart][idSurf]["TYPE"]

                if surfType == "plane":
                    # plane interaction handling
                    interface_equation_matrix_list = self.get_interface_equations_facing_planes(
                        idPart, idSurf
                    )
                    # Postprocessing
                    interface_equation_matrix_list = [
                        self.matrix_list_taylor_expansion(X) for X in interface_equation_matrix_list
                    ]
                    interface_equation_matrix_list = [
                        expa_f.multiply_elementwise(self.x_component_mask_matrix)
                        for expa_f in interface_equation_matrix_list
                    ]
                    interface_equations_plane = list(
                        set(
                            [
                                x
                                for expr in interface_equation_matrix_list
                                for x in otaf.common.extract_expressions_with_variables(expr)
                            ]
                        )
                    )
                    interface_equations.extend(interface_equations_plane)

                elif surfType == "cylinder":
                    # cylinder interaction handling
                    interface_equations.extend(
                        self.get_interface_equations_facing_cylinders(idPart, idSurf)
                    )

        return interface_equations

    def get_interface_equations_facing_planes(self, idPart: str, idSurf: str) -> List[Any]:
        """
        Generate interface equations for plane-to-plane interactions.

        This method identifies the gap matrices that define the geometric relationships
        between interacting planes and generates a list of interaction matrix loops.
        Each loop represents the transformation and gap matrices describing how two
        planes interact within the system.

        Parameters
        ----------
        idPart : str
            Identifier for the part containing the plane.
        idSurf : str
            Identifier for the surface of the part.

        Returns
        -------
        List[Any]
            List of interaction matrix loops for plane-to-plane interactions. Each element
            represents a sequence of matrices that describe the geometric relationship
            between the two interacting planes.

        Notes
        -----
        This method processes both used and unused gap matrices, identifying matches
        to ensure that all potential plane interactions are accounted for. Matching is
        performed based on part and surface identifiers.
        """
        BPSP, GMMP = otaf.constants.BASE_PART_SURF_PATTERN, otaf.constants.G_MATRIX_MSTRING_PATTERN
        self._print(f"Processing part {idPart}, surface {idSurf} for plane-to-plane interactions.")

        usedGMat = list(self.filtered_gap_matrices[idPart][idSurf]["USED"])
        usedGMatDat = [list(GMMP.fullmatch(ug).groups()) for ug in usedGMat]
        usedInteractingParts = [X[3] for X in usedGMatDat]
        usedInteractingSurfs = [X[4] for X in usedGMatDat]
        self._print("usedGMatDat", usedGMatDat)
        self._print(f"Found {len(usedGMat)} used gap matrices.")

        unusedGMat = list(self.filtered_gap_matrices[idPart][idSurf]["UNUSED"])
        unusedGMatDat = [list(GMMP.fullmatch(uug).groups()) for uug in unusedGMat]
        unusedInteractingParts = [X[3] for X in unusedGMatDat]
        unusedInteractingSurfs = [X[4] for X in unusedGMatDat]
        self._print("unusedGMatDat", unusedGMatDat)
        self._print(f"Found {len(unusedGMat)} unused gap matrices.")

        facingPlaneInteractionMatrixList = []

        for i, datGUsed in enumerate(usedGMatDat):
            idPu, idSu = datGUsed[3], datGUsed[4]
            unused_match = [
                bool(idPu == datGUnused[3] and idSu == datGUnused[4])
                for datGUnused in unusedGMatDat
            ]
            interactionMatrixLoopList = []
            for j, datGUnused in enumerate(unusedGMatDat):
                if unused_match[j]:
                    interactionMatrixLoopList.append(
                        self._generate_surface_interaction_loop(
                            unusedGMatDat[j], usedGMatDat[i], self.CLH._gap_matrix_map[usedGMat[i]]
                        )
                    )
            facingPlaneInteractionMatrixList.extend(interactionMatrixLoopList)
            if interactionMatrixLoopList:
                self._print(
                    f"Generated {len(interactionMatrixLoopList)} interaction matrix loops for current matching."
                )
            facingPlaneInteractionMatrixList.extend(interactionMatrixLoopList)

        return facingPlaneInteractionMatrixList

    def _generate_surface_interaction_loop(
        self, unusedMatch: List[str], usedMatch: List[str], usedGapMatrix: Any
    ) -> List[Any]:
        """
        Generate a surface interaction loop between two gap matrices.

        This method constructs a loop that expresses the spatial relationship
        between an unused gap matrix and a reference (used) gap matrix. It involves
        calculating transformation matrices to align the unused gap with the spatial
        configuration defined by the reference gap.

        Parameters
        ----------
        unusedMatch : List[str]
            Data extracted from the regex match of an unused gap matrix. Contains
            details about the part and surface interactions.
        usedMatch : List[str]
            Data extracted from the regex match of a used gap matrix. Serves as the
            interaction reference.
        usedGapMatrix : Any
            Gap matrix object representing the spatial relationship between the two
            surfaces.

        Returns
        -------
        List[Any]
            Sequence of matrices describing the interaction. This includes
            transformation matrices, the used gap matrix, and its inverse transformation.

        Notes
        -----
        The method constructs two transformation matrices (T1 and T2) to align
        the interacting surfaces, followed by the application of the used gap matrix
        and its inverse transformation (JCor inverse). The generated sequence of
        matrices forms a complete interaction loop.
        """
        if usedGapMatrix[0].TYPE == "G":
            GMat, JCor = usedGapMatrix
        else:
            GMat, JCor = reversed(usedGapMatrix)

        T1_name = (
            f"TP{unusedMatch[0]}{unusedMatch[1]}{unusedMatch[2]}{unusedMatch[1]}{usedMatch[2]}"
        )
        T2_name = (
            f"TP{unusedMatch[3]}{unusedMatch[4]}{usedMatch[5]}{unusedMatch[4]}{unusedMatch[5]}"
        )

        T1_info = otaf.common.parse_matrix_string(T1_name)
        T2_info = otaf.common.parse_matrix_string(T2_name)

        T1 = self.CLH.generate_transformation_matrix(T1_info)[0]
        T2 = self.CLH.generate_transformation_matrix(T2_info)[0]

        return [T1, *usedGapMatrix, T2, JCor.get_inverse()]

    def get_interface_equations_facing_cylinders(self, idPart, idSurf):
        """
        Generate interface equations for cylinder-to-cylinder interactions.

        This method identifies and processes all interface equations for a specified cylinder
        (determined by `idPart` and `idSurf`) and its interacting surfaces. It matches used
        and unused gap matrices, constructs interaction matrix loops, and computes the resulting
        interaction equations for cylinder-to-cylinder relationships.

        Parameters
        ----------
        idPart : str
            Identifier for the part containing the cylinder of interest.
        idSurf : str
            Identifier for the surface of the part of interest.

        Returns
        -------
        List[<appropriate type>]
            A list of interaction matrix loops representing the geometric relationships and
            interactions between the cylinder and its interacting surfaces. Each loop consists
            of transformation and gap matrices expanded into symbolic equations.

        Notes
        -----
        - The method distinguishes between used and unused gap matrices associated with the cylinder
          and matches them to generate interaction loops.
        - For each matched pair of used and unused gap matrices, it constructs and expands interaction
          matrix loops into symbolic equations.
        - The output can be used to analyze geometric constraints and interactions for tolerance analysis.

        Raises
        ------
        ValueError
            If the input identifiers do not match any known part or surface in the system data.

        Examples
        --------
        To generate interaction equations for a cylinder:
        >>> equations = get_interface_equations_facing_cylinders("Part1", "SurfaceA")
        >>> for eq in equations:
        ...     self._print(eq)
        """
        self._print(f"Processing part {idPart}, surface {idSurf} for cylinder-to-cylinder interactions.")

        BPSP, GMMP = otaf.constants.BASE_PART_SURF_PATTERN, otaf.constants.G_MATRIX_MSTRING_PATTERN

        usedGMat = list(self.filtered_gap_matrices[idPart][idSurf]["USED"])
        usedGMatDat = [list(GMMP.fullmatch(ug).groups()) for ug in usedGMat]
        usedInteractingParts = [X[3] for X in usedGMatDat]
        usedInteractingSurfs = [X[4] for X in usedGMatDat]
        self._print("usedGMatDat", usedGMatDat)
        self._print(f"Found {len(usedGMat)} used gap matrices.")

        unusedGMat = list(self.filtered_gap_matrices[idPart][idSurf]["UNUSED"])
        unusedGMatDat = [list(GMMP.fullmatch(uug).groups()) for uug in unusedGMat]
        unusedInteractingParts = [X[3] for X in unusedGMatDat]
        unusedInteractingSurfs = [X[4] for X in unusedGMatDat]
        self._print("unusedGMatDat", unusedGMatDat)
        self._print(f"Found {len(unusedGMat)} unused gap matrices.")

        facingCylinderInteractionEquationList = []

        for i, datGUsed in enumerate(usedGMatDat):
            idPu, idSu = datGUsed[3], datGUsed[4]
            unused_match = [
                bool(idPu == datGUnused[3] and idSu == datGUnused[4])
                for datGUnused in unusedGMatDat
            ]
            interactionEquationLoopList = []
            for j, datGUnused in enumerate(unusedGMatDat):
                if unused_match[j]:
                    self._print(
                        f"Matching used and unused gap matrices: {usedGMat[i]} with {unusedGMat[j]}"
                    )
                    cylinder_edge_center_interaction_loop = self._generate_surface_interaction_loop(
                        unusedGMatDat[j], usedGMatDat[i], self.CLH._gap_matrix_map[usedGMat[i]]
                    )
                    cylinder_edge_center_interaction_matrix = self.matrix_list_taylor_expansion(
                        cylinder_edge_center_interaction_loop
                    )
                    cylinder_edge_interaction_equations = self._cylinder_interaction_approximation(
                        idPart, idSurf, usedGMatDat[i], cylinder_edge_center_interaction_matrix
                    )
                    interactionEquationLoopList.extend(cylinder_edge_interaction_equations)

            facingCylinderInteractionEquationList.extend(interactionEquationLoopList)
            self._print(
                f"Generated {len(interactionEquationLoopList)} interaction equations for current matching."
            )
        self._print(
            f"Total interaction equations generated: {len(facingCylinderInteractionEquationList)}"
        )
        return facingCylinderInteractionEquationList

    def _cylinder_interaction_approximation(
        self, idPart: str, idSurf: str, datGUsed: List[str], cylinder_edge_center_interaction_matrix: Any
    ) -> List[Any]:
        """
        Approximate cylinder interactions using trigonometric constraints.

        This method determines whether the center of one cylinder lies within
        the interaction zone of another cylinder. It computes interaction equations
        based on the geometric relationship between the two cylinders, leveraging
        trigonometric properties to enforce constraints.

        Parameters
        ----------
        idPart : str
            Identifier for the part containing the cylinder.
        idSurf : str
            Identifier for the surface of the part.
        datGUsed : List[str]
            Data associated with the used gap matrix, providing details of the
            cylinder interaction.
        cylinder_edge_center_interaction_matrix : Any
            Transformation matrix representing the distance between the centers of
            the two interacting cylinders.

        Returns
        -------
        List[Any]
            List of interaction equations approximating the geometric constraints
            between the two cylinders.

        Notes
        -----
        The method calculates cos/sin pairs for angles between 0 and 2π, representing
        points on the circular cross-section of the cylinders. These pairs are used
        to compute the maximum interaction distance in the y-z plane. The computed
        equations ensure that this maximum distance does not exceed the difference
        in radii between the cylinders.
        """
        radius1 = self.ADP["PARTS"][datGUsed[0]][datGUsed[1]]["RADIUS"]
        radius2 = self.ADP["PARTS"][datGUsed[3]][datGUsed[4]]["RADIUS"]
        effective_radius = radius1 - radius2 if (radius1 > radius2) else radius2 - radius1
        cosSinPairs = otaf.geometry.generate_circle_points(
            1, self.CIRCLE_RESOLUTION
        )  # Generate N,2 array of coordinates
        cylinder_interaction_equations = []
        for i in range(cosSinPairs.shape[0]):
            cos_val = cosSinPairs[i, 0]
            sin_val = cosSinPairs[i, 1]
            y_val = cylinder_edge_center_interaction_matrix[1, 3]
            z_val = cylinder_edge_center_interaction_matrix[2, 3]
            equation = effective_radius - (cos_val * y_val + sin_val * z_val)
            cylinder_interaction_equations.append(equation)
        return cylinder_interaction_equations
