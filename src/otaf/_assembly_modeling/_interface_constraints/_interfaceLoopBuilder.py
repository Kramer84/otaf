from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"


import logging

import numpy as np
import sympy as sp

from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Set

from otaf.common import extract_expressions_with_variables, parse_matrix_string
from otaf.constants import G_MATRIX_MSTRING_PATTERN
from otaf.geometry import (
    LinearizationStrategy,
    circle_linearization_factor,
    generate_circle_points,
)

from ..assemblyDataProcessor import AssemblyDataProcessor
from ..firstOrderMatrixExpansion import FirstOrderMatrixExpansion

logger = logging.getLogger(__name__)


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
    linearization : LinearizationStrategy
        Polygon replacing the circular clearance zone of cylinder pairs.
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
        silence_prints=True,
        contact_gap_names: Optional[Set[str]] = None,
        linearization: LinearizationStrategy = LinearizationStrategy.CIRCUMSCRIBED,
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
        silence_prints : bool, optional
            Kept for compatibility, no effect: the debug messages go to the
            ``otaf._assembly_modeling._interface_constraints._interfaceLoopBuilder``
            logger at DEBUG level.
        contact_gap_names : set of str, optional
            Gap matrix names of the point pairs where contact can occur (see
            ``SurfaceInteractionManager.contactGapNames``). Contact constraints are
            written only for these pairs. If None, every facing pair is a contact.
        linearization : LinearizationStrategy, optional
            Polygon replacing the circular clearance zone of cylinder pairs
            (default ``CIRCUMSCRIBED``, the polygon tangent to the circle).
        """
        self.ADP = assemblyDataProcessor
        self.CLH = compatibility_loop_handling
        self.filtered_gap_matrices = filtered_gap_matrices
        self.CIRCLE_RESOLUTION = circle_resolution
        self.interface_expressions = None
        self.sp = bool(silence_prints)
        self.contact_gap_names = contact_gap_names
        self.linearization = LinearizationStrategy(linearization)
        self._diameter_feature_index: Dict[tuple, int] = {}

    def _contact_partners(
        self, used_name: str, used_dat: List[str], unused_names: List[str], unused_dats: List[List[str]]
    ) -> List[List[str]]:
        """Point pairs of the same two surfaces as `used_name` at which to write contact constraints.

        The reference pair itself is included when it is a contact location (e.g. a
        surface origin lying at an end of the overlap).
        """
        partners = [
            dat
            for name, dat in zip(unused_names, unused_dats)
            if dat[3] == used_dat[3] and dat[4] == used_dat[4] and self._is_contact(name)
        ]
        if self.contact_gap_names is not None and used_name in self.contact_gap_names:
            partners.append(used_dat)
        return partners

    def _is_contact(self, gap_name: str) -> bool:
        return self.contact_gap_names is None or gap_name in self.contact_gap_names

    def _debug(self, *args):
        logger.debug(" ".join(map(str, args)))

    def matrix_list_taylor_expansion(self, matrix_loop_list: List[Any]) -> sp.MatrixBase:
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
        return FirstOrderMatrixExpansion(matrix_loop_list).compute_first_order_expansion()

    ###################################################################################################

    def populate_interface_expressions(self) -> None:
        """Populate the interface expressions attribute with calculated interface equations."""
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
            for idSurf in surfData:
                # If len(surfData[idSurf]["USED"]) > 1 that could also mean that the cylinder is traversing multiple parts
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
                    # Order-preserving deduplication (a set would make the row order
                    # depend on the hash seed).
                    interface_equations_plane = list(
                        dict.fromkeys(
                            x
                            for expr in interface_equation_matrix_list
                            for x in extract_expressions_with_variables(expr)
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
        GMMP = G_MATRIX_MSTRING_PATTERN
        self._debug(f"Processing part {idPart}, surface {idSurf} for plane-to-plane interactions.")

        usedGMat = sorted(self.filtered_gap_matrices[idPart][idSurf]["USED"])
        usedGMatDat = [list(GMMP.fullmatch(ug).groups()) for ug in usedGMat]
        self._debug("usedGMatDat", usedGMatDat)
        self._debug(f"Found {len(usedGMat)} used gap matrices.")

        unusedGMat = sorted(self.filtered_gap_matrices[idPart][idSurf]["UNUSED"])
        unusedGMatDat = [list(GMMP.fullmatch(uug).groups()) for uug in unusedGMat]
        self._debug("unusedGMatDat", unusedGMatDat)
        self._debug(f"Found {len(unusedGMat)} unused gap matrices.")

        facingPlaneInteractionMatrixList = []
        for i, datGUsed in enumerate(usedGMatDat):
            partners = self._contact_partners(usedGMat[i], datGUsed, unusedGMat, unusedGMatDat)
            for partner in partners:
                facingPlaneInteractionMatrixList.append(
                    self._generate_surface_interaction_loop(
                        partner, datGUsed, self.CLH._gap_matrix_map[usedGMat[i]]
                    )
                )
            self._debug(f"Generated {len(partners)} interaction matrix loops for {usedGMat[i]}.")

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
            Sequence of matrices describing the interaction: T1, the used gap
            matrix with its nominal transform, and T2.

        Notes
        -----
        The loop goes from the contact point on the first surface to its origin
        (T1), across the gap exactly as the compatibility loops represent it
        (gap matrix and nominal transform JCor, in their stored order), then from
        the origin of the second surface to its contact point (T2). The translation
        of the product is the position of the second contact point in the frame of
        the first, so the gap rotation acts on the same pivot as in the
        compatibility loops.

        A trailing ``JCor`` inverse must not be appended: it re-applies the
        in-plane offset between the two surface origins after the gap rotation,
        which moves the pivot and makes the result depend on the direction in
        which the gap is used (lever arms off by that offset).
        """
        T1_name = (
            f"TP{unusedMatch[0]}{unusedMatch[1]}{unusedMatch[2]}{unusedMatch[1]}{usedMatch[2]}"
        )
        T2_name = (
            f"TP{unusedMatch[3]}{unusedMatch[4]}{usedMatch[5]}{unusedMatch[4]}{unusedMatch[5]}"
        )

        T1_info = parse_matrix_string(T1_name)
        T2_info = parse_matrix_string(T2_name)

        T1 = self.CLH.generate_transformation_matrix(T1_info)[0]
        T2 = self.CLH.generate_transformation_matrix(T2_info)[0]

        return [T1, *usedGapMatrix, T2]

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
        list of sympy.Expr
            Linearized non-interpenetration expressions, one per polygon side
            (``CIRCLE_RESOLUTION`` of them) for each contact section of each
            cylinder pair; the contact holds when each expression is non-negative.

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
        The method needs a fully built model (a ``CompatibilityLoopHandling`` and the
        surface interactions), so its example is not run as a doctest. It is
        exercised on the pin/plate assembly in ``tests/test_cylinder_interfaces.py``:

        >>> equations = builder.get_interface_equations_facing_cylinders("P1", "a")  # doctest: +SKIP
        """
        self._debug(f"Processing part {idPart}, surface {idSurf} for cylinder-to-cylinder interactions.")

        GMMP = G_MATRIX_MSTRING_PATTERN

        usedGMat = sorted(self.filtered_gap_matrices[idPart][idSurf]["USED"])
        usedGMatDat = [list(GMMP.fullmatch(ug).groups()) for ug in usedGMat]
        self._debug("usedGMatDat", usedGMatDat)
        self._debug(f"Found {len(usedGMat)} used gap matrices.")

        unusedGMat = sorted(self.filtered_gap_matrices[idPart][idSurf]["UNUSED"])
        unusedGMatDat = [list(GMMP.fullmatch(uug).groups()) for uug in unusedGMat]
        self._debug("unusedGMatDat", unusedGMatDat)
        self._debug(f"Found {len(unusedGMat)} unused gap matrices.")

        facingCylinderInteractionEquationList = []
        for i, datGUsed in enumerate(usedGMatDat):
            partners = self._contact_partners(usedGMat[i], datGUsed, unusedGMat, unusedGMatDat)
            for partner in partners:
                cylinder_edge_center_interaction_loop = self._generate_surface_interaction_loop(
                    partner, datGUsed, self.CLH._gap_matrix_map[usedGMat[i]]
                )
                cylinder_edge_center_interaction_matrix = self.matrix_list_taylor_expansion(
                    cylinder_edge_center_interaction_loop
                )
                facingCylinderInteractionEquationList.extend(
                    self._cylinder_interaction_approximation(
                        idPart, idSurf, datGUsed, cylinder_edge_center_interaction_matrix
                    )
                )
            self._debug(f"Generated interaction equations at {len(partners)} contact pairs for {usedGMat[i]}.")
        self._debug(
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
        equations ensure that this maximum distance does not exceed the radial
        clearance (`radial_clearance`) times the factor of the linearization
        strategy (`otaf.geometry.circle_linearization_factor`).
        """
        clearance = self.radial_clearance(datGUsed[0], datGUsed[1], datGUsed[3], datGUsed[4])
        rf = circle_linearization_factor(self.linearization, self.CIRCLE_RESOLUTION)
        cosSinPairs = generate_circle_points(1, self.CIRCLE_RESOLUTION)
        y_val = cylinder_edge_center_interaction_matrix[1, 3]
        z_val = cylinder_edge_center_interaction_matrix[2, 3]
        return [rf * clearance - (cos_val * y_val + sin_val * z_val) for cos_val, sin_val in cosSinPairs]

    def radial_clearance(self, part1: str, surf1: str, part2: str, surf2: str) -> Any:
        """Radial clearance of a cylinder pair, with diameter deviations if modelled.

        The clearance is ``R_hole - R_pin`` (the hole is the larger cylinder), plus
        ``(d_hole - d_pin) / 2`` for each cylinder whose ``DIAMETER_DEVIATION`` is
        true, where ``d`` is the deviation of its diameter from nominal (see
        `diameter_symbol`).
        """
        surfaces = [(part1, surf1), (part2, surf2)]
        radii = [float(self.ADP["PARTS"][p][s]["RADIUS"]) for p, s in surfaces]
        hole, pin = (0, 1) if radii[0] > radii[1] else (1, 0)
        clearance = sp.Float(radii[hole] - radii[pin])
        for index, sign in ((hole, 1), (pin, -1)):
            part, surf = surfaces[index]
            if self.ADP["PARTS"][part][surf].get("DIAMETER_DEVIATION", False):
                clearance += sign * self.diameter_symbol(part, surf) / 2
        return clearance

    def diameter_symbol(self, part: str, surf: str) -> sp.Symbol:
        """Deviation variable ``d_d_<k>`` of the diameter of cylinder `surf` of `part`.

        ``k`` is the feature index of the surface: that of its deviation matrix when
        it has one (so the diameter is grouped with the position and orientation
        deviations of the same feature), otherwise a new index after all the
        deviation matrices (e.g. a pin, which the compatibility loops cross at a
        single point and which therefore has no deviation matrix).
        """
        key = (part, surf)
        if key not in self._diameter_feature_index:
            deviation_matrices = self.CLH._deviation_matrix_map
            if f"D{part}{surf}" in deviation_matrices:
                index = deviation_matrices[f"D{part}{surf}"][0].ID
            else:
                used = [m[0].ID for m in deviation_matrices.values()]
                used += list(self._diameter_feature_index.values())
                index = max(used, default=-1) + 1
            self._diameter_feature_index[key] = index
        return sp.Symbol(f"d_d_{self._diameter_feature_index[key]}")
