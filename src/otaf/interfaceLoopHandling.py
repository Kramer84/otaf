# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = ["InterfaceLoopHandling"]

import re
import logging

from copy import copy

import numpy as np
import sympy as sp

from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Any, Set

import otaf


@beartype
class InterfaceLoopHandling:
    def __init__(self, system_data_augmented, compatibility_loop_handling, circle_resolution=8):
        self.SDA = system_data_augmented
        self.CLH = compatibility_loop_handling
        self.surfaceInteractionManager = SurfaceInteractionManager(self.SDA)
        self.surfaceInteractionManager.get_facing_point_dictionary()
        self.all_gap_matrix_names = self.generate_all_gap_matrix_names()
        self.compatibility_gap_matrix_names = self.extract_unique_gap_matrices_from_expanded_loops()
        self.filtered_gap_matrices = self.filter_gap_matrices(
            self.compatibility_gap_matrix_names, self.all_gap_matrix_names
        )

        self.interfaceLoopBuilder = InterfaceLoopBuilder(
            self.SDA, self.CLH, self.filtered_gap_matrices, circle_resolution=circle_resolution
        )
        self.interfaceLoopBuilder.populate_interface_expressions()

    @property
    def facing_point_dictionary(self):
        return self.surfaceInteractionManager.facingPointDict

    def get_interface_loop_expressions(self):
        return self.interfaceLoopBuilder.interface_expressions

    def extract_unique_gap_matrices_from_expanded_loops(self) -> Set[str]:
        """Extract unique gap matrices from expanded compatibility loops and return them as a set."""
        gap_matrices = set()
        for value in self.SDA.compatibility_loops_expanded.values():
            tokens = re.split(r"\s+|->", value)
            for token in tokens:
                if token.startswith("G"):
                    gap_matrices.add(token)

        # Sometimes we have the gap in two directions,but ony want one, soooo:
        inverse_gap_matrices = set(map(otaf.common.inverse_mstring, gap_matrices))
        intersection = gap_matrices.intersection(inverse_gap_matrices)
        intersection_list = list(intersection)
        intersection_list_index_map = {inter: i for i, inter in enumerate(intersection_list)}
        inverse_intersection_list = list(map(otaf.common.inverse_mstring, intersection_list))
        inverse_intersection_list_indexes = [
            intersection_list_index_map[inter] for inter in inverse_intersection_list
        ]
        gap_matrices_to_remove = set(
            [
                inverse_intersection_list[num]
                for index, num in enumerate(inverse_intersection_list_indexes)
                if num > index
            ]
        )
        gap_matrices -= gap_matrices_to_remove
        return gap_matrices

    def generate_all_gap_matrix_names(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Generate and return a dictionary of gap matrix names based on facing points.

        This method iterates through the facing points in the system, creating and organizing
        gap matrix names. Each gap matrix name represents a potential spatial relationship
        between different parts and surfaces in the system. The names are stored in a nested
        dictionary structure keyed by part and surface IDs.

        Additionally, it ensures that for every gap matrix, its inverse exists in the set for
        the corresponding surface on the other part. If an inverse gap matrix is missing, it is
        added to maintain consistency in the representation of spatial relationships.
        """
        POINT_PATTERN = re.compile(r"^(\d+)([a-z]+)([A-Z]\d+)$")
        gap_matrix_dict = otaf.tree()

        # Step 1: Initialize the gap matrix dictionary
        for part_id, part_data in self.SDA["PARTS"].items():
            gap_matrix_dict[part_id] = {surf_id: set() for surf_id in part_data.keys()}

        # Step 2: Populate the gap matrices
        for part_id, surfaces in self.facing_point_dictionary.items():
            for surface_id, points in surfaces.items():
                for point_id, facing_point in points.items():
                    match = POINT_PATTERN.fullmatch(facing_point)
                    if match:
                        (facing_part_id, facing_surface_id, facing_point_id) = match.groups()
                    else:
                        continue

                    gap_matrix_name = f"GP{part_id}{surface_id}{point_id}P{facing_part_id}{facing_surface_id}{facing_point_id}"
                    gap_matrix_dict[part_id][surface_id].add(gap_matrix_name)

        # Step 3: Repopulate missing gap matrices by inverting existing ones
        for part_id, surfaces in gap_matrix_dict.items():
            for surface_id, gap_matrices in surfaces.items():
                for gap_matrix in gap_matrices:
                    parsed = otaf.common.parse_matrix_string(gap_matrix)
                    inverse_gap_matrix = otaf.common.inverse_mstring(gap_matrix)
                    inv_part_id, inv_surface_id = parsed["part2"], parsed["surface2"]
                    if inverse_gap_matrix not in gap_matrix_dict[inv_part_id][inv_surface_id]:
                        gap_matrix_dict[inv_part_id][inv_surface_id].add(inverse_gap_matrix)

        return gap_matrix_dict

    def filter_gap_matrices(
        self, existing_gap_matrices: Set[str], all_gap_matrices: Dict[str, Dict[str, Set[str]]]
    ) -> Dict[str, Dict[str, Dict[str, Set[str]]]]:
        """
        Filter and organize gap matrices into used and unused categories.

        This method categorizes gap matrices into 'used' and 'unused' based on their presence
        in the existing gap matrices set. It iterates through all provided gap matrices, comparing
        them with existing ones, and then organizes them into a nested dictionary indicating
        their usage status. This only categorizes the matrices when there is at least a common
        elemnt

        Args:
            existing_gap_matrices (Set[str]): A set of existing gap matrix names.
            all_gap_matrices (Dict[str, Dict[str, Set[str]]]): A dictionary of all gap matrices,
                structured by parts and surfaces.

        Returns:
            Dict[str, Dict[str, Dict[str, Set[str]]]]: A nested dictionary categorizing gap matrices
            into 'used' and 'unused' for each part and surface.
        """
        filtered_dict = otaf.tree()
        for part, surfaces in all_gap_matrices.items():
            for surface, matrix_set in surfaces.items():
                common_elements = matrix_set.intersection(existing_gap_matrices)
                if common_elements:
                    remaining_elements = matrix_set - common_elements
                    if remaining_elements:
                        filtered_dict[part][surface]["UNUSED"] = remaining_elements
                        filtered_dict[part][surface]["USED"] = common_elements
        return filtered_dict


@beartype
class SurfaceInteractionManager:
    def __init__(self, system_data_augmented):
        self.SDA = system_data_augmented
        self.facingPointDict = otaf.tree()

    def get_facing_point_dictionary(self) -> None:
        """
        Populate 'facingPointDict' with relationships between facing points on various part surfaces.
        """
        self.facingPointDict.clear()
        for part_id, part in self.SDA["PARTS"].items():
            for surface_id, surface_data in part.items():
                self.process_surface_interactions(part_id, surface_id, surface_data)

    def process_surface_interactions(self, part_id, surface_id, surface_data):
        """
        Process interactions for a given surface.

        Args:
            part_id (str): ID of the current part.
            surface_id (str): ID of the current surface.
            surface_data (dict): Data of the current surface.
        """
        for interaction in surface_data["INTERACTIONS"]:
            (part_id_interact, surf_id_interact) = otaf.constants.BASE_PART_SURF_PATTERN.fullmatch(
                interaction
            ).groups()
            surf_data_interact = self.SDA["PARTS"][part_id_interact][surf_id_interact]
            type_current = surface_data["TYPE"]
            type_interact = surf_data_interact["TYPE"]

            if type_current == "plane" and type_interact == "plane":
                self.process_plane_plane_interaction(
                    part_id,
                    surface_id,
                    surface_data,
                    part_id_interact,
                    surf_id_interact,
                    surf_data_interact,
                )
            if type_current == "cylinder" and type_interact == "cylinder":
                self.process_cylinder_cylinder_interaction(
                    part_id,
                    surface_id,
                    surface_data,
                    part_id_interact,
                    surf_id_interact,
                    surf_data_interact,
                )

    def process_plane_plane_interaction(
        self,
        part_id_current,
        surf_id_current,
        surf_data_current,
        part_id_interact,
        surf_id_interact,
        surf_data_interact,
        point_search_radius: Union[float, int] = 0.5,
    ):
        """
        Process plane-plane interaction.

        Args:
            part_id_current (str): ID of the current part.
            surf_id_current (str): ID of the current surface.
            surf_data_current (dict): Data of the current surface.
            part_id_interact (str): ID of the interacting part.
            surf_id_interact (str): ID of the interacting surface.
            surf_data_interact (dict): Data of the interacting surface.
            point_search_radius (float|int): Radius for the search cylinder. Defaults to 0.5.
        """
        labels_current, points_current = otaf.geometry.point_dict_to_arrays(
            surf_data_current["POINTS"]
        )
        labels_interact, points_interact = otaf.geometry.point_dict_to_arrays(
            surf_data_interact["POINTS"]
        )
        if points_current.shape[0] > 2:
            if not otaf.geometry.are_points_on_2d_plane(points_current):
                otaf.exceptions.PointsNotOnPlaneError(part_id_current, surf_id_current)
        if points_interact.shape[0] > 2:
            if not otaf.geometry.are_points_on_2d_plane(points_interact):
                otaf.exceptions.PointsNotOnPlaneError(part_id_interact, surf_id_interact)

        if otaf.geometry.are_planes_facing(
            surf_data_current["FRAME"][:, 0],
            surf_data_current["ORIGIN"],
            surf_data_interact["FRAME"][:, 0],
            surf_data_interact["ORIGIN"],
        ) and otaf.geometry.are_planes_parallel(
            surf_data_current["FRAME"][:, 0], surf_data_interact["FRAME"][:, 0]
        ):
            for point_id_current, point_current in surf_data_current["POINTS"].items():
                for opp_point_id, point_interact in surf_data_interact["POINTS"].items():
                    if self.is_point_facing(
                        point_current, point_interact, surf_data_current, surf_data_interact
                    ):
                        self.facingPointDict[part_id_current][surf_id_current][
                            point_id_current
                        ] = f"{part_id_interact}{surf_id_interact}{opp_point_id}"
        else:
            sstr1 = f"P{part_id_current}{surf_id_current}"
            sstr2 = f"P{part_id_interact}{surf_id_interact}"
            logging.warning(
                f"The two interacting surfaces {sstr1} and {sstr2} are either not parallel or not facing each other."
            )

    def is_point_facing(
        self,
        point_current,
        point_interact,
        surf_data_current,
        surf_data_interact,
        point_search_radius: Union[float, int] = 0.25,  # in mm
    ):
        """
        Check if a point on one plane faces a point on another plane within a specified radius.

        Args:
            point_current (array): Coordinates of the point on the current plane.
            point_interact (array): Coordinates of the point on the interacting plane.
            surf_data_current (dict): Data of the current plane.
            surf_data_interact (dict): Data of the interacting plane.
            point_search_radius (float|int): Search radius for facing points.

        Returns:
            bool: True if the point is facing within the specified radius, False otherwise.
        """
        origin_current = surf_data_current["ORIGIN"]
        frame_current = surf_data_current["FRAME"]
        origin_interact = surf_data_interact["ORIGIN"]
        frame_interact = surf_data_interact["FRAME"]

        # Check if the line connecting point_current and point_interact intersects both planes
        intersect_point_current = otaf.geometry.line_plane_intersection(
            frame_current[:, 0], origin_current, point_interact, frame_interact[:, 0]
        )
        intersect_point_interact = otaf.geometry.line_plane_intersection(
            frame_interact[:, 0], origin_interact, point_current, frame_current[:, 0]
        )

        # Verify that intersection points are within the search radius of the original points
        distance_current = otaf.geometry.euclidean_distance(point_current, intersect_point_current)
        distance_interact = otaf.geometry.euclidean_distance(
            point_interact, intersect_point_interact
        )

        return distance_current <= point_search_radius and distance_interact <= point_search_radius

    def process_cylinder_cylinder_interaction(
        self, idPCur, idSCur, datSCur, idPInt, idSInt, datSInt
    ):
        """
        Processes interaction between two cylindrical surfaces by analyzing alignment, surface direction,
        and geometric characteristics for tolerance analysis.

        Initial checks include alignment of cylinder normals and setting or validating surface direction
        (centripetal or centrifugal). Conflicts or identical radii issues raise exceptions. Post-validation,
        the method discerns inner and outer cylinders, approximates them (inner as a line, outer with points),
        and updates `self.SDA["PARTS"]` for tolerance analysis.

        Points are generated only on the top and bottom surfaces, ensuring same-plane placement. For the
        inner cylinder, only center line top/bottom points are considered.

        Args:
            idPCur (str): ID of the current part.
            idSCur (str): ID of the current surface.
            datSCur (dict): Current surface data, including radius, origin, frame.
            idPInt (str): ID of the interacting part.
            idSInt (str): ID of the interacting surface.
            datSInt (dict): Interacting surface data, similar to `datSCur`.

        Raises:
            otaf.exceptions.NonConcentricCylindersException: For non-aligned, non-facing cylinder normals.
            ValueError: For same radii cylinders or related value errors.
            otaf.exceptions.ConflictingSurfaceDirectionsException: For conflicting cylinder surface directions.
            otaf.exceptions.CylindricalInterferenceGeometricException: For geometric interference based on surface
            directions and radii.
        """

        if not otaf.geometry.are_normals_aligned_and_facing(
            datSCur["FRAME"][:, 0], datSCur["ORIGIN"], datSInt["FRAME"][:, 0], datSInt["ORIGIN"]
        ):
            raise otaf.exceptions.NonConcentricCylindersException()  # For the moment concentricity necessary nominal case

        if not "SURFACE_DIRECTION" in datSCur and not "SURFACE_DIRECTION" in datSInt:
            if datSCur["RADIUS"] > datSInt["RADIUS"]:
                datSCur["SURFACE_DIRECTION"] = "centripetal"
                datSInt["SURFACE_DIRECTION"] = "centrifugal"
            elif datSCur["RADIUS"] < datSInt["RADIUS"]:
                datSCur["SURFACE_DIRECTION"] = "centrifugal"
                datSInt["SURFACE_DIRECTION"] = "centripetal"
            else:
                raise ValueError(
                    "Inner and outer cylinder walls have the same radii: Not yet supported."
                )

        else:
            # Check for conflicting surface directions and interfering parts
            if datSCur.get("SURFACE_DIRECTION") == datSInt.get("SURFACE_DIRECTION"):
                raise otaf.exceptions.ConflictingSurfaceDirectionsException(
                    idPCur, idSCur, idPInt, idSInt
                )
            elif (
                datSCur.get("SURFACE_DIRECTION") == "centrifugal"
                and datSCur["RADIUS"] > datSInt["RADIUS"]
            ):
                raise otaf.exceptions.CylindricalInterferenceGeometricException(
                    idPCur, idSCur, idPInt, idSInt, datSCur["RADIUS"], datSInt["RADIUS"]
                )
            elif (
                datSInt.get("SURFACE_DIRECTION") == "centrifugal"
                and datSInt["RADIUS"] > datSCur["RADIUS"]
            ):
                raise otaf.exceptions.CylindricalInterferenceGeometricException(
                    idPInt, idSInt, idPCur, idSCur, datSInt["RADIUS"], datSCur["RADIUS"]
                )
            elif datSCur["RADIUS"] == datSInt["RADIUS"]:
                raise ValueError(
                    "Inner and outer cylinder walls have the same radii: Not yet supported."
                )

        # Approximate the inner and outer cylinders
        self.approximate_cylinders_populate_facingPointDict(
            idPCur, idSCur, datSCur, idPInt, idSInt, datSInt
        )

        # self.populate_facingPointDict_cylinders()

    def approximate_cylinders_populate_facingPointDict(
        self, idPCur, idSCur, datSCur, idPInt, idSInt, datSInt
    ):
        """
        Approximate both current and interacting cylinders in the case of cylinder-cylinder interaction

        Args:
            idPCur (str): ID of the current cylinder's part.
            idSCur (str): ID of the current cylinder's surface.
            idPInt (str): ID of the interacting cylinder's part.
            idSInt (str): ID of the interacting cylinder's surface.
        """
        # The origins are always "facing", or at least interacting
        # So let's deal with that one first.
        self.facingPointDict[idPCur][idSCur][
            f"{idSCur.upper()}0"
        ] = f"{idPInt}{idSInt}{idSInt.upper()}0"

        get_max_point_id = lambda points: max(
            [int(re.compile(r"(\d+)$").search(s).group(1)) for s in points]
        )
        # Approximate Inner Cylinder
        xMaxCur = datSCur["EXTENT_LOCAL"]["x_max"]
        xMinCur = datSCur["EXTENT_LOCAL"]["x_min"]

        # Approximate Outer Cylinder
        xMaxInt = datSInt["EXTENT_LOCAL"]["x_max"]
        xMinInt = datSInt["EXTENT_LOCAL"]["x_min"]

        # if they are facing, the local max for one is local min for the other.
        normsFacing = np.dot(datSCur["FRAME"][:, 0], datSInt["FRAME"][:, 0]) < 0
        vecCur2IntGlobal = datSInt["ORIGIN"] - datSCur["ORIGIN"]
        vecCur2IntLocal = vecCur2IntGlobal @ datSCur["FRAME"]
        dOrigin = np.linalg.norm(vecCur2IntGlobal)

        logging.debug(
            f"Processing cylinder-cylinder interaction for part {idPCur} / surface {idSCur} and part {idPInt} / surface {idSInt}."
        )
        logging.debug(f"Cylinder revolution axes are {None if normsFacing else 'not'} facing.")
        logging.debug(
            f"Start cylinder to end cylinder origin vector in local {vecCur2IntLocal.round(3)} and global {vecCur2IntGlobal.round(3)} coordinates"
        )
        logging.debug(f"First cylinder local extent : {xMinCur} / {xMaxCur}")
        logging.debug(f"Second cylinder local extent : {xMinInt} / {xMaxInt}")

        if (xMaxCur < xMinCur) or (xMaxInt < xMinInt):
            raise ValueError(
                "Local extent for cylinders part {idPCur} surface {idSCur} not well defined"
            )

        if xMaxCur > xMinCur and xMaxInt > xMinInt:
            # These are top and bottom in the local frame
            topPointCur = datSCur["ORIGIN"] + datSCur["FRAME"][:, 0] * xMaxCur
            bottomPointCur = datSCur["ORIGIN"] + datSCur["FRAME"][:, 0] * xMinCur

            topPointInt = datSInt["ORIGIN"] + datSInt["FRAME"][:, 0] * xMaxInt
            bottomPointInt = datSInt["ORIGIN"] + datSInt["FRAME"][:, 0] * xMinInt

            current_points = self.SDA.get_surface_points(idPCur, idSCur)
            current_points = otaf.common.merge_with_checks(
                current_points,
                {f"{idSCur.upper()}1": topPointCur, f"{idSCur.upper()}2": bottomPointCur},
            )

            interacting_points = self.SDA.get_surface_points(idPInt, idSInt)
            interacting_points = otaf.common.merge_with_checks(
                interacting_points,
                {f"{idSInt.upper()}1": topPointInt, f"{idSInt.upper()}2": bottomPointInt},
            )

            if not normsFacing:  # they point in the same direction here
                xMaxIntInCur = xMaxInt + vecCur2IntLocal[0]
                xMinIntInCur = xMinInt + vecCur2IntLocal[0]
                xMaxCurInInt = xMaxCur - vecCur2IntLocal[0]
                xMinCurInInt = xMinCur - vecCur2IntLocal[0]
                # if the projected upper end of the interacting cylinder is lower then the current
                if xMaxIntInCur < xMaxCur:
                    # We add a new point on current part
                    topPointCurAdd = datSCur["ORIGIN"] + datSCur["FRAME"][:, 0] * xMaxIntInCur
                    if not any(
                        [
                            bool(np.array_equal(topPointCurAdd, pnt))
                            for pnt in current_points.values()
                        ]
                    ):
                        pointIndex = get_max_point_id(current_points) + 1
                        idPntCur = f"{idSCur.upper()}{pointIndex}"
                        current_points[idPntCur] = topPointCurAdd
                        pspstr = f"{idPInt}{idSInt}{idSInt.upper()}1"
                        self.facingPointDict[idPCur][idSCur][idPntCur] = pspstr
                        logging.debug("\t", "IF STATEMENT 1 NOT FACING")

                elif xMaxIntInCur > xMaxCur:
                    # We add a new point interacting part
                    topPointIntAdd = datSInt["ORIGIN"] + datSInt["FRAME"][:, 0] * xMaxCurInInt
                    if not any(
                        [
                            bool(np.array_equal(topPointIntAdd, pnt))
                            for pnt in interacting_points.values()
                        ]
                    ):
                        pointIndex = get_max_point_id(interacting_points) + 1
                        idPntInt = f"{idSInt.upper()}{pointIndex}"
                        interacting_points[idPntInt] = topPointIntAdd
                        pspstr = f"{idPCur}{idSCur}{idSCur.upper()}1"
                        self.facingPointDict[idPInt][idSInt][idPntInt] = pspstr
                        logging.debug("\t", "IF STATEMENT 2 NOT FACING")

                elif xMaxIntInCur == xMaxCur:
                    # No point needs to be added.
                    pspstr = f"{idPInt}{idSInt}{idSInt.upper()}1"
                    self.facingPointDict[idPCur][idSCur][f"{idSCur.upper()}1"] = pspstr
                    logging.debug("\t", "IF STATEMENT 3 NOT FACING")

                ####################################################################################

                # if the projected lower end of the interacting cylinder is higher then the current
                if xMinIntInCur > xMinCur:
                    # We add a new point at this location:
                    bottomPointCurAdd = datSCur["ORIGIN"] + datSCur["FRAME"][:, 0] * xMinIntInCur
                    if not any(
                        [
                            bool(np.array_equal(bottomPointCurAdd, pnt))
                            for pnt in current_points.values()
                        ]
                    ):
                        pointIndex = get_max_point_id(current_points) + 1
                        idPntCur = f"{idSCur.upper()}{pointIndex}"
                        current_points[idPntCur] = bottomPointCurAdd
                        pspstr = f"{idPInt}{idSInt}{idSInt.upper()}2"
                        self.facingPointDict[idPCur][idSCur][idPntCur] = pspstr
                        logging.debug("\t", "IF STATEMENT 4 NOT FACING")

                elif xMinIntInCur < xMinCur:
                    # We add a new point interacting part
                    bottomPointIntAdd = datSInt["ORIGIN"] + datSInt["FRAME"][:, 0] * xMinCurInInt
                    if not any(
                        [
                            bool(np.array_equal(bottomPointIntAdd, pnt))
                            for pnt in interacting_points.values()
                        ]
                    ):
                        pointIndex = get_max_point_id(interacting_points) + 1
                        idPntInt = f"{idSInt.upper()}{pointIndex}"
                        interacting_points[idPntInt] = bottomPointIntAdd
                        pspstr = f"{idPCur}{idSCur}{idSCur.upper()}2"
                        self.facingPointDict[idPInt][idSInt][idPntInt] = pspstr
                        logging.debug("\t", "IF STATEMENT 5 NOT FACING")

                elif xMinIntInCur == xMinCur:
                    # No point needs to be added.
                    pspstr = f"{idPInt}{idSInt}{idSInt.upper()}2"
                    self.facingPointDict[idPCur][idSCur][f"{idSCur.upper()}2"] = pspstr
                    logging.debug("\t", "IF STATEMENT 6 NOT FACING")
                #################################################################################

            elif normsFacing:
                # I have begune here but all the 6 case from above have to be done here.
                xMaxIntInCur = -1 * xMaxInt + vecCur2IntLocal[0]
                xMinIntInCur = -1 * xMinInt + vecCur2IntLocal[0]
                xMaxCurInInt = -1 * xMaxCur - vecCur2IntLocal[0]
                xMinCurInInt = -1 * xMinCur - vecCur2IntLocal[0]
                # Here it is inverted
                if xMaxIntInCur > xMinCur:  # Adding point tu current part
                    bottomPointCurAdd = datSCur["ORIGIN"] + datSCur["FRAME"][:, 0] * xMaxIntInCur
                    if not any(
                        [
                            bool(np.array_equal(bottomPointCurAdd, pnt))
                            for pnt in current_points.values()
                        ]
                    ):
                        pointIndex = get_max_point_id(current_points) + 1
                        idPntCur = f"{idSCur.upper()}{pointIndex}"
                        current_points[idPntCur] = bottomPointCurAdd
                        pspstr = f"{idPInt}{idSInt}{idSInt.upper()}1"
                        self.facingPointDict[idPCur][idSCur][idPntCur] = pspstr
                        logging.debug("\t", "IF STATEMENT 1 FACING")

                elif xMaxIntInCur < xMinCur:  # Adding point tu interacting part
                    bottomPointIntAdd = datInt["ORIGIN"] + datSInt["FRAME"][:, 0] * xMinCurInInt
                    if not any(
                        [
                            bool(np.array_equal(bottomPointIntAdd, pnt))
                            for pnt in interacting_points.values()
                        ]
                    ):
                        pointIndex = get_max_point_id(interacting_points) + 1
                        idPntInt = f"{idSInt.upper()}{pointIndex}"
                        interacting_points[idPntInt] = bottomPointIntAdd
                        pspstr = f"{idPCur}{idSCur}{idSCur.upper()}2"
                        self.facingPointDict[idPInt][idSInt][idPntInt] = pspstr
                        logging.debug("\t", "IF STATEMENT 2 FACING")

                elif xMaxIntInCur == xMinCur:
                    pspstr = f"{idPInt}{idSInt}{idSInt.upper()}1"
                    self.facingPointDict[idPCur][idSCur][f"{idSCur.upper()}2"] = pspstr
                    logging.debug("\t", "IF STATEMENT 3 FACING")

                ####################################################################################

                if xMinIntInCur < xMaxCur:
                    topPointCurAdd = datSCur["ORIGIN"] + datSCur["FRAME"][:, 0] * xMinIntInCur
                    if not any(
                        [
                            bool(np.array_equal(topPointCurAdd, pnt))
                            for pnt in current_points.values()
                        ]
                    ):
                        pointIndex = get_max_point_id(current_points) + 1
                        idPntCur = f"{idSCur.upper()}{pointIndex}"
                        current_points[idPntCur] = topPointCurAdd
                        pspstr = f"{idPInt}{idSInt}{idSInt.upper()}2"
                        self.facingPointDict[idPCur][idSCur][idPntCur] = pspstr
                        logging.debug("\t", "IF STATEMENT 4 FACING")

                elif xMinIntInCur > xMaxCur:
                    # We add a new lower point on the interacting part
                    topPointIntAdd = datSInt["ORIGIN"] + datSInt["FRAME"][:, 0] * xMaxCurInInt
                    if not any(
                        [
                            bool(np.array_equal(topPointIntAdd, pnt))
                            for pnt in interacting_points.values()
                        ]
                    ):
                        pointIndex = get_max_point_id(interacting_points) + 1
                        idPntInt = f"{idSInt.upper()}{pointIndex}"
                        interacting_points[idPntInt] = topPointIntAdd
                        pspstr = f"{idPCur}{idSCur}{idSCur.upper()}1"
                        self.facingPointDict[idPInt][idSInt][idPntInt] = pspstr
                        logging.debug("\t", "IF STATEMENT 5 FACING")

                elif xMinIntInCur == xMaxCur:
                    # No point needs to be added.
                    pspstr = f"{idPInt}{idSInt}{idSInt.upper()}2"
                    self.facingPointDict[idPCur][idSCur][f"{idSCur.upper()}1"] = pspstr
                    logging.debug("\t", "IF STATEMENT 6 FACING")

                #################################################################################
            try:
                self.SDA.add_surface_points(idPCur, idSCur, current_points, ignore_duplicates=True)
                self.SDA.add_surface_points(
                    idPInt, idSInt, interacting_points, ignore_duplicates=True
                )
            except Exception as e:
                error_msg = (
                    f"Error during processing cylinder approximation for parts {idPCur} and {idPInt} on surfaces {idSCur} and {idSInt}. "
                    f"Current points: {len(current_points)}, Interacting points: {len(interacting_points)}. "
                    f"Exception: {str(e)}"
                )
                # Consider logging the error message instead of printing if a logging framework is in use
                print("\t", error_msg)
                # Optionally, re-raise the exception to handle it upstream or log it in a more centralized manner
                raise e

        elif xMaxCur == xMinCur == 0.0 and xMaxInt == xMinInt == 0.0 and dOrigin == 0.0:
            pass  # No new point needs to be added

        else:
            raise NotImplementedError("You've gone too far buddy")


@beartype
class InterfaceLoopBuilder:
    x_component_mask_matrix = sp.Matrix(
        np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    )

    def __init__(
        self,
        system_data_augmented,
        compatibility_loop_handling,
        filtered_gap_matrices,
        circle_resolution=8,
    ):
        self.SDA = system_data_augmented
        self.CLH = compatibility_loop_handling
        self.filtered_gap_matrices = filtered_gap_matrices
        self.CIRCLE_RESOLUTION = circle_resolution
        self.interface_expressions = None

    def matrix_list_taylor_expansion(self, matrix_loop_list):
        return otaf.MatrixPolynomeTaylorExpansion(matrix_loop_list).construct_FO_matrix_expansion()

    ###################################################################################################

    def populate_interface_expressions(self):
        self.interface_expressions = self.get_interface_equations_from_facing_parts()

    def get_interface_equations_from_facing_parts(
        self,
    ):  # -> List[List[Union[otaf.TransformationMatrix, otaf.GapMatrix]]]
        """Generate interface loops from facing parts based on unique gap matrices."""
        interface_equations = []
        for idPart, surfData in self.filtered_gap_matrices.items():
            for idSurf, pointData in surfData.items():
                # If len(pointData["USED"]) > 1 that could also mean that the cylinder is traversing multiple parts
                # So a new implementation would be needed then.
                surfType = self.SDA["PARTS"][idPart][idSurf]["TYPE"]

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
                                for x in otaf.common.get_relevant_expressions(expr)
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

    def get_interface_equations_facing_planes(self, idPart, idSurf):
        """
        Gets all the interface equations for a plane (specified by idPart and idSurf)
        and the surfaces it's interacting with. This involves identifying used and unused
        gap matrices to generate interaction matrix loops for each plane-to-plane interaction.

        Parameters:
        - idPart (str): Identifier for the part containing the plane of interest.
        - idSurf (str): Identifier for the surface of the part of interest.

        Returns:
        List[<appropriate type>]: A list of interaction matrix loops for plane-to-plane
        interactions. Each loop is represented by a sequence of transformation and gap
        matrices that describe the geometric relationship between the two interacting planes.
        """
        BPSP, GMMP = otaf.constants.BASE_PART_SURF_PATTERN, otaf.constants.G_MATRIX_MSTRING_PATTERN
        print(f"Processing part {idPart}, surface {idSurf} for plane-to-plane interactions.")

        usedGMat = list(self.filtered_gap_matrices[idPart][idSurf]["USED"])
        usedGMatDat = [list(GMMP.fullmatch(ug).groups()) for ug in usedGMat]
        usedInteractingParts = [X[3] for X in usedGMatDat]
        usedInteractingSurfs = [X[4] for X in usedGMatDat]
        print("usedGMatDat", usedGMatDat)
        print(f"Found {len(usedGMat)} used gap matrices.")

        unusedGMat = list(self.filtered_gap_matrices[idPart][idSurf]["UNUSED"])
        unusedGMatDat = [list(GMMP.fullmatch(uug).groups()) for uug in unusedGMat]
        unusedInteractingParts = [X[3] for X in unusedGMatDat]
        unusedInteractingSurfs = [X[4] for X in unusedGMatDat]
        print("unusedGMatDat", unusedGMatDat)
        print(f"Found {len(unusedGMat)} unused gap matrices.")

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
                print(
                    f"Generated {len(interactionMatrixLoopList)} interaction matrix loops for current matching."
                )
            facingPlaneInteractionMatrixList.extend(interactionMatrixLoopList)

        return facingPlaneInteractionMatrixList

    def _generate_surface_interaction_loop(self, unusedMatch, usedMatch, usedGapMatrix):
        """
        Expresses an unused gap matrix in terms of a reference (used) gap matrix. It calculates
        the transformation matrices needed to align the unused gap with the spatial relationship
        defined by the reference gap.

        Parameters:
        - unusedMatch (list[str]): Data from the regex match of an unused gap matrix, detailing the interaction.
        - usedMatch (list[str]): Data from the regex match of a used gap matrix, serving as the interaction reference.
        - gapMatrix (otaf.GapMatrix): The gap matrix object defining the spatial relationship.

        Returns:
        List[otaf.TransformationMatrix, otaf.GapMatrix]: The sequence of transformation and gap matrices
        describing the interaction, including the nominal transformation (JCor inverse).
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
        Gets all the interface equations for a cylinder (specified by idPart and idSurf)
        and the surfaces it's interacting with. This involves identifying used and unused
        gap matrices to generate interaction matrix loops for each cylinder-to-cylinder interaction.

        Parameters:
        - idPart (str): Identifier for the part containing the cylinder of interest.
        - idSurf (str): Identifier for the surface of the part of interest.

        Returns:
        List[<appropriate type>]: A list of interaction matrix loops for cylinder-to-cylinder
        interactions. Each loop is represented by a sequence of transformation and gap
        matrices that describe the geometric relationship between the two interacting planes.
        """
        print(f"Processing part {idPart}, surface {idSurf} for cylinder-to-cylinder interactions.")

        BPSP, GMMP = otaf.constants.BASE_PART_SURF_PATTERN, otaf.constants.G_MATRIX_MSTRING_PATTERN

        usedGMat = list(self.filtered_gap_matrices[idPart][idSurf]["USED"])
        usedGMatDat = [list(GMMP.fullmatch(ug).groups()) for ug in usedGMat]
        usedInteractingParts = [X[3] for X in usedGMatDat]
        usedInteractingSurfs = [X[4] for X in usedGMatDat]
        print("usedGMatDat", usedGMatDat)
        print(f"Found {len(usedGMat)} used gap matrices.")

        unusedGMat = list(self.filtered_gap_matrices[idPart][idSurf]["UNUSED"])
        unusedGMatDat = [list(GMMP.fullmatch(uug).groups()) for uug in unusedGMat]
        unusedInteractingParts = [X[3] for X in unusedGMatDat]
        unusedInteractingSurfs = [X[4] for X in unusedGMatDat]
        print("unusedGMatDat", unusedGMatDat)
        print(f"Found {len(unusedGMat)} unused gap matrices.")

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
                    print(
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
            print(
                f"Generated {len(interactionEquationLoopList)} interaction equations for current matching."
            )
        print(
            f"Total interaction equations generated: {len(facingCylinderInteractionEquationList)}"
        )
        return facingCylinderInteractionEquationList

    def _cylinder_interaction_approximation(
        self, idPart, idSurf, datGUsed, cylinder_edge_center_interaction_matrix
    ):
        """Here using a trigonometric trick, we can verifiy if the inner cylinder approximated by
        a point, is inside the outer cylinder, by the means of a transformation matrix representing
        the distance between the two centers. The components we are interested in are the y and z
        components as they are in the plane of the cylinders section. Depending on the resolution,
        N angles between 0 and 2 pi have their corresponding cos, sin pairs calculated.
        Then for each cos/sin pair we multiply and sum the y, z and cos/sin values. This maximum
        value of all of these equations must be below the radii differerence of the cylinders.
        """
        radius1 = self.SDA["PARTS"][datGUsed[0]][datGUsed[1]]["RADIUS"]
        radius2 = self.SDA["PARTS"][datGUsed[3]][datGUsed[4]]["RADIUS"]
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
