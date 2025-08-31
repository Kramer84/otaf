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
class SurfaceInteractionManager:
    """
    Manages interactions between surfaces in a mechanical assembly.

    This class identifies and processes relationships between surfaces of parts in an assembly,
    focusing on geometrical interactions such as "facing points" and compatibility between
    interacting surfaces. It supports multiple surface types and ensures the data is structured
    for further analysis or tolerance evaluation.
    """
    def __init__(self, assemblyDataProcessor:AssemblyDataProcessor):
        self.ADP = assemblyDataProcessor
        self.facingPointDict = otaf.common.tree()

    def get_facing_point_dictionary(self) -> None:
        """
        Identify relationships between facing points on part surfaces.

        This method clears and populates the `facingPointDict` attribute with data about
        which points on one surface are "facing" or interact with points on other surfaces
        within the mechanical assembly. Relationships are based on geometrical alignment
        and compatibility.

        Notes:
            - Surfaces and points are processed based on the system data stored in `self.ADP`.
            - This method iterates over all parts and their surfaces to establish facing relationships.
        """
        self.facingPointDict.clear()
        for part_id, part in self.ADP["PARTS"].items():
            for surface_id, surface_data in part.items():
                self.process_surface_interactions(part_id, surface_id, surface_data)

    def process_surface_interactions(self, part_id, surface_id, surface_data):
        """
        Evaluate interactions for a specific surface in the assembly.

        This method identifies and processes geometrical interactions between a given surface
        and its interacting surfaces as defined in the system data. Interactions are categorized
        based on surface types (e.g., plane-plane or cylinder-cylinder).

        Args:
            part_id (str): ID of the current part to which the surface belongs.
            surface_id (str): ID of the current surface being processed.
            surface_data (dict): Detailed data describing the surface's properties and interactions.

        Notes:
            - Interaction data is retrieved from the `INTERACTIONS` field in `surface_data`.
            - Surface types determine the specific interaction processing method used.
        """
        for interaction in surface_data["INTERACTIONS"]:
            (part_id_interact, surf_id_interact) = otaf.constants.BASE_PART_SURF_PATTERN.fullmatch(
                interaction
            ).groups()
            surf_data_interact = self.ADP["PARTS"][part_id_interact][surf_id_interact]
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
        Process interactions between two plane surfaces.

        This method identifies facing relationships between points on two planar surfaces.
        It validates the geometric alignment of the planes, checks if they are parallel and facing,
        and establishes facing relationships between individual points if conditions are met.

        Args:
            part_id_current (str): ID of the current part.
            surf_id_current (str): ID of the current surface.
            surf_data_current (dict): Data of the current surface.
            part_id_interact (str): ID of the interacting part.
            surf_id_interact (str): ID of the interacting surface.
            surf_data_interact (dict): Data of the interacting surface.
            point_search_radius (float|int, optional): Radius for determining point-facing relationships. Defaults to 0.5.

        Notes:
            - Surfaces must be parallel and facing to establish relationships.
            - Points on both surfaces must be geometrically on the respective planes.
            - If surfaces are not aligned, a warning is logged, and no relationships are established.
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
        Determine if a point on one plane is facing a point on another plane.

        This method checks if the line connecting two points intersects both planes
        and verifies that the intersection points are within a specified radius
        of the original points.

        Args:
            point_current (np.ndarray): Coordinates of the point on the current plane.
            point_interact (np.ndarray): Coordinates of the point on the interacting plane.
            surf_data_current (dict): Data of the current plane.
            surf_data_interact (dict): Data of the interacting plane.
            point_search_radius (float|int, optional): Search radius for determining facing relationships. Defaults to 0.25.

        Returns:
            bool: True if the points are facing within the specified radius, False otherwise.

        Notes:
            - A point is considered "facing" if the line connecting it to the other point intersects
              both planes and the distances between the intersection points and the original points
              are within the specified radius.
            - Uses geometric calculations including plane intersections and Euclidean distance.
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
        Process interactions between two cylindrical surfaces.

        This method analyzes and validates the geometric alignment, surface direction, and radii
        of two interacting cylindrical surfaces. It determines their relationship (inner or outer),
        verifies concentricity, and generates the necessary data for tolerance analysis.

        The method assigns or validates surface directions (centripetal or centrifugal) and raises
        exceptions for conflicts, identical radii, or geometric interference. After validation, it
        updates the system data (`self.ADP["PARTS"]`) to facilitate tolerance analysis.

        Args:
            idPCur (str): ID of the current part.
            idSCur (str): ID of the current surface.
            datSCur (dict): Data of the current cylindrical surface, including radius, origin, and frame.
            idPInt (str): ID of the interacting part.
            idSInt (str): ID of the interacting surface.
            datSInt (dict): Data of the interacting cylindrical surface, similar to `datSCur`.

        Raises:
            otaf.exceptions.NonConcentricCylindersException: If cylinder normals are not aligned and facing.
            otaf.exceptions.ConflictingSurfaceDirectionsException: If both cylinders have conflicting surface directions.
            otaf.exceptions.CylindricalInterferenceGeometricException: If there is geometric interference
                based on surface directions and radii.
            ValueError: If the inner and outer cylinders have the same radii, which is not supported.

        Notes:
            - Concentricity between cylinders is required for nominal cases.
            - Cylinders are classified as inner or outer based on their radii and surface direction.
            - Points are generated for only the top and bottom surfaces of the cylinders, with the inner
              cylinder approximated as a line and the outer cylinder using points on its top/bottom surfaces.
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
        Approximate cylinders and populate the facing point dictionary.

        This method handles the geometric approximation of cylinder-cylinder interactions.
        It determines whether the current and interacting cylinders are facing, computes
        their extents, and updates the facing point dictionary with relevant information
        about their interactions.

        Parameters
        ----------
        idPCur : str
            Identifier for the current cylinder's part.
        idSCur : str
            Identifier for the current cylinder's surface.
        datSCur : dict
            Data structure containing information about the current cylinder.
            Must include the following keys:
                - "EXTENT_LOCAL" : dict
                    Contains `x_max` and `x_min` defining the local extent of the cylinder.
                - "FRAME" : numpy.ndarray
                    Local frame (3x3 matrix) of the cylinder.
                - "ORIGIN" : numpy.ndarray
                    Origin (3D coordinates) of the cylinder.
        idPInt : str
            Identifier for the interacting cylinder's part.
        idSInt : str
            Identifier for the interacting cylinder's surface.
        datSInt : dict
            Data structure containing information about the interacting cylinder.
            Structure must be similar to `datSCur`.

        Raises
        ------
        ValueError
            If the local extent (`x_min` or `x_max`) of either cylinder is invalid (e.g., `x_max < x_min`).
        NotImplementedError
            If the cylinders have unusual configurations that are not supported.
        Exception
            For errors in updating surface points, including duplicate points or invalid data.

        Notes
        -----
        - The method identifies whether the revolution axes of the cylinders are facing or not by
          computing the dot product of their respective local frame's x-axes.
        - Updates the facing point dictionary with the following format:
            `self.facingPointDict[idPCur][idSCur][point_id] = corresponding_point`
        - Calculates new points based on cylinder extents (`x_min`, `x_max`) and their interaction vectors.
        - Uses logging to provide detailed debug-level information about the computation process.

        Warnings
        --------
        Ensure that the `datSCur` and `datSInt` dictionaries are properly structured and include
        the necessary keys. Missing or malformed data may result in unexpected errors.

        Examples
        --------
        Given two cylinders with the following attributes:
        - Cylinder 1 (`idPCur="P1"`, `idSCur="A"`) has local extents `x_min=0` and `x_max=10`
          and origin `[0, 0, 0]`.
        - Cylinder 2 (`idPInt="P2"`, `idSInt="B"`) has local extents `x_min=5` and `x_max=15`
          and origin `[10, 0, 0]`.

        The method will:
        1. Compute whether the revolution axes of the two cylinders are facing.
        2. Determine the interaction vector and update the local extents accordingly.
        3. Add new points for any overlap or interaction between the cylinders.
        4. Populate `self.facingPointDict` with the relationship between corresponding points.
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

            current_points = self.ADP.get_surface_points(idPCur, idSCur)
            current_points = otaf.common.merge_with_checks(
                current_points,
                {f"{idSCur.upper()}1": topPointCur, f"{idSCur.upper()}2": bottomPointCur},
            )

            interacting_points = self.ADP.get_surface_points(idPInt, idSInt)
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
                        logging.debug("\t", "IF STATEMENT 1, NOT FACING")

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
                        logging.debug("\t", "IF STATEMENT 2, NOT FACING")

                elif xMaxIntInCur == xMaxCur:
                    # No point needs to be added.
                    pspstr = f"{idPInt}{idSInt}{idSInt.upper()}1"
                    self.facingPointDict[idPCur][idSCur][f"{idSCur.upper()}1"] = pspstr
                    logging.debug("\t", "IF STATEMENT 3, NOT FACING")

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
                        logging.debug("\t", "IF STATEMENT 4, NOT FACING")

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
                        logging.debug("\t", "IF STATEMENT 5, NOT FACING")

                elif xMinIntInCur == xMinCur:
                    # No point needs to be added.
                    pspstr = f"{idPInt}{idSInt}{idSInt.upper()}2"
                    self.facingPointDict[idPCur][idSCur][f"{idSCur.upper()}2"] = pspstr
                    logging.debug("\t", "IF STATEMENT 6, NOT FACING")
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
                        logging.debug("\t", "IF STATEMENT 1, FACING")

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
                        logging.debug("\t", "IF STATEMENT 2, FACING")

                elif xMaxIntInCur == xMinCur:
                    pspstr = f"{idPInt}{idSInt}{idSInt.upper()}1"
                    self.facingPointDict[idPCur][idSCur][f"{idSCur.upper()}2"] = pspstr
                    logging.debug("\t", "IF STATEMENT 3, FACING")

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
                        logging.debug("\t", "IF STATEMENT 4, FACING")

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
                        logging.debug("\t", "IF STATEMENT 5, FACING")

                elif xMinIntInCur == xMaxCur:
                    # No point needs to be added.
                    pspstr = f"{idPInt}{idSInt}{idSInt.upper()}2"
                    self.facingPointDict[idPCur][idSCur][f"{idSCur.upper()}1"] = pspstr
                    logging.debug("\t", "IF STATEMENT 6, FACING")

                #################################################################################
            try:
                self.ADP.add_surface_points(idPCur, idSCur, current_points, ignore_duplicates=True)
                self.ADP.add_surface_points(
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
