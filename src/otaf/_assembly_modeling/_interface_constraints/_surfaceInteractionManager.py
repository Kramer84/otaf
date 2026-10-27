from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"


import re
import logging

import numpy as np

from beartype import beartype
from beartype.typing import Union

from otaf.common import tree, merge_with_checks
from otaf.constants import BASE_PART_SURF_PATTERN
from otaf.geometry import point_dict_to_arrays, are_points_on_2d_plane, are_planes_facing, are_planes_parallel, are_lines_collinear, interval_overlap, planar_overlap_vertices
from otaf.exceptions import PointsNotOnPlaneError, NonConcentricCylindersException, ConflictingSurfaceDirectionsException, CylindricalInterferenceGeometricException, NoAxialOverlapException, NoPlanarOverlapException

from ..assemblyDataProcessor import AssemblyDataProcessor

# Two points closer than this (in model length units) are the same point.
POINT_MATCH_ATOL = 1e-9


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
        self.facingPointDict = tree()
        # Gap matrix names (both directions) of point pairs where contact can occur,
        # i.e. the vertices of the overlap of two interacting surfaces. The other
        # facing pairs (surface origins) only serve as loop references.
        self.contactGapNames = set()

    def get_facing_point_dictionary(self) -> None:
        """
        Identify relationships between facing points on part surfaces.

        This method clears and populates the `facingPointDict` attribute with data about
        which points on one surface are "facing" or interact with points on other surfaces
        within the mechanical assembly. Relationships are based on geometrical alignment
        and compatibility.

        Notes
        -----
            - Surfaces and points are processed based on the system data stored in `self.ADP`.
            - This method iterates over all parts and their surfaces to establish facing relationships.
        """
        self.facingPointDict.clear()
        self.contactGapNames.clear()
        for part_id, part in self.ADP["PARTS"].items():
            for surface_id, surface_data in part.items():
                self.process_surface_interactions(part_id, surface_id, surface_data)

    def process_surface_interactions(self, part_id, surface_id, surface_data):
        """
        Evaluate interactions for a specific surface in the assembly.

        This method identifies and processes geometrical interactions between a given surface
        and its interacting surfaces as defined in the system data. Interactions are categorized
        based on surface types (e.g., plane-plane or cylinder-cylinder).

        Parameters
        ----------
            part_id (str): ID of the current part to which the surface belongs.
            surface_id (str): ID of the current surface being processed.
            surface_data (dict): Detailed data describing the surface's properties and interactions.

        Notes
        -----
            - Interaction data is retrieved from the `INTERACTIONS` field in `surface_data`.
            - Surface types determine the specific interaction processing method used.
        """
        for interaction in surface_data["INTERACTIONS"]:
            (part_id_interact, surf_id_interact) = BASE_PART_SURF_PATTERN.fullmatch(
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
            elif type_current == "cylinder" and type_interact == "cylinder":
                self.process_cylinder_cylinder_interaction(
                    part_id,
                    surface_id,
                    surface_data,
                    part_id_interact,
                    surf_id_interact,
                    surf_data_interact,
                )
            else:
                raise NotImplementedError(
                    f"Interaction between a {type_current} (P{part_id}{surface_id}) and a "
                    f"{type_interact} (P{part_id_interact}{surf_id_interact}) is not supported."
                )


    def _find_or_add_point(self, part_id: str, surf_id: str, point: np.ndarray) -> str:
        """Return the label of the surface point at `point`, creating it if needed.

        Reusing an existing point (e.g. the surface origin) when a contact location
        coincides with it avoids duplicate coordinates on the surface.
        """
        points = self.ADP.get_surface_points(part_id, surf_id)
        for label, existing in points.items():
            if np.linalg.norm(np.asarray(existing, dtype=float) - point) <= POINT_MATCH_ATOL:
                return label
        index = max(int(re.search(r"(\d+)$", label).group(1)) for label in points) + 1
        label = f"{surf_id.upper()}{index}"
        self.ADP.add_surface_points(part_id, surf_id, {label: point})
        return label

    def _register_facing_pair(self, part_a, surf_a, label_a, part_b, surf_b, label_b, contact: bool) -> None:
        """Record that point `label_a` of surface a faces point `label_b` of surface b."""
        self.facingPointDict[part_a][surf_a].setdefault(label_a, set()).add(f"{part_b}{surf_b}{label_b}")
        if contact:
            self.contactGapNames.add(f"GP{part_a}{surf_a}{label_a}P{part_b}{surf_b}{label_b}")
            self.contactGapNames.add(f"GP{part_b}{surf_b}{label_b}P{part_a}{surf_a}{label_a}")

    def _register_origin_pair(self, part_a, surf_a, part_b, surf_b) -> None:
        """Pair the two surface origins, used as reference by the compatibility loops."""
        origin_a, origin_b = f"{surf_a.upper()}0", f"{surf_b.upper()}0"
        self._register_facing_pair(part_a, surf_a, origin_a, part_b, surf_b, origin_b, contact=False)

    def process_plane_plane_interaction(
        self,
        part_id_current,
        surf_id_current,
        surf_data_current,
        part_id_interact,
        surf_id_interact,
        surf_data_interact,
    ):
        """
        Pair the contact points of two facing planes.

        The two faces are projected onto the plane of the current surface and
        intersected. The vertices of the overlap are where contact can occur: a
        point is placed at each vertex on both surfaces (reusing an existing point
        at that location) and the two points are paired. Points of either face that
        lie outside the overlap are not paired. The surface origins are paired as
        well, as reference for the compatibility loops.

        In 2D assemblies the faces are segments and the overlap has 2 end points;
        in 3D each face is taken as the convex hull of its points (see
        :func:`otaf.geometry.planar_overlap_vertices`).

        Parameters
        ----------
        part_id_current : str
            ID of the current part.
        surf_id_current : str
            ID of the current surface.
        surf_data_current : dict
            Data of the current surface.
        part_id_interact : str
            ID of the interacting part.
        surf_id_interact : str
            ID of the interacting surface.
        surf_data_interact : dict
            Data of the interacting surface.

        Raises
        ------
        PointsNotOnPlaneError
            If the points of a surface are not coplanar.
        NoPlanarOverlapException
            If the two faces do not overlap.

        Notes
        -----
        Surfaces that are not parallel and facing are skipped with a warning.
        """
        _, points_current = point_dict_to_arrays(surf_data_current["POINTS"])
        _, points_interact = point_dict_to_arrays(surf_data_interact["POINTS"])
        if points_current.shape[0] > 2 and not are_points_on_2d_plane(points_current):
            raise PointsNotOnPlaneError(part_id_current, surf_id_current)
        if points_interact.shape[0] > 2 and not are_points_on_2d_plane(points_interact):
            raise PointsNotOnPlaneError(part_id_interact, surf_id_interact)

        frame = np.asarray(surf_data_current["FRAME"], dtype=float)
        normal, origin = frame[:, 0], np.asarray(surf_data_current["ORIGIN"], dtype=float)
        if not (
            are_planes_facing(normal, origin, surf_data_interact["FRAME"][:, 0], surf_data_interact["ORIGIN"])
            and are_planes_parallel(normal, surf_data_interact["FRAME"][:, 0])
        ):
            logging.warning(
                f"The two interacting surfaces P{part_id_current}{surf_id_current} and "
                f"P{part_id_interact}{surf_id_interact} are either not parallel or not facing each other."
            )
            return

        # In-plane coordinates of both faces in the current surface frame (projection along the normal).
        in_plane = frame[:, 1:]
        vertices = planar_overlap_vertices(
            (points_current - origin) @ in_plane, (points_interact - origin) @ in_plane, atol=POINT_MATCH_ATOL
        )
        if len(vertices) == 0:
            raise NoPlanarOverlapException(part_id_current, surf_id_current, part_id_interact, surf_id_interact)

        self._register_origin_pair(part_id_current, surf_id_current, part_id_interact, surf_id_interact)
        origin_interact = np.asarray(surf_data_interact["ORIGIN"], dtype=float)
        for vertex in vertices:
            point_current = origin + in_plane @ vertex
            point_interact = point_current + normal * np.dot(origin_interact - point_current, normal)
            label_current = self._find_or_add_point(part_id_current, surf_id_current, point_current)
            label_interact = self._find_or_add_point(part_id_interact, surf_id_interact, point_interact)
            self._register_facing_pair(
                part_id_current, surf_id_current, label_current,
                part_id_interact, surf_id_interact, label_interact,
                contact=True,
            )

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

        Parameters
        ----------
            idPCur (str): ID of the current part.
            idSCur (str): ID of the current surface.
            datSCur (dict): Data of the current cylindrical surface, including radius, origin, and frame.
            idPInt (str): ID of the interacting part.
            idSInt (str): ID of the interacting surface.
            datSInt (dict): Data of the interacting cylindrical surface, similar to `datSCur`.

        Raises
        ------
            NonConcentricCylindersException: If cylinder normals are not aligned and facing.
            ConflictingSurfaceDirectionsException: If both cylinders have conflicting surface directions.
            CylindricalInterferenceGeometricException: If there is geometric interference
                based on surface directions and radii.
            ValueError: If the inner and outer cylinders have the same radii, which is not supported.

        Notes
        -----
            - Concentricity between cylinders is required for nominal cases.
            - Cylinders are classified as inner or outer based on their radii and surface direction.
            - Points are generated for only the top and bottom surfaces of the cylinders, with the inner
              cylinder approximated as a line and the outer cylinder using points on its top/bottom surfaces.
        """
        # Checking if nominal cylinder axes are well aligned
        if not are_lines_collinear(
            datSCur["FRAME"][:, 0], datSCur["ORIGIN"], datSInt["FRAME"][:, 0], datSInt["ORIGIN"]
        ):
            raise NonConcentricCylindersException(idPCur, idSCur, idPInt, idSInt)  # For the moment concentricity necessary nominal case

        # Determining if a feature is a hole or a axis if not pre-defined.
        if not "SURFACE_DIRECTION" in datSCur and not "SURFACE_DIRECTION" in datSInt:
            if datSCur["RADIUS"] > datSInt["RADIUS"]:
                datSCur["SURFACE_DIRECTION"] = "centripetal"
                datSInt["SURFACE_DIRECTION"] = "centrifugal"
            elif datSCur["RADIUS"] < datSInt["RADIUS"]:
                datSCur["SURFACE_DIRECTION"] = "centrifugal"
                datSInt["SURFACE_DIRECTION"] = "centripetal"
            else:
                raise ValueError(
                    "Inner and outer cylinder walls have the same radii: Not (yet) supported."
                )

        # Checking surface direction validity if already defined
        else:
            # Check for conflicting surface directions and interfering parts
            if datSCur.get("SURFACE_DIRECTION") == datSInt.get("SURFACE_DIRECTION"):
                raise ConflictingSurfaceDirectionsException(
                    idPCur, idSCur, idPInt, idSInt
                )
            elif (
                datSCur.get("SURFACE_DIRECTION") == "centrifugal"
                and datSCur["RADIUS"] > datSInt["RADIUS"]
            ):
                raise CylindricalInterferenceGeometricException(
                    idPCur, idSCur, idPInt, idSInt, datSCur["RADIUS"], datSInt["RADIUS"]
                )
            elif (
                datSInt.get("SURFACE_DIRECTION") == "centrifugal"
                and datSInt["RADIUS"] > datSCur["RADIUS"]
            ):
                raise CylindricalInterferenceGeometricException(
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

    def approximate_cylinders_populate_facingPointDict(
        self, idPCur, idSCur, datSCur, idPInt, idSInt, datSInt
    ):
        """
        Pair the contact points of two coaxial cylinders.

        The axial extents of both cylinders (``EXTENT_LOCAL``, relative to their
        origin along their own axis) are expressed along the axis of the current
        cylinder and intersected. A point is placed at each end of the overlap on
        both axes (reusing an existing point at that location, e.g. an origin lying
        at a cylinder end) and the two points are paired. The surface origins are
        paired as well, as reference for the compatibility loops.

        Parameters
        ----------
        idPCur : str
            Identifier for the current cylinder's part.
        idSCur : str
            Identifier for the current cylinder's surface.
        datSCur : dict
            Data of the current cylinder: ``FRAME`` (axis in the first column),
            ``ORIGIN`` and ``EXTENT_LOCAL`` (``x_min``, ``x_max``).
        idPInt : str
            Identifier for the interacting cylinder's part.
        idSInt : str
            Identifier for the interacting cylinder's surface.
        datSInt : dict
            Data of the interacting cylinder, as for `datSCur`.

        Raises
        ------
        ValueError
            If a local extent is not well defined (``x_max < x_min``).
        NoAxialOverlapException
            If the cylinders do not overlap along their axis.

        Notes
        -----
        Cylinders of zero length (``x_min == x_max == 0``) with coinciding origins
        are paired at their origins only; that pair is then a contact.
        """
        extents = []
        for idP, idS, dat in ((idPCur, idSCur, datSCur), (idPInt, idSInt, datSInt)):
            x_min, x_max = float(dat["EXTENT_LOCAL"]["x_min"]), float(dat["EXTENT_LOCAL"]["x_max"])
            if x_max < x_min:
                raise ValueError(f"Local extent for cylinder part {idP} surface {idS} not well defined")
            extents.append((x_min, x_max))

        axis_cur = np.asarray(datSCur["FRAME"], dtype=float)[:, 0]
        axis_int = np.asarray(datSInt["FRAME"], dtype=float)[:, 0]
        origin_cur = np.asarray(datSCur["ORIGIN"], dtype=float)
        origin_int = np.asarray(datSInt["ORIGIN"], dtype=float)
        logging.debug(
            f"Processing cylinder-cylinder interaction for part {idPCur} / surface {idSCur} "
            f"and part {idPInt} / surface {idSInt}."
        )

        if extents[0] == (0.0, 0.0) and extents[1] == (0.0, 0.0):
            if np.linalg.norm(origin_int - origin_cur) > POINT_MATCH_ATOL:
                raise NoAxialOverlapException(idPCur, idSCur, idPInt, idSInt)
            self._register_facing_pair(
                idPCur, idSCur, f"{idSCur.upper()}0", idPInt, idSInt, f"{idSInt.upper()}0", contact=True
            )
            return

        # Interacting extent expressed along the current axis (the axes may be opposed).
        ends_int = [origin_int + axis_int * x for x in extents[1]]
        span_int = tuple(float(np.dot(p - origin_cur, axis_cur)) for p in ends_int)
        overlap = interval_overlap(extents[0], span_int, atol=POINT_MATCH_ATOL)
        if overlap is None:
            raise NoAxialOverlapException(idPCur, idSCur, idPInt, idSInt)

        self._register_origin_pair(idPCur, idSCur, idPInt, idSInt)
        for x in overlap:
            point = origin_cur + axis_cur * x  # on the common axis, shared by both cylinders
            label_cur = self._find_or_add_point(idPCur, idSCur, point)
            label_int = self._find_or_add_point(idPInt, idSInt, point)
            self._register_facing_pair(idPCur, idSCur, label_cur, idPInt, idSInt, label_int, contact=True)
