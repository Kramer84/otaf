# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = ["AssemblyDataProcessor"]

import re
import logging

from copy import copy

import numpy as np
import sympy as sp
import trimesh as tr

from trimesh import viewer
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Any, Set, Optional

import otaf


@beartype
class AssemblyDataProcessor:
    """
    Class to manage and validate mechanical system data for assembly representation.

    This class transforms a minimal representation of mechanical system data into a structured
    format usable by the rest of the code. It ensures data integrity and validates compatibility
    with defined requirements, including surfaces, points, interactions, and constraints.

    Parameters
    ----------
    system_data : dict, optional
        A nested dictionary representing mechanical system data. If not provided,
        an empty structure is initialized.

    Attributes
    ----------
    system_data : dict
        The validated and potentially augmented system data.
    compatibility_loops_expanded : None or dict
        Placeholder for compatibility loop expansions, initialized as None.
    """
    def __init__(self, system_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None) -> None:
        """
        Initialize a `AssemblyDataProcessor` instance.

        Parameters
        ----------
        system_data : dict, optional
            The input system data dictionary. If None, an empty dictionary structure
            is created and initialized.
        """
        self.system_data = (
            system_data if system_data is not None else self._initialize_empty_system_data()
        )
        self.validate_system_data_structure()
        self.compatibility_loops_expanded = None

    def __getitem__(self, key):
        """
        Retrieve data associated with a key from the system data.

        Parameters
        ----------
        key : str
            The key for accessing system data.

        Returns
        -------
        Any
            The value associated with the specified key.
        """
        return self.system_data[key]

    def __setitem__(self, key, val):
        """
        Set a value in the system data for a given key.

        Parameters
        ----------
        key : str
            The key for the data to be set.
        val : Any
            The value to assign to the key in the system data.
        """
        self.system_data.__setitem__(key, val)

    def __repr__(self):
        """
        Generate a string representation of the system data.

        Returns
        -------
        str
            A string representation of the system data dictionary.
        """
        return self.system_data.__repr__()

    def get_surface_points(self, part_id: str, surf_id: str):
        """
        Retrieve a copy of the points associated with a specific surface.

        Parameters
        ----------
        part_id : str
            The ID of the part containing the surface.
        surf_id : str
            The ID of the surface.

        Returns
        -------
        dict
            A copy of the points associated with the surface.

        Raises
        ------
        KeyError
            If the specified part or surface ID is not found.
        """
        try:
            return copy(self["PARTS"][part_id][surf_id].setdefault("POINTS", {}))
        except KeyError as e:
            raise e

    def _initialize_empty_system_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Initialize an empty system data dictionary structure.

        Returns
        -------
        dict
            A nested dictionary with keys for parts, loops, and global constraints.
        """
        logging.info("No system data dictionary wa passed, initializing empty one")
        return {"PARTS": {}, "LOOPS": {"COMPATIBILITY": {}}, "GLOBAL_CONSTRAINTS": "3D"}

    def validate_system_data_structure(self):
        """
        Validate the structure and integrity of the system data.

        Ensures that the required keys and nested structures exist and conform to expected
        standards, including part IDs, surface labels, types, points, interactions, and constraints.

        Raises
        ------
        otaf.exceptions.MissingKeyError
            If a required key is missing.
        otaf.exceptions.InvalidPartLabelError
            If part labels are not integers.
        otaf.exceptions.InvalidSurfaceLabelError
            If surface labels are not lowercase alphabetic strings.
        otaf.exceptions.UnsupportedSurfaceTypeError
            If the surface type is not supported.
        otaf.exceptions.MissingSurfaceTypeKeyError
            If a surface lacks a "TYPE" key.
        otaf.exceptions.MissingOriginPointError
            If no origin point is defined for a surface.
        otaf.exceptions.InvalidInteractionFormatError
            If an interaction format is invalid.
        otaf.exceptions.InvalidSurfaceDirectionError
            If a surface direction is invalid.
        otaf.exceptions.InvalidGlobalConstraintError
            If the global constraint is not supported.
        """
        logging.info("Validating system data dictionary structure.")
        required_top_level_keys = ["PARTS", "LOOPS", "GLOBAL_CONSTRAINTS"]
        for key in required_top_level_keys:
            if key not in self.system_data:
                raise otaf.exceptions.MissingKeyError(key, "system_data")
            else:
                if key == "PARTS":
                    part_keys = list(self.system_data["PARTS"].keys())
                    if not all(
                        map(lambda x: x.isdigit(), part_keys)
                    ):  # Check if all part labels are integers
                        raise otaf.exceptions.InvalidPartLabelError()
                    for part_id in part_keys:
                        surf_keys = list(self.system_data["PARTS"][part_id].keys())
                        if not all(
                            map(lambda x: x.islower() and x.isalpha(), surf_keys)
                        ):  # Check if all surface labels are lowercase ascii
                            raise otaf.exceptions.InvalidSurfaceLabelError()
                        for surf_id in surf_keys:
                            part_surf_dict = self.system_data["PARTS"][part_id][surf_id]
                            surf_params = list(part_surf_dict.keys())
                            if "TYPE" in surf_params:
                                if not part_surf_dict["TYPE"] in otaf.constants.BASE_SURFACE_TYPES:
                                    raise otaf.exceptions.UnsupportedSurfaceTypeError(
                                        part_id, surf_id, part_surf_dict["TYPE"]
                                    )
                            else:
                                raise otaf.exceptions.MissingSurfaceTypeKeyError(surf_id, part_id)

                            if "POINTS" not in surf_params:
                                self.generate_points_for_surface(part_id, surf_id)

                            else:
                                point_dict = part_surf_dict["POINTS"]
                                self.validate_point_dict(point_dict)
                                label_array, point_array = otaf.geometry.point_dict_to_arrays(
                                    point_dict
                                )
                                label_origin = list(
                                    filter(
                                        otaf.constants.SURF_ORIGIN_PATTERN.fullmatch, label_array
                                    )
                                )
                                if label_origin:
                                    part_surf_dict["ORIGIN"] = np.array(
                                        point_dict[label_origin[0]], dtype="float64"
                                    )
                                else:
                                    raise otaf.exceptions.MissingOriginPointError(part_id, surf_id)

                            if "INTERACTIONS" in surf_params:
                                for interaction in part_surf_dict["INTERACTIONS"]:
                                    if not otaf.constants.BASE_PART_SURF_PATTERN.fullmatch(
                                        interaction
                                    ):
                                        raise otaf.exceptions.InvalidInteractionFormatError(
                                            interaction
                                        )

                            if "SURFACE_DIRECTION" in surf_params:
                                if (
                                    part_surf_dict["SURFACE_DIRECTION"]
                                    not in otaf.constants.SURFACE_DIRECTIONS
                                ):
                                    raise otaf.exceptions.InvalidSurfaceDirectionError(
                                        part_surf_dict["SURFACE_DIRECTION"]
                                    )

                if key == "GLOBAL_CONSTRAINTS":
                    if (
                        self.system_data["GLOBAL_CONSTRAINTS"]
                        not in otaf.constants.GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF.keys()
                    ):
                        raise otaf.exceptions.InvalidGlobalConstraintError(
                            otaf.constants.GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF.keys()
                        )
                if key == "LOOPS":
                    loop_keys = self.system_data["LOOPS"].keys()
                    pass  # Not nice enough to continue working on this...

    def validate_point_dict(self, point_dict: Dict[str, Union[Tuple, np.ndarray]]):
        """
        Validate a dictionary of points for compliance with labeling, formatting, and uniqueness rules.

        Parameters
        ----------
        point_dict : Dict[str, Union[Tuple, numpy.ndarray]]
            A dictionary where keys are point labels and values are their corresponding coordinates.

        Raises
        ------
        otaf.exceptions.LabelPatternError
            If any labels do not match the required pattern.
        otaf.exceptions.LabelPrefixError
            If label prefixes are inconsistent across the points.
        otaf.exceptions.UniqueLabelSuffixError
            If label suffixes are not unique across the points.
        otaf.exceptions.NonUniqueCoordinatesError
            If point coordinates are not unique. Includes information about duplicate points in the error message.

        Notes
        -----
        - Labels must conform to the `SURF_POINT_PATTERN` defined in the `otaf.constants` module.
        - Labels must have consistent prefixes (e.g., "A", "B", etc.) and unique numeric suffixes.
        - The coordinates for all points must be unique.
        """
        label_array, point_array = otaf.geometry.point_dict_to_arrays(point_dict)
        label_pattern_compliance = list(
            filter(otaf.constants.SURF_POINT_PATTERN.fullmatch, label_array)
        )
        if len(label_pattern_compliance) != label_array.shape[0]:
            raise otaf.exceptions.LabelPatternError()
        label_prefixes = list(map(lambda X: re.findall(r"^[A-Z]+", X)[0], label_array))
        if len(list(set(label_prefixes))) > 1:
            raise otaf.exceptions.LabelPrefixError()
        label_suffixes = list(map(lambda X: int(re.findall(r"[0-9]+$", X)[0]), label_array))
        if len(list(set(label_suffixes))) < label_array.shape[0]:
            raise otaf.exceptions.UniqueLabelSuffixError()
        # unique_points = np.unique(point_array, axis=0)
        # if unique_points.shape[0] < point_array.shape[0]:
        #     raise otaf.exceptions.NonUniqueCoordinatesError()
        # Finding duplicates with their labels and coordinates
        point_to_labels = {}
        for label, point in zip(
            label_array, map(tuple, point_array)
        ):  # Convert numpy arrays to tuples for hashing
            point_to_labels.setdefault(point, []).append(label)

        duplicates = {}
        for point, labels in point_to_labels.items():
            if len(labels) > 1:
                duplicates[point] = labels

        if duplicates:
            # Transforming point keys back to their original numpy array form for the error message
            duplicates_formatted = {
                str(np.array(point)): labels for point, labels in duplicates.items()
            }
            raise otaf.exceptions.NonUniqueCoordinatesError(duplicates_formatted)

    def add_surface_points(
        self,
        part_id: str,
        surf_id: str,
        point_dict: Dict[str, Union[Tuple, np.ndarray, List]],
        ignore_duplicates: bool = False,
    ):
        """
        Add surface points to a specified part and surface, ensuring uniqueness and validating the input.

        Parameters
        ----------
        part_id : str
            The ID of the part to which the surface belongs.
        surf_id : str
            The ID of the surface to which the points will be added.
        point_dict : Dict[str, Union[Tuple, numpy.ndarray, List]]
            A dictionary where keys are point labels and values are their corresponding coordinates.
        ignore_duplicates : bool, optional
            If True, ignore duplicate points that match exactly with existing points (default is False).

        Raises
        ------
        otaf.exceptions.PartNotFoundError
            If the specified part ID is not found in the system data.
        otaf.exceptions.SurfaceNotFoundError
            If the specified surface ID is not found in the given part.
        otaf.exceptions.DuplicatePointError
            If a point with the same label or coordinates already exists and duplicates are not ignored.
        otaf.exceptions.LabelPatternError
            If any labels in the point dictionary do not match the required pattern.
        otaf.exceptions.LabelPrefixError
            If label prefixes are inconsistent across the points.
        otaf.exceptions.UniqueLabelSuffixError
            If label suffixes are not unique across the points.
        otaf.exceptions.NonUniqueCoordinatesError
            If point coordinates are not unique.

        Notes
        -----
        - Surface points are stored under the "POINTS" key of the surface's dictionary.
        - If the label of a point matches the origin pattern (`SURF_ORIGIN_PATTERN`), the point is set as the surface's origin.
        """
        if part_id not in self.system_data["PARTS"]:
            raise otaf.exceptions.PartNotFoundError(part_id)
        if surf_id not in self.system_data["PARTS"][part_id]:
            raise otaf.exceptions.SurfaceNotFoundError(surf_id, part_id)

        # Ensure the "POINTS" key exists for the given surface
        points = self.system_data["PARTS"][part_id][surf_id].setdefault("POINTS", {})

        # Validate each point in point_dict
        self.validate_point_dict(point_dict)

        for new_point_name, new_point_value in point_dict.items():
            new_point_array = np.array(new_point_value, dtype="float64")

            if new_point_name in points:
                if ignore_duplicates and np.array_equal(points[new_point_name], new_point_array):
                    continue
                raise otaf.exceptions.DuplicatePointError(
                    point_name=new_point_name,
                    surf_id=surf_id,
                    part_id=part_id,
                    new_point_value=new_point_array,
                    existing_point_value=points[new_point_name],
                    exact_match=True,
                )

            for existing_point_name, existing_point_value in points.items():
                if np.array_equal(existing_point_value, new_point_array):
                    raise otaf.exceptions.DuplicatePointError(
                        point_name=new_point_name,
                        surf_id=surf_id,
                        part_id=part_id,
                        existing_point=existing_point_name,
                        new_point_value=new_point_array,
                        existing_point_value=existing_point_value,
                        exact_match=True,
                    )

            # Add the new point
            points[new_point_name] = new_point_array

            if otaf.constants.SURF_ORIGIN_PATTERN.fullmatch(new_point_name):
                self.system_data["PARTS"][part_id][surf_id]["ORIGIN"] = new_point_array

    def generate_points_for_surface(self, part_id: str, surf_id: str) -> None:
        """
        Generate points for a surface based on its type and characteristics.

        This method determines the surface type and dispatches the task to the appropriate
        surface-specific point generation method.

        Parameters
        ----------
        part_id : str
            The ID of the part containing the surface.
        surf_id : str
            The ID of the surface for which points will be generated.

        Raises
        ------
        otaf.exceptions.MissingKeyError
            If required keys are missing from the surface data.
        otaf.exceptions.UnsupportedSurfaceTypeError
            If the surface type is not supported.
        """
        surf_data = self.system_data["PARTS"][part_id][surf_id]
        otaf.common.validate_dict_keys(
            surf_data,
            ["TYPE", "FRAME", "ORIGIN", "INTERACTIONS"],
            f"Surface dictionary for surface {surf_id} on part {part_id}",
            otaf.constants.SURFACE_DICT_VALUE_CHECKS,
        )

        if surf_data["TYPE"] == "plane":
            self._generate_points_for_plane(part_id, surf_id)
        elif surf_data["TYPE"] == "cylinder":
            self._generate_points_for_cylinder(part_id, surf_id)
        else:
            raise otaf.exceptions.UnsupportedSurfaceTypeError(part_id, surf_id, surf_data["TYPE"])

    def _generate_points_for_plane(self, part_id: str, surf_id: str) -> None:
        """
        Generate points for a plane surface.

        Handles surfaces defined by contours in the global frame or by specified extents.

        Parameters
        ----------
        part_id : str
            The ID of the part containing the surface.
        surf_id : str
            The ID of the plane surface for which points will be generated.

        Raises
        ------
        otaf.exceptions.GeometricConditionError
            If the points in the "CONTOUR_GLOBAL" data are not coplanar.
        otaf.exceptions.DuplicatePointError
            If duplicate points are detected during addition.
        """
        surf_data = self.system_data["PARTS"][part_id][surf_id]
        key_prefix = f"{surf_id.upper()}"
        point_dict = {key_prefix + "0": surf_data["ORIGIN"]}
        if "CONTOUR_GLOBAL" in surf_data:
            # The contour of the surface has been defined in the global frame
            NP = surf_data["CONTOUR_GLOBAL"].shape[0]
            points = surf_data["CONTOUR_GLOBAL"]
            if not otaf.geometry.are_points_on_2d_plane(points):
                raise otaf.exceptions.GeometricConditionError("are_points_on_2d_plane")
            contour_point_dict = {
                key_prefix + str(key): value for key, value in zip(list(range(1, NP + 1)), points)
            }
            point_dict.update(contour_point_dict)
        self.add_surface_points(part_id, surf_id, point_dict)

    def _generate_points_for_cylinder(self, part_id: str, surf_id: str) -> None:
        """
        Generate points for a cylindrical surface.

        Handles surfaces defined by contours in the global frame or by specified extents.

        Parameters
        ----------
        part_id : str
            The ID of the part containing the surface.
        surf_id : str
            The ID of the cylindrical surface for which points will be generated.

        Raises
        ------
        otaf.exceptions.DuplicatePointError
            If duplicate points are detected during addition.
        """
        surf_data = self.system_data["PARTS"][part_id][surf_id]
        key_prefix = f"{surf_id.upper()}"
        point_dict = {key_prefix + "0": surf_data["ORIGIN"]}
        if "CONTOUR_GLOBAL" in surf_data:
            # The contour of the surface has been defined in the global frame
            NP = surf_data["CONTOUR_GLOBAL"].shape[0]
            contour_point_dict = {
                key_prefix + str(key): value
                for key, value in zip(list(range(1, NP + 1)), surf_data["CONTOUR_GLOBAL"])
            }
            point_dict.update(contour_point_dict)
        self.add_surface_points(part_id, surf_id, point_dict)

    def generate_expanded_loops(self) -> None:
        """
        Generate the full representation of compatibility loops and store them as an attribute.

        This method processes the compact descriptions of compatibility loops in the system data
        and expands them into their detailed forms.

        Attributes
        ----------
        compatibility_loops_expanded : dict
            A dictionary where keys are loop names and values are the fully expanded loop strings.
        """
        self.compatibility_loops_expanded = {
            loop_name: self._expand_loop(compact_loop_str)
            for loop_name, compact_loop_str in self.system_data["LOOPS"]["COMPATIBILITY"].items()
        }

    def _expand_loop(self, compact_loop_str: str) -> str:
        """
        Expand a compact loop string into its full representation.

        The compact loop string describes a sequence of transformations and deviations in a mechanical system.
        This method validates and processes the elements of the loop and generates the expanded sequence.

        Parameters
        ----------
        compact_loop_str : str
            A string representing the loop in compact form.

        Returns
        -------
        str
            The expanded loop as a string.

        Raises
        ------
        ValueError
            If any loop element is invalid or if the loop order is inconsistent.
        """
        split_compact_loop = compact_loop_str.split(" -> ")
        for loop_element in split_compact_loop:
            self._validate_loop_element(loop_element)
        part_surf_point = [
            otaf.constants.LOC_STAMP_PATTERN.search(loop_element).groups()
            for loop_element in split_compact_loop
        ]
        first_inter = self._check_loop_order(part_surf_point)
        expanded_loop = self._generate_expanded_loop(part_surf_point)
        expanded_loop += self._handle_loop_closure(part_surf_point, first_inter)
        return " -> ".join(expanded_loop)

    def _validate_loop_element(self, loop_element: str) -> None:
        """
        Validate an individual loop element against the expected format.

        Parameters
        ----------
        loop_element : str
            A string representing an element of the loop.

        Raises
        ------
        ValueError
            If the loop element does not match the expected pattern.
        """
        if not otaf.constants.LOOP_ELEMENT_PATTERN.fullmatch(loop_element):
            raise ValueError(f"Invalid loop element: {loop_element}")

    def _check_loop_order(self, part_surf_point: List[Tuple[str, str, str]]) -> bool:
        """
        Check the order of loop elements to ensure the first transformation is valid.

        This method verifies whether the loop starts and ends on the same part
        based on whether the first transformation is inter-part or intra-part.

        Parameters
        ----------
        part_surf_point : List[Tuple[str, str, str]]
            A list of tuples describing part-surface-point combinations in the loop.

        Returns
        -------
        bool
            True if the first transformation is inter-part, False otherwise.

        Raises
        ------
        ValueError
            If the loop order is invalid.
        """
        if part_surf_point[0][0] != part_surf_point[1][0]:
            if part_surf_point[0][0] != part_surf_point[-1][0]:
                raise ValueError(
                    "Invalid loop order: If first transformation is inter-part, the loop must start and end on the same part."
                )
            return True
        else:
            if part_surf_point[0][0] == part_surf_point[-1][0]:
                raise ValueError(
                    "Invalid loop order: If first transformation is intra-part, the loop must not start and end on different parts."
                )
            return False

    def _generate_expanded_loop(self, part_surf_point: List[Tuple[str, str, str]]) -> List[str]:
        """
        Generate the main sequence of the expanded loop.

        This method processes the list of part-surface-point combinations to create the expanded loop,
        excluding the closure.

        Parameters
        ----------
        part_surf_point : List[Tuple[str, str, str]]
            A list of tuples describing part-surface-point combinations in the loop.

        Returns
        -------
        List[str]
            A list of strings representing the main sequence of the expanded loop.
        """
        return [
            (
                f"TP{ps}{ss}{pts}{se}{pte}"
                if ps == pe
                else f"D{ps}{ss} -> GP{ps}{ss}{pts}P{pe}{se}{pte} -> Di{pe}{se}"
            )
            for (ps, ss, pts), (pe, se, pte) in zip(part_surf_point[:-1], part_surf_point[1:])
        ]

    def _handle_loop_closure(
        self, part_surf_point: List[Tuple[str, str, str]], first_inter: bool
    ) -> List[str]:
        """
        Handle the special case of loop closure in an expanded loop.

        This method generates the closure portion of the loop based on whether the first transformation
        is inter-part or intra-part.

        Parameters
        ----------
        part_surf_point : List[Tuple[str, str, str]]
            A list of tuples describing part-surface-point combinations in the loop.
        first_inter : bool
            Indicates whether the first transformation is inter-part.

        Returns
        -------
        List[str]
            A list of strings representing the closure portion of the expanded loop.
        """
        p_start, s_start, pt_start = part_surf_point[0]
        p_end, s_end, pt_end = part_surf_point[-1]
        if first_inter:
            return [f"TP{p_start}{s_end}{pt_end}{s_start}{pt_start}"]
        else:
            return [
                f"D{p_end}{s_end}",
                f"GP{p_end}{s_end}{pt_end}P{p_start}{s_start}{pt_start}",
                f"Di{p_start}{s_start}",
            ]

    def generate_sphere_clouds(
        self,
        radius: Union[float, int] = 0.5,
        global_translation: Union[list, np.ndarray] = np.array([0, 0, 0]),
    ) -> list:
        """
        Generate sphere representations for each set of points in the system data.

        This method creates a list of 3D spheres representing the points in the system data,
        grouped by parts and surfaces. Each group of points is assigned a unique color.

        Parameters
        ----------
        radius : Union[float, int], optional
            The radius of the spheres (default is 0.5).
        global_translation : Union[list, numpy.ndarray], optional
            A 3D vector representing a global translation applied to all points
            (default is [0, 0, 0]).

        Returns
        -------
        list
            A list of `trimesh` sphere objects for each part and surface.

        Notes
        -----
        - The method uses a color palette to assign unique colors to the spheres
          for each part and surface.
        - The generated spheres are translated globally based on the `global_translation` parameter.
        """
        sphere_clouds = []
        color_index = 0

        for part_id, surfaces in self.system_data["PARTS"].items():
            for surf_id, surface_data in surfaces.items():
                if "POINTS" in surface_data:
                    points = np.array(list(surface_data["POINTS"].values()))
                    color_hex = otaf.plotting.color_palette_3[
                        color_index % len(otaf.plotting.color_palette_3)
                    ]
                    color_rgba = otaf.plotting.hex_to_rgba(color_hex)

                    spheres = otaf.plotting.spheres_from_point_cloud(
                        points,
                        radius=radius,
                        color=color_rgba,
                        global_translation=global_translation,
                    )
                    sphere_clouds.extend(spheres)

                    color_index += 1

        return sphere_clouds

    def generate_functional_planes(
        self,
    ) -> list:
        """
        Generate planes representing each planar feature in the assembly

        Parameters
        ----------

        Returns
        -------
        list
            A list of `trimesh` objects for each part and surface.

        Notes
        -----
        - The method uses a color palette to assign unique colors to the spheres
          for each part and surface.
        - The generated spheres are translated globally based on the `global_translation` parameter.
        """
        trimesh_planes = []
        color_index = 0

        for part_id, surfaces in self.system_data["PARTS"].items():
            for surf_id, surface_data in surfaces.items():
                if "POINTS" in surface_data and surface_data['TYPE']=='plane':
                    vertices = np.array(list(surface_data["POINTS"].values()))
                    color_hex = otaf.plotting.color_palette_3[
                        color_index % len(otaf.plotting.color_palette_3)
                    ]
                    color_rgba = otaf.plotting.hex_to_rgba(color_hex)

                    spheres = otaf.plotting.spheres_from_point_cloud(
                        points,
                        radius=radius,
                        color=color_rgba,
                        global_translation=global_translation,
                    )
                    sphere_clouds.extend(spheres)

                    color_index += 1

        return sphere_clouds


    def get_notebook_scene_sphere_clouds(self, radius=0.5, background_hex_color="e6e6e6"):
        """
        Create a 3D notebook scene with sphere clouds representing the system's points.

        This method generates a 3D scene containing spheres for the system's points and
        renders it as a scene compatible with Jupyter Notebook.

        Parameters
        ----------
        radius : float, optional
            The radius of the spheres in the scene (default is 0.5).
        background_hex_color : str, optional
            The hexadecimal color code for the scene's background (default is "e6e6e6").

        Returns
        -------
        scene
            A scene object rendered in a format suitable for Jupyter Notebook.
        """
        sphere_list = self.generate_sphere_clouds(radius=radius)
        scene = tr.Scene([*sphere_list])
        return otaf.plotting.trimesh_scene_as_notebook_scene(scene, background_hex_color)
