# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = ["SystemDataAugmented"]

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
class SystemDataAugmented:
    """
    This class manages the representation and integrity of mechanical system data,
    particularly focusing on the expansion of compact loop descriptions and validation
    of mechanical system elements.

    Attributes:
        system_data (dict): Nested dictionaries representing system data.
        compatibility_loops_expanded (dict): Contains the complete form of compatibility loops.
    """

    def __init__(self, system_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None) -> None:
        self.system_data = (
            system_data if system_data is not None else self._initialize_empty_system_data()
        )
        self.validate_system_data_structure()
        self.compatibility_loops_expanded = None

    def __getitem__(self, key):
        return self.system_data[key]

    def __setitem__(self, key, val):
        self.system_data.__setitem__(key, val)

    def __repr__(self):
        return self.system_data.__repr__()

    def get_surface_points(self, part_id: str, surf_id: str):
        """Returns a copy of a set of surface points"""
        try:
            return copy(self["PARTS"][part_id][surf_id].setdefault("POINTS", {}))
        except KeyError as e:
            raise e

    def _initialize_empty_system_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        logging.info("No system data dictionary wa passed, initializing empty one")
        return {"PARTS": {}, "LOOPS": {"COMPATIBILITY": {}}, "GLOBAL_CONSTRAINTS": "3D"}

    def validate_system_data_structure(self):
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
        Main method to generate points for a surface based on its characteristics.
        Dispatches to specific surface type methods.
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
        Generates points for a plane surface.
        Handles different definitions like contour arrays or extents.
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
        Generates points for a cylindrical surface based on origin, extent, etc.
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
        """Generate the complete form of the compatibility loops and stores them in an attribute."""
        self.compatibility_loops_expanded = {
            loop_name: self._expand_loop(compact_loop_str)
            for loop_name, compact_loop_str in self.system_data["LOOPS"]["COMPATIBILITY"].items()
        }

    def _expand_loop(self, compact_loop_str: str) -> str:
        """
        Expands a compact loop string into its full representation. The loop string is a compact
        description of a series of transformations and deviations in a mechanical system.

        Args:
            compact_loop_str (str): A string representing the loop in compact form.

        Returns:
            str: The complete for of the loop as a string.

        Raises:
            ValueError: If any of the loop elements is invalid.
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
        """Check if the loop element matches the expected pattern."""
        if not otaf.constants.LOOP_ELEMENT_PATTERN.fullmatch(loop_element):
            raise ValueError(f"Invalid loop element: {loop_element}")

    def _check_loop_order(self, part_surf_point: List[Tuple[str, str, str]]) -> bool:
        """
        Checks the order of loop elements, to ensure the first transformation is inter-part.

        Args:
            part_surf_point (List[Tuple[str, str, str]]): A list of tuples describing part-surface-point combinations.

        Returns:
            bool: True if the first transformation is inter-part, False otherwise.

        Raises:
            ValueError: If the loop order is invalid.
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
        """Generate the main expanded loop sequence."""
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
        """Handle special cases for loop closure."""
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
        """Generates sphere representations for each set of points in the system data.

        Returns:
            list: A list of trimesh sphere objects for each part and surface.
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

    def get_notebook_scene_sphere_clouds(self, radius=0.5, background_hex_color="e6e6e6"):
        sphere_list = self.generate_sphere_clouds(radius=radius)
        scene = tr.Scene([*sphere_list])
        return otaf.plotting.trimesh_scene_as_notebook_scene(scene, background_hex_color)
