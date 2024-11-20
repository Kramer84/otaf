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
from ._interfaceLoopBuilder import InterfaceLoopBuilder
from ._surfaceInteractionManager import SurfaceInteractionManager

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
        gap_matrix_dict = otaf.common.tree()

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
        filtered_dict = otaf.common.tree()
        for part, surfaces in all_gap_matrices.items():
            for surface, matrix_set in surfaces.items():
                common_elements = matrix_set.intersection(existing_gap_matrices)
                if common_elements:
                    remaining_elements = matrix_set - common_elements
                    if remaining_elements:
                        filtered_dict[part][surface]["UNUSED"] = remaining_elements
                        filtered_dict[part][surface]["USED"] = common_elements
        return filtered_dict
