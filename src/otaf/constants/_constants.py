# -*- coding: utf-8 -*-
"""
_constants.py

Defines constants, regex patterns, and utility dictionaries for surface and gap modeling.

This file includes the following:
- Definitions of base surface types, directions, and contact types.
- Regular expressions for validating naming patterns of surfaces, parts, and transformations.
- Mappings for constraints, degrees of freedom (DOFs), and nullified components.
- Dictionaries for validating surface properties and basis transformations.

Example:
    >>> from otaf.constants import BASE_SURFACE_TYPES
    >>> BASE_SURFACE_TYPES
    ['plane', 'cylinder', 'cone', 'sphere']

Author: Kramer84
"""

__author__ = "Kramer84"
__all__ = [
    "BASE_SURFACE_TYPES",
    "SURFACE_DIRECTIONS",
    "CONTACT_TYPES",
    "SURF_POINT_PATTERN",
    "SURF_ORIGIN_PATTERN",
    "BASE_PART_SURF_PATTERN",
    "LOOP_ELEMENT_PATTERN",
    "LOC_STAMP_PATTERN",
    "T_MATRIX_PATTERN",
    "D_MATRIX_PATTERN1",
    "D_MATRIX_PATTERN2",
    "D_MATRIX_PATTERN3",
    "D_MATRIX_MSTRING_PATTERN",
    "G_MATRIX_PATTERN1",
    "G_MATRIX_PATTERN2",
    "G_MATRIX_MSTRING_PATTERN",
    "GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF",
    "SURF_TYPE_TO_DEVIATION_DOF",
    "GLOBAL_CONSTRAINTS_TO_GAP_DOF",
    "CONTACT_TYPE_TO_GAP_DOF",
    "GAP_TYPE_TO_NULLIFIED_NOMINAL_COMPONENTS",
    "BASIS_DICT",
    "SURFACE_DICT_VALUE_CHECKS",
]

import re
import sympy as sp
import numpy as np
from beartype import beartype
from beartype.typing import List, Union


# -----------------------------------------------------------------------------
# Constant Types and Descriptions
# -----------------------------------------------------------------------------

BASE_SURFACE_TYPES = ["plane", "cylinder", "cone", "sphere"]
"""List of supported surface types for modeling."""

SURFACE_DIRECTIONS = ["centripetal", "centrifugal"]
"""List of valid surface directions for interactions."""

CONTACT_TYPES = ["FIXED", "SLIDING", "FLOATING"]
"""Types of contact interactions between surfaces."""


# -----------------------------------------------------------------------------
# Regular Expression Patterns
# -----------------------------------------------------------------------------

SURF_POINT_PATTERN = re.compile(r"[A-Z]+[0-9]+$")
"""Regex to validate surface point names, e.g., 'AA01'."""

SURF_ORIGIN_PATTERN = re.compile(r"[A-Z]+0+$")
"""Regex to validate surface origin names, e.g., 'AA0000'."""

# Patterns for parts and surfaces
BASE_PART_SURF_PATTERN = re.compile(r"^P(\d+)([a-z]+)$")
"""Regex for part-surface naming pattern, e.g., 'P1a'."""

LOOP_ELEMENT_PATTERN = re.compile(r"^P\d+[a-z]+[A-Z]+\d+$")
"""Regex for loop element names, e.g., 'P1aA5'."""

LOC_STAMP_PATTERN = re.compile(r"^P(\d+)([a-z]+)([A-Z]+\d+)$")
"""Regex for location stamp naming, e.g., 'P1aA5'."""

# Patterns for transformation and deviation matrices
T_MATRIX_PATTERN = re.compile(r"^TP(\d+)([a-z]+)([A-Z]+\d+)([a-z]+)([A-Z]+\d+)$")
"""Regex for transformation matrix naming, e.g., 'TP1aA0bB99'."""

D_MATRIX_PATTERN1 = re.compile(r"^D(i*)(\d+)([a-z]+)$")
"""Regex for deviation matrix type 1, e.g., 'D1a' or 'Di1a'."""

D_MATRIX_PATTERN2 = re.compile(r"^D(i*)(\d+)([a-z]+)([A-Z]+\d+)$")
"""Regex for deviation matrix type 2, e.g., 'D1aA5'."""

D_MATRIX_PATTERN3 = re.compile(r"^D(i*)(\d+)([a-z]+)(\d+)([a-z]+)$")
"""Regex for deviation matrix type 3, e.g., 'D1a5b'."""

D_MATRIX_MSTRING_PATTERN = re.compile(r"^D(\d+)([a-z]+)$")
"""Regex for deviation matrix simplified naming, e.g., 'D1a'."""

# Patterns for gap matrices
G_MATRIX_PATTERN1 = re.compile(
    r"^GP(i*)(\d+)([a-z]+)([A-Z]+\d+)P(\d+)([a-z]+)([A-Z]+\d+)$"
)
"""Regex for gap matrix type 1, e.g., 'GPi1aA0P2aA0'."""

G_MATRIX_PATTERN2 = re.compile(r"^GP(i*)(\d+)([a-z]+)P(\d+)([a-z]+)$")
"""Regex for gap matrix type 2, e.g., 'GPi1aP2a'."""

G_MATRIX_MSTRING_PATTERN = re.compile(
    r"^GP(\d+)([a-z]+)([A-Z]+\d+)P(\d+)([a-z]+)([A-Z]+\d+)$"
)
"""Regex for gap matrix simplified naming, e.g., 'GP1aA0P2aA0'."""


# -----------------------------------------------------------------------------
# Mappings for Constraints and Degrees of Freedom
# -----------------------------------------------------------------------------

GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF = {
    "3D": {"translations_2remove": "", "rotations_2remove": ""},
    "2D_NX": {"translations_2remove": "x", "rotations_2remove": "yz"},
    "2D_NY": {"translations_2remove": "y", "rotations_2remove": "xz"},
    "2D_NZ": {"translations_2remove": "z", "rotations_2remove": "xy"},
}
"""Mapping global constraints to degrees of freedom for deviations."""

SURF_TYPE_TO_DEVIATION_DOF = {
    "plane-none": {"translations": "x", "rotations": "yz"},
    "plane-perfect": {"translations": "", "rotations": ""},
    "plane-translation": {"translations": "x", "rotations": ""},
    "plane-rotation": {"translations": "", "rotations": "yz"},
    "cylinder-none": {"translations": "yz", "rotations": "yz"},  # x along the centerline
    "cylinder-perfect": {"translations": "", "rotations": ""},
    "cylinder-translation": {"translations": "yz", "rotations": ""},
    "cylinder-rotation": {"translations": "", "rotations": "yz"},
}
"""Mapping surface types to degrees of freedom for deviations."""

GLOBAL_CONSTRAINTS_TO_GAP_DOF = {
    "3D": {"translations_blocked": "", "rotations_blocked": ""},
    "2D_NX": {"translations_blocked": "x", "rotations_blocked": "yz"},
    "2D_NY": {"translations_blocked": "y", "rotations_blocked": "xz"},
    "2D_NZ": {"translations_blocked": "z", "rotations_blocked": "xy"},
}
"""Mapping global constraints to blocked degrees of freedom for gaps."""

CONTACT_TYPE_TO_GAP_DOF = {
    "plane-plane-floating": {"translations_blocked": "", "rotations_blocked": ""},
    "plane-plane-sliding": {"translations_blocked": "x", "rotations_blocked": "yz"},
    "plane-plane-fixed": {"translations_blocked": "xyz", "rotations_blocked": "xyz"},
    "cylinder-cylinder-floating": {"translations_blocked": "", "rotations_blocked": ""},
    "cylinder-cylinder-sliding": {"translations_blocked": "yz", "rotations_blocked": "yz"},
    "cylinder-cylinder-fixed": {"translations_blocked": "xyz", "rotations_blocked": "xyz"},
}
"""Mapping surface types to blocked degrees of freedom for gaps."""

GAP_TYPE_TO_NULLIFIED_NOMINAL_COMPONENTS = {
    "plane-plane": {"nullify_x": True, "nullify_y": False, "nullify_z": False},
    "cylinder-cylinder": {"nullify_x": True, "nullify_y": True, "nullify_z": True},
}
"""Mapping gap type interaction to gap components that will be nullified in the nominal/perfect gap matrix"""


# -----------------------------------------------------------------------------
# Additional Dictionaries and Validation Functions
# -----------------------------------------------------------------------------

BASIS_DICT = {
    "1": {
        "MATRIX": [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "AXIS": "x",
        "VARIABLE_D": "u_d",
        "VARIABLE_G": "u_g",
    },
    "2": {
        "MATRIX": [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        "AXIS": "y",
        "VARIABLE_D": "v_d",
        "VARIABLE_G": "v_g",
    },
    "3": {
        "MATRIX": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
        "AXIS": "z",
        "VARIABLE_D": "w_d",
        "VARIABLE_G": "w_g",
    },
    "4": {
        "MATRIX": [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        "AXIS": "x",
        "VARIABLE_D": "alpha_d",
        "VARIABLE_G": "alpha_g",
    },
    "5": {
        "MATRIX": [[0, 0, 1, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0]],
        "AXIS": "y",
        "VARIABLE_D": "beta_d",
        "VARIABLE_G": "beta_g",
    },
    "6": {
        "MATRIX": [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "AXIS": "z",
        "VARIABLE_D": "gamma_d",
        "VARIABLE_G": "gamma_g",
    },
}
"""Components of the matrix basis used for modelling defects and rotations with small displacement hypothesis"""

SURFACE_DICT_VALUE_CHECKS = {
    "TYPE": lambda x: x in BASE_SURFACE_TYPES,
    "FRAME": lambda x: (
        np.allclose(np.dot(np.array(x), np.array(x).T), np.identity(3))
        and np.isclose(np.linalg.det(np.array(x)), 1)
    ),
    "ORIGIN": lambda x: np.array(x).shape == (3,),
    "INTERACTIONS": lambda xL: all(BASE_PART_SURF_PATTERN.fullmatch(x) is not None for x in xL),
}
"""Validation functions for surface property values."""
