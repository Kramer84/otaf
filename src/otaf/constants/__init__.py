"""Constants, regex patterns, and utilities for surface and gap modeling.

This module defines base surface types, directions, contact types, and
regular expressions for validating naming patterns of surfaces and parts.
It also includes mappings for constraints and degrees of freedom (DOFs).

Examples
--------
>>> from otaf.constants import BASE_SURFACE_TYPES
>>> print(BASE_SURFACE_TYPES)
['plane', 'cylinder', 'cone', 'sphere']
"""

from __future__ import annotations
# -*- coding: utf-8 -*-

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
import numpy as np
# -----------------------------------------------------------------------------
# Constant Types and Descriptions
# -----------------------------------------------------------------------------

BASE_SURFACE_TYPES = ["plane", "cylinder", "cone", "sphere"]
"""List of supported surface types for geometric modeling.

Accepted types include ``'plane'``, ``'cylinder'``, ``'cone'``, and
``'sphere'``. Only ``'plane'`` and ``'cylinder'`` are currently 
supported for interactions.
"""

SURFACE_DIRECTIONS = ["centripetal", "centrifugal"]
"""List of valid surface orientation vectors for revolutionary surfaces."""

CONTACT_TYPES = ["FIXED", "SLIDING", "FLOATING"]
"""List of cinematic contact interaction types between mating parts."""


# -----------------------------------------------------------------------------
# Regular Expression Patterns
# -----------------------------------------------------------------------------

SURF_POINT_PATTERN = re.compile(r"[A-Z]+[0-9]+$")
"""Regex to validate surface point names, e.g., ``AA01``."""

SURF_ORIGIN_PATTERN = re.compile(r"[A-Z]+0+$")
"""Regex to validate surface origin names, e.g., ``AA0000``."""

# Patterns for parts and surfaces
BASE_PART_SURF_PATTERN = re.compile(r"^P(\d+)([a-z]+)$")
"""Regex pattern verifying structural part-surface tokens.

Matches naming schemes such as ``P1a``, capturing the target part index and the
surface identifier string.
"""

LOOP_ELEMENT_PATTERN = re.compile(r"^P\d+[a-z]+[A-Z]+\d+$")
"""Regex to validate kinematic loop element identifiers (e.g., ``P1aA5``)."""

LOC_STAMP_PATTERN = re.compile(r"^P(\d+)([a-z]+)([A-Z]+\d+)$")
"""Regex to validate positioning location stamp tokens (e.g., ``P1aA5``)."""

# Patterns for transformation and deviation matrices
T_MATRIX_PATTERN = re.compile(r"^TP(\d+)([a-z]+)([A-Z]+\d+)([a-z]+)([A-Z]+\d+)$")
"""Regex for coordinate transformation matrix matching (e.g.,
``TP1aA0bB99``).
"""

D_MATRIX_PATTERN1 = re.compile(r"^D(i*)(\d+)([a-z]+)$")
"""Regex for deviation matrix type 1, e.g., ``D1a`` or ``Di1a``."""

D_MATRIX_PATTERN2 = re.compile(r"^D(i*)(\d+)([a-z]+)([A-Z]+\d+)$")
"""Regex for deviation matrix type 2, e.g., ``D1aA5``."""

D_MATRIX_PATTERN3 = re.compile(r"^D(i*)(\d+)([a-z]+)(\d+)([a-z]+)$")
"""Regex for deviation matrix type 3, e.g., ``D1a5b``."""

D_MATRIX_MSTRING_PATTERN = re.compile(r"^D(\d+)([a-z]+)$")
"""Regex for deviation matrix simplified naming, e.g., ``D1a``."""

# Patterns for gap matrices
G_MATRIX_PATTERN1 = re.compile(
    r"^GP(i*)(\d+)([a-z]+)([A-Z]+\d+)P(\d+)([a-z]+)([A-Z]+\d+)$"
)
"""Regex for gap matrix type 1, e.g., ``GPi1aA0P2aA0``."""

G_MATRIX_PATTERN2 = re.compile(r"^GP(i*)(\d+)([a-z]+)P(\d+)([a-z]+)$")
"""Regex for gap matrix type 2, e.g., ``GPi1aP2a``."""

G_MATRIX_MSTRING_PATTERN = re.compile(
    r"^GP(\d+)([a-z]+)([A-Z]+\d+)P(\d+)([a-z]+)([A-Z]+\d+)$"
)
"""Regex for gap matrix simplified naming, e.g., ``GP1aA0P2aA0``."""


# -----------------------------------------------------------------------------
# Mappings for Constraints and Degrees of Freedom
# -----------------------------------------------------------------------------

GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF = {
    "3D": {"translations_2remove": "", "rotations_2remove": ""},
    "2D_NX": {"translations_2remove": "x", "rotations_2remove": "yz"},
    "2D_NY": {"translations_2remove": "y", "rotations_2remove": "xz"},
    "2D_NZ": {"translations_2remove": "z", "rotations_2remove": "xy"},
}
"""Mapping from dimensional constraints to dropped deviation DOFs.

Structure
---------
dict of str to dict of str to str
    Maps dimensional constraints (e.g., ``'2D_NX'``) to target strings
    identifying coordinate axes removed from calculations.

:meta hide-value:
"""

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
"""Mapping from feature topology combinations to active deviation DOFs.

Structure
---------
dict of str to dict of str to str
    Maps joint feature configurations (e.g., ``'plane-none'``) to active
    ``'translations'`` and ``'rotations'`` coordinate axis symbols.

:meta hide-value:
"""

GLOBAL_CONSTRAINTS_TO_GAP_DOF = {
    "3D": {"translations_blocked": "", "rotations_blocked": ""},
    "2D_NX": {"translations_blocked": "x", "rotations_blocked": "yz"},
    "2D_NY": {"translations_blocked": "y", "rotations_blocked": "xz"},
    "2D_NZ": {"translations_blocked": "z", "rotations_blocked": "xy"},
}
"""Mapping from system spatial limits to constrained clearance gap DOFs.

Structure
---------
dict of str to dict of str to str
    Maps systemic dimensions to coordinate strings specifying blocked linear
    and rotational components.

:meta hide-value:
"""

CONTACT_TYPE_TO_GAP_DOF = {
    "plane-plane-floating": {"translations_blocked": "", "rotations_blocked": ""},
    "plane-plane-sliding": {"translations_blocked": "x", "rotations_blocked": "yz"},
    "plane-plane-fixed": {"translations_blocked": "xyz", "rotations_blocked": "xyz"},
    "cylinder-cylinder-floating": {"translations_blocked": "", "rotations_blocked": ""},
    "cylinder-cylinder-sliding": {"translations_blocked": "yz", "rotations_blocked": "yz"},
    "cylinder-cylinder-fixed": {"translations_blocked": "xyz", "rotations_blocked": "xyz"},
}
"""Mapping from local interface boundary conditions to locked clearance DOFs.

Structure
---------
dict of str to dict of str to str
    Maps localized contact pairs (e.g., ``'plane-plane-sliding'``) to locked
    directional components.

:meta hide-value:
"""

GAP_TYPE_TO_NULLIFIED_NOMINAL_COMPONENTS = {
    "plane-plane": {"nullify_x": True, "nullify_y": False, "nullify_z": False},
    "cylinder-cylinder": {"nullify_x": True, "nullify_y": True, "nullify_z": True},
}
"""Mapping from gap topology variants to inactive perfect matrix items.

Structure
---------
dict of str to dict of str to bool
    Determines whether individual translational tracking axes must be
    nullified within nominal configuration models.

:meta hide-value:
"""


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
"""Matrix generators forming the mathematical $SE(3)$ displacement basis.

Provides projections, axes, and symbolic variable associations mapping minor
variational transformations under small-displacement hypotheses.

Structure
---------
dict of str to dict
    Indexed by coordinate transformation entry tags (``'1'`` through
    ``'6'``). Inner components hold:

    - ``'MATRIX'`` : list of list of int
      The coordinate translation or rotation generator step layout block.
    - ``'AXIS'`` : str
      The target axis identifier symbol (``'x'``, ``'y'``, or ``'z'``).
    - ``'VARIABLE_D'`` : str
      The designated notation for tracking positional part deviation metrics.
    - ``'VARIABLE_G'`` : str
      The designated notation for tracking feature clearance gap metrics.

:meta hide-value:
"""

SURFACE_DICT_VALUE_CHECKS = {
    "TYPE": lambda x: x in BASE_SURFACE_TYPES,
    "FRAME": lambda x: (
        np.allclose(np.dot(np.array(x), np.array(x).T), np.identity(3))
        and np.isclose(np.linalg.det(np.array(x)), 1)
    ),
    "ORIGIN": lambda x: np.array(x).shape == (3,),
    "INTERACTIONS": lambda xL: all(BASE_PART_SURF_PATTERN.fullmatch(x) is not None for x in xL),
}
"""Validation lambdas checking individual parameter values for surfaces.

Structure
---------
dict of str to callable
    Maps metadata validation tokens (``'TYPE'``, ``'FRAME'``, ``'ORIGIN'``,
    ``'INTERACTIONS'``) to diagnostic execution rules checking conformance.

:meta hide-value:
"""
