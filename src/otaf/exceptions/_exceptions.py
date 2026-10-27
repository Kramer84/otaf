from __future__ import annotations

__author__ = "Kramer84"
__all__ = [
    "MissingSurfaceTypeKeyError",
    "MissingKeyError",
    "InvalidPartLabelError",
    "InvalidSurfaceLabelError",
    "UnsupportedSurfaceTypeError",
    "InvalidInteractionFormatError",
    "InvalidSurfaceDirectionError",
    "InvalidGlobalConstraintError",
    "PointDictionaryError",
    "MissingOriginPointError",
    "NonUniqueLabelError",
    "PartNotFoundError",
    "SurfaceNotFoundError",
    "DuplicatePointError",
    "LabelPatternError",
    "LabelPrefixError",
    "UniqueLabelSuffixError",
    "NonUniqueCoordinatesError",
    "GeometricConditionError",
    "PointsNotOnPlaneError",
    "NonConcentricCylindersException",
    "ConflictingSurfaceDirectionsException",
    "CylindricalInterferenceGeometricException",
    "NoAxialOverlapException",
    "InvalidAffineTransformException",
    "_raise_missing_dependency",
]
import numpy as np

from otaf.constants import BASE_SURFACE_TYPES, SURFACE_DIRECTIONS


def _raise_missing_dependency(library_name, method_name):
    raise ImportError(
        f"The '{library_name}' library is required for '{method_name}'. Please install it with `pip install {library_name}`."
    )


class MissingSurfaceTypeKeyError(KeyError):
    """Raised when a surface has no ``TYPE`` key.

    Parameters
    ----------
    part_id : str
        ID of the part.
    surf_id : str
        ID of the surface.
    """

    def __init__(self, part_id, surf_id):
        super().__init__(
            f"Surface type key 'TYPE' is missing for surface {surf_id} in part {part_id}."
        )


class MissingKeyError(KeyError):
    """Raised when a required key is missing from an input dictionary.

    Parameters
    ----------
    key : str
        Name of the missing key.
    dictionary_name : str
        Name of the dictionary that should contain it.
    """

    def __init__(self, key, dictionary_name):
        super().__init__(
            f"Required key '{key}' is missing in '{dictionary_name}' dictionary."
        )


class InvalidPartLabelError(ValueError):
    """Raised when a part label is not the string form of an integer."""

    def __init__(self):
        super().__init__(
            "All parts must be labelled by the string representation of an integer."
        )


class InvalidSurfaceLabelError(ValueError):
    """Raised when a surface label is not a lowercase alphabetical string."""

    def __init__(self):
        super().__init__(
            "All surfaces must be labelled by a lowercase alphabetical character."
        )


class InvalidInteractionFormatError(ValueError):
    """Raised when an ``INTERACTIONS`` entry does not match ``P<part><surface>`` (e.g. ``P2a``).

    Parameters
    ----------
    interaction : str
        The offending interaction string.
    """

    def __init__(self, interaction: str):
        message = f"The interaction '{interaction}' does not match the expected format. Ensure it conforms to the required regex pattern."
        super().__init__(message)


class InvalidSurfaceDirectionError(ValueError):
    """Raised when ``SURFACE_DIRECTION`` is not one of ``otaf.constants.SURFACE_DIRECTIONS``.

    Parameters
    ----------
    surf_dir : str
        The offending direction.
    """

    def __init__(self, surf_dir: str):
        message = f"The surface direction '{surf_dir}' is not one of the expected ones: {SURFACE_DIRECTIONS}."
        super().__init__(message)


class UnsupportedSurfaceTypeError(ValueError):
    """Raised when a surface ``TYPE`` is not one of ``otaf.constants.BASE_SURFACE_TYPES``.

    Parameters
    ----------
    part_id : str
        ID of the part.
    surf_id : str
        ID of the surface.
    surf_type : str
        The unsupported type.
    """

    def __init__(self, part_id, surf_id, surf_type):
        message = f"The type {surf_type} defined for part {part_id} on surface {surf_id} is not a standard surface type and not supported.\n Standard types: {BASE_SURFACE_TYPES}"
        super().__init__(message)


class InvalidGlobalConstraintError(ValueError):
    """Raised when ``GLOBAL_CONSTRAINTS`` is not one of the available presets.

    Parameters
    ----------
    valid_constraints : iterable of str
        The accepted preset names.
    """

    def __init__(self, valid_constraints):
        message = f"The global constraint can only be one the following: {list(valid_constraints)}"
        super().__init__(message)


class PointDictionaryError(ValueError):
    """Base class of the errors raised for an invalid ``POINTS`` dictionary.

    Parameters
    ----------
    message : str
        Description of the problem.
    """

    def __init__(self, message):
        super().__init__(message)


class MissingOriginPointError(PointDictionaryError):
    """Raised when the ``POINTS`` dictionary of a surface has no point for the surface origin.

    Parameters
    ----------
    part_id : str
        ID of the part.
    surf_id : str
        ID of the surface.
    """

    def __init__(self, part_id, surf_id):
        message = f"A point dictionary has been passed for part {part_id} surface {surf_id} but does not contain a point for the surface origin."
        super().__init__(message)


class NonUniqueLabelError(PointDictionaryError):
    """Raised when point labels of a surface are not unique.

    Parameters
    ----------
    message : str
        Description of the problem.
    """

    def __init__(self, message):
        super().__init__(message)


class PartNotFoundError(ValueError):
    """Raised when a part ID does not exist in the assembly data.

    Parameters
    ----------
    part_id : str
        The unknown part ID.
    """

    def __init__(self, part_id):
        super().__init__(f"Part ID {part_id} not found.")


class SurfaceNotFoundError(ValueError):
    """Raised when a surface ID does not exist in a part.

    Parameters
    ----------
    surf_id : str
        The unknown surface ID.
    part_id : str
        ID of the part searched.
    """

    def __init__(self, surf_id, part_id):
        super().__init__(f"Surface ID {surf_id} not found in Part ID {part_id}.")


class DuplicatePointError(ValueError):
    """Raised when a point name or its coordinates duplicate an existing point of the surface.

    Parameters
    ----------
    point_name : str
        Name of the new point.
    surf_id : str
        ID of the surface.
    part_id : str
        ID of the part.
    existing_point : str, optional
        Name of the existing point with similar coordinates (``exact_match=False``).
    origin : bool, optional
        Whether the point is the surface origin.
    new_point_value : array_like, optional
        Coordinates of the new point, shown in the message.
    existing_point_value : array_like, optional
        Coordinates of the existing point, shown in the message.
    exact_match : bool, optional
        True when the name itself is duplicated, False when only the coordinates are.
    """

    def __init__(
        self,
        point_name,
        surf_id,
        part_id,
        existing_point=None,
        origin=False,
        new_point_value=None,
        existing_point_value=None,
        exact_match=True,
    ):
        value_message = ""
        if new_point_value is not None and existing_point_value is not None:
            formatted_new_value = np.array2string(
                np.array(new_point_value), precision=2, separator=","
            )
            formatted_existing_value = np.array2string(
                np.array(existing_point_value), precision=2, separator=","
            )
            value_message = f" New value: {formatted_new_value}. Existing value: {formatted_existing_value}."
        if exact_match:
            message = f"Duplicate point name '{point_name}' for surface {surf_id}, part {part_id}.\n{value_message}"
        else:
            message = f"Point '{point_name}' conflicts with existing point '{existing_point}' on surface {surf_id}, part {part_id} due to similar coordinates.\n{value_message}"
        super().__init__(message)


class LabelPatternError(ValueError):
    """Raised when point labels do not match the expected pattern (capital letters followed by an integer)."""

    def __init__(self):
        super().__init__(
            "Some labels in the point dict do not comply to the expected pattern."
        )


class LabelPrefixError(ValueError):
    """Raised when the point labels of a surface do not share the same letter prefix."""

    def __init__(self):
        super().__init__(
            "Some labels in the point dict do not comply to the expected pattern, all prefixes should be identical."
        )


class UniqueLabelSuffixError(ValueError):
    """Raised when integer suffixes of point labels coincide once leading zeros are removed (``A000`` and ``A0``)."""

    def __init__(self):
        super().__init__(
            "Some labels in the point dict do not comply to the expected pattern, all integer suffixes must be unique. Here A000 and A0 would be identical, and cannot be used together."
        )


class NonUniqueCoordinatesError(ValueError):
    """Raised when two points of a surface have the same coordinates.

    Parameters
    ----------
    duplicates : dict, optional
        Mapping from coordinates to the labels sharing them, shown in the message.
    """

    def __init__(self, duplicates=None):
        if duplicates is None:
            message = "All coordinates in the point dictionary must be unique."
        else:
            duplicates_str = "; ".join(
                [
                    f"for label(s) {', '.join(labels)} we have {coords}"
                    for coords, labels in duplicates.items()
                ]
            )
            message = f"All coordinates in the point dictionary must be unique. {duplicates_str}"
        super().__init__(message)


class GeometricConditionError(Exception):
    """Raised when a geometric condition required by the modelling is not met.

    Parameters
    ----------
    condition_name : str
        Name of the condition.
    """

    def __init__(self, condition_name):
        super().__init__(f"Geometric condition '{condition_name}' is not met.")


class PointsNotOnPlaneError(Exception):
    """Raised when the points given for a plane surface are not coplanar.

    Parameters
    ----------
    part_id : str
        ID of the part.
    surf_id : str
        ID of the surface.
    """

    def __init__(self, part_id, surf_id):
        super().__init__(
            f"Points for expected plane, surface {surf_id} on part {part_id} are not on same plane"
        )


class NonConcentricCylindersException(Exception):
    """Exception raised for cylinders that are not concentric."""

    def __init__(self, part_id1, surface_id1, part_id2, surface_id2):
        super().__init__(
            f"Cylinders are not concentric for: Part {part_id1}, Surface {surface_id1} and Part {part_id2}, Surface {surface_id2}"
        )


class ConflictingSurfaceDirectionsException(Exception):
    """Exception raised for conflicting surface directions in cylinder interactions."""

    def __init__(self, part_id1, surface_id1, part_id2, surface_id2):
        message = f"Conflicting surface directions between Part {part_id1}, Surface {surface_id1} and Part {part_id2}, Surface {surface_id2}"
        super().__init__(message)


class CylindricalInterferenceGeometricException(Exception):
    """Exception raised for geometric inconsistencies in cylindrical interactions."""

    def __init__(self, part_id1, surface_id1, part_id2, surface_id2, radius1, radius2):
        message = f"Geometric inconsistency or interference in cylindrical interaction: Part {part_id1}, Surface {surface_id1} (Radius: {radius1}) and Part {part_id2}, Surface {surface_id2} (Radius: {radius2})"
        super().__init__(message)


class NoAxialOverlapException(Exception):
    """Exception raised when two interacting cylinders have no physical overlap along their axis."""

    def __init__(self, part_id1, surface_id1, part_id2, surface_id2):
        message = f"No axial overlap in cylindrical interaction: Part {part_id1}, Surface {surface_id1} and Part {part_id2}, Surface {surface_id2} are separated along their axis."
        super().__init__(message)


class NoPlanarOverlapException(Exception):
    """Exception raised when two interacting planes do not overlap once projected onto each other."""

    def __init__(self, part_id1, surface_id1, part_id2, surface_id2):
        message = f"No overlap in planar interaction: Part {part_id1}, Surface {surface_id1} and Part {part_id2}, Surface {surface_id2} do not face each other over any area."
        super().__init__(message)


class InvalidAffineTransformException(Exception):
    """
    Exception raised when a matrix is not a valid affine transformation matrix.

    Attributes
    ----------
    matrix : np.ndarray
        The matrix that caused the exception.
    message : str
        Explanation of the error.
    """

    def __init__(
        self, matrix, message="Matrix is not a valid affine transformation matrix."
    ):
        self.matrix = matrix
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}\nMatrix:\n{self.matrix}"
