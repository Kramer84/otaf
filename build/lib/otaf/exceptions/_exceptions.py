# -*- coding: utf-8 -*-
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
    "InvalidAffineTransformException",
]

import otaf
import numpy as np


class MissingSurfaceTypeKeyError(KeyError):
    def __init__(self, part_id, surf_id):
        super().__init__(
            f"Surface type key 'TYPE' is missing for surface {surf_id} in part {part_id}."
        )


class MissingKeyError(KeyError):
    def __init__(self, key, dictionary_name):
        super().__init__(f"Required key '{key}' is missing in '{dictionary_name}' dictionary.")


class InvalidPartLabelError(ValueError):
    def __init__(self):
        super().__init__("All parts must be labelled by the string representation of an integer.")


class InvalidSurfaceLabelError(ValueError):
    def __init__(self):
        super().__init__("All surfaces must be labelled by a lowercase alphabetical character.")


class InvalidInteractionFormatError(ValueError):
    def __init__(self, interaction: str):
        message = f"The interaction '{interaction}' does not match the expected format. Ensure it conforms to the required regex pattern."
        super().__init__(message)


class InvalidSurfaceDirectionError(ValueError):
    def __init__(self, surf_dir: str):
        message = f"The surface direction '{surf_dir}' is not one of the expected ones: {otaf.constants.SURFACE_DIRECTIONS}."
        super().__init__(message)


class UnsupportedSurfaceTypeError(ValueError):
    def __init__(self, part_id, surf_id, surf_type):
        message = f"The type {surf_type} defined for part {part_id} on surface {surf_id} is not a standard surface type and not supported.\n Standard types: {otaf.BASE_SURFACE_TYPES}"
        super().__init__(message)


class InvalidGlobalConstraintError(ValueError):
    def __init__(self, valid_constraints):
        message = f"The global constraint can only be one the following: {list(valid_constraints)}"
        super().__init__(message)


class PointDictionaryError(ValueError):
    def __init__(self, message):
        super().__init__(message)


class MissingOriginPointError(PointDictionaryError):
    def __init__(self, part_id, surf_id):
        message = f"A point dictionary has been passed for part {part_id} surface {surf_id} but does not contain a point for the surface origin."
        super().__init__(message)


class NonUniqueLabelError(PointDictionaryError):
    def __init__(self, message):
        super().__init__(message)


class PartNotFoundError(ValueError):
    def __init__(self, part_id):
        super().__init__(f"Part ID {part_id} not found.")


class SurfaceNotFoundError(ValueError):
    def __init__(self, surf_id, part_id):
        super().__init__(f"Surface ID {surf_id} not found in Part ID {part_id}.")


class DuplicatePointError(ValueError):
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
            value_message = (
                f" New value: {formatted_new_value}. Existing value: {formatted_existing_value}."
            )

        if exact_match:
            message = f"Duplicate point name '{point_name}' for surface {surf_id}, part {part_id}.\n{value_message}"
        else:
            message = (
                f"Point '{point_name}' conflicts with existing point '{existing_point}' "
                f"on surface {surf_id}, part {part_id} due to similar coordinates.\n{value_message}"
            )

        super().__init__(message)


class LabelPatternError(ValueError):
    def __init__(self):
        super().__init__("Some labels in the point dict do not comply to the expected pattern.")


class LabelPrefixError(ValueError):
    def __init__(self):
        super().__init__(
            "Some labels in the point dict do not comply to the expected pattern, all prefixes should be identical."
        )


class UniqueLabelSuffixError(ValueError):
    def __init__(self):
        super().__init__(
            "Some labels in the point dict do not comply to the expected pattern, all integer suffixes must be unique. Here A000 and A0 would be identical, and cannot be used together."
        )


class NonUniqueCoordinatesError(ValueError):
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
    def __init__(self, condition_name):
        super().__init__(f"Geometric condition '{condition_name}' is not met.")


class PointsNotOnPlaneError(Exception):
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


class InvalidAffineTransformException(Exception):
    """
    Exception raised when a matrix is not a valid affine transformation matrix.

    Attributes:
        matrix (np.ndarray): The matrix that caused the exception.
        message (str): Explanation of the error.
    """

    def __init__(self, matrix, message="Matrix is not a valid affine transformation matrix."):
        self.matrix = matrix
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}\nMatrix:\n{self.matrix}"
