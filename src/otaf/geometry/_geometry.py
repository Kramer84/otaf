# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "is_affine_transformation_matrix",
    "tfrt",
    "points_in_cylinder",
    "points_in_cylinder_vect",
    "plane_parameters",
    "euclidean_distance",
    "angle_between_vectors",
    "point_in_hull",
    "line_plane_intersection",
    "transform_point",
    "closest_point_on_plane",
    "project_vector_onto_plane",
    "closest_point_on_line",
    "point_plane_distance",
    "point_to_segment_distance",
    "are_planes_coincident",
    "are_planes_parallel",
    "are_planes_perpendicular",
    "are_planes_facing",
    "angle_between_planes",
    "distance_between_planes",
    "centroid",
    "are_points_on_2d_plane",
    "point_dict_to_arrays",
    "generate_circle_points",
    "generate_circle_points_3d",
    "calculate_cylinder_surface_frame",
    "compute_bounding_box",
    "is_bounding_box_within",
    "do_bounding_boxes_overlap",
    "are_normals_aligned_and_facing",
    "calculate_scalar_projection_factor",
]

import math
import numpy as np

from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Any, Set, Iterable

import otaf


@beartype
def is_affine_transformation_matrix(M: np.ndarray, raise_exception=False) -> bool:
    """
    Checks if the given 4x4 matrix is an affine transformation matrix with a valid rotation component.

    Parameters:
        M (np.ndarray): A 4x4 NumPy array representing the matrix to be checked.
        raise_exception (bool): If True, raises an exception when the matrix is invalid.
                                If False, returns False instead.

    Returns:
        bool: True if the matrix is an affine transformation matrix, False otherwise.

    Raises:
        otaf.exceptions.InvalidAffineTransformException: If the matrix is not a valid affine transformation matrix
                                         and raise_exception is True.
    """
    if M.shape != (4, 4):
        if raise_exception:
            raise otaf.exceptions.InvalidAffineTransformException(M, "Matrix is not 4x4.")
        return False

    if not np.array_equal(M[3, :], np.array([0, 0, 0, 1])):
        if raise_exception:
            raise otaf.exceptions.InvalidAffineTransformException(
                M, "Last row of the matrix is not [0, 0, 0, 1]."
            )
        return False

    P = M[:3, :3]

    if not np.allclose(np.dot(P, P.T), np.identity(3)):
        if raise_exception:
            raise otaf.exceptions.InvalidAffineTransformException(
                M, "Top-left 3x3 submatrix is not a valid rotation matrix."
            )
        return False

    if not np.isclose(np.linalg.det(P), 1):
        if raise_exception:
            raise otaf.exceptions.InvalidAffineTransformException(
                M, "Determinant of the top-left 3x3 submatrix is not 1."
            )
        return False

    return True


@beartype
def transformation_from_rotation_translation(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Create a 4x4 transformation matrix from a 3x3 rotation matrix and a translation vector.

    Args:
        rotation (np.ndarray): 3x3 rotation matrix.
        translation (np.ndarray): Translation vector.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    M = np.eye(4)
    M[:3, :3] = rotation  # .T  We need(n't) this to make it work.
    M[:3, 3] = translation
    return M


def tfrt(*args):
    """Alias for transformation_from_rotation_translation function."""
    return transformation_from_rotation_translation(*args)


@beartype
def points_in_cylinder(
    pt1: np.ndarray, pt2: np.ndarray, r: Union[float, int], q: np.ndarray
) -> bool:
    """Return True if point q is inside cylinder defined by pt1, pt2, and r. False otherwise."""
    vec = pt2 - pt1
    vec_norm = np.linalg.norm(vec)
    const = r * vec_norm
    delta_q1 = q - pt1
    delta_q2 = q - pt2
    dot1 = np.dot(delta_q1, vec)
    dot2 = np.dot(delta_q2, vec)
    cross_product_norm = np.linalg.norm(np.cross(delta_q1, vec))
    return bool(dot1 >= 0 and dot2 <= 0 and cross_product_norm / vec_norm <= r)


@beartype
def points_in_cylinder_vect(
    pt1: np.ndarray, pt2: np.ndarray, r: Union[float, int], q: np.ndarray
) -> np.ndarray:
    """Return boolean array for points in q being inside cylinder defined by pt1, pt2, and r."""
    vec = pt2 - pt1
    vec_norm = np.linalg.norm(vec)
    const = r * vec_norm
    delta_q1 = q - pt1
    delta_q2 = q - pt2
    dot1 = np.dot(delta_q1, vec)
    dot2 = np.dot(delta_q2, vec)
    cross_product_norm = np.linalg.norm(np.cross(delta_q1, vec), axis=1)
    return (dot1 >= 0) & (dot2 <= 0) & (cross_product_norm <= const)


@beartype
def plane_parameters(point_on_plane: np.ndarray, rotation_matrix: np.ndarray) -> tuple:
    """Return plane equation parameters using point and rotation matrix."""
    normal_vector_base_frame = np.dot(rotation_matrix.T, [1, 0, 0])
    distance_to_plane = -np.dot(normal_vector_base_frame, point_on_plane)
    A, B, C = normal_vector_base_frame
    return A, B, C, distance_to_plane


@beartype
def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Return Euclidean distance between point1 and point2."""
    return np.linalg.norm(point1 - point2)


# @beartype
# def angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
#    """Return angle in radians between vectors 'vec1' and 'vec2'."""
#    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#    return np.arccos(np.clip(cos_theta, -1, 1))#


@beartype
def angle_between_vectors(
    vec1: np.ndarray, vec2: np.ndarray, assume_normalized: bool = False
) -> float:
    """Return the angle in radians between vectors 'vec1' and 'vec2'."""

    if not assume_normalized:
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # Check if either vector is zero or close to zero
        if np.isclose(norm_vec1, 0.0) or np.isclose(norm_vec2, 0.0):
            raise ValueError("One or both vectors are close to zero length, angle is undefined.")

        cos_theta = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
    else:
        cos_theta = np.dot(vec1, vec2)

    return np.arccos(np.clip(cos_theta, -1, 1))


@beartype
def point_in_hull(point: np.ndarray, hull: np.ndarray) -> bool:
    """Return True if 'point' is inside 'hull'. False otherwise."""
    hull = ConvexHull(hull)
    new_hull = ConvexHull(np.concatenate([hull.points, [point]]))
    return np.allclose(hull.volume, new_hull.volume)


@beartype
def line_plane_intersection(
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
) -> np.ndarray:
    """Find the intersection of a line and a plane."""
    d = np.dot(plane_point, plane_normal)
    t = (d - np.dot(line_point, plane_normal)) / np.dot(line_direction, plane_normal)
    return line_point + t * line_direction


@beartype
def transform_point(point: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """Transform a point using a given transformation matrix."""
    return np.dot(transformation_matrix, np.append(point, 1))[:-1]


@beartype
def closest_point_on_plane(
    plane_normal: np.ndarray, plane_point: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """Return the closest point on a plane to a given point."""
    d = -np.dot(plane_point, plane_normal)
    t = -(np.dot(plane_normal, point) + d) / np.dot(plane_normal, plane_normal)
    return point + t * plane_normal


@beartype
def project_vector_onto_plane(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Project a vector onto a plane defined by its normal."""
    normalized_plane_normal = plane_normal / np.linalg.norm(plane_normal)
    proj_onto_normal = np.dot(vector, normalized_plane_normal) * normalized_plane_normal
    projection_onto_plane = vector - proj_onto_normal
    return projection_onto_plane


@beartype
def closest_point_on_line(
    line_point1: np.ndarray, line_point2: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """Return the closest point on a line segment to a given point."""
    line_vec = line_point2 - line_point1
    point_vec = point - line_point1
    t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
    return line_point1 + t * line_vec


@beartype
def point_plane_distance(
    plane_normal: np.ndarray, plane_point: np.ndarray, point: np.ndarray
) -> float:
    """Calculate the distance from a point to a plane."""
    return np.dot(plane_normal, point - plane_point) / np.linalg.norm(plane_normal)


@beartype
def point_to_segment_distance(
    point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray
) -> Union[float, np.ndarray]:
    """Calculate the shortest distance from a point or array of points to a line segment."""
    return np.linalg.norm(
        np.cross(segment_end - segment_start, point - segment_start, axis=-1), axis=-1
    ) / np.linalg.norm(segment_end - segment_start)


@beartype
def are_planes_coincident(
    normal1: np.ndarray,
    point1: np.ndarray,
    normal2: np.ndarray,
    point2: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """
    Check if two planes are coincident within a specified tolerance.

    Args:
        normal1 (np.ndarray): The normal vector of the first plane.
        point1 (np.ndarray): A point on the first plane.
        normal2 (np.ndarray): The normal vector of the second plane.
        point2 (np.ndarray): A point on the second plane.
        tolerance (float): The tolerance within which the planes are considered coincident.

    Returns:
        bool: True if planes are coincident within the tolerance, False otherwise.
    """
    # Normalize the normal vectors
    norm1 = normal1 / np.linalg.norm(normal1)
    norm2 = normal2 / np.linalg.norm(normal2)
    # Check if the normalized normals are parallel (same or opposite direction)
    if not np.allclose(norm1, norm2, atol=tolerance) and not np.allclose(
        norm1, -norm2, atol=tolerance
    ):
        return False
    # Distance of point2 from the plane formed by norm1 and point1
    distance = np.abs(np.dot(norm1, point2 - point1) / np.linalg.norm(norm1))
    return distance < tolerance


@beartype
def are_planes_parallel(normal1: np.ndarray, normal2: np.ndarray, tolerance: float = 1e-8) -> bool:
    """Check if two planes are parallel within a tolerance."""
    return bool(np.allclose(np.abs(np.dot(normal1, normal2)), 1, atol=tolerance))


@beartype
def are_planes_perpendicular(
    normal1: np.ndarray, normal2: np.ndarray, tolerance: float = 1e-8
) -> bool:
    """Check if two planes are perpendicular within a tolerance."""
    return bool(np.allclose(np.dot(normal1, normal2), 0, atol=tolerance))


@beartype
def are_planes_facing(
    normal1: np.ndarray,
    point1: np.ndarray,
    normal2: np.ndarray,
    point2: np.ndarray,
    atol: float = 1e-9,
    max_angle: float = 0.26,
) -> bool:
    """
    Check if translating the origin of each plane along its normal intersects the other plane.

    Args:
        normal1 (np.ndarray): Normal of the first plane.
        point1 (np.ndarray): A point on the first plane.
        normal2 (np.ndarray): Normal of the second plane.
        point2 (np.ndarray): A point on the second plane.
        atol (float): Absolute tolerance for the intersection check.

    Returns:
        bool: True if translation along normals of each plane intersects the other plane, False otherwise.
    """

    if not (
        math.pi - max_angle < angle_between_vectors(normal1, normal2) < math.pi + max_angle
    ):  # Approx 15°
        return False
    # Calculate the vector between the two points
    point_vector = point2 - point1
    if np.linalg.norm(point_vector) > atol:
        return bool(np.dot(point_vector, normal1) > atol and np.dot(-point_vector, normal2) > atol)
    else:
        return True


@beartype
def angle_between_planes(normal1: np.ndarray, normal2: np.ndarray) -> float:
    """Calculate the angle between two planes."""
    return np.arccos(np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2)))


@beartype
def distance_between_planes(
    normal1: np.ndarray, point1: np.ndarray, normal2: np.ndarray, point2: np.ndarray
) -> float:
    """Calculate the distance between two planes, returns 0 if they intersect."""
    if are_planes_parallel(normal1, normal2):
        return abs(np.dot(normal1, (point1 - point2))) / np.linalg.norm(normal1)
    else:
        return 0.0  # Planes intersect


@beartype
def centroid(arr: np.ndarray) -> np.ndarray:
    """Calculate the centroid of a numpy array."""
    sum_vals = np.sum(arr, axis=0)
    length = arr.shape[0]
    centroid = sum_vals / length
    return centroid


@beartype
def are_points_on_2d_plane(
    points: np.ndarray, return_normal: bool = False
) -> Union[Tuple[bool, np.ndarray], bool]:
    """Check if all points are on the same 2D plane.

    Parameters:
        points (np.ndarray): An array of points (NxD) where N is the number of points and D is the dimension.

    Returns:
        bool: True if all points are on the same 2D plane, False otherwise."""
    if len(points) < 3:
        raise ValueError("At least 3 points are required to determine a plane.")

    p0, p1, p2 = points[:3]
    normal_vector = np.cross(p1 - p0, p2 - p0)
    dot_products = np.dot(points - p0, normal_vector)
    if return_normal:
        return np.allclose(dot_products, 0), normal_vector
    return np.allclose(dot_products, 0)


@beartype
def point_dict_to_arrays(point_dict: Dict[str, Union[np.ndarray, Tuple, List]]):
    try:
        label_array = np.array(list(point_dict.keys()), dtype=str)
        point_array = np.stack(list(point_dict.values()), dtype="float64")
    except Exception as e:
        print(f"Error converting to arrays for point_dict: {point_dict}")
        raise e
    return label_array, point_array


@beartype
def generate_circle_points(
    radius: Union[float, int],
    num_points: int,
    center: Iterable[Union[float, int]] = (0, 0),
    start_angle: Union[float, int] = 0,
    rnd: int = 9,
) -> np.ndarray:
    """Generates N points around a circle with a given starting angle.

    Args:
        radius (float): Radius of the circle.
        num_points (int): Number of points to generate.
        center (Tuple[float, float], optional): (x, y) coordinates of the center of the circle. Defaults to (0, 0).
        start_angle (float, optional): Starting angle in degrees, where 0 degrees is along the positive x-axis. Defaults to 0.

    Returns:
        np.ndarray: Numpy array of shape (N, 2), where each row is an (x, y) point.
    """
    start_angle_rad = np.deg2rad(start_angle)
    angles = np.linspace(start_angle_rad, start_angle_rad + 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.array(np.column_stack((x, y)), dtype="float64").round(rnd)


@beartype
def generate_circle_points_3d(
    radius: Union[float, int],
    num_points: int,
    center: Iterable[Union[float, int]] = (0, 0, 0),
    normal: Iterable[Union[float, int]] = (0, 0, 1),
    start_angle: Union[float, int] = 0,
) -> np.ndarray:
    """Generates N points around a circle in 3D space with a given starting angle.

    Args:
        radius (float): Radius of the circle.
        num_points (int): Number of points to generate.
        center (Iterable[Union[float, int]]): (x, y, z) coordinates of the center of the circle. Defaults to (0, 0, 0).
        normal (Iterable[Union[float, int]]): Normal vector of the circle's plane. Defaults to (0, 0, 1).
        start_angle (float): Starting angle in degrees, where 0 degrees is along the positive x-axis. Defaults to 0.

    Returns:
        np.ndarray: Numpy array of shape (N, 3), where each row is an (x, y, z) point.
    """
    # Create a circle in the XY-plane
    start_angle_rad = np.deg2rad(start_angle)
    angles = np.linspace(start_angle_rad, start_angle_rad + 2 * np.pi, num_points, endpoint=False)
    circle_points = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.zeros(num_points))
    )

    # Calculate the rotation from the XY-plane to the plane defined by the normal vector
    rotation_vector = np.cross([0, 0, 1], normal)
    rotation_angle = np.arccos(np.dot([0, 0, 1], normal) / np.linalg.norm(normal))
    rotation = Rotation.from_rotvec(
        rotation_angle * rotation_vector / np.linalg.norm(rotation_vector)
    )
    rotated_points = rotation.apply(circle_points)

    # Translate the points to the specified center
    return np.array(rotated_points + np.array(center), dtype="float64").round(9)


@beartype
def calculate_cylinder_surface_frame(
    axis_translation: float, axis_rotation: float, radius: float, use_interior_normal: bool = False
) -> np.ndarray:
    """Calculate transformation matrix to position a frame on the cylinder surface.

    Args:
        axis_translation (float): Displacement along the cylinder's axis.
        axis_rotation (float): Rotation angle around the axis (in radians).
        radius (float): Radius of the cylinder.

    Returns:
        np.ndarray: Transformation matrix for the cylinder surface frame.
    """
    # Check for radius value
    if radius < 0:
        raise ValueError("Radius must be non-negative.")
    # Check for reasonable rotation angle
    if not -np.pi * 2 <= axis_rotation <= np.pi * 2:
        raise ValueError("Rotation angle should be within -2π to 2π for stability.")

    # Check for extremely large translations
    if abs(axis_translation) > 1e6:  # 1e6 is an example threshold
        raise ValueError("Axis translation is too large.")

    cos_theta = np.cos(axis_rotation)
    sin_theta = np.sin(axis_rotation)

    translation_x = np.array(
        [[1, 0, 0, axis_translation], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    rotation_x = np.array(
        [[1, 0, 0, 0], [0, cos_theta, -sin_theta, 0], [0, sin_theta, cos_theta, 0], [0, 0, 0, 1]]
    )

    translation_y = np.array([[1, 0, 0, 0], [0, 1, 0, radius], [0, 0, 1, 0], [0, 0, 0, 1]])

    if use_interior_normal:
        rotation_z = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        rotation_z = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Combine all transformation matrices efficiently
    # result_frame = rotation_z @ translation_y @ translation_x @ rotation_x
    result_frame = np.linalg.multi_dot([rotation_x, translation_x, translation_y, rotation_z])

    return result_frame


@beartype
def compute_bounding_box(points: np.ndarray) -> np.ndarray:
    """
    Computes the bounding box for a set of multi-dimensional points.

    Args:
        points (np.ndarray): A NumPy array of points, shape (N, M).

    Returns:
        np.ndarray: The bounding box, shape (M, 2), with min and max values per dimension.
    """
    return np.array([points.min(axis=0), points.max(axis=0)]).T


@beartype
def is_bounding_box_within(bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
    """
    Checks if one bounding box is completely contained within another.

    Args:
        bbox1 (np.ndarray): The first bounding box, shape (M, 2).
        bbox2 (np.ndarray): The second bounding box, shape (M, 2).

    Returns:
        bool: True if bbox1 is completely within bbox2, False otherwise.
    """
    return np.all(bbox1[:, 0] >= bbox2[:, 0]) and np.all(bbox1[:, 1] <= bbox2[:, 1])


@beartype
def do_bounding_boxes_overlap(bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
    """
    Checks if two multidimensional bounding boxes overlap.

    Args:
        bbox1 (np.ndarray): The first bounding box, shape (M, 2).
        bbox2 (np.ndarray): The second bounding box, shape (M, 2).

    Returns:
        bool: True if the bounding boxes overlap, False otherwise.
    """
    return np.all(np.maximum(bbox1[:, 0], bbox2[:, 0]) <= np.minimum(bbox1[:, 1], bbox2[:, 1]))


@beartype
def are_normals_aligned_and_facing(
    normal1: np.ndarray,
    point1: np.ndarray,
    normal2: np.ndarray,
    point2: np.ndarray,
    atol: float = 1e-9,
) -> bool:
    """
    Check if two vectors are opposed and facing each other.

    Vectors are considered to be facing each other if they are aligned in
    opposite directions and if extending them would cause them to intersect.

    Args:
        normal1 (np.ndarray): The first normal.
        point1 (np.ndarray): The point from which the first normal originates.
        normal2 (np.ndarray): The second normal.
        point2 (np.ndarray): The point from which the second normal originates.
        tolerance (float): Tolerance for considering normals as opposed.

    Returns:
        bool: True if normals are opposed and facing each other, False otherwise.
    """
    if not math.isclose(abs(np.dot(normal1, normal2)), 1):
        return False
    point_vector = point2 - point1
    return bool(np.dot(point_vector, normal1) * np.dot(point_vector, normal2) <= atol)


@beartype
def calculate_scalar_projection_factor(
    vector_to_project: np.ndarray, reference_vector: np.ndarray
) -> float:
    """
    Calculate the scalar projection factor of one vector onto another vector.

    Args:
        vector_to_project (np.ndarray): The vector to be projected.
        reference_vector (np.ndarray): The vector onto which the first vector is projected.

    Returns:
        float: The scalar projection factor.

    Explanation:
        The scalar projection factor represents the length of the projection of the
        vector_to_project onto the reference_vector. It is calculated as the dot
        product of the two vectors divided by the square of the length of the
        reference_vector.
    """
    return np.dot(vector_to_project, reference_vector) / np.linalg.norm(reference_vector) ** 2
