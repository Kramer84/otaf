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
    Check if a 4x4 matrix is a valid affine transformation matrix with a valid rotation component.

    An affine transformation matrix must meet the following criteria:
    - It is a 4x4 matrix.
    - Its last row is `[0, 0, 0, 1]`.
    - The top-left 3x3 submatrix is a valid rotation matrix (orthonormal and determinant equals 1).

    Parameters
    ----------
    M : numpy.ndarray
        A 4x4 NumPy array representing the matrix to be checked.
    raise_exception : bool, optional
        If True, raises an exception when the matrix is invalid. If False, returns False instead (default is False).

    Returns
    -------
    bool
        True if the matrix is an affine transformation matrix, False otherwise.

    Raises
    ------
    otaf.exceptions.InvalidAffineTransformException
        If the matrix is not a valid affine transformation matrix and `raise_exception` is True.

    Notes
    -----
    - The validity of the rotation matrix is checked using orthonormality and determinant conditions.
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
    """
    Create a 4x4 transformation matrix from a 3x3 rotation matrix and a translation vector.

    The transformation matrix has the following structure:
        [ R | t ]
        [---|---]
        [ 0 | 1 ]
    where R is the 3x3 rotation matrix, and t is the translation vector.

    Parameters
    ----------
    rotation : numpy.ndarray
        A 3x3 NumPy array representing the rotation matrix.
    translation : numpy.ndarray
        A 1D NumPy array representing the translation vector.

    Returns
    -------
    numpy.ndarray
        A 4x4 NumPy array representing the transformation matrix.

    Notes
    -----
    - The input rotation matrix is assumed to be valid (orthonormal and determinant equals 1).
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
    """
    Check if a point is inside a cylinder defined by two points and a radius.

    The cylinder is defined by its axis (the line segment between `pt1` and `pt2`) and its radius `r`.
    This function checks if the given point `q` lies inside this cylinder.

    Parameters
    ----------
    pt1 : numpy.ndarray
        A 1D NumPy array representing one endpoint of the cylinder's axis.
    pt2 : numpy.ndarray
        A 1D NumPy array representing the other endpoint of the cylinder's axis.
    r : float or int
        The radius of the cylinder.
    q : numpy.ndarray
        A 1D NumPy array representing the point to be checked.

    Returns
    -------
    bool
        True if the point `q` is inside the cylinder, False otherwise.

    Notes
    -----
    - The check involves verifying that the point lies between `pt1` and `pt2` along the axis and
      that its perpendicular distance to the axis is less than or equal to the cylinder's radius.
    """
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
    """
    Check if multiple points are inside a cylinder defined by two points and a radius.

    The cylinder is defined by its axis (the line segment between `pt1` and `pt2`) and its radius `r`.
    This function checks for each point in `q` whether it lies inside this cylinder.

    Parameters
    ----------
    pt1 : numpy.ndarray
        A 1D NumPy array representing one endpoint of the cylinder's axis.
    pt2 : numpy.ndarray
        A 1D NumPy array representing the other endpoint of the cylinder's axis.
    r : float or int
        The radius of the cylinder.
    q : numpy.ndarray
        A 2D NumPy array where each row represents a point to be checked.

    Returns
    -------
    numpy.ndarray
        A 1D boolean NumPy array where each entry is True if the corresponding point in `q` is
        inside the cylinder, and False otherwise.

    Notes
    -----
    - This function is vectorized to efficiently handle multiple points in `q`.
    - The check involves verifying that each point lies between `pt1` and `pt2` along the axis
      and that its perpendicular distance to the axis is less than or equal to the cylinder's radius.
    """
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
    """
    Compute the equation parameters of a plane given a point on the plane and a rotation matrix.

    The plane equation is defined as `Ax + By + Cz + D = 0`, where:
    - `A`, `B`, and `C` are the components of the normal vector to the plane.
    - `D` is the distance to the plane along the normal vector.

    Parameters
    ----------
    point_on_plane : numpy.ndarray
        A 1D NumPy array representing a point on the plane.
    rotation_matrix : numpy.ndarray
        A 3x3 rotation matrix representing the plane's orientation in space.

    Returns
    -------
    tuple
        A tuple `(A, B, C, D)` representing the parameters of the plane equation.

    Notes
    -----
    - The normal vector is computed in the base frame using the rotation matrix.
    - The distance `D` is computed as the negative dot product of the normal vector and the point on the plane.
    """
    normal_vector_base_frame = np.dot(rotation_matrix.T, [1, 0, 0])
    distance_to_plane = -np.dot(normal_vector_base_frame, point_on_plane)
    A, B, C = normal_vector_base_frame
    return A, B, C, distance_to_plane


@beartype
def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two points.

    Parameters
    ----------
    point1 : numpy.ndarray
        A 1D NumPy array representing the first point.
    point2 : numpy.ndarray
        A 1D NumPy array representing the second point.

    Returns
    -------
    float
        The Euclidean distance between `point1` and `point2`.

    Notes
    -----
    - The Euclidean distance is calculated as the L2 norm of the difference between the points.
    """
    return np.linalg.norm(point1 - point2)

@beartype
def angle_between_vectors(
    vec1: np.ndarray, vec2: np.ndarray, assume_normalized: bool = False
) -> float:
    """
    Calculate the angle in radians between two vectors.

    Parameters
    ----------
    vec1 : numpy.ndarray
        A 1D NumPy array representing the first vector.
    vec2 : numpy.ndarray
        A 1D NumPy array representing the second vector.
    assume_normalized : bool, optional
        If True, assumes the input vectors are already normalized, skipping normalization (default is False).

    Returns
    -------
    float
        The angle between `vec1` and `vec2` in radians.

    Raises
    ------
    ValueError
        If one or both vectors have zero or near-zero magnitude, making the angle undefined.

    Notes
    -----
    - The angle is computed using the dot product formula:
      `cos(theta) = (vec1 . vec2) / (||vec1|| * ||vec2||)`.
    - The result is clamped to the range [-1, 1] to account for floating-point inaccuracies.
    """
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
    """
    Determine if a point is inside a convex hull.

    Parameters
    ----------
    point : numpy.ndarray
        A 1D NumPy array representing the point to check.
    hull : numpy.ndarray
        A 2D NumPy array representing the vertices of the convex hull.

    Returns
    -------
    bool
        True if the `point` is inside the convex hull, False otherwise.

    Notes
    -----
    - This function constructs a convex hull from the input points and the test point.
      If the hull volume does not change after adding the test point, the point is inside the hull.
    """
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
    """
    Compute the intersection point of a line and a plane.

    Parameters
    ----------
    plane_normal : numpy.ndarray
        A 1D NumPy array representing the normal vector of the plane.
    plane_point : numpy.ndarray
        A 1D NumPy array representing a point on the plane.
    line_point : numpy.ndarray
        A 1D NumPy array representing a point on the line.
    line_direction : numpy.ndarray
        A 1D NumPy array representing the direction vector of the line.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array representing the intersection point of the line and the plane.

    Raises
    ------
    ZeroDivisionError
        If the line is parallel to the plane (i.e., the direction vector is orthogonal to the plane normal).
    """
    d = np.dot(plane_point, plane_normal)
    t = (d - np.dot(line_point, plane_normal)) / np.dot(line_direction, plane_normal)
    return line_point + t * line_direction


@beartype
def transform_point(point: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Transform a point using a given transformation matrix.

    Parameters
    ----------
    point : numpy.ndarray
        A 1D NumPy array representing the point to transform.
    transformation_matrix : numpy.ndarray
        A 4x4 transformation matrix.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array representing the transformed point.

    Notes
    -----
    - The function assumes homogeneous coordinates for the transformation.
    - The transformation matrix must be a 4x4 affine matrix.
    """
    return np.dot(transformation_matrix, np.append(point, 1))[:-1]


@beartype
def closest_point_on_plane(
    plane_normal: np.ndarray, plane_point: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """
    Find the closest point on a plane to a given point.

    Parameters
    ----------
    plane_normal : numpy.ndarray
        A 1D NumPy array representing the normal vector of the plane.
    plane_point : numpy.ndarray
        A 1D NumPy array representing a point on the plane.
    point : numpy.ndarray
        A 1D NumPy array representing the point to find the closest point for.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array representing the closest point on the plane to the given point.

    Notes
    -----
    - The closest point is determined by projecting the input point onto the plane.
    """
    d = -np.dot(plane_point, plane_normal)
    t = -(np.dot(plane_normal, point) + d) / np.dot(plane_normal, plane_normal)
    return point + t * plane_normal


@beartype
def project_vector_onto_plane(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Project a vector onto a plane defined by its normal vector.

    Parameters
    ----------
    vector : numpy.ndarray
        A 1D NumPy array representing the vector to be projected.
    plane_normal : numpy.ndarray
        A 1D NumPy array representing the normal vector of the plane.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array representing the projection of the vector onto the plane.

    Notes
    -----
    - The projection removes the component of the vector that is parallel to the plane normal.
    """
    normalized_plane_normal = plane_normal / np.linalg.norm(plane_normal)
    proj_onto_normal = np.dot(vector, normalized_plane_normal) * normalized_plane_normal
    projection_onto_plane = vector - proj_onto_normal
    return projection_onto_plane


@beartype
def closest_point_on_line(
    line_point1: np.ndarray, line_point2: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """
    Find the closest point on a line segment to a given point.

    Parameters
    ----------
    line_point1 : numpy.ndarray
        A 1D NumPy array representing one endpoint of the line segment.
    line_point2 : numpy.ndarray
        A 1D NumPy array representing the other endpoint of the line segment.
    point : numpy.ndarray
        A 1D NumPy array representing the point to find the closest point for.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array representing the closest point on the line segment to the given point.

    Notes
    -----
    - The closest point is computed using vector projections.
    """
    line_vec = line_point2 - line_point1
    point_vec = point - line_point1
    t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
    return line_point1 + t * line_vec


@beartype
def point_plane_distance(
    plane_normal: np.ndarray, plane_point: np.ndarray, point: np.ndarray
) -> float:
    """
    Calculate the shortest distance from a point to a plane.

    Parameters
    ----------
    plane_normal : numpy.ndarray
        A 1D NumPy array representing the normal vector of the plane.
    plane_point : numpy.ndarray
        A 1D NumPy array representing a point on the plane.
    point : numpy.ndarray
        A 1D NumPy array representing the point for which the distance is to be calculated.

    Returns
    -------
    float
        The shortest distance from the point to the plane.

    Notes
    -----
    - The distance is positive if the point lies above the plane and negative if below,
      relative to the plane's normal vector.
    """
    return np.dot(plane_normal, point - plane_point) / np.linalg.norm(plane_normal)


@beartype
def point_to_segment_distance(
    point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray
) -> Union[float, np.ndarray]:
    """
    Calculate the shortest distance from a point or array of points to a line segment.

    Parameters
    ----------
    point : numpy.ndarray
        A 1D or 2D NumPy array representing the point(s) to compute the distance for.
    segment_start : numpy.ndarray
        A 1D NumPy array representing the start point of the line segment.
    segment_end : numpy.ndarray
        A 1D NumPy array representing the end point of the line segment.

    Returns
    -------
    Union[float, np.ndarray]
        The shortest distance from the point(s) to the line segment. Returns a float for a single point
        or a NumPy array for multiple points.
    """
    # Ensure all inputs are at least 2D for consistent broadcasting
    point = np.atleast_2d(point)
    segment_start = np.atleast_2d(segment_start)
    segment_end = np.atleast_2d(segment_end)

    # Compute segment and point vectors
    segment_vector = segment_end - segment_start
    point_vector = point - segment_start

    # Compute the cross product and distances
    cross_product = np.cross(segment_vector, point_vector)
    distances = np.abs(cross_product) / np.linalg.norm(segment_vector, axis=1)

    # If the input was a single point, return a scalar
    return distances[0] if point.shape[0] == 1 else distances


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

    Parameters
    ----------
    normal1 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the first plane.
    point1 : numpy.ndarray
        A 1D NumPy array representing a point on the first plane.
    normal2 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the second plane.
    point2 : numpy.ndarray
        A 1D NumPy array representing a point on the second plane.
    tolerance : float, optional
        The tolerance within which the planes are considered coincident (default is 1e-8).

    Returns
    -------
    bool
        True if the planes are coincident within the specified tolerance, False otherwise.

    Notes
    -----
    - Two planes are considered coincident if their normal vectors are parallel (same or opposite direction)
      and the distance between the planes is within the specified tolerance.
    """
    norm1 = normal1 / np.linalg.norm(normal1)
    norm2 = normal2 / np.linalg.norm(normal2)
    if not np.allclose(norm1, norm2, atol=tolerance) and not np.allclose(
        norm1, -norm2, atol=tolerance
    ):
        return False
    distance = np.abs(np.dot(norm1, point2 - point1) / np.linalg.norm(norm1))
    return bool(distance < tolerance)


@beartype
def are_planes_parallel(normal1: np.ndarray, normal2: np.ndarray, tolerance: float = 1e-8) -> bool:
    """
    Check if two planes are parallel within a specified tolerance.

    Parameters
    ----------
    normal1 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the first plane.
    normal2 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the second plane.
    tolerance : float, optional
        The tolerance within which the planes are considered parallel (default is 1e-8).

    Returns
    -------
    bool
        True if the planes are parallel within the specified tolerance, False otherwise.

    Notes
    -----
    - Two planes are parallel if the dot product of their normal vectors is approximately ±1.
    """
    return bool(np.allclose(np.abs(np.dot(normal1, normal2)), 1, atol=tolerance))


@beartype
def are_planes_perpendicular(
    normal1: np.ndarray, normal2: np.ndarray, tolerance: float = 1e-8
) -> bool:
    """
    Check if two planes are perpendicular within a specified tolerance.

    Parameters
    ----------
    normal1 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the first plane.
    normal2 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the second plane.
    tolerance : float, optional
        The tolerance within which the planes are considered perpendicular (default is 1e-8).

    Returns
    -------
    bool
        True if the planes are perpendicular within the specified tolerance, False otherwise.

    Notes
    -----
    - Two planes are perpendicular if the dot product of their normal vectors is approximately 0.
    """
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

    This method determines whether the planes are "facing" each other, meaning their normals are
    approximately opposite, and translation along the normal of one plane intersects the other.

    Parameters
    ----------
    normal1 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the first plane.
    point1 : numpy.ndarray
        A 1D NumPy array representing a point on the first plane.
    normal2 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the second plane.
    point2 : numpy.ndarray
        A 1D NumPy array representing a point on the second plane.
    atol : float, optional
        Absolute tolerance for the intersection check (default is 1e-9).
    max_angle : float, optional
        Maximum allowed deviation from π radians (180°) for the planes to be considered facing (default is 0.26).

    Returns
    -------
    bool
        True if the planes are facing each other and the translated origins intersect, False otherwise.

    Notes
    -----
    - The method uses the angle between the normals to determine alignment and checks
      the position of the points relative to each plane.
    - The `max_angle` parameter controls the angular tolerance, approximately ±15° by default.
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
    """
    Calculate the angle between two planes.

    Parameters
    ----------
    normal1 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the first plane.
    normal2 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the second plane.

    Returns
    -------
    float
        The angle between the planes in radians.

    Notes
    -----
    - The angle is calculated using the dot product of the normal vectors.
    """
    return np.arccos(np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2)))


@beartype
def distance_between_planes(
    normal1: np.ndarray, point1: np.ndarray, normal2: np.ndarray, point2: np.ndarray
) -> float:
    """
    Calculate the distance between two parallel planes.

    Parameters
    ----------
    normal1 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the first plane.
    point1 : numpy.ndarray
        A 1D NumPy array representing a point on the first plane.
    normal2 : numpy.ndarray
        A 1D NumPy array representing the normal vector of the second plane.
    point2 : numpy.ndarray
        A 1D NumPy array representing a point on the second plane.

    Returns
    -------
    float
        The distance between the two planes. Returns 0.0 if the planes are not parallel (i.e., they intersect).

    Notes
    -----
    - The distance is computed only if the planes are parallel.
    - For intersecting planes, the distance is considered 0.
    """
    if are_planes_parallel(normal1, normal2):
        return abs(np.dot(normal1, (point1 - point2))) / np.linalg.norm(normal1)
    else:
        return 0.0  # Planes intersect


@beartype
def centroid(arr: np.ndarray) -> np.ndarray:
    """
    Calculate the centroid of a set of points.

    Parameters
    ----------
    arr : numpy.ndarray
        A 2D NumPy array where each row represents a point.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array representing the centroid of the points.

    Notes
    -----
    - The centroid is calculated as the mean of all points along each axis.
    """
    sum_vals = np.sum(arr, axis=0)
    length = arr.shape[0]
    centroid = sum_vals / length
    return centroid


@beartype
def are_points_on_2d_plane(
    points: np.ndarray, return_normal: bool = False
) -> Union[Tuple[bool, np.ndarray], bool]:
    """
    Check if all points lie on the same 2D plane.

    Parameters
    ----------
    points : numpy.ndarray
        A 2D NumPy array of shape (N, D), where N is the number of points and D is the dimension.
    return_normal : bool, optional
        If True, returns the normal vector of the plane in addition to the boolean result (default is False).

    Returns
    -------
    Union[Tuple[bool, numpy.ndarray], bool]
        - A boolean indicating whether all points lie on the same 2D plane.
        - If `return_normal` is True, also returns the normal vector of the plane.

    Raises
    ------
    ValueError
        If fewer than 3 points are provided.

    Notes
    -----
    - The method determines coplanarity by computing the normal vector of the plane formed by the first three points
      and checking if all other points lie on that plane.
    """
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
    """
    Convert a dictionary of points into separate arrays of labels and coordinates.

    Parameters
    ----------
    point_dict : dict
        A dictionary where keys are labels (str) and values are points (list, tuple, or numpy.ndarray).

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        - A 1D NumPy array of labels as strings.
        - A 2D NumPy array of points as floats.

    Raises
    ------
    Exception
        If the conversion to arrays fails, the original exception is raised with additional context.

    Notes
    -----
    - The labels are converted into a NumPy array of strings.
    - The points are stacked into a 2D NumPy array with dtype `float64`.
    """
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
    """
    Generate points evenly distributed around a circle in 2D space.

    Parameters
    ----------
    radius : float or int
        The radius of the circle.
    num_points : int
        The number of points to generate.
    center : Iterable[Union[float, int]], optional
        The (x, y) coordinates of the circle's center (default is (0, 0)).
    start_angle : float or int, optional
        The starting angle in degrees, where 0 degrees is along the positive x-axis (default is 0).
    rnd : int, optional
        Number of decimal places to round the output points (default is 9).

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape (num_points, 2), where each row represents an (x, y) point on the circle.

    Notes
    -----
    - The points are evenly distributed along the circle's circumference.
    - The starting angle allows customization of the initial point's position relative to the x-axis.
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
    """
    Generate points evenly distributed around a circle in 3D space.

    Parameters
    ----------
    radius : float or int
        The radius of the circle.
    num_points : int
        The number of points to generate.
    center : Iterable[Union[float, int]], optional
        The (x, y, z) coordinates of the circle's center (default is (0, 0, 0)).
    normal : Iterable[Union[float, int]], optional
        The normal vector defining the plane of the circle (default is (0, 0, 1)).
    start_angle : float or int, optional
        The starting angle in degrees, where 0 degrees is along the positive x-axis (default is 0).

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape (num_points, 3), where each row represents an (x, y, z) point on the circle.

    Notes
    -----
    - The circle is initially generated in the XY-plane and then rotated to align with the specified normal vector.
    - The starting angle allows customization of the initial point's position relative to the x-axis in the circle's plane.
    """
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
    """
    Calculate a transformation matrix to position a frame on a cylinder's surface.

    The transformation positions a frame on the cylinder surface based on a specified displacement
    along the axis, a rotation angle around the axis, and the cylinder radius.

    Parameters
    ----------
    axis_translation : float
        The displacement along the cylinder's axis.
    axis_rotation : float
        The rotation angle around the axis, in radians.
    radius : float
        The radius of the cylinder.
    use_interior_normal : bool, optional
        If True, the normal points inward toward the cylinder's axis. If False, the normal points outward (default is False).

    Returns
    -------
    numpy.ndarray
        A 4x4 transformation matrix for positioning a frame on the cylinder's surface.

    Raises
    ------
    ValueError
        If the radius is negative or the rotation angle is outside the range -2π to 2π.

    Notes
    -----
    - The transformation includes translations and rotations to align the frame with the cylinder's geometry.
    - The direction of the normal vector can be controlled using the `use_interior_normal` parameter.
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative.")
    if not -np.pi * 2 <= axis_rotation <= np.pi * 2:
        raise ValueError("Rotation angle should be within -2π to 2π for stability.")


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
    Compute the bounding box for a set of multi-dimensional points.

    The bounding box is defined by the minimum and maximum values along each dimension.

    Parameters
    ----------
    points : numpy.ndarray
        A 2D NumPy array of shape (N, M), where N is the number of points and M is the dimensionality.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array of shape (M, 2), where each row represents [min, max] values for a dimension.

    Notes
    -----
    - The bounding box is calculated independently for each dimension of the input points.
    """
    return np.array([points.min(axis=0), points.max(axis=0)]).T


@beartype
def is_bounding_box_within(bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
    """
    Check if one bounding box is completely contained within another.

    Parameters
    ----------
    bbox1 : numpy.ndarray
        A 2D NumPy array of shape (M, 2), representing the first bounding box.
        Each row represents [min, max] values for a dimension.
    bbox2 : numpy.ndarray
        A 2D NumPy array of shape (M, 2), representing the second bounding box.
        Each row represents [min, max] values for a dimension.

    Returns
    -------
    bool
        True if `bbox1` is completely within `bbox2`, False otherwise.

    Notes
    -----
    - This function checks that the minimum values of `bbox1` are greater than or equal to the
      minimum values of `bbox2` and that the maximum values of `bbox1` are less than or equal to
      the maximum values of `bbox2`.
    """
    return np.all(bbox1[:, 0] >= bbox2[:, 0]) and np.all(bbox1[:, 1] <= bbox2[:, 1])


@beartype
def do_bounding_boxes_overlap(bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
    """
    Check if two multidimensional bounding boxes overlap.

    Parameters
    ----------
    bbox1 : numpy.ndarray
        A 2D NumPy array of shape (M, 2), where each row represents [min, max] values for a dimension.
    bbox2 : numpy.ndarray
        A 2D NumPy array of shape (M, 2), where each row represents [min, max] values for a dimension.

    Returns
    -------
    bool
        True if the bounding boxes overlap, False otherwise.

    Notes
    -----
    - Bounding boxes overlap if their intervals intersect in all dimensions.
    """
    return bool(np.all(np.maximum(bbox1[:, 0], bbox2[:, 0]) <= np.minimum(bbox1[:, 1], bbox2[:, 1])))


@beartype
def are_normals_aligned_and_facing(
    normal1: np.ndarray,
    point1: np.ndarray,
    normal2: np.ndarray,
    point2: np.ndarray,
    atol: float = 1e-9,
) -> bool:
    """
    Check if two normal vectors are aligned in opposite directions and facing each other.

    Vectors are considered to be facing each other if they are approximately anti-parallel
    (opposed) and the points from which they originate are positioned such that their extensions
    would intersect.

    Parameters
    ----------
    normal1 : numpy.ndarray
        A 1D NumPy array representing the first normal vector.
    point1 : numpy.ndarray
        A 1D NumPy array representing the origin of the first normal vector.
    normal2 : numpy.ndarray
        A 1D NumPy array representing the second normal vector.
    point2 : numpy.ndarray
        A 1D NumPy array representing the origin of the second normal vector.
    atol : float, optional
        Absolute tolerance for considering the normals to be facing each other (default is 1e-9).

    Returns
    -------
    bool
        True if the normals are opposed and facing each other, False otherwise.

    Notes
    -----
    - This function checks that the dot product of the normals is approximately -1
      (indicating they are anti-parallel).
    - The vectors from `point1` to `point2` are used to ensure the normals are facing each other.
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
    Calculate the scalar projection factor of one vector onto another.

    The scalar projection factor represents the length of the projection of one vector
    onto another, divided by the length of the reference vector.

    Parameters
    ----------
    vector_to_project : numpy.ndarray
        A 1D NumPy array representing the vector to be projected.
    reference_vector : numpy.ndarray
        A 1D NumPy array representing the reference vector onto which the first vector is projected.

    Returns
    -------
    float
        The scalar projection factor.

    Notes
    -----
    - The scalar projection factor is calculated as the dot product of the two vectors
      divided by the squared norm of the reference vector.
    """
    return np.dot(vector_to_project, reference_vector) / np.linalg.norm(reference_vector) ** 2
