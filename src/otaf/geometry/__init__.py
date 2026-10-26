"""Geometric primitives and spatial analysis utilities for the OTAF project."""

from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"

# Explicitly import from _geometry
from ._geometry import (
    angle_between_planes,
    angle_between_vectors,
    are_lines_collinear,
    are_normals_aligned_and_facing,
    are_planes_coincident,
    are_planes_facing,
    are_planes_parallel,
    are_planes_perpendicular,
    are_points_on_2d_plane,
    calculate_cylinder_surface_frame,
    calculate_scalar_projection_factor,
    centroid,
    closest_point_on_line,
    closest_point_on_plane,
    compute_bounding_box,
    distance_between_planes,
    do_bounding_boxes_overlap,
    euclidean_distance,
    generate_circle_points,
    generate_circle_points_3d,
    is_affine_transformation_matrix,
    is_bounding_box_within,
    line_plane_intersection,
    plane_parameters,
    point_dict_to_arrays,
    point_in_hull,
    point_plane_distance,
    point_to_segment_distance,
    points_in_cylinder,
    points_in_cylinder_vect,
    project_vector_onto_plane,
    rotation_matrix_from_vectors,
    tfrt,
    transform_point,
)

# Explicitly import from _meshes
from ._meshes import (
    open_cylinder_mesh,
    spheres_from_point_cloud,
    surface_from_planar_contour,
)

# Define the strict public namespace
__all__ = [
    "angle_between_planes",
    "angle_between_vectors",
    "are_lines_collinear",
    "are_normals_aligned_and_facing",
    "are_planes_coincident",
    "are_planes_facing",
    "are_planes_parallel",
    "are_planes_perpendicular",
    "are_points_on_2d_plane",
    "calculate_cylinder_surface_frame",
    "calculate_scalar_projection_factor",
    "centroid",
    "closest_point_on_line",
    "closest_point_on_plane",
    "compute_bounding_box",
    "distance_between_planes",
    "do_bounding_boxes_overlap",
    "euclidean_distance",
    "generate_circle_points",
    "generate_circle_points_3d",
    "is_affine_transformation_matrix",
    "is_bounding_box_within",
    "line_plane_intersection",
    "open_cylinder_mesh",
    "plane_parameters",
    "point_dict_to_arrays",
    "point_in_hull",
    "point_plane_distance",
    "point_to_segment_distance",
    "points_in_cylinder",
    "points_in_cylinder_vect",
    "project_vector_onto_plane",
    "rotation_matrix_from_vectors",
    "spheres_from_point_cloud",
    "surface_from_planar_contour",
    "tfrt",
    "transform_point",
]