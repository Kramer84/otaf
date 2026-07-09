from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "spheres_from_point_cloud",
    "open_cylinder_mesh",
    "surface_from_planar_contour",
]

import numpy as np

from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    import trimesh

from otaf.geometry import are_points_on_2d_plane, rotation_matrix_from_vectors
import otaf.exceptions as otaf_exceptions


def spheres_from_point_cloud(
    pc: np.ndarray,
    radius: float = 1.0,
    color: np.ndarray | list[int] | tuple[int, int, int, int] = [
        100,
        100,
        100,
        255,
    ],
    global_translation: np.ndarray | list[float] = np.array([0, 0, 0]),
) -> list[trimesh.Trimesh]:
    """Generate Trimesh icospheres at each point in a point cloud.

    Instantiates a sphere mesh for each coordinate position, applies
    a uniform color, and shifts each position by an optional global
    translation vector.

    Parameters
    ----------
    pc : array_like
        An (N, 3) array containing the coordinates of the point cloud.
    radius : float, default 1.0
        The radius of each generated sphere.
    color : array_like, default [100, 100, 100, 255]
        The RGBA color applied to the vertices of the spheres.
    global_translation : array_like, default [0, 0, 0]
        A 3D offset vector added to every point's position.

    Returns
    -------
    list of trimesh.Trimesh
        A list of individual Trimesh icosphere objects.

    Raises
    ------
    ImportError
        If the ``trimesh`` library is not installed.
    """
    try :
        import trimesh
    except ImportError :
        raise otaf_exceptions._raise_missing_dependency("trimesh", "spheres_from_point_cloud")

    spheres = []
    for i in range(pc.shape[0]):
        sph = trimesh.creation.icosphere(radius=radius)
        sph.visual.vertex_colors = color
        sph.apply_translation(pc[i] + global_translation)
        spheres.append(sph)
    return spheres



def open_cylinder_mesh(
    radius: float, 
    height: float, 
    transform: np.ndarray, 
    sections: int = 64
) -> trimesh.Trimesh:
    """Create an open-ended cylinder mesh with a transformation.

    Generates a standard cylinder, removes the cap faces connected
    to the top and bottom center vertices, and applies a 4x4 spatial
    transformation matrix.

    Parameters
    ----------
    radius : float
        The radius of the cylinder.
    height : float
        The height of the cylinder along its local Z-axis.
    transform : array_like
        A 4x4 homogeneous transformation matrix to position and
        rotate the mesh.
    sections : int, default 64
        The number of radial facets used to approximate the circular
        cross-section.

    Returns
    -------
    trimesh.Trimesh
        The open-ended cylindrical mesh.

    Raises
    ------
    ImportError
        If the ``trimesh`` library is not installed.
    """
    try :
        import trimesh
    except ImportError :
        raise otaf_exceptions._raise_missing_dependency("trimesh", "open_cylinder_mesh")
    temp_cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    # Identify cap center vertices (local Z is +/- height/2)
    z_limit = height / 2.0
    center_v_indices = np.where(
        (np.abs(temp_cyl.vertices[:, 0]) < 1e-5) &
        (np.abs(temp_cyl.vertices[:, 1]) < 1e-5) &
        (np.abs(np.abs(temp_cyl.vertices[:, 2]) - z_limit) < 1e-5)
    )[0]
    # Filter faces: keep only those that DON'T use a cap center vertex
    face_mask = ~np.any(np.isin(temp_cyl.faces, center_v_indices), axis=1)
    # Process=False prevents trimesh from trying to "fix" the open ends
    cyl = trimesh.Trimesh(vertices=temp_cyl.vertices, faces=temp_cyl.faces[face_mask], process=False)
    cyl.remove_unreferenced_vertices()
    # Apply global transform
    cyl.apply_transform(transform)
    return cyl

def surface_from_planar_contour(
    vertices: np.ndarray | list,
    segments: np.ndarray | list | None = None,
) -> trimesh.Trimesh:
    """Triangulate a 3D planar point contour to generate a surface mesh.

    Projects the 3D coordinates onto a local 2D plane, performs a
    Constrained Delaunay Triangulation using the ``triangle`` library,
    and then transforms the resulting mesh vertices back to their
    original 3D orientation.

    Parameters
    ----------
    vertices : array_like
        An (N, 3) array of 3D coordinates representing the boundary
        of the contour.
    segments : array_like, optional
        An optional (M, 2) array of vertex indices defining fixed
        boundary lines or holes that must be respected during
        triangulation. Default is None.

    Returns
    -------
    trimesh.Trimesh
        The generated flat surface mesh.

    Raises
    ------
    ImportError
        If either the ``triangle`` or ``trimesh`` libraries are not
        installed.
    ValueError
        If the input `segments` do not have a shape ending in 2, or
        if the input `vertices` do not all lie on the same 2D plane.
    """
    try :
        import triangle
    except ImportError :
        raise otaf_exceptions._raise_missing_dependency("triangle", "surface_from_planar_contour")

    try :
        import trimesh
    except ImportError :
        raise otaf_exceptions._raise_missing_dependency("trimesh", "surface_from_planar_contour")

    vertices = np.array(vertices)

    if segments is not None:
        segments = np.array(segments)
        if segments.shape[-1] != 2:
            raise ValueError(f"The segment array has to be of shape (Mx2), not {segments.shape}")

    on_plane, normal = are_points_on_2d_plane(vertices, return_normal=True)
    if not on_plane:
        raise ValueError(
            "The provided vertices are not on the same 2D plane. Triangulation not possible."
        )

    # RotationMatrixLocalPlaneToGlobal
    R = rotation_matrix_from_vectors(normal, np.array([0, 0, 1]))
    vertices_proj = (R @ vertices.T).T
    print(vertices_proj)
    assert np.array_equal(
        are_points_on_2d_plane(vertices_proj,True)[1], np.array([0, 0, 1])
    ), "Some problem"
    vertices_proj_2d = vertices_proj[:, [0, 1]]

    tri = dict(vertices=vertices_proj_2d)
    if segments is not None:
        tri["segments"] = segments

    triangulation = triangle.triangulate(tri, opts="C")
    vertices_tri_2d_proj = np.array(triangulation["vertices"])
    # Back in 3D with extra dimension for inverse transormation
    vertices_tri_3d_proj = np.column_stack(
        [vertices_tri_2d_proj, np.zeros((vertices_tri_2d_proj.shape[0],))]
    )
    vertices_tri_3D = (R.T @ vertices_tri_3d_proj.T).T
    triangles = np.array(triangulation["triangles"])

    planar_mesh = trimesh.Trimesh(vertices=vertices_tri_3D, faces=triangles)
    return planar_mesh