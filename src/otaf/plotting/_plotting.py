# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "plot_points_3D",
    "trimesh_scene_as_notebook_scene",
    "spheres_from_point_cloud",
    "create_open_cylinder",
    "create_surface_from_planar_contour",
    "hex_to_rgba",
    "plot_single_transform",
    "plot_transform_sequence",
    "plot_best_worst_results",
    "plot_best_worst_input_data",
    "print_sample_in_deviation_domain",
    "set_graph_legends",
    "plot_rect_part",
    "plot_deviated_surfs",
    "calculate_sample_in_deviation_domain",
    "plot_deviation_domain",
    "arrange_axes_in_grid",
    "compare_jacobians",
    "pair_plot",
    "plot_combined_CDF",
    "save_plot",
    "plot_gld_pbox_cdf"
]

import os
import re
import math
from time import time
import logging
from itertools import product

import numpy as np
import sympy as sp
from scipy.optimize import fminbound

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon, Rectangle
import trimesh as tr
from trimesh import viewer
from pytransform3d.plot_utils import make_3d_axis
import pytransform3d.transformations as pytr
import pytransform3d.rotations as pyrot
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional
import otaf


@beartype
def plot_points_3D(
    points_dict: Dict[str, Tuple[float, float, float]],
    color: str = "r",
    marker: str = "o",
    azim: float = 0,
) -> None:
    """
    Create a 3D scatter plot of points.

    Args:
        points_dict (dict): A dictionary where keys are labels and values are 3D points (tuples of floats).
        color (str, optional): Color of the markers. Defaults to 'r' (red).
        marker (str, optional): Marker style. Defaults to 'o' (circle).
        azim (float, optional): Azimuthal viewing angle. Defaults to 0 (front view).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for label, point in points_dict.items():
        ax.scatter(point[0], point[1], point[2], c=color, marker=marker)
        # ax.text(point[0], point[1], point[2], label, fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal", adjustable="datalim")
    ax.view_init(vertical_axis="y", azim=azim)
    plt.show()


def trimesh_scene_as_notebook_scene(scene, background_hex_color="e6e6e6"):
    notebook_scene = viewer.notebook.scene_to_notebook(scene)
    notebook_scene.data = notebook_scene.data.replace(
        "scene.background=new THREE.Color(0xffffff)",
        f"scene.background=new THREE.Color(0x{background_hex_color})",
    )
    return notebook_scene


@beartype
def spheres_from_point_cloud(
    pc: np.ndarray,
    radius: float = 1.0,
    color: Union[Tuple[int, int, int, float], List[Union[float, int]], np.ndarray] = [
        0.1,
        0.1,
        0.1,
        1.0,
    ],
    global_translation: Union[List[Union[float, int]], np.ndarray] = np.array([0, 0, 0]),
) -> List[tr.Trimesh]:
    """Create a list of spheres from a point cloud.

    Args:
        pc (np.ndarray): The point cloud as an Nx3 numpy array, where N is the number of points.
        radius (float, optional): The radius of the spheres. Defaults to 1.0.
        color (List[float], optional): The color of the spheres as an RGBA list. Defaults to [0.1, 0.1, 0.1, 1.0].
        global_translation (np.ndarray, optional): A translation vector to apply to all spheres. Defaults to [0, 0, 0].

    Returns:
        List[tr Trimesh]: A list of tr Trimesh objects representing the spheres.
    """
    spheres = [tr.creation.icosphere(radius=radius, face_colors=color) for i in range(pc.shape[0])]
    spheres = [spheres[i].apply_translation(pc[i] + global_translation) for i in range(pc.shape[0])]
    return spheres


def create_open_cylinder(radius, segment, **kwargs):
    """Creates a trimesh object containing a cylinder open on both ends"""
    cylinder = tr.creation.cylinder(radius=radius, segment=segment, **kwargs)
    # Get all points not on outer cylinder surface
    vertex_axis_distance = otaf.geometry.point_to_segment_distance(
        cylinder.vertices, np.array(segment[0]), np.array(segment[1])
    )
    vertex_not_outer_args = np.squeeze(np.argwhere(vertex_axis_distance.round(5) < radius))
    faces_to_remove = np.zeros(cylinder.faces.shape[0])
    for v in vertex_not_outer_args:
        faces_to_remove += np.where(cylinder.faces == v, 1, 0).sum(axis=-1)
    faces_to_remove = np.where(faces_to_remove > 0, 1, 0)
    cylinder.update_faces((1 - faces_to_remove).astype(bool))
    return cylinder


def create_surface_from_planar_contour(vertices, segments):
    import triangle
    vertices = np.array(vertices)
    segments = np.array(segments)
    on_plane, normal = otaf.geometry.are_points_on_2d_plane(vertices, return_normal=True)
    if not on_plane:
        raise ValueError(
            "The provided vertices are not on the same 2D plane. Triangulation not possible."
        )
    if not segments.shape[-1] == 2:
        raise ValueError(f"The segment array has to be of shape (Mx2) not {segments.shape}")
    # RotationMatrixLocalPlaneToGlobal
    R = pyrot.matrix_from_two_vectors(normal, np.array([0, 0, 1]))
    vertices_proj = np.dot(vertices, R.T)
    print(vertices_proj)
    assert np.array_equal(
        otaf.geometry.are_points_on_2d_plane(vertices_proj[:, :-1]), np.array([0, 0, 1])
    ), "Some problem"
    vertices_proj_2D = vertices_proj[:, [0, 1]]
    tri = dict(vertices=vertices_proj_2D, segments=segments)
    triangulation = triangle.triangulate(tri, opts="C")
    vertices_tri_2D_proj = np.array(triangulation["vertices"])
    # Back in 3D with extra dimension for inverse transormation
    vertices_tri_3D_proj = np.column_stack(
        [vertices_tri_2D_proj, np.zeros((vertices_tri_2D_proj.shape[0],))]
    )
    vertices_tri_3D = R.T @ vertices_tri_3D_proj
    triangles = np.array(triangulation["triangles"])

    planar_mesh = tr.Trimesh(vertices=vertices_tri_3D, faces=triangles)
    return planar_mesh


@beartype
def hex_to_rgba(
    hex_color: str, alpha: float = 1.0, as_float: bool = False
) -> Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int]]:
    """Converts a hexadecimal color code to RGBA values.

    Args:
        hex_color (str): The hexadecimal color code (e.g., '#ffffff').
        alpha (float): The alpha (transparency) value between 0.0 and 1.0 (default is 1.0).
        as_float (bool): If True, returns RGBA values as floats between 0 and 1. If False (default), returns integers.

    Returns:
        tuple: A tuple containing the RGBA values as floats (or integers if as_float is False)
               and the alpha value as a float (e.g., (1.0, 1.0, 1.0, 1.0)).
    """
    # Remove the '#' if it exists in the input
    hex_color = hex_color.lstrip("#")

    # Convert the hex values to integers
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    if as_float:
        r /= 255.0
        g /= 255.0
        b /= 255.0

    return (r, g, b, alpha)


def plot_single_transform(transform, figsize=(8, 8), start_color="red", end_color="blue", azim=0):
    """
    Plots a single transformation matrix in a 3D space.

    Args:
    - transform (np.ndarray or similar): A 4x4 transformation matrix.
    - figsize (tuple): Size of the figure.
    - origin_color (str): Color of the origin point.
    """
    transform = np.asarray(transform)  # Ensure the transform is a NumPy array
    ax = make_3d_axis(figsize[0], 111)

    ax.scatter(0.0, 0.0, 0.0, color=start_color, s=25)
    ax.text(0, 0, 0, "(0, 0, 0)")

    pytr.plot_transform(ax)

    # Plotting the origin with a distinct color
    end = transform[:3, 3]
    ax.scatter(*end, color=end_color, s=25)  # Adjust size with 's' parameter
    ax.text(*end, "({}, {}, {})".format(*map(lambda x: round(x, 2), list(end))))

    pytr.plot_transform(ax, A2B=transform)

    # ax.view_init(vertical_axis="y", azim=azim)
    ax.set_aspect("equal")
    ax.grid(True, which="both")
    plt.tight_layout()
    plt.show()


def plot_transform_sequence(
    transforms, figsize=(8, 8), origin_color="blue", start_color="red", end_color="blue", azim=0
):
    """
    Plots a sequence of transformation matrices, showing the transformation
    of a point across multiple frames.

    Args:
    - transforms (list of np.ndarray or similar): A list of 4x4 transformation matrices.
    - figsize (tuple): Size of the figure.
    - origin_color (str): Color of the origin point in each frame.
    """
    ax = make_3d_axis(figsize[0], 111)
    point = np.array([0, 0, 0, 1])  # Starting point

    ax.scatter(0.0, 0.0, 0.0, color=start_color, s=25)
    ax.text(0, 0, 0, "(0, 0, 0)")

    pytr.plot_transform(ax)

    for transform in transforms:
        transform = np.asarray(transform)  # Convert to NumPy array
        pytr.plot_transform(ax, A2B=transform)
        point = pytr.transform(transform, point)
        ax.scatter(*point[:3], color=origin_color, s=100)  # Adjust size with 's' parameter

    plt.show()


def plot_best_worst_results(
    best_results,
    worst_results,
    xlabel="Sample Index",
    ylabel="Failure rate",
    title="Best and Worst Results Plot",
    figsize=(10, 6),
    save_as_png=False,
    save_path="images",
    filename="best_worst_results.png",
    dpi=600,
):
    """Plot the best and worst results.

    Args:
        best_results (array-like): Array containing the best results.
        worst_results (array-like): Array containing the worst results.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Sample Index'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Failure rate'.
        title (str, optional): Title of the plot. Defaults to 'Best and Worst Results Plot'.
        figsize (tuple, optional): Size of the figure (width, height) in inches. Defaults to (10, 6).
        save_as_png (bool, optional): Whether to save the plot as a PNG. Defaults to False.
        save_path (str, optional): Path to save the plot. Defaults to 'images'.
        filename (str, optional): Name of the file to save the plot. Defaults to 'best_worst_results.png'.
        dpi (int, optional): Resolution of the saved plot. Defaults to 600 dpi.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the best results
    ax.plot(best_results, label="Best Results", marker="o", linestyle="-", linewidth=2)

    # Plot the worst results
    ax.plot(worst_results, label="Worst Results", marker="x", linestyle="-", linewidth=2)

    # Add labels and legend
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)

    # Add grid for better readability
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Improve layout spacing
    plt.tight_layout()

    # Save the plot as a PNG if requested
    if save_as_png:
        # Create 'images' directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the figure
        fig.savefig(os.path.join(save_path, filename), dpi=dpi, bbox_inches="tight")
        print(f"Plot saved as {filename} in the '{save_path}' directory.")

    # Show the plot
    plt.show()


def plot_best_worst_input_data(
    best_data,
    worst_data,
    variable_labels,
    figsize=(16, 8),
    labels=True,
    save_as_png=False,
    save_path="images",
    filename="best_worst_input_data.png",
    dpi=600,
):
    """Plot the best and worst 5% input data.

    Args:
        best_data (list): List of input data for the best 5%.
        worst_data (list): List of input data for the worst 5%.
        variable_labels (list): List of variable labels.
        figsize (tuple, optional): Size of the figure (width, height) in inches. Defaults to (16, 8).
        labels (bool, optional): Whether to label samples. Defaults to True.
        save_as_png (bool, optional): Whether to save the plot as a PNG. Defaults to False.
        save_path (str, optional): Path to save the plot. Defaults to 'images'.
        filename (str, optional): Name of the file to save the plot. Defaults to 'best_worst_input_data.png'.
        dpi (int, optional): Resolution of the saved plot. Defaults to 600 dpi.
    """
    # Create LaTeX formatted labels
    variable_labels_tex = [f"${sp.printing.latex(sp.Symbol(var))}$" for var in variable_labels]

    # Create separate figures and axes for the best and worst 5% plots
    fig_best, ax_best = plt.subplots(figsize=figsize)
    fig_worst, ax_worst = plt.subplots(figsize=figsize)

    # Plot the input data for the best 5%
    for i, sample in enumerate(best_data):
        label = f"Sample {i + 1}" if labels else None
        if i < 3:
            ax_best.plot(sample, "k-", linewidth=0.69)
        ax_best.plot(sample, "o", label=label)

    # Plot the input data for the worst 5%
    for i, sample in enumerate(worst_data):
        label = f"Sample {i + 1}" if labels else None
        if i > len(worst_data) - 4:
            ax_worst.plot(sample, "k-", linewidth=0.69)
        ax_worst.plot(sample, "o", label=label)

    # Rotate x-axis labels for better readability
    ax_best.tick_params(axis="x", rotation=45)
    ax_worst.tick_params(axis="x", rotation=45)

    # Add labels and legend for the best 5% plot
    ax_best.set_xlabel("Variable Label")
    ax_best.set_ylabel("Input Value")
    ax_best.set_title("Best 5% Input Data Plot")
    ax_best.legend()

    # Add labels and legend for the worst 5% plot
    ax_worst.set_xlabel("Variable Label")
    ax_worst.set_ylabel("Input Value")
    ax_worst.set_title("Worst 5% Input Data Plot")
    ax_worst.legend()

    # Set y-axis limits
    ax_worst.set_ylim(0, 1)
    ax_best.set_ylim(0, 1)

    # Define a color mapping dictionary for variables with the same index
    color_mapping = {}
    color_idx = 0
    for var_label in variable_labels:
        index = re.match(".*?([0-9]+)$", str(var_label)).group(1)  # Extract the index part
        if index not in color_mapping:
            color_idx += 1
            color_mapping[index] = [0.5, 0.5, 0.5, 0.3]  # Placeholder color mapping

    for i, var_label in enumerate(variable_labels):
        var_index = int(re.match(".*?([0-9]+)$", str(var_label)).group(1))

        # Process best 5%
        var_data_best = [sample[variable_labels.index(var_label)] for sample in best_data]
        min_val_best, max_val_best = min(var_data_best), max(var_data_best)

        # Plot vertical lines for best 5%
        ax_best.axvline(
            x=i, ymin=min_val_best, ymax=max_val_best, color="black", linestyle="--", alpha=0.9
        )
        ax_best.axvspan(i - 0.5, i + 0.5, facecolor=color_mapping[str(var_index)], alpha=0.3)

        # Process worst 5%
        var_data_worst = [sample[variable_labels.index(var_label)] for sample in worst_data]
        min_val_worst, max_val_worst = min(var_data_worst), max(var_data_worst)

        # Plot vertical lines for worst 5%
        ax_worst.axvline(
            x=i, ymin=min_val_worst, ymax=max_val_worst, color="black", linestyle="--", alpha=0.9
        )
        ax_worst.axvspan(i - 0.5, i + 0.5, facecolor=color_mapping[str(var_index)], alpha=0.3)

        # Plot vertical lines representing the span of the other axis (worst on best, best on worst)
        ax_best.axvline(
            x=i + 0.15,
            ymin=min_val_worst,
            ymax=max_val_worst,
            color="red",
            linestyle="dashdot",
            alpha=0.5,
            linewidth=1.5,
        )
        ax_worst.axvline(
            x=i + 0.15,
            ymin=min_val_best,
            ymax=max_val_best,
            color="red",
            linestyle="dashdot",
            alpha=0.5,
            linewidth=1.5,
        )

    # Set the x-axis ticks and labels
    ax_best.set_xticks(list(range(len(variable_labels))))
    ax_worst.set_xticks(list(range(len(variable_labels))))
    ax_best.set_xticklabels(variable_labels_tex)
    ax_worst.set_xticklabels(variable_labels_tex)

    # Save the plots as PNGs if requested
    if save_as_png:
        # Create save path if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save both the best and worst figures
        fig_best.savefig(os.path.join(save_path, f"best_{filename}"), dpi=dpi, bbox_inches="tight")
        fig_worst.savefig(
            os.path.join(save_path, f"worst_{filename}"), dpi=dpi, bbox_inches="tight"
        )
        print(
            f"Plots saved as best_{filename} and worst_{filename} in the '{save_path}' directory."
        )

    # Show the plots
    plt.show()


def print_sample_in_deviation_domain(
    ax,
    pos_u,
    theta,
    lambda_pos,
    lambda_theta,
    t_max,
    theta_max,
    ratio=1,
    r=None,
    xlabel="U",
    ylabel="Theta",
    remove_ticks=False,
):
    """
    Plots a sample in the deviation domain on the provided matplotlib axis with customizable labels and tick options.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot.
    pos_u (array-like): Position data.
    theta (array-like): Theta data.
    lambda_pos (float): Lambda position scaling factor.
    lambda_theta (float): Lambda theta scaling factor.
    t_max (float): Maximum value for t (U domain).
    theta_max (float): Maximum value for theta (Theta domain).
    ratio (float, optional): Aspect ratio for the plot. Defaults to 1.
    r (float, optional): Additional parameter to display. Defaults to None.
    xlabel (str, optional): Label for the x-axis. Defaults to "U".
    ylabel (str, optional): Label for the y-axis. Defaults to "Theta".
    remove_ticks (bool, optional): If True, removes axis ticks. Defaults to False.

    Returns:
    matplotlib.axes.Axes: The axis with the plot.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Plotting boundary lines
    ax.plot([-t_max, 0], [0, theta_max], "r-", lw=3)
    ax.plot([0, t_max], [theta_max, 0], "r-", lw=3)
    ax.plot([t_max, 0], [0, -theta_max], "r-", lw=3)
    ax.plot([0, -t_max], [-theta_max, 0], "r-", lw=3)

    # Scatter plot of the data points
    ax.scatter(lambda_pos * pos_u, lambda_theta * theta)

    # Set the axis bounds
    ax.set_xbound([-t_max * 1.1, t_max * 1.1])
    ax.set_ybound([-theta_max * 1.1, theta_max * 1.1])

    # Adjust aspect ratio
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    xspan = xright - xleft
    yspan = ytop - ybottom
    ax.set_aspect(abs(xspan / yspan) * ratio)

    # Position text labels on the plot
    x_txt = xleft + 0.0125 * xspan
    y_txt_fnc = lambda i: ytop - 0.1 * (1 + i) * yspan

    if r is not None:
        ax.text(x_txt, y_txt_fnc(2), rf"$r: {r}$", fontsize=15, fontweight="bold")

    # Optionally remove ticks
    if remove_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(True)
    return ax


def set_graph_legends(graph, x_title="", y_title="", title="", legends=None, colors=None):
    """
    Sets titles, legends, and colors for the given graph.

    Parameters:
    graph (object): The graph object to be modified.
    x_title (str, optional): Title for the x-axis. Defaults to an empty string.
    y_title (str, optional): Title for the y-axis. Defaults to an empty string.
    title (str, optional): Title for the graph. Defaults to an empty string.
    legends (list, optional): List of legend labels. Defaults to None.
    colors (list, optional): List of colors for the graph elements. Defaults to None.

    Returns:
    object: The modified graph object.
    """
    if x_title:
        graph.setXTitle(x_title)
    if y_title:
        graph.setYTitle(y_title)
    if title:
        graph.setTitle(title)
    if legends:
        graph.setLegends(legends)
    if colors:
        graph.setColors(colors)

    return graph


def plot_rect_part(ax, scale_factor=100):
    """
    Plots the rectangular part of the example and changes the aspect ratio,
    so that it is 2 times higher and 2 times shorter.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot.
    scale_factor (float, optional): Scaling factor for the arrows. Defaults to 100.

    Returns:
    matplotlib.axes.Axes: The axis with the plot.
    """
    t_ = 10  # Ensure t_ is defined for arrow scaling

    ax.plot([0, 0], [0, 10], "k-")
    ax.plot([0, 100], [10, 10], "k-")
    ax.plot([100, 100], [10, 0], "k--")
    ax.plot([100, 0], [0, 0], "k-")
    ax.set_xlim(-30, 120)
    ax.set_ylim(-1, 13)

    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 0.5)

    ax.arrow(
        100,
        11.5,
        -t_ * scale_factor,
        0,
        head_width=0.1,
        head_length=0.2,
        overhang=0,
        linewidth=2,
        color="r",
        length_includes_head=True,
    )
    ax.arrow(
        100,
        11.5,
        t_ * scale_factor,
        0,
        head_width=0.1,
        head_length=0.2,
        overhang=0,
        linewidth=2,
        color="r",
        length_includes_head=True,
    )
    ax.text(95, 12, f"2t*{scale_factor}", color="r", fontsize="x-large", fontweight="bold")

    ax.text(
        -23,
        5,
        "A",
        color="black",
        fontsize="xx-large",
        fontweight="bold",
        bbox=dict(facecolor="none", edgecolor="black"),
    )
    ax.arrow(
        -10.5,
        5.25,
        9,
        0,
        head_width=0.4,
        head_length=1,
        overhang=0,
        linewidth=4,
        color="k",
        length_includes_head=True,
    )

    return ax


def plot_deviated_surfs(ax, se_pos, se_rot, lambda_, scale_factor):
    """
    Adds on the rectangular part a set of deviated lines to
    visualize the geometrical distribution of the defects and the impact
    of the allocations. The defects are multiplied by a scaleFactor.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot.
    se_pos (array-like): Positional deviations.
    se_rot (array-like): Rotational deviations.
    lambda_ (float): Scaling factor for the deviations.
    scale_factor (float): Factor to scale the deviations.

    Returns:
    matplotlib.axes.Axes: The axis with the plot.
    """
    se_pos = np.squeeze(se_pos)
    se_rot = np.squeeze(se_rot)
    l = 10  # height of surface
    x_line = np.array([100, 100])
    y_line = np.array([10, 0.0])  # stays constant

    assert len(se_pos) == len(se_rot), "The length of se_pos and se_rot must be equal"

    for i in range(len(se_pos)):
        x_ltemp = x_line + (lambda_ * se_pos[i]) * scale_factor
        x_ltemp[0] -= ((1 - lambda_) * se_rot[i] * 0.5 * l) * scale_factor
        x_ltemp[1] += ((1 - lambda_) * se_rot[i] * 0.5 * l) * scale_factor
        ax.plot(x_ltemp, y_line, "b-")

    return ax


def calculate_sample_in_deviation_domain(pos_u, theta, lambda_pos, lambda_theta):
    """
    Calculates the sample data in the deviation domain.

    Parameters:
    pos_u (array-like): Position data.
    theta (array-like): Theta data.
    lambda_pos (float): Lambda position scaling factor.
    lambda_theta (float): Lambda theta scaling factor.

    Returns:
    np.ndarray: 2D sample data with scaled position and theta.
    """
    scaled_pos = lambda_pos * pos_u
    scaled_theta = lambda_theta * theta
    sample = np.vstack((scaled_pos, scaled_theta)).T
    return sample


def plot_deviation_domain(
    ax,
    sample,
    x_label="X",
    y_label="Y",
    x_bounds=(-0.15, 0.15),
    y_bounds=(-1, 1),
    ratio_bounds=1.5,
    r=None,
    tick_spacing=[0.25, 0.25],
):
    """
    Plots a sample in the deviation domain on the provided matplotlib axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot.
    sample (array-like): 2D sample data with scaled position and theta.
    x_label (str, optional): Label for the x-axis. Defaults to "X".
    y_label (str, optional): Label for the y-axis. Defaults to "Y".
    x_bounds (tuple, optional): Bounds for the x-axis. Defaults to (-0.15, 0.15).
    y_bounds (tuple, optional): Bounds for the y-axis. Defaults to (-1, 1).
    Returns:
    matplotlib.axes.Axes: The axis with the plot.
    """
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Define the polygon vertices
    polygon_points = np.array(
        [[x_bounds[0], 0], [0, y_bounds[1]], [x_bounds[1], 0], [0, y_bounds[0]]]
    )

    # Create a polygon patch
    polygon = Polygon(polygon_points, closed=True, edgecolor="r", facecolor="none", lw=2)
    ax.add_patch(polygon)

    # Scatter plot of the data points
    inside = []
    outside = []
    for point in sample:
        if polygon.get_path().contains_point(point):
            inside.append(point)
        else:
            outside.append(point)

    inside = np.array(inside)
    outside = np.array(outside)

    # Plot points
    if len(inside) > 0:
        ax.scatter(inside[:, 0], inside[:, 1], c="b")
    if len(outside) > 0:
        ax.scatter(outside[:, 0], outside[:, 1], c="k")

    # Set the axis bounds
    ax.set_xbound([x * ratio_bounds for x in x_bounds])
    ax.set_ybound([y * ratio_bounds for y in y_bounds])

    # Adjust aspect ratio
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    xspan = xright - xleft
    yspan = ytop - ybottom
    ax.set_aspect(abs(xspan / yspan))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[0]))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[1]))

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    # Display count of points outside the polygon
    n_failed = len(outside)
    ax.text(
        0.05,
        0.95,
        f"$n_{{failed}} = {n_failed}$",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
    )

    return ax


def arrange_axes_in_grid(axes_list):
    """
    Arrange a list of matplotlib axes in a 2D grid as close to a square as possible.

    Parameters:
    axes_list (list): List of matplotlib axes to be arranged in the grid.
    """
    num_axes = len(axes_list)
    num_cols = math.ceil(math.sqrt(num_axes))
    num_rows = math.ceil(num_axes / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Flatten axs array for easier indexing
    axs = axs.flatten()

    for i, ax in enumerate(axes_list):
        # Get the subplot axis
        subplot_ax = axs[i]

        # Transfer properties from the original axis to the subplot axis
        for line in ax.get_lines():
            subplot_ax.plot(
                line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color()
            )

        # Transfer other attributes (titles, labels, etc.)
        subplot_ax.set_title(ax.get_title())
        subplot_ax.set_xlabel(ax.get_xlabel())
        subplot_ax.set_ylabel(ax.get_ylabel())
        subplot_ax.legend()
        subplot_ax.grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()


def compare_jacobians(
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    plot_type: str = "side_by_side",
    class_labels: np.ndarray = None,
):
    """
    Compare two Jacobians visually using bar plots with optional class-based background coloring.

    Parameters:
    jacobian1 (np.ndarray): First Jacobian array of shape (32,).
    jacobian2 (np.ndarray): Second Jacobian array of shape (32,).
    plot_type (str): Either 'side_by_side' or 'overlay' to choose plot style.
    class_labels (np.ndarray): Array of class labels to apply background coloring. Must be same length as Jacobians.

    Returns:
    None: Displays a bar plot for visual comparison.
    """
    if jacobian1.shape != jacobian2.shape or jacobian1.shape != (32,):
        raise ValueError("Both Jacobians must be 1D arrays of length 32.")

    if class_labels is not None and len(class_labels) != len(jacobian1):
        raise ValueError("The class labels must have the same length as the Jacobians.")

    indices = np.arange(32)  # Indices for x-axis

    plt.figure(figsize=(10, 6))

    # Plot the background colors based on class_labels
    if class_labels is not None:
        for idx in range(len(indices)):
            class_color = otaf.plotting.color_palette_2[
                class_labels[idx] % len(otaf.plotting.color_palette_2)
            ]
            plt.gca().add_patch(
                Rectangle(
                    (idx - 0.5, min(min(jacobian1), min(jacobian2))),
                    1,
                    max(max(jacobian1), max(jacobian2)) - min(min(jacobian1), min(jacobian2)),
                    color=class_color,
                    alpha=0.2,
                    lw=0,
                )
            )

    if plot_type == "side_by_side":
        bar_width = 0.4
        plt.bar(indices - bar_width / 2, jacobian1, width=bar_width, label="Jacobian 1")
        plt.bar(indices + bar_width / 2, jacobian2, width=bar_width, label="Jacobian 2")
    elif plot_type == "overlay":
        plt.bar(indices, jacobian1, width=0.4, label="Jacobian 1", alpha=0.7)
        plt.bar(indices, jacobian2, width=0.4, label="Jacobian 2", alpha=0.7, bottom=jacobian1)
    else:
        raise ValueError("plot_type must be either 'side_by_side' or 'overlay'")

    plt.xlabel("Index")
    plt.ylabel("Jacobian Value")
    plt.title("Comparison of Jacobians")
    plt.legend()
    plt.tight_layout()
    plt.show()


def pair_plot(data, labels=None, subset_labels=None, plot_half="both", hide_diag=False, color_by=None):
    """
    Creates a pair plot of all combinations of variables in the data array, with additional customization options.

    Parameters:
    - data: 2D numpy array or similar where each column corresponds to a variable.
    - labels: List of variable names corresponding to the columns of data.
              If None, default labels will be used as 'Var1', 'Var2', etc.
    - subset_labels: List of variable names to plot a specific subset of the pair plot.
                     If None, all variables will be plotted.
    - plot_half: Whether to plot the full matrix ('both'), only the lower triangular ('lower'),
                 or only the upper triangular ('upper'). Defaults to 'both'.
    - hide_diag: If True, the diagonal plots will be hidden.
    - color_by: If provided, the data points will be colored based on the values of this column (array-like).

    Returns:
    - A customized pair plot of all variable combinations.
    """
    # Ensure the data is a 2D array
    if len(data.shape) != 2:
        raise ValueError("Input data must be a 2D array-like structure")

    num_vars = data.shape[1]

    # If labels are not provided, generate default labels
    if labels is None:
        labels = [f"Var{i+1}" for i in range(num_vars)]

    # Check if the number of labels matches the number of variables in the data
    if len(labels) != num_vars:
        raise ValueError("The length of labels must match the number of columns in data")

    # Subset the data and labels if needed
    if subset_labels is not None:
        subset_indices = [labels.index(lbl) for lbl in subset_labels if lbl in labels]
        data = data[:, subset_indices]
        labels = subset_labels
        num_vars = len(subset_indices)

    fig, axes = plt.subplots(num_vars, num_vars, figsize=(12, 12))

    # Define color for points if color_by is provided
    if color_by is not None:
        color_data = color_by
    else:
        color_data = "blue"

    for i, j in product(range(num_vars), repeat=2):
        if plot_half == "lower" and i < j:
            axes[i, j].axis('off')
            continue
        if plot_half == "upper" and i > j:
            axes[i, j].axis('off')
            continue

        if i == j:
            if not hide_diag:
                axes[i, j].hist(
                    data[:, i], bins=20, alpha=0.6, color="gray"
                )  # Show histograms on the diagonal
        else:
            sc = axes[i, j].scatter(data[:, j], data[:, i], c=color_data, alpha=0.5)

            # Optional: Add a regression line (optional, but can be useful for trends)
            if color_by is None:
                m, b = np.polyfit(data[:, j], data[:, i], 1)
                axes[i, j].plot(data[:, j], m*data[:, j] + b, color='red')

        if j == 0:
            axes[i, j].set_ylabel(labels[i])
        if i == num_vars - 1:
            axes[i, j].set_xlabel(labels[j])

    plt.tight_layout()
    plt.show()


def plot_combined_CDF(distributions, x_min, x_max, color_list=None, legend_list=None):
    """Plot combined CDFs for multiple distributions."""
    color_list = color_list or []
    legend_list = legend_list or []

    graph_full = distributions[0].drawCDF(x_min, x_max)

    if color_list:
        graph_full.setColors(color_list[0])
    if legend_list:
        graph_full.setLegends(legend_list[0])

    for i, dist in enumerate(distributions[1:], start=1):
        graph_temp = dist.drawCDF(x_min, x_max)
        if color_list:
            graph_temp.setColors(color_list[i])
        if legend_list:
            graph_temp.setLegends(legend_list[i])
        graph_full.add(graph_temp)

    return graph_full



def save_plot(fig=None, ax=None, filename='plot', folder='.', file_format='png', dpi=300, **kwargs):
    """
    Saves a Matplotlib figure or axis to a specified location with customizable parameters.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to save. Default is None.
        ax (matplotlib.axes.Axes): The axis to save. Default is None.
        filename (str): The name of the file (with or without extension). Default is 'plot'.
        folder (str): The folder where the file will be saved. Default is current directory '.'.
        file_format (str): The format to save the file (e.g., 'png', 'pdf', 'svg'). Default is 'png'.
        dpi (int): The dots per inch (DPI) for the output file. Default is 300.
        **kwargs: Other optional keyword arguments for plt.savefig().

    Returns:
        str: The full path of the saved file.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Determine the figure to save
    if fig is None and ax is not None:
        fig = ax.get_figure()
    elif fig is None and ax is None:
        raise ValueError("Either a figure or axis must be provided.")

    # Check if the filename has an extension
    if '.' in filename:
        file_format = filename.split('.')[-1]  # Infer file format from the extension
    else:
        # Append the default or specified format if no extension is provided
        filename = f"{filename}.{file_format}"

    # Construct the full file path
    file_path = os.path.join(folder, filename)

    # Save the figure
    fig.savefig(file_path, format=file_format, dpi=dpi, **kwargs)

    return file_path


def plot_gld_pbox_cdf(gld_obj, lower_params, upper_params, x_values, xtol=1e-5, labels=None, colors=('tab:blue', 'tab:orange'), fill_color='gray', alpha=0.3, xlabel="X", ylabel="P", title="Probability-Box of the CDF"):
    """
    Plot a Probability-Box (P-Box) using two sets of GLD parameters representing the lower and upper bounds of the CDF.

    Parameters
    ----------
    gld_obj : object
        An instance of the GLD class from gldpy.
    lower_params : array-like
        Parameters for the lower bound GLD.
    upper_params : array-like
        Parameters for the upper bound GLD.
    x_values : array-like
        X values where the P-box should be computed.
    xtol : float, optional
        Tolerance for numerical CDF computation. Default is 1e-5.
    labels : tuple of str, optional
        Labels for the lower and upper bound CDFs. Default is None.
    colors : tuple of str, optional
        Colors for the lower and upper bound CDFs. Default is ('tab:blue', 'tab:orange').
    fill_color : str, optional
        Color for the filled P-box region. Default is 'gray'.
    alpha : float, optional
        Transparency of the filled region. Default is 0.3.
    """

    # Compute CDF values for the given x_values
    lower_cdf = gld_obj.CDF_num(x_values, lower_params, xtol=xtol)
    upper_cdf = gld_obj.CDF_num(x_values, upper_params, xtol=xtol)

    # Plot P-box
    plt.figure(dpi=150)
    plt.grid(True)
    plt.plot(x_values, lower_cdf, color=colors[0], label=labels[0] if labels else 'Lower bound')
    plt.plot(x_values, upper_cdf, color=colors[1], label=labels[1] if labels else 'Upper bound')
    plt.fill_between(x_values, lower_cdf, upper_cdf, color=fill_color, alpha=alpha)

    # Labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.show()
