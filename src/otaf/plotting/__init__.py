"""Codes to plot the various objects, results, shapes, distributions, etc."""

from __future__ import annotations

__author__ = "Kramer84"
from ._color_palettes import color_palette_1, color_palette_2, color_palette_3
from ._core import (
    DEFAULT_FIG_STYLE,
    arrange_axes_in_grid,
    hex_to_rgba,
    trimesh_scene_as_notebook_scene,
)
from ._deviation_domain import plot_deviation_domain
from ._part_visualizations import plot_deviated_surfs, plot_rect_part
from ._topology_plots import generate_topological_tikz
from ._uncertainty_plots import plot_ensemble_gld_pbox_cdf, plot_gld_pbox_cdf

__all__ = [
    "DEFAULT_FIG_STYLE",
    "arrange_axes_in_grid",
    "color_palette_1",
    "color_palette_2",
    "color_palette_3",
    "generate_topological_tikz",
    "hex_to_rgba",
    "plot_deviated_surfs",
    "plot_deviation_domain",
    "plot_ensemble_gld_pbox_cdf",
    "plot_gld_pbox_cdf",
    "plot_rect_part",
    "trimesh_scene_as_notebook_scene",
]
