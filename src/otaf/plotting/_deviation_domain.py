from __future__ import annotations

__author__ = "Kramer84"
__all__ = ["plot_deviation_domain"]
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from beartype import beartype
from beartype.typing import Any, Optional, Sequence, Tuple
from matplotlib.patches import Polygon

from ._core import _merge_style


def plot_deviation_domain(
    ax: Any,
    sample: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
    x_bounds: Tuple[float, float] = (-0.15, 0.15),
    y_bounds: Tuple[float, float] = (-1, 1),
    ratio_bounds: float = 1.5,
    tick_spacing: Sequence[float] = (0.25, 0.25),
    style: Optional[dict] = None,
    **style_kwargs,
) -> Any:
    """
    Plot sample coordinates on a Matplotlib axis with a diamond-shaped tolerance polygon.

    Draws a 4-sided boundary polygon using the specified limits, splits data points
    into groups based on whether they fall inside or outside the polygon boundaries,
    colors them using the global style configuration, and prints the failure count.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object where the polygon and data points will be rendered.
    sample : np.ndarray
        An (N, 2) array containing the coordinate points to plot.
    x_label : str, default="X"
        The text label for the horizontal X-axis.
    y_label : str, default="Y"
        The text label for the vertical Y-axis.
    x_bounds : Tuple[float, float], default=(-0.15, 0.15)
        The minimum and maximum horizontal intercept coordinates for the polygon.
    y_bounds : Tuple[float, float], default=(-1, 1)
        The minimum and maximum vertical intercept coordinates for the polygon.
    ratio_bounds : float, default=1.5
        A multiplier used to pad the outer limits of the plot viewport.
    tick_spacing : Sequence[float], default=(0.25, 0.25)
        The major locator stepping interval for the X and Y axes ticks respectively.
    style : dict, optional
        Base style dictionary.
    **style_kwargs :
        Overrides for DEFAULT_FIG_STYLE.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axis object containing the rendered polygon, scatter plots, and text.
    """
    cfg = _merge_style(style, **style_kwargs)
    c_poly = cfg["colors"]["c2"]
    c_inside = cfg["colors"]["c1"]
    c_outside = cfg["colors"].get("c7", "#7f7f7f")
    f_size = cfg.get("font_size", 10)
    if cfg.get("grid", True):
        ax.grid(True)
    if cfg.get("labels", True):
        ax.set_xlabel(x_label, fontsize=f_size)
        ax.set_ylabel(y_label, fontsize=f_size)
    polygon_points = np.array(
        [[x_bounds[0], 0], [0, y_bounds[1]], [x_bounds[1], 0], [0, y_bounds[0]]]
    )
    polygon = Polygon(
        polygon_points, closed=True, edgecolor=c_poly, facecolor="none", lw=2
    )
    ax.add_patch(polygon)
    inside = []
    outside = []
    for point in sample:
        if polygon.get_path().contains_point(point):
            inside.append(point)
        else:
            outside.append(point)
    inside = np.array(inside)
    outside = np.array(outside)
    if len(inside) > 0:
        ax.scatter(inside[:, 0], inside[:, 1], color=c_inside)
    if len(outside) > 0:
        ax.scatter(outside[:, 0], outside[:, 1], color=c_outside)
    ax.set_xbound([x * ratio_bounds for x in x_bounds])
    ax.set_ybound([y * ratio_bounds for y in y_bounds])
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    xspan = xright - xleft
    yspan = ytop - ybottom
    ax.set_aspect(abs(xspan / yspan))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[0]))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[1]))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    n_failed = len(outside)
    ax.text(
        0.05,
        0.95,
        f"$n_{{failed}} = {n_failed}$",
        transform=ax.transAxes,
        fontsize=f_size + 4,
        verticalalignment="top",
    )
    return ax
