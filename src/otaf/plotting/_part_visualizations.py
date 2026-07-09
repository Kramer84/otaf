from __future__ import annotations

__author__ = "Kramer84"
__all__ = ["plot_rect_part", "plot_deviated_surfs"]
import numpy as np
from beartype.typing import Any, Optional

from ._core import _merge_style

def plot_rect_part(
    ax: Any, scale_factor: float = 100, style: Optional[dict] = None, **style_kwargs
) -> Any:
    """
    Plot the rectangular component of the example mesh or model on an existing axis.

    Draws a bounding rectangle using line segments, modifies the visual aspect
    ratio to be twice as tall and half as wide, and overlays dimensional tracking
    arrows along with text annotations. Uses global styling for colors and fonts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object where the geometric lines and annotations are drawn.
    scale_factor : float, default=100
        A multiplier applied to scale the horizontal span of the indicator arrows.
    style : dict, optional
        Base style dictionary.
    **style_kwargs :
        Overrides for DEFAULT_FIG_STYLE.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axis object containing the rendered layout elements.
    """
    cfg = _merge_style(style, **style_kwargs)
    c_main = cfg["colors"].get("c7", "#7f7f7f")
    c_accent = cfg["colors"]["c2"]
    f_size = cfg.get("font_size", 10)
    t_ = 10
    ax.plot([0, 0], [0, 10], color=c_main, linestyle="-")
    ax.plot([0, 100], [10, 10], color=c_main, linestyle="-")
    ax.plot([100, 100], [10, 0], color=c_main, linestyle="--")
    ax.plot([100, 0], [0, 0], color=c_main, linestyle="-")
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
        color=c_accent,
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
        color=c_accent,
        length_includes_head=True,
    )
    ax.text(
        95,
        12,
        f"2t*{scale_factor}",
        color=c_accent,
        fontsize=f_size + 4,
        fontweight="bold",
    )
    ax.text(
        -23,
        5,
        "A",
        color=c_main,
        fontsize=f_size + 6,
        fontweight="bold",
        bbox=dict(facecolor="none", edgecolor=c_main),
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
        color=c_main,
        length_includes_head=True,
    )
    return ax


def plot_deviated_surfs(
    ax: Any,
    se_pos: Any,
    se_rot: Any,
    lambda_: float,
    scale_factor: float,
    style: Optional[dict] = None,
    **style_kwargs,
) -> Any:
    """
    Overlay deviated profile lines onto a plot to visualize geometric defects.

    Iterates through paired positional and rotational variations, scales them by
    the provided modifiers, and draws calculated deviation lines across a constant
    vertical span on an existing Matplotlib axis using global styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object where the geometric lines will be drawn.
    se_pos : Any
        An array or sequence containing positional deviation tracking data.
    se_rot : Any
        An array or sequence containing rotational deviation tracking data.
    lambda_ : float
        A weight multiplier balancing positional vs rotational defect components.
    scale_factor : float
        A scalar multiplier used to visually exaggerate the defects for readability.
    style : dict, optional
        Base style dictionary.
    **style_kwargs :
        Overrides for DEFAULT_FIG_STYLE.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axis object containing the plotted deviation lines.

    Raises
    ------
    AssertionError
        If the lengths of the `se_pos` and `se_rot` arrays do not match.
    """
    cfg = _merge_style(style, **style_kwargs)
    c_dev = cfg["colors"]["c1"]
    se_pos = np.squeeze(se_pos)
    se_rot = np.squeeze(se_rot)
    assert len(se_pos) == len(se_rot), "The length of se_pos and se_rot must be equal"
    l = 10
    x_line = np.array([100.0, 100.0])
    y_line = np.array([10.0, 0.0])
    for i in range(len(se_pos)):
        x_ltemp = x_line + lambda_ * se_pos[i] * scale_factor
        x_ltemp[0] -= (1 - lambda_) * se_rot[i] * 0.5 * l * scale_factor
        x_ltemp[1] += (1 - lambda_) * se_rot[i] * 0.5 * l * scale_factor
        ax.plot(x_ltemp, y_line, color=c_dev, linestyle="-")
    return ax
