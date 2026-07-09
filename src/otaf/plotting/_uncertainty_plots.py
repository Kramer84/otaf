from __future__ import annotations

__author__ = "Kramer84"
__all__ = ["plot_gld_pbox_cdf", "plot_ensemble_gld_pbox_cdf"]
import matplotlib.pyplot as plt
import numpy as np
from beartype.typing import Any, Optional, Tuple

from ._core import _finalize_figure, _merge_style, _new_fig_ax


def plot_gld_pbox_cdf(
    gld_obj: Any,
    lower_params: Any,
    upper_params: Any,
    x_values: Any,
    xtol: float = 1e-05,
    labels: Optional[Tuple[str, str]] = None,
    alpha: float = 0.3,
    xlabel: str = "X",
    ylabel: str = "P",
    title: str = "Probability-Box of the CDF",
    ax: Optional[Any] = None,
    style: Optional[dict] = None,
    **style_kwargs,
) -> Any:
    """
    Plot a Probability-Box (P-Box) by shading the region between two GLD cumulative distributions.

    Computes the numerical CDF arrays for both the lower and upper Generalized Lambda
    Distribution parameters, plots them as bounding lines on a designated Matplotlib axis,
    and fills the area between them to visualize the uncertainty envelope. Now utilizes
    global styling configurations.

    Parameters
    ----------
    gld_obj : Any
        An instance of the GLD class used to calculate numerical CDF values.
    lower_params : Any
        An array or sequence of parameters defining the lower-bound distribution.
    upper_params : Any
        An array or sequence of parameters defining the upper-bound distribution.
    x_values : Any
        An array or sequence of horizontal coordinate points where the curves are evaluated.
    xtol : float, default=1e-5
        The numerical convergence tolerance threshold passed to the CDF evaluation function.
    labels : Optional[Tuple[str, str]], default=None
        An optional pair of string labels used to identify the bounding curves in the plot legend.
    alpha : float, default=0.3
        The opacity level (ranging from 0.0 to 1.0) applied to the filled background region.
    xlabel : str, default="X"
        The text label for the horizontal X-axis.
    ylabel : str, default="P"
        The text label for the vertical Y-axis.
    title : str, default="Probability-Box of the CDF"
        The main header title displayed above the plot grid.
    style : dict, optional
        Base style dictionary to pass to the configuration merger.
    **style_kwargs :
        Overrides for DEFAULT_FIG_STYLE (e.g., dpi=150, save=True).

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The initialized figure and axes objects.
    """
    cfg = _merge_style(style, **style_kwargs)
    if ax is None:
        fig, ax = _new_fig_ax(cfg)
    else:
        fig = ax.get_figure()
    lower_cdf = gld_obj.CDF_num(x_values, lower_params, xtol=xtol)
    upper_cdf = gld_obj.CDF_num(x_values, upper_params, xtol=xtol)
    c_lower = cfg["colors"]["c1"]
    c_upper = cfg["colors"]["c2"]
    c_fill = cfg["colors"].get("c7", "#7f7f7f")
    label_low = labels[0] if labels else "Lower bound"
    label_high = labels[1] if labels else "Upper bound"
    ax.plot(x_values, lower_cdf, color=c_lower, label=label_low)
    ax.plot(x_values, upper_cdf, color=c_upper, label=label_high)
    ax.fill_between(x_values, lower_cdf, upper_cdf, color=c_fill, alpha=alpha)
    if cfg.get("labels", True):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if cfg.get("title", True):
        ax.set_title(title)
    if cfg.get("legend", True):
        ax.legend()
    return _finalize_figure(fig, ax, cfg)


def plot_ensemble_gld_pbox_cdf(
    gld_obj,
    param_list,
    x_values,
    ax=None,
    style=None,
    labels=("Lower bound", "Upper bound"),
    fill_color="gray",
    alpha=0.3,
    xlabel="Slack",
    ylabel="Probability of Failure ($P_f$)",
    title="Probability-Box of the CDF",
    add_zoom=False,
    zoom_x_range=(-0.05, 0.05),
    zoom_y_range=(1e-06, 0.1),
    inset_log_y=False,
    **style_kwargs,
):
    """
    Plot a P-Box using custom global styling and saving options.

    Parameters
    ----------
    ...
    style : dict, optional
        Base style dictionary.
    **style_kwargs :
        Overrides for DEFAULT_FIG_STYLE (e.g., save=True, save_path='my_plot.png')
    """

    def compute_fast_pbox_bounds(param_list, x_values):
        p_values = np.linspace(1e-06, 1 - 1e-06, 3000)
        p = p_values[np.newaxis, :]
        p0, p1, p2, p3 = (
            param_list[:, 0:1],
            param_list[:, 1:2],
            param_list[:, 2:3],
            param_list[:, 3:4],
        )
        term1 = (1 - p2) * (p**p3 - 1) / p3
        term2 = p2 * ((1 - p) ** p3 - 1) / p3
        q_matrix = p0 + (term1 - term2) * p1
        composite_min_q = q_matrix.min(axis=0)
        composite_max_q = q_matrix.max(axis=0)
        upper_cdf = np.interp(x_values, composite_min_q, p_values, left=0.0, right=1.0)
        lower_cdf = np.interp(x_values, composite_max_q, p_values, left=0.0, right=1.0)
        return (lower_cdf, upper_cdf)

    cfg = _merge_style(style, **style_kwargs)
    if ax is None:
        fig, ax = _new_fig_ax(cfg)
    else:
        fig = ax.get_figure()
    lower_cdf, upper_cdf = compute_fast_pbox_bounds(param_list, x_values)
    c1 = cfg["colors"]["c1"]
    c2 = cfg["colors"]["c2"]
    ax.plot(x_values, lower_cdf, color=c1, label=labels[0], linewidth=1.5)
    ax.plot(x_values, upper_cdf, color=c2, label=labels[1], linewidth=1.5)
    ax.fill_between(
        x_values,
        lower_cdf,
        upper_cdf,
        color=fill_color,
        alpha=alpha,
        label="Epistemic Uncertainty",
    )
    ax.axvline(
        0.0,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Failure Threshold (Slack = 0)",
    )
    if cfg.get("labels", True):
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
    if cfg.get("title", True):
        ax.set_title(title, pad=15)
    if add_zoom:
        axins = ax.inset_axes([0.075, 0.25, 0.3, 0.3])
        axins.patch.set_facecolor("white")
        axins.patch.set_alpha(1.0)
        axins.set_zorder(100)
        axins.grid(False)
        axins.plot(x_values, lower_cdf, color=c1, linewidth=1.5)
        axins.plot(x_values, upper_cdf, color=c2, linewidth=1.5)
        axins.fill_between(
            x_values, lower_cdf, upper_cdf, color=fill_color, alpha=alpha
        )
        axins.axvline(0.0, color="red", linestyle="--", linewidth=1)
        axins.set_xlim(zoom_x_range)
        axins.set_ylim(zoom_y_range)
        if inset_log_y:
            axins.set_yscale("log")
        axins.tick_params(
            axis="both", which="major", labelsize=cfg.get("font_size", 9) - 1
        )
        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.2, alpha=0.6)
    if cfg.get("legend", True):
        ax.legend(loc="upper left", framealpha=0.9)
    return _finalize_figure(fig, ax, cfg)
