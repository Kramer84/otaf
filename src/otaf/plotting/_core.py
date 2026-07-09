from __future__ import annotations

__author__ = "Kramer84"
__all__ = [
    "DEFAULT_FIG_STYLE",
    "_cm_to_inch",
    "_merge_style",
    "_setup_mpl_from_style",
    "_new_fig_ax",
    "_finalize_figure",
    "trimesh_scene_as_notebook_scene",
    "hex_to_rgba",
    "arrange_axes_in_grid",
]
import math

import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import TYPE_CHECKING, Any, List, Tuple, Union

if TYPE_CHECKING:
    import trimesh
import otaf.exceptions as otaf_exceptions

DEFAULT_FIG_STYLE = {
    "figsize_cm": (16, 9),
    "font_size": 10,
    "font_family": "serif",
    "usetex": False,
    "colors": {
        "c1": "#1f77b4",
        "c2": "#d62728",
        "c3": "#2ca02c",
        "c4": "#9467bd",
        "c5": "#8c564b",
        "c6": "#e377c2",
        "c7": "#7f7f7f",
        "c8": "#bcbd22",
    },
    "grid": True,
    "legend": True,
    "tight_layout": True,
    "title": True,
    "labels": True,
    "save": False,
    "save_path": None,
    "dpi": 300,
    "transparent": True,
    "show": True,
    "xlim": None,
    "ylim": None,
}


def _cm_to_inch(cm):
    """Convert a (width, height) tuple from centimeters to inches."""
    if cm is None:
        return None
    return tuple((c / 2.54 for c in cm))


def _merge_style(style=None, **overrides):
    """
    Merge DEFAULT_FIG_STYLE, a user style dict, and per-call overrides.

    Parameters
    ----------
    style : dict or None
        Base style dict shared across figures.
    **overrides
        Per-call overrides like figsize_cm=..., dpi=..., legend=False, etc.
    """
    cfg = DEFAULT_FIG_STYLE.copy()
    cfg["colors"] = DEFAULT_FIG_STYLE["colors"].copy()
    if style is not None:
        for k, v in style.items():
            if k == "colors":
                cfg["colors"].update(v)
            else:
                cfg[k] = v
    for k, v in overrides.items():
        if v is not None:
            if k == "colors":
                cfg["colors"].update(v)
            else:
                cfg[k] = v
    return cfg


def _setup_mpl_from_style(cfg):
    """
    Apply global Matplotlib settings for fonts / LaTeX, using
    pdfLaTeX + Libertinus (libertinus-type1 + libertinust1math).
    """
    rc = {
        "font.size": cfg.get("font_size", 9),
        "font.family": cfg.get("font_family", "serif"),
    }
    if cfg.get("usetex", False):
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = (
            "\n\\usepackage[T1]{fontenc}\n\\usepackage{amsmath}\n\\usepackage{libertinus}        % wrapper, picks libertinus-type1 under pdfLaTeX\n\\usepackage{libertinust1math}  % Libertinus Math for pdfLaTeX\n"
        )
    else:
        rc["text.usetex"] = False
    plt.rcParams.update(rc)


def _new_fig_ax(cfg):
    """Create a new (fig, ax) using the style config."""
    _setup_mpl_from_style(cfg)
    figsize = _cm_to_inch(cfg.get("figsize_cm")) if cfg.get("figsize_cm") else None
    fig, ax = plt.subplots(figsize=figsize)
    return (fig, ax)


def _finalize_figure(fig, ax, cfg):
    """Apply grid, x/y limits, saving and showing."""
    if cfg.get("grid", True):
        ax.grid(True)
    if cfg.get("xlim") is not None:
        ax.set_xlim(*cfg["xlim"])
    if cfg.get("ylim") is not None:
        ax.set_ylim(*cfg["ylim"])
    if cfg.get("tight_layout", True):
        fig.tight_layout()
    if cfg.get("save", False) and cfg.get("save_path") is not None:
        fig.savefig(
            cfg["save_path"],
            dpi=cfg.get("dpi", 300),
            transparent=cfg.get("transparent", True),
            bbox_inches="tight",
        )
    if cfg.get("show", True):
        plt.show()
    return (fig, ax)


def trimesh_scene_as_notebook_scene(
    scene: trimesh.Scene, background_hex_color: str = "e6e6e6"
) -> Any:
    """
    Convert a Trimesh scene into an interactive widget for a Jupyter notebook.

    Generates the HTML/JS representation of the scene and modifies the underlying
    Three.js canvas setup to apply a custom background color.

    Parameters
    ----------
    scene : trimesh.Scene
        The Trimesh scene object to be displayed.
    background_hex_color : str, default="e6e6e6"
        The hexadecimal color code (without the '#' prefix) for the background.

    Returns
    -------
    Any
        The interactive notebook widget containing the rendered scene.

    Raises
    ------
    ImportError
        If trimesh viewer dependencies are not installed.
    """
    try:
        from trimesh import viewer
    except ImportError:
        raise otaf_exceptions._raise_missing_dependency(
            "trimesh", "trimesh_scene_as_notebook_scene"
        )
    notebook_scene = viewer.notebook.scene_to_notebook(scene)
    notebook_scene.data = notebook_scene.data.replace(
        "scene.background=new THREE.Color(0xffffff)",
        f"scene.background=new THREE.Color(0x{background_hex_color})",
    )
    return notebook_scene


@beartype
def hex_to_rgba(
    hex_color: str, alpha: float = 1.0, as_float: bool = False
) -> Tuple[Union[float, int], Union[float, int], Union[float, int], float]:
    """
    Convert a hexadecimal color string to an RGBA tuple.

    Strips any leading '#' character, extracts the red, green, and blue components,
    and optionally normalizes the RGB values to floats before appending the alpha channel.

    Parameters
    ----------
    hex_color : str
        The hexadecimal color code string (e.g., '#ffffff' or 'e6e6e6').
    alpha : float, default=1.0
        The alpha (transparency) value, typically between 0.0 and 1.0.
    as_float : bool, default=False
        If True, returns RGB values as floats scaled between 0.0 and 1.0.
        If False, returns them as integers between 0 and 255.

    Returns
    -------
    Tuple[Union[float, int], Union[float, int], Union[float, int], float]
        A tuple containing the Red, Green, Blue, and Alpha values in order.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    if as_float:
        r /= 255.0
        g /= 255.0
        b /= 255.0
    return (r, g, b, alpha)


def arrange_axes_in_grid(axes_list: List[Any]) -> None:
    """
    Arrange a list of existing Matplotlib axes into a new, compact 2D grid plot.

    Calculates the optimal number of rows and columns to form a near-square layout,
    creates a new figure, copies lines and labels from the source axes into the
    subplots, and deletes any unused grid positions.

    Parameters
    ----------
    axes_list : List[Any]
        A list of Matplotlib axis objects whose lines and metadata should be copied.
    """
    num_axes = len(axes_list)
    num_cols = math.ceil(math.sqrt(num_axes))
    num_rows = math.ceil(num_axes / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axs = axs.flatten()
    for i, ax in enumerate(axes_list):
        subplot_ax = axs[i]
        for line in ax.get_lines():
            subplot_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                label=line.get_label(),
                color=line.get_color(),
            )
        subplot_ax.set_title(ax.get_title())
        subplot_ax.set_xlabel(ax.get_xlabel())
        subplot_ax.set_ylabel(ax.get_ylabel())
        subplot_ax.legend()
        subplot_ax.grid(True)
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()
