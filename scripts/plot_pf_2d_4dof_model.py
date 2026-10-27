import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from gldpy import GLD
import otaf

# Import the model definition you provided
from otaf.example_models import model1


DEFAULT_FIG_STYLE = {
    "figsize_cm": (16, 9),          # width, height in cm
    "font_size": 13,
    "font_family": "serif",
    "usetex": True,                # set True if you want LaTeX-rendered text

    # Generic color slots you can reuse everywhere
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
    "title": False,
    "labels": False,
    
    # Saving options
    "save": False,
    "save_path": None,
    "dpi": 300,
    "transparent": True,

    # Display options
    "show": True,

    # Optional overrides for limits
    "xlim": None,
    "ylim": None,
}

def _cm_to_inch(cm):
    """Convert a (width, height) tuple from centimeters to inches."""
    if cm is None:
        return None
    return tuple(c / 2.54 for c in cm)

def _merge_style(style=None, **overrides):
    """
    Merge DEFAULT_FIG_STYLE, a user style dict, and per-call overrides.
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
    Apply global Matplotlib settings for fonts / LaTeX.
    """
    rc = {
        "font.size": cfg.get("font_size", 9),
        "font.family": cfg.get("font_family", "serif"),
    }

    if cfg.get("usetex", False):
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"""
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{libertinus}
\usepackage{libertinust1math}
"""
    else:
        rc["text.usetex"] = False

    plt.rcParams.update(rc)


def _new_fig_ax(cfg):
    """Create a new (fig, ax) using the style config."""
    _setup_mpl_from_style(cfg)
    figsize = _cm_to_inch(cfg.get("figsize_cm")) if cfg.get("figsize_cm") else None
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def _finalize_figure(fig, ax, cfg):
    """Apply grid, x/y limits, saving and showing."""
    if cfg.get("grid", True):
        ax.grid(True, linestyle=":", alpha=0.6)

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

    return fig, ax

def format_prob_tex(p: float, digits: int = 2, notation: int = 1) -> str:
    """Return a LaTeX-friendly scientific notation string for a probability."""
    if p == 0:
        return "0"
    exp = int(np.floor(np.log10(p)))
    mant = p / 10**exp
    # Use nested f-string interpolation to dynamically set the decimal precision
    if notation == 1 :
        return rf"{mant:.{digits}f} \times 10^{{{exp}}}"
    if notation == 2 :
        return rf"{mant:.{digits}f}e^{{{exp}}}"


def build_peak_highlight_cmap(base_cmap_name: str = "viridis", n_colors: int = 200, highlight_color: str = "#FF0000"):
    """
    Builds a colormap of n_colors where the bottom n_colors-1 colors
    come from base_cmap and the top color is set to highlight_color.
    """
    base_cmap = plt.get_cmap(base_cmap_name)
    colors = base_cmap(np.linspace(0.0, 1.0, n_colors - 1))
    red_rgba = np.array([mcolors.to_rgba(highlight_color)])
    spliced_colors = np.vstack([colors, red_rgba])
    return ListedColormap(spliced_colors)


def plot_single_pf_grid(
    U4_grid,
    U5_grid,
    PF_grid,
    alpha,
    style=None,
    norm_type="power",
    n_levels=200,
    highlight_pct=0.05,
    highlight_mode="value",
    highlight_color="#FF0000",
    base_cmap="viridis",
    log_min_floor=1e-5,
    **kwargs,
):
    """
    Plots a single Probability of Failure grid with:
      - 200 contour levels
      - Pure red highlight for peak probabilities mapped strictly to the top bin.
      - Dynamic gradient resolution (log, power, or linear)
    """
    cfg = _merge_style(style, **kwargs)
    fig, ax = _new_fig_ax(cfg)

    vmax = float(np.max(PF_grid))
    
    # 1. Determine meaningful vmin to prevent dynamic range collapse
    pos_vals = PF_grid[PF_grid > 0]
    if len(pos_vals) > 0:
        vmin = float(max(np.min(pos_vals), log_min_floor))
    else:
        vmin = log_min_floor
    
    if vmax <= vmin:
        vmax = vmin + 1e-4

    # 2. Compute the threshold boundary for the red highlight
    pct = highlight_pct if highlight_pct < 1.0 else highlight_pct / 100.0

    if highlight_mode == "quantile":
        threshold_val = float(np.percentile(PF_grid, 100.0 * (1.0 - pct)))
    else:
        # Value-based: top pct band near the maximum
        threshold_val = vmax - pct * (vmax - vmin)

    # Ensure threshold does not collapse levels
    threshold_val = max(vmin + 1e-6, min(threshold_val, vmax - 1e-6))

    # 3. Build Level Spacing strictly for n_levels total bins
    # We need n_levels + 1 boundaries to create exactly n_levels bins
    if norm_type == "log":
        sub_levels = np.geomspace(vmin, threshold_val, n_levels)
    elif norm_type == "power":
        # Power law clustering at the top (expanding contrast for high Pf)
        t = np.linspace(0, 1, n_levels)
        sub_levels = threshold_val - (threshold_val - vmin) * ((1 - t)**2)
    else:
        sub_levels = np.linspace(vmin, threshold_val, n_levels)

    # Append absolute maximum to form the final red bin
    levels = np.append(sub_levels, vmax)

    # Create colormap with exactly n_levels colors
    custom_cmap = build_peak_highlight_cmap(
        base_cmap_name=base_cmap,
        n_colors=n_levels,
        highlight_color=highlight_color
    )
    
    # Force 1-to-1 mapping of bin index to colormap index
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=n_levels)

    # Clip data strictly into [vmin, vmax] for contouring
    pf_plot = np.clip(PF_grid, vmin, vmax)

    # 4. Render Contour Plot
    contour = ax.contourf(
        U4_grid,
        U5_grid,
        pf_plot,
        levels=levels,
        cmap=custom_cmap,
        norm=norm,
        extend="neither"
    )

    # 5. Colorbar Setup
    # Limit ticks to 6 evenly spaced markers to avoid rendering 200 black lines
    cb_ticks = np.linspace(vmin, vmax, 6)
    
    # Wrap existing format_prob_tex in math-mode delimiters ($...$)
    def tex_tick_formatter(x, pos):
        return f"${format_prob_tex(x, digits=1, notation=2)}$"

    cbar = fig.colorbar(
        contour, 
        ax=ax, 
        ticks=cb_ticks, 
        format=FuncFormatter(tex_tick_formatter)
    )

    # 6. Apply Labels and Titles
    usetex = cfg.get("usetex", False)
    if cfg.get("labels", True):
        if usetex:
            cbar.set_label(rf"Probability of Failure ($P_f$) [$\alpha = {alpha}$]")
            ax.set_xlabel(r"$\lambda_{d,1}$")
            ax.set_ylabel(r"$\lambda_{d,2}$")
        else:
            cbar.set_label(f"Probability of Failure (Pf) [alpha = {alpha}]")
            ax.set_xlabel("u_d_4 Allocation Scale (0=Max Rot, 1=Max Trans)")
            ax.set_ylabel("u_d_5 Allocation Scale (0=Max Rot, 1=Max Trans)")

    if cfg.get("title", True):
        peak_str = format_prob_tex(vmax) if usetex else f"{vmax:.2e}"
        if usetex:
            ax.set_title(rf"$\alpha = {alpha}$, Max $P_f \approx {peak_str}$")
        else:
            ax.set_title(f"Failure Probability (alpha = {alpha}, Peak Pf approx {peak_str})")

    fig, ax = _finalize_figure(fig, ax, cfg)
    return fig, ax

###################################################

def get_model_evaluator(sample, mu_vect, neural_model):
    def evaluate(x):
        x_transformed = (sample - mu_vect) * x + mu_vect
        prediction = neural_model.evaluate_model_non_standard_space(x_transformed)
        return np.squeeze(prediction.numpy())
    return evaluate

def main():
    parser = argparse.ArgumentParser(description="Grid Evaluation for 4-DOF OTAF Model")
    parser.add_argument("--surrogate-path", type=str, default="./model1_4_dof_surrogate.pth", help="Path to the .pth surrogate file")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0], help="List of failure limits/alphas")
    parser.add_argument("--grid-res", type=int, default=20, help="Resolution of the 2D grid")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the Monte Carlo sample")
    parser.add_argument("--mc-size", type=int, default=100000, help="Monte Carlo sample size")
    parser.add_argument("--norm-type", type=str, choices=["log", "linear", "power"], default="log", help="Contour normalization type")
    parser.add_argument("--highlight-pct", type=float, default=0.05, help="Top percentage to color in red (e.g. 0.05 or 5.0)")
    parser.add_argument("--highlight-mode", type=str, choices=["value", "quantile"], default="value", help="Method to compute peak threshold")
    args = parser.parse_args()

    # 1. Setup Models and Distributions
    print("Loading surrogate model and distributions...")
    model_sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(args.surrogate_path)
    
    jointDist, symbols, max_std_vect, mu_vect = model1.get_distribution_params()
    otaf.common.set_seed(args.seed)
    sample_gld = np.array(jointDist.getSample(args.mc_size))
    
    model_eval_fn = get_model_evaluator(sample_gld, mu_vect, model_sur)
    gld = GLD('VSL')

    # 2. Construct the Grid
    print(f"Building {args.grid_res}x{args.grid_res} evaluation grid for alphas: {args.alphas}...")
    u4_scales = np.linspace(0, 1, args.grid_res)
    u5_scales = np.linspace(0, 1, args.grid_res)
    
    U4_grid, U5_grid = np.meshgrid(u4_scales, u5_scales)
    PF_grids = {alpha: np.zeros_like(U4_grid) for alpha in args.alphas}

    # 3. Evaluate Each Point on the Grid
    for i in range(args.grid_res):
        for j in range(args.grid_res):
            u4 = U4_grid[i, j]
            u5 = U5_grid[i, j]
            
            # Analytically determine the boundary gamma scales
            gamma4 = np.sqrt(1.0 - u4**2)
            gamma5 = np.sqrt(1.0 - u5**2)
            
            x_scaled = np.array([u4, gamma4, u5, gamma5])
            
            # Evaluate model via surrogate ONCE per spatial coordinate
            eval_vals = model_eval_fn(x_scaled)
            
            # Compute GLD parameters ONCE per spatial coordinate
            gld_params = gld.fit_LMM(eval_vals, disp_fit=False, disp_optimizer=False)
            
            # Apply the evaluated distribution against all alpha thresholds
            for alpha in args.alphas:
                if np.any(np.isnan(gld_params)):
                    # Fallback to empirical probability if GLD fails
                    pf_raw = (eval_vals < alpha).mean()
                else:
                    try:
                        pf_raw = gld.CDF_num(alpha, gld_params, xtol=1e-6)
                    except Exception:
                        pf_raw = (eval_vals < alpha).mean()
                
                # Extract scalar float (gld.CDF_num returns a tuple (cdf_val, abserr))
                if isinstance(pf_raw, (tuple, list, np.ndarray)):
                    pf_val = float(pf_raw[0])
                else:
                    pf_val = float(pf_raw)
                    
                PF_grids[alpha][i, j] = float(np.clip(pf_val, 0.0, 1.0))
                
        print(f"Row {i+1}/{args.grid_res} completed.")

    # 4. Plot the Results
    print("Generating individual contour plots...")
    
    for alpha in args.alphas:
        plot_filename = f"Pf_Grid_alpha_{alpha}.png"
        print(f"Saving {plot_filename}...")
        
        fig, ax = plot_single_pf_grid(
            U4_grid, 
            U5_grid, 
            PF_grids[alpha], 
            alpha=alpha,
            norm_type=args.norm_type,
            n_levels=200,
            highlight_pct=args.highlight_pct,
            highlight_mode=args.highlight_mode,
            highlight_color="#FF0000",
            base_cmap="viridis",
            log_min_floor=1e-5,
            save=True, 
            save_path=plot_filename, 
            dpi=600,
            show=False,
            usetex=True,
            transparent=True
        )
        
        plt.close(fig)

if __name__ == "__main__":
    main()
