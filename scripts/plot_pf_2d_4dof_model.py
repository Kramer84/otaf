import argparse
import numpy as np
import matplotlib.pyplot as plt
from gldpy import GLD
import otaf
from matplotlib.colors import LogNorm

# Import the model definition you provided
from otaf.example_models import model1


DEFAULT_FIG_STYLE = {
    "figsize_cm": (16, 9),          # width, height in cm
    "font_size": 10,
    "font_family": "serif",
    "usetex": False,                # set True if you want LaTeX-rendered text

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
    "title": True,
    "labels":True,
    
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
\usepackage{libertinus}        % wrapper, picks libertinus-type1 under pdfLaTeX
\usepackage{libertinust1math}  % Libertinus Math for pdfLaTeX
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

    return fig, ax

def format_prob_tex(p: float) -> str:
    """Return a LaTeX-friendly scientific notation string for a probability."""
    if p == 0:
        return "0"
    exp = int(np.floor(np.log10(p)))
    mant = p / 10**exp
    return rf"{mant:.2f} \times 10^{{{exp}}}"


def plot_single_pf_grid(U4_grid, U5_grid, PF_grid, alpha, style=None, use_lognorm=True, **kwargs):
    """
    Plots a single Probability of Failure grid for a given alpha limit.
    """
    # 1. Merge configurations
    cfg = _merge_style(style, **kwargs)
    
    # 2. Create Figure and Axes
    fig, ax = _new_fig_ax(cfg)
    
    # 3. Plot Data
    pf_safe = np.clip(PF_grid, 1e-16, 1.0)
    
    if use_lognorm:
        try:
            contour = ax.contourf(
                U4_grid, U5_grid, pf_safe, 
                levels=50, 
                cmap='viridis',
                norm=LogNorm(vmin=pf_safe.min(), vmax=pf_safe.max())
            )
        except ValueError:
            use_lognorm = False
            
    if not use_lognorm:
        contour = ax.contourf(
            U4_grid, U5_grid, PF_grid, 
            levels=50, 
            cmap='viridis'
        )
    
    # 4. Colorbar
    cbar = fig.colorbar(contour, ax=ax)
    
    # 5. Apply Labels and Titles based on usetex preference
    usetex = cfg.get("usetex", False)
    
    if cfg.get("labels", True):
        if usetex:
            cbar.set_label(rf"Probability of Failure ($P_f$) [$\alpha = {alpha}$]")
            ax.set_xlabel(r"$u_{d,4}$ Allocation Scale")
            ax.set_ylabel(r"$u_{d,5}$ Allocation Scale")
        else:
            cbar.set_label(f"Probability of Failure (Pf) [alpha = {alpha}]")
            ax.set_xlabel("u_d_4 Allocation Scale (0=Max Rot, 1=Max Trans)")
            ax.set_ylabel("u_d_5 Allocation Scale (0=Max Rot, 1=Max Trans)")

    if cfg.get("title", True):
        if usetex:
            ax.set_title(rf"Failure Probability ($\alpha = {alpha}$)")
        else:
            ax.set_title(f"Failure Probability (alpha = {alpha})")
            
    # 6. Finalize (Grid, Limits, Save, Show)
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
    parser.add_argument("--mc-size", type=int, default=100000, help="Monte Carlo sample size")
    args = parser.parse_args()

    # 1. Setup Models and Distributions
    print("Loading surrogate model and distributions...")
    model_sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(args.surrogate_path)
    
    jointDist, symbols, max_std_vect, mu_vect = model1.get_distribution_params()
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
                    pf = np.where(eval_vals < alpha, 1, 0).mean()
                else:
                    pf = gld.CDF_num(alpha, gld_params, xtol=1e-6)
                    
                PF_grids[alpha][i, j] = pf
                
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
            save=True, 
            use_lognorm=False,
            save_path=plot_filename, 
            dpi=600,
            show=False,
            usetex=True,
            transparent=True
        )
        
        # Free up memory explicitly after saving each plot
        plt.close(fig)

if __name__ == "__main__":
    main()