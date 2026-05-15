import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gldpy import GLD

# --- Global Plot Style Configuration ---
# --- Global plot style configuration ----------------------------------------

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
        rc["text.latex.preamble"] = r"""
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{libertinus}        % wrapper, picks libertinus-type1 under pdfLaTeX
\usepackage{libertinust1math}  % Libertinus Math for pdfLaTeX
"""
        # DO *NOT* load fontspec/unicode-math here
        # DO *NOT* try to switch engine to lualatex; usetex uses pdfTeX.
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

# --- Data Loading and Parsing ---

def parse_numpy_string(x):
    """Safely converts string representations of numpy arrays back to arrays."""
    if isinstance(x, str) and "[" in x:
        clean_str = re.sub(r'[\[\]\n]', '', x)
        return np.fromstring(clean_str, sep=' ')
    return x


def load_and_clean_data(csv_path):
    """Loads the CSV and filters out rows with violated constraints or bounds."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    for col in df.columns:
        df[col] = df[col].apply(parse_numpy_string)
        
    valid_mask = (df["bounds_respected"] == True) & (df["constraints_respected"] == True)
    return df[valid_mask].copy()

def get_valid_gld_params(df):
    """Extracts a clean 2D numpy array of GLD parameters."""
    params = np.stack(df["GLD_PARAMS"].values)
    # Filter out invalid GLD fits (e.g., Lambda 4 close to 0 or invalid scale)
    safe_mask = (np.abs(params[:, 3]) > 1e-8) & (params[:, 1] > 0)
    return params[safe_mask]

# --- Analysis Functions ---

def get_all_valid_allocations(df):
    """
    Extracts all valid parameter allocations from the DataFrame, keeping 
    their origin metadata (which slack threshold they were optimized at).
    """
    allocations = []
    for idx, row in df.iterrows():
        params = parse_numpy_string(row['GLD_PARAMS'])
        
        # Verify GLD parameters are mathematically valid for CDF evaluation
        if isinstance(params, np.ndarray) and len(params) == 4:
            if np.abs(params[3]) > 1e-8 and params[1] > 0:
                allocations.append({
                    'point': parse_numpy_string(row['point']),
                    'gld_params': params,
                    'optimized_at_slack': row['FAILURE_SLACK'],
                })
    return allocations

def get_dominant_variables(allocation_point, lower_tol=0.05, upper_tol=0.95):
    """Filters allocations to show which variables are pegged to their geometric limits."""
    fmi = np.where(allocation_point < lower_tol, 0.0, allocation_point)
    fma = np.where(fmi > upper_tol, 1.0, fmi)
    return np.round(fma, 4)

def generate_summary_report(df, model_name, target_eval_slack=0.0):
    """
    Evaluates ALL discovered allocations at the target_eval_slack to find 
    the true global minimum and maximum probability of failure (the P-Box bounds).
    """
    gld = GLD('VSL')
    print(f"\n" + "="*60)
    print(f" SUMMARY REPORT: {model_name.upper()}")
    print(f" Target Evaluation Slack: {target_eval_slack}")
    print("="*60)
    
    allocations = get_all_valid_allocations(df)
    
    if not allocations:
        print("No valid GLD parameters found for this model. Skipping.")
        return

    global_min_pf = np.inf
    global_max_pf = -np.inf
    best_min_alloc = None
    best_max_alloc = None
    
    # Sweep every allocation found across all runs
    for alloc in allocations:
        try:
            pf = gld.CDF_num(target_eval_slack, list(alloc['gld_params']), xtol=1e-6)[0]
            
            if pf < global_min_pf:
                global_min_pf = pf
                best_min_alloc = alloc
                
            if pf > global_max_pf:
                global_max_pf = pf
                best_max_alloc = alloc
        except Exception:
            pass # Skip if GLD numerical evaluation fails for a specific point

    # Calculate magnitude difference
    if global_min_pf > 0 and global_max_pf > 0:
        mag_diff = np.log10(global_max_pf) - np.log10(global_min_pf)
    else:
        mag_diff = np.nan
        
    print(f"-> Global Minimum Pf : {global_min_pf:.3e} (Found during opt at slack {best_min_alloc['optimized_at_slack']})")
    print(f"-> Global Maximum Pf : {global_max_pf:.3e} (Found during opt at slack {best_max_alloc['optimized_at_slack']})")
    print(f"-> Magnitude Diff    : {mag_diff:.2f} orders of magnitude")
    
    print("\n[Dominant Variables - True Min Pf Allocation]")
    print(get_dominant_variables(best_min_alloc["point"]))
    
    print("\n[Dominant Variables - True Max Pf Allocation]")
    print(get_dominant_variables(best_max_alloc["point"]))
    print("="*60 + "\n")

# --- Plotting ---

def compute_fast_pbox_bounds(param_list, x_values):
    p_values = np.linspace(1e-6, 1 - 1e-6, 3000)
    p = p_values[np.newaxis, :]       
    p0, p1, p2, p3 = param_list[:, 0:1], param_list[:, 1:2], param_list[:, 2:3], param_list[:, 3:4]

    term1 = (1 - p2) * (p**p3 - 1) / p3
    term2 = p2 * ((1 - p)**p3 - 1) / p3
    q_matrix = p0 + (term1 - term2) * p1  

    composite_min_q = q_matrix.min(axis=0)
    composite_max_q = q_matrix.max(axis=0)

    upper_cdf = np.interp(x_values, composite_min_q, p_values, left=0.0, right=1.0)
    lower_cdf = np.interp(x_values, composite_max_q, p_values, left=0.0, right=1.0)
    return lower_cdf, upper_cdf

def plot_ensemble_gld_pbox_cdf(param_list, x_values, 
                               style=None, labels=('Lower bound', 'Upper bound'),
                               fill_color='gray', alpha=0.3,
                               xlabel="Slack Threshold", ylabel="Probability of Failure (Pf)",
                               title="Probability-Box of the CDF",
                               add_zoom=False, zoom_x_range=(-0.05, 0.05), zoom_y_range=(1e-6, 0.1),
                               inset_log_y=False, **style_kwargs):
    
    cfg = _merge_style(style, **style_kwargs)
    fig, ax = _new_fig_ax(cfg)

    lower_cdf, upper_cdf = compute_fast_pbox_bounds(param_list, x_values)

    c1, c2 = cfg['colors']['c1'], cfg['colors']['c2']
    
    ax.plot(x_values, lower_cdf, color=c1, label=labels[0], linewidth=1.5)
    ax.plot(x_values, upper_cdf, color=c2, label=labels[1], linewidth=1.5)
    ax.fill_between(x_values, lower_cdf, upper_cdf, color=fill_color, alpha=alpha, label='Epistemic Uncertainty')
    ax.axvline(0.0, color='red', linestyle='--', linewidth=1, label='Failure Threshold (Slack = 0)')

    if cfg.get("labels", True):
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
    if cfg.get("title", True):
        ax.set_title(title, pad=15)

    if add_zoom:
        axins = ax.inset_axes([0.1, 0.2, 0.3, 0.3])
        axins.plot(x_values, lower_cdf, color=c1, linewidth=1.5)
        axins.plot(x_values, upper_cdf, color=c2, linewidth=1.5)
        axins.fill_between(x_values, lower_cdf, upper_cdf, color=fill_color, alpha=alpha)
        axins.axvline(0.0, color='red', linestyle='--', linewidth=1)
        
        axins.set_xlim(zoom_x_range)
        axins.set_ylim(zoom_y_range)
        if inset_log_y:
            axins.set_yscale('log')
        
        axins.tick_params(axis='both', which='major', labelsize=cfg.get("font_size", 9) - 1)
        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.2, alpha=0.6)

    if cfg.get("legend", True):
        ax.legend(loc='upper left', framealpha=0.9)

    return _finalize_figure(fig, ax, cfg)

# --- CLI Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze OTAF Optimization Results.")
    parser.add_argument("--models", nargs="+", required=True, help="Models to analyze (e.g., model1_4_dof)")
    parser.add_argument("--input-dir", type=str, default=".", help="Directory containing the CSV files")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--eval-slack", type=float, default=0.0, help="Target slack threshold for Pf evaluation")
    parser.add_argument("--plot-min", type=float, default=-0.10, help="Min x-axis bound for P-Box plot")
    parser.add_argument("--plot-max", type=float, default=0.15, help="Max x-axis bound for P-Box plot")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating the plots")
    
    args = parser.parse_args()
    
    for m_name in args.models:
        csv_path = os.path.join(args.input_dir, f"OptimizationResults_{m_name}.csv")
        
        try:
            df = load_and_clean_data(csv_path)
        except Exception as e:
            print(f"Error loading {m_name}: {e}")
            continue
            
        generate_summary_report(df, m_name, target_eval_slack=args.eval_slack)
        
        if not args.no_plot:
            gld_params = get_valid_gld_params(df)
            x_range = np.linspace(args.plot_min, args.plot_max, 2000)
            
            save_path = os.path.join(args.output_dir, f"pbox_{m_name}.png")
            
            plot_ensemble_gld_pbox_cdf(
                gld_params, x_range,
                add_zoom=True,
                zoom_x_range=(-0.025, 0.015), 
                zoom_y_range=(-1e-6, 0.04),
                title=f"Probability-Box of the CDF ({m_name})",
                save=True, 
                save_path=save_path, 
                dpi=600,
                show=False,
                usetex=False 
            )
            print(f"Plot saved to {save_path}")

"""
python analyze_results.py --models model1_4_dof model2_16_dof model3_30_dof model4_50_dof 
"""