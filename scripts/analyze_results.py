import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gldpy import GLD
import otaf

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


def load_and_clean_data(csv_path, strict_tol=5e-4):
    """
    Loads the CSV and STRICTLY enforces the EQUALITY constraint.
    This overrides the tracker's blind spot for negative constraint values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    df = pd.read_csv(csv_path, index_col=0)
    
    for col in ["point", "GLD_PARAMS", "constraint_values"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_numpy_string)
            
    def is_strictly_on_boundary(c_vals):
        if not isinstance(c_vals, np.ndarray) or len(c_vals) == 0:
            return False
            
        # A constraint is valid if it is exactly at the capability target (active)
        on_boundary = np.abs(c_vals) <= strict_tol
        
        # OR if it is completely inactive/fixed to zero, which yields exactly -1.0
        # (Using a small tolerance for floating point inaccuracies around -1.0)
        inactive = np.abs(c_vals + 1.0) <= 1e-4 
        
        # Ensures ALL features are either exactly at target OR explicitly deactivated
        return np.all(on_boundary | inactive)
        
    if "constraint_values" in df.columns:
        df["strict_constraints"] = df["constraint_values"].apply(is_strictly_on_boundary)
        valid_mask = (df["bounds_respected"]) & (df["strict_constraints"])
    else:
        # Fallback if constraint_values isn't present
        valid_mask = (df["bounds_respected"]) & (df["constraints_respected"])
        
    return df[valid_mask].copy()

def get_valid_gld_params(df):
    """Extracts a clean 2D numpy array of GLD parameters."""
    params = np.stack(df["GLD_PARAMS"].values)
    # Filter out invalid GLD fits (e.g., Lambda 4 close to 0 or invalid scale)
    safe_mask = (np.abs(params[:, 3]) > 1e-8) & (params[:, 1] > 0)
    return params[safe_mask]

# --- Analysis Functions ---

def format_latex_allocation(vector, label="min", cols=4):
    """
    Generates a colored LaTeX bmatrix for a given allocation vector.
    Colors: >= 0.8 (red), >= 0.6 (orange), <= 0.2 (green), else black.
    """
    lines = [r"\begin{equation}", f"    \\mathbf{{x}}_{{{label}}} = \\begin{{bmatrix}}"]
    
    for i in range(0, len(vector), cols):
        chunk = vector[i:i+cols]
        row_strs = []
        for val in chunk:
            if val >= 0.8:
                color = "red"
            elif val >= 0.6:
                color = "orange"
            elif val <= 0.2:
                color = "green"
            else:
                color = "black"
            
            # Formatting to match the specific visual alignment in LaTeX
            v_round = np.round(val, 4)
            if v_round == 0.0:
                v_str = "0.    "
            elif v_round == 1.0:
                v_str = "1.    "
            else:
                # Keep up to 4 decimals, strip trailing zeros, pad to 6 chars
                v_str = f"{v_round:.4f}".rstrip('0')
                if v_str.endswith('.'):
                    v_str += "    "
                v_str = v_str.ljust(6, ' ')
            
            row_strs.append(f"\\textcolor{{{color}}}{{{v_str}}}")
        
        row_line = " & ".join(row_strs)
        if i + cols < len(vector):
            row_line += " \\\\"
        lines.append("    " + row_line)
    
    lines.append("    \\end{bmatrix}^T")
    lines.append(f"\\label{{eq:{label}_standards}}")
    lines.append(r"\end{equation}")
    
    return "\n".join(lines)

def get_all_valid_allocations(df):
    """Extracts all valid parameter allocations from the strictly filtered DataFrame."""
    allocations = []
    for idx, row in df.iterrows():
        params = row['GLD_PARAMS']
        
        # Verify GLD parameters are mathematically valid for CDF evaluation
        if isinstance(params, np.ndarray) and len(params) == 4:
            if np.abs(params[3]) > 1e-8 and params[1] > 0:
                allocations.append({
                    'point': row['point'],
                    'gld_params': params,
                    'optimized_at_slack': row.get('FAILURE_SLACK', 'unknown'),
                    'constraint_val_max': np.max(np.abs(row.get('constraint_values', [0])))
                })
    return allocations

def get_dominant_variables(allocation_point, lower_tol=0.05, upper_tol=0.95):
    """Filters allocations to show which variables are pegged to their geometric limits."""
    fmi = np.where(allocation_point < lower_tol, 0.0, allocation_point)
    fma = np.where(fmi > upper_tol, 1.0, fmi)
    return np.round(fma, 4)

def generate_summary_report(df, model_name, target_eval_slack=0.0):
    """
    Evaluates ALL true boundary allocations at the target_eval_slack to find 
    the absolute global minimum and maximum probability of failure (the P-Box bounds).
    Returns the generated report as a formatted string.
    """
    gld = GLD('VSL')
    
    lines = []
    lines.append("\n" + "="*70)
    lines.append(f" SUMMARY REPORT: {model_name.upper()}")
    lines.append(f" Target Evaluation Slack: {target_eval_slack}")
    lines.append(f" Valid Boundary Points Analyzed: {len(df)}")
    lines.append("="*70)
    
    allocations = get_all_valid_allocations(df)
    
    if not allocations:
        lines.append("No valid boundary points found. The optimization may have never hit the boundary.")
        return "\n".join(lines)

    global_min_pf = np.inf
    global_max_pf = -np.inf
    best_min_alloc = None
    best_max_alloc = None
    
    # Sweep every strictly valid allocation found across all runs
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
            pass 

    if global_min_pf > 0 and global_max_pf > 0:
        mag_diff = np.log10(global_max_pf) - np.log10(global_min_pf)
    else:
        mag_diff = np.nan
        
    lines.append(f"-> Global Minimum Pf : {global_min_pf:.3e} (Found during opt at slack {best_min_alloc['optimized_at_slack']})")
    lines.append(f"-> Global Maximum Pf : {global_max_pf:.3e} (Found during opt at slack {best_max_alloc['optimized_at_slack']})")
    lines.append(f"-> Magnitude Diff    : {mag_diff:.2f} orders of magnitude")
    
    lines.append("\n[Dominant Variables - True Min Pf Allocation]")
    lines.append(str(get_dominant_variables(best_min_alloc["point"])))
    
    lines.append("\n[Dominant Variables - True Max Pf Allocation]")
    lines.append(str(get_dominant_variables(best_max_alloc["point"])))
    lines.append("="*70 + "\n")
    
    # --- LaTeX Output ---
    lines.append("### LaTeX Matrices")
    lines.append("```latex")
    lines.append(format_latex_allocation(best_min_alloc["point"], label="\\min", cols=4))
    lines.append("")
    lines.append(format_latex_allocation(best_max_alloc["point"], label="\\max", cols=4))
    lines.append("```\n")
    
    return "\n".join(lines)

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

def plot_ensemble_gld_pbox_cdf(gld_obj, param_list, x_values, 
                               style=None, 
                               labels=('Lower bound', 'Upper bound'),
                               fill_color='gray', alpha=0.3,
                               xlabel="Slack", ylabel="Probability of Failure ($P_f$)",
                               title="Probability-Box of the CDF",
                               add_zoom=False, zoom_x_range=(-0.05, 0.05), zoom_y_range=(1e-6, 0.1),
                               inset_log_y=False,
                               **style_kwargs):
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
    # 1. Merge style and initialize figure using your helper functions
    cfg = _merge_style(style, **style_kwargs)
    fig, ax = _new_fig_ax(cfg)

    lower_cdf, upper_cdf = compute_fast_pbox_bounds(param_list, x_values)

    # 3. Plot main P-box using colors from your config
    c1 = cfg['colors']['c1']
    c2 = cfg['colors']['c2']
    
    ax.plot(x_values, lower_cdf, color=c1, label=labels[0], linewidth=1.5)
    ax.plot(x_values, upper_cdf, color=c2, label=labels[1], linewidth=1.5)
    ax.fill_between(x_values, lower_cdf, upper_cdf, color=fill_color, alpha=alpha, label='Epistemic Uncertainty')
    ax.axvline(0.0, color='red', linestyle='--', linewidth=1, label='Failure Threshold (Slack = 0)')

    # Labels and Titles (controlled by cfg toggle if desired, or passed strings)
    if cfg.get("labels", True):
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
    if cfg.get("title", True):
        ax.set_title(title, pad=15)

    # 4. Zoom Inset Logic
    if add_zoom:
        axins = ax.inset_axes([0.075, 0.25, 0.3, 0.3])

        axins.patch.set_facecolor('white')
        axins.patch.set_alpha(1.0)
        axins.set_zorder(100)
        
        # 2. Turn OFF the grid inside the inset so it doesn't look like the main grid bleeding through
        axins.grid(False)

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

    # 5. Legend
    if cfg.get("legend", True):
        ax.legend(loc='upper left', framealpha=0.9)

    # 6. Finalize (handles grid, tight_layout, save, and show)
    return _finalize_figure(fig, ax, cfg)

# --- CLI Execution ---
# --- CLI Execution ---
if __name__ == "__main__":
    import os
    import argparse
    import numpy as np
    from gldpy import GLD
    import otaf

    parser = argparse.ArgumentParser(description="Analyze OTAF Optimization Results.")
    parser.add_argument("--models", nargs="+", required=True, help="Models to analyze (e.g., model1_4_dof)")
    parser.add_argument("--input-dir", type=str, default=".", help="Directory containing the CSV files")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--eval-slack", type=float, default=0.0, help="Target slack threshold for Pf evaluation")
    parser.add_argument("--strict-tol", type=float, default=5e-4, help="Tolerance for strict constraint equality checking")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating the plots")
    parser.add_argument("--reduced-mode", action="store_true", help="Analyze the reduced variable subsets for model3 and model4")
    
    # Dynamic Plot Arguments
    parser.add_argument("--plot-mins", nargs="+", default=["-0.10"], help="Min x-axis bound per model")
    parser.add_argument("--plot-maxs", nargs="+", default=["0.15"], help="Max x-axis bound per model")
    parser.add_argument("--zoom-x-mins", nargs="+", default=["-0.025"], help="Zoom x min per model")
    parser.add_argument("--zoom-x-maxs", nargs="+", default=["0.015"], help="Zoom x max per model")
    parser.add_argument("--zoom-y-mins", nargs="+", default=["-1e-6"], help="Zoom y min per model")
    parser.add_argument("--zoom-y-maxs", nargs="+", default=["0.04"], help="Zoom y max per model")
    
    args = parser.parse_args()
    
    available_models = {
        "model1_4_dof": otaf.example_models.model1,
        "model2_16_dof": otaf.example_models.model2,
        "model3_30_dof": otaf.example_models.model3,
        "model4_50_dof": otaf.example_models.model4
    }

    # Helper to safely extract positional arguments
    def get_arg(arg_list, idx, type_func=float):
        if not arg_list: 
            return None
        val = arg_list[idx] if idx < len(arg_list) else arg_list[-1]
        return type_func(val)
    
    gld = GLD('VSL')
    
    for i, m_name in enumerate(args.models):
        if m_name not in available_models:
            print(f"Skipping unknown model: {m_name}")
            continue
            
        model = available_models[m_name]
        
        # Determine if we are analyzing a reduced subset
        is_reduced = args.reduced_mode and m_name in ["model3_30_dof", "model4_50_dof"]
        file_suffix = "_reduced" if is_reduced else ""
        
        # Determine actual dimension for the plot title
        if is_reduced:
            if m_name == "model3_30_dof":
                actual_dim = 26
            elif m_name == "model4_50_dof":
                actual_dim = 32 # Active indices 6 to 37 (37 - 6 + 1 = 32)
        else:
            actual_dim = model.dim

        csv_path = os.path.join(args.input_dir, f"OptimizationResults_{m_name}{file_suffix}.csv")
        
        try:
            df = load_and_clean_data(csv_path, strict_tol=args.strict_tol)
        except Exception as e:
            print(f"Error loading {m_name} (File: {csv_path}): {e}")
            continue
            
        # 1. Generate and print the report
        report_str = generate_summary_report(df, m_name, target_eval_slack=args.eval_slack)
        
        # 2. Save the report to a Markdown file
        md_save_path = os.path.join(args.output_dir, f"Summary_{m_name}{file_suffix}.md")
        with open(md_save_path, "w", encoding="utf-8") as md_file:
            md_file.write(report_str)
        print(f"Summary report saved to {md_save_path}")
        print(report_str)
        if not args.no_plot and not df.empty:
            gld_params = get_valid_gld_params(df)
            
            p_min = get_arg(args.plot_mins, i)
            p_max = get_arg(args.plot_maxs, i)
            zx_min = get_arg(args.zoom_x_mins, i)
            zx_max = get_arg(args.zoom_x_maxs, i)
            zy_min = get_arg(args.zoom_y_mins, i)
            zy_max = get_arg(args.zoom_y_maxs, i)
            
            x_range = np.linspace(p_min, p_max, 2000)
            save_path = os.path.join(args.output_dir, f"pbox_{m_name}{file_suffix}.png")
            
            plot_ensemble_gld_pbox_cdf(
                gld,
                gld_params, 
                x_range,
                add_zoom=True,
                zoom_x_range=(zx_min, zx_max), 
                zoom_y_range=(zy_min, zy_max),
                title=f"P-Box (model {actual_dim} dim)",
                save=True, 
                save_path=save_path, 
                dpi=600,
                show=False,
                usetex=False,
                transparent=False
            )
            print(f"Plot saved to {save_path}")

"""
python analyze_results.py --models model1_4_dof model2_16_dof model3_30_dof model4_50_dof --strict-tol 1e-2 --plot-mins -0.05 -0.05 -0.1 -0.05  --plot-maxs 0.2 0.15 0.2 0.15 --zoom-x-mins -0.025 -0.015 -0.06 -0.025 --zoom-x-maxs 0.02 0.015 0.015 0.015 --zoom-y-mins -1e-6 --zoom-y-maxs 0.02 0.015 0.07 0.03
python analyze_results.py --models model3_30_dof model4_50_dof --reduced-mode --strict-tol 1e-3 --plot-mins -0.1 -0.05  --plot-maxs 0.2 0.15 --zoom-x-mins -0.06 -0.01 --zoom-x-maxs 0.015 0.01 --zoom-y-mins -1e-6 --zoom-y-maxs 0.07 0.001

python analyze_results.py --models model4_50_dof --reduced-mode --no-plot --strict-tol 5e-4 --plot-mins -0.05  --plot-maxs 0.15 --zoom-x-mins -0.01 --zoom-x-maxs 0.01 --zoom-y-mins -1e-6 --zoom-y-maxs 0.001
python analyze_results.py --models model4_50_dof --reduced-mode --no-plot --strict-tol 1e-5 --plot-mins -0.05  --plot-maxs 0.15 --zoom-x-mins -0.01 --zoom-x-maxs 0.01 --zoom-y-mins -1e-6 --zoom-y-maxs 0.001
python analyze_results.py --models model4_50_dof --reduced-mode --no-plot --strict-tol 5e-5 --plot-mins -0.05  --plot-maxs 0.15 --zoom-x-mins -0.01 --zoom-x-maxs 0.01 --zoom-y-mins -1e-6 --zoom-y-maxs 0.001
python analyze_results.py --models model4_50_dof --reduced-mode --no-plot --strict-tol 2e-5 --plot-mins -0.05  --plot-maxs 0.15 --zoom-x-mins -0.01 --zoom-x-maxs 0.01 --zoom-y-mins -1e-6 --zoom-y-maxs 0.001


"""