import argparse
import numpy as np
from scipy.optimize import NonlinearConstraint, Bounds, minimize
import otaf
from gldpy import GLD

# Here will be the set of codes to load the neural network based surrogates, 
# define the constraints of the credal sets
# and the find the max/min prob of failure, 
# mainly here we will save verything in csv files
# and then  load it later somewhere else to do the plotting and the rest of the analysis


def optimize_scaling_vector(
    constraint_fn, 
    n_vars: int, 
    x_warm: np.ndarray = None, 
    labels: list = None, 
    verbose: bool = True,
    maxiter: int = 500,
    ftol: float = 1e-10
) -> np.ndarray:
    """
    Finds the independent scaling factors x in [0, 1]^n_vars closest to x=1 
    that satisfy the provided constraint function.
    """
    # If no warm start is provided, default to a uniform conservative scale
    if x_warm is None:
        x_warm = np.full(n_vars, 0.5)
        
    # Verify warm start
    c_warm = np.array(constraint_fn(x_warm))
    if verbose:
        status = 'feasible ✓' if np.max(c_warm) <= 1e-8 else 'infeasible, solver will correct'
        print(f"[{n_vars}-dim] Warm start — max constraint: {np.max(c_warm):.6f} ({status})")

    # Optimisation
    result = minimize(
        fun=lambda x: np.sum((x - 1.0) ** 2),
        jac=lambda x: 2.0 * (x - 1.0),
        x0=x_warm,
        method="SLSQP",
        bounds=Bounds(lb=1e-7, ub=1.0, keep_feasible=True),
        constraints={
            "type": "ineq",
            "fun": lambda x: -np.array(constraint_fn(x)) # scipy expects c(x) >= 0
            # Note: Omitted 'jac' to let SLSQP compute finite differences safely 
            # for vector-valued outputs, avoiding approx_fprime broadcasting errors.
        },
        options={"ftol": ftol, "maxiter": maxiter, "disp": verbose},
    )

    x_opt = result.x
    c_opt = np.array(constraint_fn(x_opt))

    if verbose:
        print(f"\nResult — mean: {x_opt.mean():.4f}, min: {x_opt.min():.4f}, max: {x_opt.max():.4f}")
        
        print("\nConstraint values (must all be <= 0):")
        # Handle labels mapping if provided
        iter_labels = labels if (labels and len(labels) == len(c_opt)) else [f"Cons_{i}" for i in range(len(c_opt))]
        for label, val in zip(iter_labels, c_opt):
            print(f"  {label:8s}: {val:+.2e}  {'✓' if val <= 1e-8 else '✗'}")

        print(f"\nFeasible : {(c_opt <= 1e-8).all()}")
        if not result.success:
            print(f"Solver note: {result.message}")

    return x_opt

def get_model_evaluator(sample, mu_vect, neural_model): 
    """
    Creates and returns a function to evaluate the surrogate model.
    
    Args:
        sample: The sample data used for transformation
        mu_vect: The mean vector for scaling
        neural_model: The pre-trained neural model object
        
    Returns:
        A function that accepts 'x' and returns the model evaluation.
    """
    def evaluate(x):
        """
        Inner function to evaluate the surrogate model at point x.
        
        Args:
            x: Input values (normalized 0 to 1)
        """
        # Transform x to the original space
        # Calculation: x_transformed = (sample - mu_vect) * x + mu_vect
        x_transformed = (sample - mu_vect) * x + mu_vect
        # Get prediction and squeeze to remove singleton dimensions
        prediction = neural_model.evaluate_model_non_standard_space(x_transformed)
        return np.squeeze(prediction.numpy())
    return evaluate

@otaf.optimization.scaling(scale_factor=1.0)
def optimization_function(
        x, 
        failure_slack=0.0,
        gld=None,
        model=None, 
        experiment_key=None, 
        tracker=None, 
        logprob=False, 
        minimize=True):
    # Determine mode label and output multiplier based on the minimize flag
    mode_label = "mini" if minimize else "maxi"
    multiplier = 1 if minimize else -1

    slack = model(x)
    gld_params = gld.fit_LMM(slack, disp_fit=False, disp_optimizer=False)
    
    fp_slack = np.where(slack < failure_slack, 1, 0).mean()
    fp_gld = np.nan
    
    if np.any(np.isnan(gld_params)):
        print("GLD Parameters are NaN, returning estimated Pf")
        fp_out = fp_slack
    else:
        fp_gld = gld.CDF_num(failure_slack, gld_params, xtol=1e-6)
        fp_out = fp_gld
        
    print(f"Pf ({mode_label}) is {fp_out}, GLD Pf is {fp_gld}, estimated PF is {fp_slack} , log prob is {np.log(1e-16+fp_out)} ")
    
    tracker.update_objective_data(
        exp_key=experiment_key, 
        x=x, 
        fp_gld=fp_gld, 
        fp_slack=fp_slack, 
        gld_params=gld_params, 
        failure_slack=failure_slack,
    )
    
    if logprob:
        return multiplier * np.log(1e-16 + fp_out)
    return multiplier * fp_out

def pf_min_max_optimizer(
        failure_slack=0.0, 
        tracker=None, 
        experiment_key=None, 
        logprob=True, 
        model_eval_fn=None, 
        constraint_factory=None,
        x0=None,
        dim=None,
        maxiter=10000):
    
    # 1. Instantiate GLD here so it can be passed to the objective function
    gld = GLD('VSL')
    
    normalized_bounds = Bounds(1e-9, 1.0, keep_feasible=True)
    print(f"\n--- Starting optimization sequence for Slack: {failure_slack} ---\n")
    
    # 2. Fix the args tuple to strictly match optimization_function's signature
    # (failure_slack, gld, model, experiment_key, tracker, logprob, minimize)
    
    res_maxi = minimize(
        optimization_function, x0,
        args=(failure_slack, gld, model_eval_fn, experiment_key, tracker, logprob, False),
        method="COBYQA", 
        jac=None, 
        bounds=normalized_bounds,
        constraints=constraint_factory(tracker, experiment_key, is_minimization=False),
        options={
            "f_target": -np.inf if logprob else -1.01, 
            "maxiter": maxiter,
            "maxfev": maxiter,
            "feasibility_tol": 1e-5,
            "initial_tr_radius": np.sqrt(dim)/np.sqrt(2),
            "final_tr_radius": 0.001,
            "disp": False,
            "scale": False
        }
    )
    print('\nMaximization result:\n', res_maxi.message, 'Fun:', res_maxi.fun,"\n")
    
    res_mini = minimize(
        optimization_function, x0, 
        args=(failure_slack, gld, model_eval_fn, experiment_key, tracker, logprob, True),
        method="COBYQA", 
        jac=None, 
        bounds=normalized_bounds,
        constraints=constraint_factory(tracker, experiment_key, is_minimization=True),
        options={
            "f_target": -np.inf if logprob else -0.01,
            "maxiter": maxiter,
            "maxfev": maxiter,
            "feasibility_tol": 1e-5,
            "initial_tr_radius": np.sqrt(dim)/np.sqrt(2),
            "final_tr_radius": 0.001,
            "disp": False,
            "scale": False
        }
    )

    print('\nMinimization result:\n', res_mini.message, 'Fun:', res_mini.fun,"\n")

    # Retrieve using the tracker
    data_min = tracker.get_data(experiment_key, res_mini.x)
    data_max = tracker.get_data(experiment_key, res_maxi.x)    
    return (res_mini.x, res_maxi.x), (data_min['GLD_PARAMS'], data_max['GLD_PARAMS']), (data_min['FP_GLD'], data_max['FP_GLD'])


def run_reduced_analysis(m_name, model_class, args, base_model_fun, max_std_vect, slacks):
    """
    Standalone runner for reduced-dimension models.
    Uses closures to zero-pad inputs for the MLP and constraints, 
    ensuring the tracker strictly logs the reduced dimensional space.
    """
    full_dim = model_class.dim
    if m_name == "model3_30_dof":
        active_indices = list(range(26))
    elif m_name == "model4_50_dof":
        active_indices = list(range(6, 38))
    else:
        raise ValueError(f"No reduced mapping defined for {m_name}")
        
    opt_dim = len(active_indices)
    
    tracker = otaf.optimization.OptimizationTracker(
        bounds=Bounds(1e-7, 1.0), constraint_tolerance=1e-5, precision_decimals=8
    )

    # 1. Wrap the MLP Evaluator
    def reduced_model_eval(x_reduced):
        x_full = np.zeros(full_dim)
        x_full[active_indices] = x_reduced
        return base_model_fun(x_full)

    # 2. Wrap the Constraints & Handle Tracker Manually
    def get_reduced_constraint_fun(tr, expKey):
        # Instantiate base constraint WITHOUT the tracker so it doesn't log the full_dim vector
        base_cons = model_class.getScaledCredalSetConstraintsFunction(
            max_std_vect, tracker=None, experiment_key=None
        )
        
        def wrapped_cons(x_reduced):
            x_full = np.zeros(full_dim)
            x_full[active_indices] = x_reduced
            c_vals = base_cons(x_full)
            
            # Manually track the reduced vector so dimensions align with objective logs
            if tr is not None:
                tr.update_constraint_data(exp_key=expKey, x=x_reduced, constraints=c_vals)
            return c_vals
            
        return wrapped_cons

    def create_directional_constraint(tr, expKey, is_minimization):
        lb = 0.0 if is_minimization else -np.inf
        ub = np.inf if is_minimization else 0.0
        return NonlinearConstraint(
            fun=get_reduced_constraint_fun(tr, expKey),
            lb=lb,
            ub=ub,
            keep_feasible=True
        )

    print(f"\nFinding feasible start (x0) for reduced {m_name} ({opt_dim} dimensions)...")
    dummy_cons = get_reduced_constraint_fun(None, None)
    x0_reduced = optimize_scaling_vector(dummy_cons, n_vars=opt_dim)

    for slack in slacks:
        pf_min_max_optimizer(
            failure_slack=slack, 
            tracker=tracker, 
            experiment_key=f"{m_name}_slack_{slack}", 
            logprob=True, 
            model_eval_fn=reduced_model_eval,       # Passes the wrapped closure
            constraint_factory=create_directional_constraint, 
            x0=x0_reduced,
            dim=opt_dim,                            # Optimizer operates in reduced space
            maxiter=args.maxiter
        )
        
    df = tracker.to_dataframe()
    out_path = f"{args.output_dir}/OptimizationResults_{m_name}_reduced.csv"
    df.to_csv(out_path)
    print(f"\nSaved REDUCED results for {m_name} to {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imprecise Bounds Analysis for OTAF Models.")
    parser.add_argument("--models", nargs="+", required=True, help="Models to run (e.g., model1_4_dof)")
    parser.add_argument("--slacks", nargs="+", required=True, help="Comma-separated slacks per model")
    parser.add_argument("--surrogate-dir", type=str, default=".", help="Directory containing .pth files")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save CSV results")
    parser.add_argument("--maxiter", type=int, default=5000, help="Max iterations for COBYQA")
    parser.add_argument("--mc-size", type=int, default=100000, help="Monte Carlo sample size for GLD")
    parser.add_argument("--reduced-mode", action="store_true", help="Run reduced variable subsets for model3 and model4")

    args = parser.parse_args()

    available_models = {
        "model1_4_dof": otaf.example_models.model1,
        "model2_16_dof": otaf.example_models.model2,
        "model3_30_dof": otaf.example_models.model3,
        "model4_50_dof": otaf.example_models.model4
    }

    for i, m_name in enumerate(args.models):
        print("\n=========================================")
        print(f"PROCESSING {m_name}")
        if args.reduced_mode:
            print("MODE: REDUCED VARIABLE SUBSET")
        print("=========================================")
        
        if m_name not in available_models:
            print(f"Skipping unknown model: {m_name}")
            continue
            
        model = available_models[m_name]
        
        # 1. Shared Setup (Neural Network & Space definition)
        model_path = f"{args.surrogate_dir}/{m_name}_surrogate.pth"
        model_sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(model_path)
        
        jointDist, symbols, max_std_vect, mu_vect = model.getDistributionParams()
        sample_gld = np.array(jointDist.getSample(args.mc_size))
        
        # Base full-dimensional model evaluator
        base_model_fun = get_model_evaluator(sample_gld, mu_vect, model_sur)

        slack_str = args.slacks[i] if i < len(args.slacks) else args.slacks[-1]
        slacks = [float(s) for s in slack_str.split(',')]

        # 2. Branch Logic
        if args.reduced_mode and m_name in ["model3_30_dof", "model4_50_dof"]:
            run_reduced_analysis(m_name, model, args, base_model_fun, max_std_vect, slacks)
        else:
            # Standard Flow
            if args.reduced_mode:
                print(f"No reduced mapping for {m_name}, falling back to full dimension.")
                
            tracker = otaf.optimization.OptimizationTracker(bounds=Bounds(1e-7, 1.0), constraint_tolerance=1e-5, precision_decimals=8)
            
            def create_directional_constraint(tr, expKey, is_minimization):
                lb = 0.0 if is_minimization else -np.inf
                ub = np.inf if is_minimization else 0.0
                return NonlinearConstraint(
                    fun=model.getScaledCredalSetConstraintsFunction(max_std_vect, tr, expKey),
                    lb=lb,
                    ub=ub,
                    keep_feasible=True
                )

            print("\nFinding feasible start (x0)...")
            x0 = optimize_scaling_vector(
                model.getScaledCredalSetConstraintsFunction(max_std_vect), 
                n_vars=model.dim
            )

            for slack in slacks:
                pf_min_max_optimizer(
                    failure_slack=slack, 
                    tracker=tracker, 
                    experiment_key=f"{m_name}_slack_{slack}", 
                    logprob=True, 
                    model_eval_fn=base_model_fun, 
                    constraint_factory=create_directional_constraint, 
                    x0=x0,
                    dim=model.dim,
                    maxiter=args.maxiter
                )
                
            df = tracker.to_dataframe()
            out_path = f"{args.output_dir}/OptimizationResults_{m_name}.csv"
            df.to_csv(out_path)
            print(f"\nSaved results for {m_name} to {out_path}")

"""
python imprecise_bounds_analysis.py \
    --models model1_4_dof model2_16_dof model3_30_dof model4_50_dof \
    --slacks "0.0,0.025,0.05,0.075" "0.0,0.025,0.05,0.075" "0.0,0.025,0.05,0.075" "0.0,0.025,0.05,0.075" \
    --maxiter 5000 \
    --mc-size 100000

python imprecise_bounds_analysis.py \
    --models model3_30_dof model4_50_dof \
    --slacks "0.0,0.025,0.05,0.075" "0.0,0.025,0.05,0.075" \
    --maxiter 5000 \
    --mc-size 100000 \
    --reduced-mode
"""