import copy
import ast
import math
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from functools import partial
from scipy.optimize import NonlinearConstraint, Bounds, minimize, approx_fprime
import pandas as pd

from IPython import display

import torch
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
        status = 'feasible ✓' if np.max(c_warm) <= 0 else 'infeasible, solver will correct'
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
            print(f"  {label:8s}: {val:+.2e}  {'✓' if val <= 0 else '✗'}")

        print(f"\nFeasible : {(c_opt <= 0).all()}")
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

    x_eval = np.concatenate([np.zeros(6), x, np.zeros(12)])
    slack = model(x_eval)
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
        credal_constraints=None,
        x0=None,
        dim=None):
    # Initial guess
    normalized_bounds = Bounds(1e-9, 1.0, keep_feasible=True) #Bounds on the allocation
    print(f"\nStarting optimization sequence with failure slack :{failure_slack} \n")
    # Perform the local optimization using COBYQA directly
    res_maxi = minimize(
        optimization_function, x0,
        args=(failure_slack, model_eval_fn, experiment_key, tracker,logprob, False),
        method="COBYQA", 
        jac=None, 
        bounds=normalized_bounds,
        constraints = credal_constraints(tracker, experiment_key),
        options={
            "f_target": -np.inf if logprob else -1.01, 
            "maxiter": 10000,
            "maxfev": 10000,
            "feasibility_tol": 1e-5,
            "initial_tr_radius": np.sqrt(dim)/np.sqrt(2),  # Scaled up slightly for the [0, 1] space
            "final_tr_radius": 0.001,
            "disp": False,
            "scale": False
        }
    )
    print('\nMaximization result:\n', res_maxi, '\n')
    
    # Perform the local optimization using COBYQA directly
    res_mini = minimize(
        optimization_function, x0, 
        args=(failure_slack, model_eval_fn, experiment_key, tracker,logprob, True),
        method="COBYQA", 
        jac=None, 
        bounds=normalized_bounds,
        constraints = credal_constraints(tracker, experiment_key),
        options={
            "f_target": -np.inf if logprob else -0.01,
            "maxiter": 10000,
            "maxfev": 10000,
            "feasibility_tol": 1e-5,
            "initial_tr_radius": np.sqrt(dim)/np.sqrt(2),  # Scaled up slightly for the [0, 1] space
            "final_tr_radius": 0.001,
            "disp": False,
            "scale": False
        }
    )

    print("\nMinimization result:\n", res_mini, '\n')

    # Retrieve using the tracker
    data_min = tracker.get_data(experiment_key, res_mini.x)
    data_max = tracker.get_data(experiment_key, res_maxi.x)
    df = tracker.to_dataframe()
    df.to_csv("OptimizationResults4Pins2Plates.csv")
    return (res_mini.x, res_maxi.x), (data_min['GLD_PARAMS'], data_max['GLD_PARAMS']), (data_min['FP_GLD'], data_max['FP_GLD'])




if __name__ == "__main__":
    gld = GLD('VSL')
    normalized_bounds = Bounds(1e-7, 1.0, keep_feasible=True) #bounds on the std devition allocation

    available_models = {
        "model1_4_dof": otaf.example_models.model1,
        "model2_16_dof": otaf.example_models.model2,
        "model3_30_dof": otaf.example_models.model3,
        "model4_50_dof": otaf.example_models.model4
    }

    #LEt's just show how it is done for the model1:
    m_name = "model1_4_dof"
    model = available_models[m_name]
    model_sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(f"{m_name}_surrogate.pth")
    model_dim = model.dim
    tracker_1= otaf.optimization.OptimizationTracker(bounds=normalized_bounds, constraint_tolerance=1e-5, precision_decimals=8)
    jointDistribution1, symbols1, max_std_vect, mu_vect = model.getDistributionParams()
    
    SIZE_MC_PF = 100000 #int(1e6) #1e4
    sample_gld = np.array(jointDistribution1.getSample(SIZE_MC_PF))
    
    model_fun = get_model_evaluator(sample_gld, mu_vect, model_sur)


    nonLinearConstraint = lambda tracker, expKey: NonlinearConstraint(
        fun = model.getScaledCredalSetConstraintsFunction(max_std_vect, tracker, expKey),
        lb  = -1e-5,  # Lower bound slack
        ub  =  1e-5,  # Upper bound slack
        keep_feasible = True,
    )

    #let's obtain a good starting point:
    x0 = optimize_scaling_vector(model.getScaledCredalSetConstraintsFunction(max_std_vect), model_dim)

    # Now we need  series of slack values on which to optimize on... 
    slacks = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
    for slack in slacks:
        res_x, res_gld, res_fp = pf_min_max_optimizer(
            failure_slack=slack, 
            tracker=tracker_1, 
            experiment_key=f"exp_slack_{slack}", 
            logprob=True, 
            model_eval_fn=model_fun,  # Ensure this matches the parameter name in your def
            credal_constraints=nonLinearConstraint, # Ensure this matches too
            x0=x0,
            dim=model_dim
        )