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

def load_surrogate_model(model_path: str) -> torch.nn.Module:
    """
    Load a surrogate model from a given path.

    Args:
        model_path (str): The path to the saved surrogate model
    """
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

def model_evaluation(x, sample, mu_vect, neural_model): 
    """
    Evaluate the surrogate model at a given point.

    Args:
        x: The input point values between 0 and 1, which will be transformed to the original space using the mean vector
        sample: The sample data

    Returns:
        The output of the surrogate model
    """
    # Surrogate ai model
    x = (sample-mu_vect)*x + mu_vect
    return  np.squeeze(neural_model.evaluate_model_non_standard_space(x).numpy())

@otaf.optimization.scaling(scale_factor=1.0)
def optimization_function(
        x, 
        failure_slack=0.0, 
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
        tracker=tracker, 
        experiment_key=None, 
        logprob=True, 
        model=None, 
        credal_constraints=None,
        x0=None,
        dim=None):
    # Initial guess
    normalized_bounds = Bounds(1e-9, 1.0, keep_feasible=True) #Bounds on the allocation
    print(f"\nStarting optimization sequence with failure slack :{failure_slack} \n")
    # Perform the local optimization using COBYQA directly
    res_maxi = minimize(
        optimization_function, x0,
        args=(failure_slack, model, experiment_key, tracker,logprob, False),
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
        args=(failure_slack, model, experiment_key, tracker,logprob, True),
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





tracker = otaf.optimization.OptimizationTracker(bounds=bounds, constraint_tolerance=1e-4, precision_decimals=8)
