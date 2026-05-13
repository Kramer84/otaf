from __future__ import annotations

import re
import copy
import numpy as np
from enum import Enum
from typing import Tuple, Sequence
import argparse
import sys

import otaf
import sympy as sp
import torch

#These sripts will load the models, find suitable tolerance values
#Then train the neural network, then save it.
#Let's start with some generalistic functions.

class HyperparameterTuning:
    def __init__(self, 
                 system_of_constraints, 
                 distribution_function, 
                 dimension, 
                 sample_size=10000,
                 error=None,
                 tol=None,
                 mult=None, 
                 sample_multiplier=None):
        """Class to find the right tolerance value
        and the right multiplicator so that:
        for the basic tolerance value we have 5% failure
        with the multiplicator we have 15 % failures (we'll may pass these as parameters.
        
        system_of_constraints is on of the 4 models in otaf.example_models. The functions qre
        usuqlly called getSystemOfConstraintsAssemblyModel()
        distribution_function is the function that returns the vector of random variables,
        the variable names and vector of standard deviations and means.)
        The sample multiplier is a matrix performing a variable change"""
        self.target_failure_base = 0.05
        self.target_failure_mult = 0.15
        self.system_of_constraints = system_of_constraints
        self.distribution_function = distribution_function
        self.dim = dimension
        self.error = error if error is not None else 0.0001
        self.tol = tol if tol is not None else 0.1 #First we optimimze this here
        self.mult = mult if mult is not None else 1.0 #Then we optimize this here
        self.sample_size = sample_size #Just to get the ratios

    def solve_system_of_constraints_ratio(self, sample, bounds=None, n_cpu=-2, progress_bar=True, batch_size=500, dtype="float32"):
        """This function will solve the system of constraints for the given sample and return the ratio of failures."""
        results = otaf.uncertainty.compute_gap_optimizations_on_sample_batch(
            self.system_of_constraints,
            sample,
            bounds=bounds,
            n_cpu=n_cpu,
            progress_bar=progress_bar,
            batch_size=batch_size,
            dtype=dtype
        )
        return np.where(results[:,-1]<0,1,0).sum()/self.sample_size

    def generate_sample_from_tolerance(self, tolerance, seed=3462346):
        """This function will generate a sample from the distribution function with the given tolerance and multiplicator."""
        joint_distribution, _, _, _ = self.distribution_function(tol=tolerance, capa=1.0)
        np.random.seed(seed)
        sample = np.array(joint_distribution.getSample(self.sample_size),dtype="float32")
        return sample

    def optimize_tolerance(self, tol_range=(0.1, 0.4)):
        """Let's search for the value of self.tol so that failure ratio is close to  0.05
        Let's say with a tolerance of 0.0025.
        And then we set self.tol with that value.
        
        Using a Bisection (Binary) search assuming higher tol -> higher failure ratio.
        """
        low, high = tol_range
        target = self.target_failure_base
        acceptable_error = 0.0025
        max_iterations = 50
        
        best_tol = (low + high) / 2.0
        best_error = float('inf')
        
        for i in range(max_iterations):
            mid = (low + high) / 2.0
            sample = self.generate_sample_from_tolerance(mid)
            ratio = self.solve_system_of_constraints_ratio(sample)
            
            error = ratio - target
            abs_error = abs(error)
            
            # Keep track of the best result in case we don't perfectly hit the threshold
            if abs_error < best_error:
                best_error = abs_error
                best_tol = mid
            
            # If we are within the acceptable error, we stop and succeed
            if abs_error <= acceptable_error:
                self.tol = mid
                return self.tol
            
            # Bisection logic:
            # If the current ratio is too high, we need a tighter (lower) tolerance
            if ratio > target:
                high = mid
            # If the current ratio is too low, we need a looser (higher) tolerance
            else:
                low = mid
                
            # If the search interval becomes smaller than our minimum step/precision, break early
            # We can utilize your tol_step here as the minimum resolution limit, or a smaller epsilon
            if (high - low) < 1e-5:
                break

        # Fallback if the loop finishes without hitting the exact threshold
        self.tol = best_tol
        print(f"Warning: Bisection stopped without reaching the 0.0025 threshold. "
              f"Best found tolerance is {self.tol:.5f} with a failure ratio error of {best_error:.5f}.")
        
        return self.tol

    def generate_sample_from_multiplicator(self, mult, seed=3462346):
        """Generates a sample by applying a multiplier to the base distribution's dispersion.
        This uses the already optimized self.tol."""
        # 1. Get the base distribution using the optimized tolerance
        base_dist, _, _, _ = self.distribution_function(tol=self.tol, capa=1.0)
        
        # 2. Apply the multiplier to increase dispersion
        dist = otaf.distribution.multiply_composed_distribution_standard_with_constants(
            base_dist, [mult]*self.dim)
        # 3. Draw the sample
        np.random.seed(seed)
        sample = np.array(dist.getSample(self.sample_size), dtype="float32")
        return sample

    def optimize_multiplicator(self, mult_range=(1.0, 10.0)):
        """Bisection search for self.mult so that the failure ratio is close to 0.15.
        Assumes self.tol has already been optimized.
        Assumes a higher multiplier -> higher dispersion -> higher failure ratio.
        """
        if getattr(self, 'tol', 0.1) == 0.1:
            print("Note: self.tol appears to be at its default value. Ensure optimize_tolerance is run first.")

        low, high = mult_range
        target = self.target_failure_mult  # 0.15
        acceptable_error = self.error
        max_iterations = 50
        
        best_mult = (low + high) / 2.0
        best_error = float('inf')
        
        for i in range(max_iterations):
            mid = (low + high) / 2.0
            
            # Generate sample and compute failure ratio for current multiplier
            sample = self.generate_sample_from_multiplicator(mid)
            ratio = self.solve_system_of_constraints_ratio(sample)
            
            error = ratio - target
            abs_error = abs(error)
            
            if abs_error < best_error:
                best_error = abs_error
                best_mult = mid
            
            if abs_error <= acceptable_error:
                self.mult = mid
                return self.mult
            
            # Bisection logic
            if ratio > target:
                # Too many failures, reduce the multiplier
                high = mid
            else:
                # Too few failures, increase the multiplier
                low = mid
                
            if (high - low) < 1e-5:
                break

        self.mult = best_mult
        print(f"Warning: Bisection stopped without reaching the 0.0025 threshold for the multiplier. "
              f"Best found multiplier is {self.mult:.5f} with a failure ratio error of {best_error:.5f}.")
        
        return self.mult
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for OTAF models.")
    
    parser.add_argument(
        "--models", nargs="+", 
        default=["model1_4_dof", "model2_16_dof", "model4_50_dof"],
        help="Names of the models to optimize. E.g., model1_4_dof model2_16_dof"
    )
    parser.add_argument(
        "--tols", nargs="+", type=float, 
        help="Starting tolerance (tol) for each model. Pass one value to apply to all, or one per model."
    )
    parser.add_argument(
        "--mults", nargs="+", type=float, 
        help="Starting multiplicator (mult) for each model. Pass one value to apply to all, or one per model."
    )
    parser.add_argument(
        "--errors", nargs="+", type=float, 
        help="Target error margin for each model. Pass one value to apply to all, or one per model."
    )
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int, 
        help="Sample size for each model. Pass one value to apply to all, or one per model."
    )
    parser.add_argument(
        "--tol-ranges", nargs="+", type=str, 
        help="Tolerance search range for each model formatted as 'low,high' (e.g., 0.1,0.4). Pass one to apply to all, or one per model."
    )
    parser.add_argument(
        "--mult-ranges", nargs="+", type=str, 
        help="Multiplicator search range for each model formatted as 'low,high' (e.g., 1.0,10.0). Pass one to apply to all, or one per model."
    )

    args = parser.parse_args()

    available_models = {  #previous optimization results:
        "model1_4_dof": otaf.example_models.model1, #tol: 0.31 mult: 1.35 
        "model2_16_dof": otaf.example_models.model2, #tol: 0.16  mult: 1.21
        "model3_30_dof": otaf.example_models.model3, #tol:  mult: 
        "model4_50_dof": otaf.example_models.model4 #tol: 0.21  mult: 1.15
    }

    # Helper function to map list arguments to the current model index
    def get_param(param_list, index, default):
        if not param_list:
            return default
        if len(param_list) == 1:
            return param_list[0]
        if index < len(param_list):
            return param_list[index]
        print(f"Warning: Not enough arguments provided for index {index}. Falling back to default: {default}")
        return default

    for i, model_name in enumerate(args.models):
        if model_name not in available_models:
            print(f"Error: Model '{model_name}' not found. Available models: {list(available_models.keys())}")
            continue

        print(f"Optimizing hyperparameters for {model_name}...")
        
        # Fetch parameters for the current model index
        tol = get_param(args.tols, i, None)
        mult = get_param(args.mults, i, None)
        error = get_param(args.errors, i, None)
        sample_size = get_param(args.sample_sizes, i, 10000)
        
        tol_range_str = get_param(args.tol_ranges, i, "0.1,0.4")
        mult_range_str = get_param(args.mult_ranges, i, "1.0,10.0")

        try:
            tol_range = tuple(map(float, tol_range_str.split(',')))
            mult_range = tuple(map(float, mult_range_str.split(',')))
        except ValueError:
            print(f"Error parsing ranges for {model_name}. Please ensure format is 'low,high'. Exiting.")
            sys.exit(1)

        model_module = available_models[model_name]
        system_of_constraints = model_module.getSystemOfConstraintsAssemblyModel()
        distribution_function = model_module.getDistributionParams
        dimension = int(model_module.dim)
        
        tuner = HyperparameterTuning(
            system_of_constraints=system_of_constraints, 
            distribution_function=distribution_function, 
            dimension=dimension,
            sample_size=sample_size,
            error=error,
            tol=tol,
            mult=mult
        )
        
        print(f"Optimizing tolerance (range: {tol_range})...")
        optimal_tol = tuner.optimize_tolerance(tol_range=tol_range)
        print(f"Optimal tolerance for {model_name}: {optimal_tol:.5f}")
        
        print(f"Optimizing multiplicator (range: {mult_range})...")
        optimal_mult = tuner.optimize_multiplicator(mult_range=mult_range)
        print(f"Optimal multiplicator for {model_name}: {optimal_mult:.5f}")
        
        print(f"Finished optimizing {model_name}.\n")
        