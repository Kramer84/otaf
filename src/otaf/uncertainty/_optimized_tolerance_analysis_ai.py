# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = ["SocAssemblyAnalysisOptimized"]

import numpy as np
from typing import List, Optional, Union
import logging

import otaf


class SocAssemblyAnalysisOptimized:
    """Class to do do system of constraint based tolerance analysis for assembly, using a binary
    classifier to pre-treat the inputs for the optimizer, to evaluate less points.
    """

    def __init__(
        self,
        binary_classifier,
        constraint_matrix_generator,
        X_optimization,
        y_optimization,
        calculate_true_positives=True,  # Even if we think a point is positive we still calculate it.
    ):
        self.binary_classifier = binary_classifier
        self.constraint_matrix_generator = constraint_matrix_generator
        self.X_optimization = X_optimization
        self.y_optimization = y_optimization
        self.opt_res_fn_tn = None
        self.opt_res_fp_tp = None
        self.calculate_true_positives = calculate_true_positives

    def __repr__(self):
        """
        Representation of the class providing information on threshold optimization and other details.

        Returns:
            str: String representation of the class.
        """
        repr_str = (
            f"SocAssemblyAnalysisOptimized(\n"
            f"  Binary Classifier: {self.binary_classifier.__class__.__name__}\n"
            f"  Constraint Matrix Generator: {self.constraint_matrix_generator.__class__.__name__}\n"
            f"  X Optimization Shape: {self.X_optimization.shape}\n"
            f"  Y Optimization Shape: {self.y_optimization.shape}\n"
            f"  Optimize Results (FN/TN):\n"
            f"    Best Failure Threshold: {self.opt_res_fn_tn['best_failure_threshold'] if self.opt_res_fn_tn else 'N/A'}\n"
            f"    Best Success Threshold: {self.opt_res_fn_tn['best_success_threshold'] if self.opt_res_fn_tn else 'N/A'}\n"
            f"    Confusion Matrix: \n"
            f"      TN: {self.opt_res_fn_tn['evaluation']['true_negatives'] if self.opt_res_fn_tn else 'N/A'}\n"
            f"      FP: {self.opt_res_fn_tn['evaluation']['false_positives'] if self.opt_res_fn_tn else 'N/A'}\n"
            f"      FN: {self.opt_res_fn_tn['evaluation']['false_negatives'] if self.opt_res_fn_tn else 'N/A'}\n"
            f"      TP: {self.opt_res_fn_tn['evaluation']['true_positives'] if self.opt_res_fn_tn else 'N/A'}\n"
            f"  Optimize Results (FP/TP):\n"
            f"    Best Failure Threshold: {self.opt_res_fp_tp['best_failure_threshold'] if self.opt_res_fp_tp else 'N/A'}\n"
            f"    Best Success Threshold: {self.opt_res_fp_tp['best_success_threshold'] if self.opt_res_fp_tp else 'N/A'}\n"
            f"    Confusion Matrix: \n"
            f"      TN: {self.opt_res_fp_tp['evaluation']['true_negatives'] if self.opt_res_fp_tp else 'N/A'}\n"
            f"      FP: {self.opt_res_fp_tp['evaluation']['false_positives'] if self.opt_res_fp_tp else 'N/A'}\n"
            f"      FN: {self.opt_res_fp_tp['evaluation']['false_negatives'] if self.opt_res_fp_tp else 'N/A'}\n"
            f"      TP: {self.opt_res_fp_tp['evaluation']['true_positives'] if self.opt_res_fp_tp else 'N/A'}\n"
            f")"
        )
        return repr_str

    def evaluate_binary_classifier(self, X, batch_size=50000, return_on_gpu=False):
        import torch as tor
        self.binary_classifier.eval()
        with tor.no_grad():
            return (
                self.binary_classifier.evaluate_model(
                    tor.tensor(X, dtype=tor.float32),
                    batch_size=batch_size,
                    return_on_gpu=return_on_gpu,
                )
                .cpu()
                .detach()
                .numpy()
            )

    def optimize_thresholds(self, bounds=[-5.0, 5.0], **kwargs):
        """ """
        # Step 1: Predict true negatives and true positives
        pred_class = self.evaluate_binary_classifier(self.X_optimization, **kwargs)
        ground_truth_binary = np.where(
            self.y_optimization[:, -1] < 0, 1, 0
        )  # ground truth for the failures

        # True positives are failures !!! And true negatives successes !!!
        # Step 2: Optimize thresholds
        # Optimizing thresholds for maximizing true negatives and minimizing false negatives
        self.opt_res_fn_tn = otaf.surrogate.optimize_thresholds_with_alpha(
            pred_class,
            ground_truth_binary,
            bounds=bounds,
            optimize_for="minimize_fn_maximize_tn",
            equality_decision="failure",
            optimal_ratio=1e-3,
        )
        # Optimizing thresholds for maximizing true positives and minimizing false positives
        self.opt_res_fp_tp = otaf.surrogate.optimize_thresholds_with_alpha(
            pred_class,
            ground_truth_binary,
            bounds=bounds,
            optimize_for="minimize_fp_maximize_tp",
            equality_decision="success",
            optimal_ratio=1e-2,
        )

    def binary_input_array_classification(self, X, **kwargs):
        pred_class = self.evaluate_binary_classifier(X, **kwargs)

        pred_failures_corr_fn_tn = otaf.surrogate.generate_corrected_binary_predictions(
            pred_class,
            self.opt_res_fn_tn["best_failure_threshold"],
            self.opt_res_fn_tn["best_success_threshold"],
        )
        pred_failures_corr_fp_tp = otaf.surrogate.generate_corrected_binary_predictions(
            pred_class,
            self.opt_res_fp_tp["best_failure_threshold"],
            self.opt_res_fp_tp["best_success_threshold"],
            equality_decision="success",
        )

        true_negatives = np.squeeze(np.argwhere(pred_failures_corr_fn_tn == 0))
        true_positives = np.squeeze(np.argwhere(pred_failures_corr_fp_tp == 1))

        return true_positives, true_negatives

    def soc_optimization_sample(
        self,
        X,
        C=None,
        bounds=None,
        batch_size=1000,
        n_cpu=1,
        progress_bar=False,
        verbose=0,
        dtype="float32",
        batch_size_ai=100000,
    ):
        """
        Perform optimization on samples excluding the ones classified as true negatives or true positives.

        Args:
            X (np.ndarray): Input array of shape NxD.
            bounds (Union[list[list[float]], np.ndarray], optional): Bounds for the variables.
            batch_size (int, optional): Batch size for parallel processing.
            n_cpu (int, optional): Number of CPUs to use for parallel processing.
            progress_bar (bool, optional): Whether to show a progress bar.

        Returns:
            np.ndarray: Combined results of predictions and optimizations.
        """
        true_positives, true_negatives = self.binary_input_array_classification(
            X, batch_size=batch_size_ai
        )

        # Step 3: Filter out predicted true negatives and true positives
        mask = np.ones(X.shape[0], dtype=bool)
        mask[true_negatives] = False
        if not self.calculate_true_positives:
            # We don't calculate true positives
            mask[true_positives] = False
        X_to_optimize = X[mask]

        logging.info(
            f"Calculating {mask.sum():05d} points out of {X.shape[0]} so {100*(1-mask.sum()/X.shape[0]):.4f}% were predicted. With {np.squeeze(true_positives).size:03d} predicted failures."
        )

        combined_results = np.zeros(X.shape[0])
        combined_results[true_negatives] = 0
        if not self.calculate_true_positives:
            combined_results[true_positives] = 1

        slack_values = None
        # Step 4: Send remaining samples to compute_gap_optimizations_on_sample_batch
        if mask.sum() > 0:
            lp_optimization_results = otaf.uncertainty.compute_gap_optimizations_on_sample_batch(
                constraint_matrix_generator=self.constraint_matrix_generator,
                deviation_array=X_to_optimize,
                C=C,
                bounds=bounds,
                batch_size=batch_size,
                n_cpu=n_cpu,
                progress_bar=progress_bar,
                verbose=verbose,
                dtype=dtype,
            )

            slack_values = np.array([x[-1] for x in lp_optimization_results])
            binarized_slack = np.where(slack_values < 0, 1, 0)  # failures are 1, successes are 0

            combined_results[mask] = binarized_slack

        return combined_results, (X_to_optimize, slack_values)
