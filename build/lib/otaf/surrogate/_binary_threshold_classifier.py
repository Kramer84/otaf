# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = [
    "BinaryClassificationModel",
    "generate_corrected_binary_predictions",
    "evaluate_binary_predictions",
    "optimize_thresholds_with_alpha",
    "plot_confusion_matrix",
]

import os
import logging
import re
import copy

from time import time

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import tqdm
import openturns as ot

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.optimize import minimize, basinhopping, shgo

from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import R2Score

import otaf


DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class BinaryClassificationModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        X,
        y,
        slack_threshold=0.1,
        pred_threshold=0.0,  # If > 0 class 1, if negative class 2
        clamping=False,
        clamping_threshold=1.0,
        finish_critertion_epoch=20,
        loss_finish=1e-16,
        metric_finish=0.999,
        max_epochs=100,
        batch_size=100,
        train_size=0.7,
        input_normalization=True,
        noise_dims=None,
        noise_level=0.0,
        scheduler=None,
        device=DEVICE,
        display_progress_disable=False,
        squeeze_labels=False,
        labels_to_long=False,
        use_dual_target=False,  # For cross entropy loss
        save_path=None,
    ):
        super().__init__()

        self.register_buffer("input_dim", torch.tensor(input_dim, dtype=int, requires_grad=False))
        self.register_buffer("output_dim", torch.tensor(output_dim, dtype=int, requires_grad=False))

        self.register_buffer(
            "input_normalization",
            torch.tensor(input_normalization, dtype=bool, requires_grad=False),
        )

        if input_dim == 1:
            X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        self.X_raw = copy.deepcopy(X)
        self.y_raw = copy.deepcopy(y)

        self.register_buffer("X_mean", torch.tensor(X.mean(axis=0), requires_grad=False))
        self.register_buffer(
            "X_std",
            torch.tensor(np.where(X.std(axis=0) == 0.0, 1, X.std(axis=0)), requires_grad=False),
        )

        if input_normalization:
            self.X = (torch.tensor(self.X_raw, dtype=torch.float32) - self.X_mean) / self.X_std
        else:
            self.X = torch.tensor(self.X_raw, dtype=torch.float32)

        self.y = torch.tensor(
            self.y_raw, dtype=torch.float32
        )  # No normalization for output needed for classification

        self.clamping_threshold = clamping_threshold
        self.clamping = clamping

        # Finish criterions
        self.finish_critertion_epoch = finish_critertion_epoch
        self.loss_finish = loss_finish
        self.metric_finish = metric_finish

        self.train_size = train_size

        self.get_train_test_data()

        # Define the layers
        self.fc1 = otaf.surrogate.get_base_relu_mlp_model(
            input_dim, 128
        )  # nn.Linear(input_dim + 1, 128)  # +1 for the threshold input
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output for prediction

        self.model = nn.Sequential(self.fc1, self.fc2, self.fc3, nn.Sigmoid())

        self.register_buffer(
            "slack_threshold", torch.tensor(slack_threshold, dtype=float, requires_grad=False)
        )
        self.register_buffer(
            "pred_threshold", torch.tensor(pred_threshold, dtype=float, requires_grad=False)
        )

        self.register_buffer(
            "squeeze_labels", torch.tensor(squeeze_labels, dtype=bool, requires_grad=False)
        )
        self.register_buffer(
            "labels_to_long", torch.tensor(labels_to_long, dtype=bool, requires_grad=False)
        )
        self.register_buffer(
            "use_dual_target", torch.tensor(use_dual_target, dtype=bool, requires_grad=False)
        )

        # Training parameters
        self.n_epochs = max_epochs
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        self.metric = confusion_matrix

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.device = device
        self.noise_dims = noise_dims
        self.noise_level = noise_level
        self.display_progress_disable = display_progress_disable
        self.history_loss = []
        self.history_metric = []
        self.best_loss = float("inf")
        self.best_metric = 0.0
        self.best_weights = copy.deepcopy(self.state_dict())
        self.save_path = save_path

    def forward(self, x):
        # Forward pass through the network
        pred = self.model(x)
        return pred

    def get_train_test_data(self):
        # Train-test split of the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, train_size=self.train_size, shuffle=True
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def add_new_data_points(self, new_X, new_y):
        # Ensure the new data points have the correct shape
        self.to("cpu")
        if new_X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected new_X to have {self.input_dim} features, but got {new_X.shape[1]}"
            )
        if new_y.ndim == 1:
            new_y = new_y.reshape(-1, 1)
        # Normalize new data points if necessary
        if self.input_normalization:
            new_X = (torch.tensor(new_X, dtype=torch.float32) - self.X_mean) / self.X_std
        else:
            new_X = torch.tensor(new_X, dtype=torch.float32)
        # Convert new_y to tensor
        new_y = torch.tensor(new_y, dtype=torch.float32)
        # Append new data points to the existing raw data
        self.X_raw = np.vstack([self.X_raw, new_X])
        self.y_raw = np.vstack([self.y_raw, new_y])
        # Update normalized X
        if self.input_normalization:
            self.X = (torch.tensor(self.X_raw, dtype=torch.float32) - self.X_mean) / self.X_std
        else:
            self.X = torch.tensor(self.X_raw, dtype=torch.float32)
        # Update y
        self.y = torch.tensor(self.y_raw, dtype=torch.float32)
        # Re-split the data into training and testing sets
        self.get_train_test_data()

    def train_model(self):
        self.train()
        self.to(self.device)
        X_test = self.X_test.to(self.device)
        y_test = self.y_test.to(self.device)

        for epoch in range(self.n_epochs):
            self.train(True)
            permutation = torch.randperm(self.X_train.size()[0])
            with tqdm.tqdm(
                permutation, unit="batch", mininterval=0.05, disable=self.display_progress_disable
            ) as bar:
                bar.set_description(f"Epoch {epoch + 1:03d}")
                for i in range(0, self.X_train.size()[0], self.batch_size):
                    indices = permutation[i : i + self.batch_size]
                    X_batch = self.X_train[indices].to(self.device)
                    y_batch = self.y_train[indices].to(self.device)

                    # Add noise if needed
                    if self.noise_dims and self.noise_level > 0:
                        X_batch[:, self.noise_dims] = otaf.surrogate.add_gaussian_noise(
                            X_batch[:, self.noise_dims], self.noise_level
                        )

                    self.optimizer.zero_grad()
                    y_pred = self.forward(X_batch)

                    y_batch_binary = torch.where(y_batch < self.slack_threshold, 1.0, 0.0).float()

                    y_batch_binary_dual = torch.cat(
                        (
                            torch.where(y_batch < self.slack_threshold, 1.0, 0.0),
                            torch.where(y_batch >= self.slack_threshold, 1.0, 0.0),
                        ),
                        dim=1,
                    ).float()

                    y_batch_binary, y_batch_binary_dual = (
                        (y_batch_binary.long(), y_batch_binary_dual.long())
                        if self.labels_to_long
                        else (y_batch_binary, y_batch_binary_dual)
                    )

                    # print("y_pred.shape:", y_pred.shape)
                    # print("y_batch_binary_dual.shape:", y_batch_binary_dual.shape)

                    if self.use_dual_target:
                        loss = self.criterion(
                            y_pred,
                            (
                                y_batch_binary_dual.squeeze()
                                if self.squeeze_labels
                                else y_batch_binary_dual
                            ),
                        )
                    else:
                        loss = self.criterion(
                            y_pred,
                            y_batch_binary.squeeze() if self.squeeze_labels else y_batch_binary,
                        )
                    loss.backward()
                    self.optimizer.step()

                    bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])

                    if self.clamping:
                        for p in self.parameters():
                            p.data.clamp_(-self.clamping_threshold, self.clamping_threshold)

            # Validation
            self.eval()
            with torch.no_grad():
                y_pred = self.forward(X_test)
                y_test_binary = torch.where(y_test < self.slack_threshold, 1, 0)

                y_test_binary_dual = torch.cat(
                    (
                        torch.where(y_test < self.slack_threshold, 1.0, 0.0),
                        torch.where(y_test >= self.slack_threshold, 1.0, 0.0),
                    ),
                    dim=1,
                )

                y_test_binary, y_test_binary_dual = (
                    (y_test_binary.long(), y_test_binary_dual.long())
                    if self.labels_to_long
                    else (y_test_binary, y_test_binary_dual)
                )

                if self.use_dual_target:
                    val_loss = self.criterion(
                        y_pred,
                        y_test_binary_dual.squeeze() if self.squeeze_labels else y_test_binary_dual,
                    )
                else:
                    val_loss = self.criterion(
                        y_pred, y_test_binary.squeeze() if self.squeeze_labels else y_test_binary
                    )

                y_pred_class = torch.where(y_pred > self.pred_threshold, 1, 0)
                val_loss = self.criterion(
                    y_pred, y_test_binary.squeeze() if self.squeeze_labels else y_test_binary
                )

                ratio_predicted = (y_pred_class == y_test_binary).float().mean()  # We change that

                if self.output_dim > 1:
                    cm = confusion_matrix(
                        y_test_binary.cpu().numpy(), y_pred_class[:, 1].cpu().numpy()
                    )
                else:
                    cm = confusion_matrix(y_test_binary.cpu().numpy(), y_pred_class.cpu().numpy())
                tn, fp, fn, tp = cm.ravel()
                val_metric = min(1.0, float(fn / (tn + 1e-9)))  # ratio of false negatives

            self.history_loss.append(val_loss.item())
            self.history_metric.append(val_metric)

            print(
                f"\tEpoch {epoch + 1:03d}, Val Loss: {val_loss.item():.6f}, Ratio predicted: {ratio_predicted.item():.6f}, Ratio FN/TN {val_metric:.6f}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}"
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss.item()
                self.best_metric = val_metric
                self.best_weights = copy.deepcopy(self.state_dict())

            if self.scheduler:
                self.scheduler.step()

            if self.training_stopping_criterion(epoch):
                break

        print(
            f"Finished training with best loss {self.best_loss:.6f} and metric of {self.best_metric:.6f}"
        )

        X_test = self.X_test.cpu()
        y_test = self.y_test.cpu()
        torch.cuda.empty_cache()

        # Restore the best model
        self.load_state_dict(self.best_weights)

    def evaluate_model(self, x, batch_size=50000, return_on_gpu=False):
        """
        Evaluates the model in non-standard space, processing the input in batches.

        Args:
            x (array-like or torch.Tensor): Input data to evaluate.
            batch_size (int, optional): Size of batches for evaluation. Default is 50000.
            return_on_gpu (bool, optional): Whether to return the result on the GPU. Default is False.

        Returns:
            torch.Tensor: The model's output.
        """
        # Standardize the input
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if self.input_normalization:
            x = (x.cpu() - self.X_mean.cpu()) / self.X_std.cpu()

        # Ensure the model is in eval mode and sent to the right device
        self.eval()
        self.to(DEVICE)

        # Prepare to collect the results
        results = []

        try:
            # Evaluate in batches
            with torch.no_grad():
                for i in range(0, len(x), batch_size):
                    batch = x[i : i + batch_size].to(DEVICE)
                    output = self(batch)

                    # Move to CPU if necessary
                    if not return_on_gpu:
                        output = output.cpu()

                    results.append(output)

            # Concatenate all batch results into a single tensor
            final_result = torch.cat(results, dim=0)
        finally:
            # Ensure to free up the GPU memory
            del batch, output
            torch.cuda.empty_cache()

        return final_result

    def training_stopping_criterion(self, epoch):
        """
        Determines whether training should stop based on various criteria including:
        - Minimum loss tolerance.
        - Gradient changes.
        - R2 score achievement.

        Args:
        epoch (int): The current epoch number.

        Returns:
        bool: True if training should stop, False otherwise.
        """

        # Check various conditions to determine if training should stop
        if epoch > self.finish_critertion_epoch:
            if self.best_loss <= self.loss_finish:
                print("Stopping: Loss below threshold.")
                return True
            elif self.best_metric >= self.metric_finish:
                print("Stopping: R2 score is high enough.")
                return True
        return False

    def plot_results(
        self, save_as_png=False, save_path="images", filename="ai_training_results.png", dpi=600
    ):
        # Create the figure and two axes
        fig = plt.figure(figsize=(10, 6))  # Increased figure size for better readability
        ax1 = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)

        # Plot loss on the first axis
        ax1.plot(
            self.history_loss, color="C0", linewidth=2
        )  # Increased line width for better visibility
        ax1.set_xlabel("Epoch", color="C0", fontsize=14)  # Increased font size
        ax1.set_ylabel("Loss", color="C0", fontsize=14)  # Increased font size
        ax1.tick_params(axis="x", colors="C0", labelsize=12)  # Adjusted tick label size
        ax1.tick_params(axis="y", colors="C0", labelsize=12)

        # Plot metric on the second axis
        ax2.plot(
            self.history_metric, color="C1", linewidth=2
        )  # Increased line width for better visibility
        ax2.yaxis.tick_right()
        ax2.set_ylabel("Metric", color="C1", fontsize=14)  # Increased font size
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(axis="y", colors="C1", labelsize=12)
        ax2.set_xticks([])

        # Add grid for better readability
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Improve layout spacing
        plt.tight_layout()

        # Save the plot as a PNG if required
        if save_as_png:
            # Create 'images' directory if it doesn't exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save the figure
            fig.savefig(os.path.join(save_path, filename), dpi=dpi, bbox_inches="tight")
            print(f"Plot saved as {filename} in the 'images' directory.")

        # Show the plot
        plt.show()

    def save_model(self):
        torch.save(self.state_dict(), self.save_path)

    def load_model(self):
        self.load_state_dict(torch.load(self.save_path))
        self.best_weights = copy.deepcopy(self.state_dict())


def generate_corrected_binary_predictions(
    pred_probs, failure_threshold=-0.5, success_threshold=0.5, equality_decision="failure"
):
    """
    Generates corrected binary predictions for failures and successes based on threshold values.

    Parameters:
    pred_probs (np.ndarray): Array of predicted class probabilities with shape (n_samples, 2).
    failure_threshold (float): Threshold value for classifying failures.
    success_threshold (float): Threshold value for classifying successes.
    equality_decision (str): What is the outcome when both success and failure are positive? "failure" or "success"

    Returns:
    np.ndarray: Corrected binary predictions.
    """
    if equality_decision == "failure":
        eq_outcome = 1
    elif equality_decision == "success":
        eq_outcome = 0
    else:
        raise ValueError("equality_decision is either 'success' or 'failure'")

    # Generate binary predictions for failures
    pred_failures = np.where(pred_probs[:, 1] >= failure_threshold, 1, 0)
    # Generate binary predictions for successes
    pred_success = np.where(pred_probs[:, 0] >= success_threshold, 1, 0)
    # Correct failures and successes. If both have the same outcome, it automatically becomes eq_outcome
    corrected_predictions = np.where(pred_failures == pred_success, eq_outcome, pred_failures)
    return corrected_predictions


def evaluate_binary_predictions(predictions, labels):
    """
    Evaluates binary predictions against ground truth labels and computes confusion matrix and accuracy.

    Parameters:
    predictions (np.ndarray): Array of binary predictions (1 for positive class, 0 for negative class).
    labels (np.ndarray): Array of ground truth binary labels (1 for positive class, 0 for negative class).

    Returns:
    dict: Dictionary containing confusion matrix, accuracy, and detailed metrics.
    """
    # Ensure inputs are numpy arrays
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    # Compute accuracy
    accuracy = (tn + tp) / (tn + fp + fn + tp)

    # Compute other metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    return metrics


def optimize_thresholds_with_alpha(
    pred_probs,
    ground_truth,
    bounds=[-5.0, 5.0],
    optimize_for="minimize_fn_maximize_tn",
    equality_decision="failure",
    optimal_ratio=1 / 10000,
):
    """
    Optimizes the failure and success thresholds based on the specified objective using basin hopping.

    Parameters:
    pred_probs (np.ndarray): Array of predicted class probabilities with shape (n_samples, 2).
    ground_truth (np.ndarray): Array of true binary labels.
    bounds (list): Bounds for the threshold values, default is [-5, 5].
    optimize_for (str): Objective for optimization, "minimize_fn_maximize_tn" or "minimize_fp_maximize_tp".
    equality_decision (str): What is the outcome when both success and failure are positive? "failure" or "success".
    optimal_ratio (float): Ratio for penalizing false negatives/positives, default is 1/10000.

    Returns:
    dict: Best thresholds, alpha (currently unused), and corresponding evaluation metrics.
    """

    # Informative message at the start
    print(
        "Using basin hopping with cobyla to optimize thresholds for minimizing classification errors."
    )

    def objective(params):
        """
        Objective function to minimize based on the specified optimization goal.
        """
        failure_threshold, success_threshold = params
        predictions = generate_corrected_binary_predictions(
            pred_probs, failure_threshold, success_threshold, equality_decision
        )
        evaluation = evaluate_binary_predictions(predictions, ground_truth)

        fn = evaluation["false_negatives"]
        tn = evaluation["true_negatives"]
        tp = evaluation["true_positives"]
        fp = evaluation["false_positives"]

        # Normalize the objective function results
        total_positives = fn + tp
        total_negatives = tn + fp

        if optimize_for == "minimize_fn_maximize_tn":
            return (fn / optimal_ratio - tn) / total_negatives
        elif optimize_for == "minimize_fp_maximize_tp":
            return (fp / optimal_ratio - tp) / total_positives
        else:
            raise ValueError(
                "Invalid value for optimize_for. Choose either 'minimize_fn_maximize_tn' or 'minimize_fp_maximize_tp'."
            )

    # Setup for the basin hopping minimizer
    minimizer_kwargs = {
        "method": "COBYLA",
        "bounds": [bounds, bounds],  # Applying bounds to both failure and success thresholds
        "options": {
            "disp": False,
            "maxiter": 1000,
            "tol": 1e-6,
        },  # Turn off display for optimization steps
    }

    # Basin hopping optimization
    result = basinhopping(
        objective,
        x0=np.random.uniform(bounds[0], bounds[1], 2),  # Random initial guess within bounds
        minimizer_kwargs=minimizer_kwargs,
        niter=50,  # Number of iterations for basin hopping
        disp=False,  # Suppressing verbose output
        T=1.0,  # Temperature parameter for randomness in basin hopping
        stepsize=2.5,  # Step size for random displacement
        niter_success=10,  # Stop if no improvement after this many iterations
    )

    # Extracting the best found thresholds
    best_failure_threshold, best_success_threshold = result.x

    # Generate predictions and evaluate using the best thresholds
    best_predictions = generate_corrected_binary_predictions(
        pred_probs, best_failure_threshold, best_success_threshold, equality_decision
    )
    best_evaluation = evaluate_binary_predictions(best_predictions, ground_truth)

    # Returning the result in a dictionary format
    return {
        "best_failure_threshold": best_failure_threshold,
        "best_success_threshold": best_success_threshold,
        "best_alpha": "None",  # Currently unused; placeholder for future use
        "evaluation": best_evaluation,
    }


def plot_confusion_matrix(cm):
    ConfusionMatrixDisplay(cm).plot()
