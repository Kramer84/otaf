# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = ["DualNetworkPredictor"]

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
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import R2Score
from torch.utils.data import TensorDataset, random_split

import otaf

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class DualNetworkPredictor(nn.Module):
    """
    A PyTorch module that combines two neural networks to predict slack from defects.

    This module first predicts gaps from defects using the first neural network,
    then combines the original input array with the predicted gaps to predict slack
    using the second neural network.

    Attributes:
        model_defects_to_gaps (nn.Module): The first neural network model.
        model_gap_def_to_slack (nn.Module): The second neural network model.
    """

    def __init__(self, model_defects_to_gaps, model_gap_def_to_slack):
        super().__init__()
        self.model_defects_to_gaps = model_defects_to_gaps
        self.model_gap_def_to_slack = model_gap_def_to_slack

        self.X_def = torch.tensor(self.model_defects_to_gaps.X_raw)
        self.y_gap = torch.tensor(self.model_defects_to_gaps.y_raw)
        self.y_slack = torch.tensor(self.model_gap_def_to_slack.y_raw)

        self.register_buffer(
            "mean_def", self.model_defects_to_gaps.X_mean.clone().detach().requires_grad_(False)
        )
        self.register_buffer(
            "std_def", self.model_defects_to_gaps.X_std.clone().detach().requires_grad_(False)
        )
        self.register_buffer(
            "mean_gap", self.model_defects_to_gaps.y_mean.clone().detach().requires_grad_(False)
        )
        self.register_buffer(
            "std_gap", self.model_defects_to_gaps.y_std.clone().detach().requires_grad_(False)
        )
        self.register_buffer(
            "mean_slack", self.model_gap_def_to_slack.y_mean.clone().detach().requires_grad_(False)
        )
        self.register_buffer(
            "std_slack", self.model_gap_def_to_slack.y_std.clone().detach().requires_grad_(False)
        )

        self.metric = R2Score()

        # Hold the best model
        self.best_R2 = 0
        self.best_loss = np.inf  # init to infinity
        self.best_weights = copy.deepcopy(self.state_dict())
        self.history_gaps = []
        self.history_slack = []

    def load_best_weights(self):
        self.load_state_dict(self.best_weights)

    def restore_start_weights(self):
        self.model_defects_to_gaps.load_model()
        self.model_gap_def_to_slack.load_model()
        self.best_weights = copy.deepcopy(self.state_dict())

    def get_train_test_data(self, test_size=0.2, random_seed=42):
        """
        Splits the dataset into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
            random_seed (int): Seed for random number generator. Default is 42.

        Returns:
            tuple: Training and testing datasets (X_train, y_train_gaps, y_train_slack, X_test, y_test_gaps, y_test_slack).
        """
        dataset = TensorDataset(self.X_def, self.y_gap, self.y_slack)
        total_size = len(dataset)
        test_size = int(total_size * test_size)
        train_size = total_size - test_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed)
        )

        self.X_train = train_dataset.dataset.tensors[0][train_dataset.indices]
        self.y_train_gaps = train_dataset.dataset.tensors[1][train_dataset.indices]
        self.y_train_slack = train_dataset.dataset.tensors[2][train_dataset.indices]

        self.X_test = test_dataset.dataset.tensors[0][test_dataset.indices]
        self.y_test_gaps = test_dataset.dataset.tensors[1][test_dataset.indices]
        self.y_test_slack = test_dataset.dataset.tensors[2][test_dataset.indices]

    def forward(self, x):
        """
        Forward pass through the combined networks.

        Args:
            x (torch.Tensor): Input tensor containing normalized defect features.
            batch_size (int, optional): Batch size for processing. Default is 250000.

        Returns:
            torch.Tensor: Predicted slack values.
        """

        if not self.model_defects_to_gaps.input_normalization:  # We assume x is in std space
            pred_gaps = self.model_defects_to_gaps(x * self.std_def + self.mean_def)
        else:
            pred_gaps = self.model_defects_to_gaps(x)

        if not self.model_defects_to_gaps.output_normalization:
            pred_gaps = (pred_gaps - self.mean_gap) / self.std_gap

        # Combine the original input array with the predicted gaps
        if not self.model_gap_def_to_slack.input_normalization:
            combined_array = torch.cat(
                (x * self.std_def + self.mean_def, pred_gaps * self.std_gap + self.mean_gap), dim=1
            )
        else:
            combined_array = torch.cat((x, pred_gaps), dim=1)

        pred_slack = self.model_gap_def_to_slack(combined_array)

        # if not self.model_gap_def_to_slack.output_normalization: # If the output is not normalized we normalize?
        #    pred_slack = (pred_slack-self.mean_slack)/self.std_slack

        return pred_slack

    def evaluate_in_batches(self, x, batch_size=50000, return_on_gpu=False):
        """
        Helper function to evaluate the combined model in batches.

        Args:
            x (torch.Tensor): Input tensor in real space
            batch_size (int, optional): Batch size for processing. Default is 50000.
            return_on_gpu (bool, optional): Whether to return the result on the GPU. Default is False.

        Returns:
            torch.Tensor: The concatenated output of the model evaluated in batches.
        """
        self.load_best_weights()
        self.eval()
        self.to(DEVICE)
        results = []

        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = (
                    torch.tensor(x[i : i + batch_size], dtype=torch.float32).to(DEVICE)
                    - self.mean_def
                ) / self.std_def  # Normalizing
                output = self(batch)
                if self.model_gap_def_to_slack.output_normalization:
                    output = output * self.std_slack + self.mean_slack
                if not return_on_gpu:
                    output = output.cpu()
                results.append(output)

                # Free up the GPU memory
                del batch, output
            torch.cuda.empty_cache()

        final_result = torch.cat(results, dim=0)

        return final_result

    def brute_force_probability_of_failure(
        self,
        composed_distribution,
        N_MC_MAX=int(1e9),
        N_GEN_MAX=int(1e7),
        batch_size=500000,
        PF_STAB=1e-6,
        threshold=0.0,
    ):
        """
        Brute-force calculation of the probability of failure.

        Args:
            x (torch.Tensor): Input tensor in real space.
            threshold (float): The threshold below which a prediction is considered a failure.
            batch_size (int, optional): Batch size for processing. Default is 50000.
            N_MC_MAX (int, optional): Maximum number of Monte Carlo samples to evaluate. Default is 1e9.
            PF_STAB (float, optional): Stability threshold for stopping criterion. Default is 1e-6.

        Returns:
            float: The probability of failure.
        """
        N_FIN = 0  # Final size of Monte Carlo
        means = list(composed_distribution.getMean())
        stnds = list(composed_distribution.getStandardDeviation())
        torchDist = torch.distributions.Normal(torch.Tensor(means), torch.Tensor(stnds))
        self.load_best_weights()
        with torch.no_grad():
            self.to(DEVICE)
            self.eval()

            failures = torch.Tensor().to(torch.int8).to_sparse().to(DEVICE)
            pf_array = np.array([], dtype="float32")

            for i in range(N_MC_MAX // N_GEN_MAX):
                sample = torchDist.sample((N_GEN_MAX,)).cpu()
                sample = (sample - self.mean_def.cpu()) / self.std_def.cpu()

                for j in range(0, N_GEN_MAX, batch_size):
                    batch = sample[j : j + batch_size].to(DEVICE)
                    output = self(batch)
                    if self.model_gap_def_to_slack.output_normalization:
                        output = output * self.std_slack + self.mean_slack

                    output = (
                        torch.squeeze(torch.where(output < threshold, 1, 0))
                        .to(torch.int8)
                        .to_sparse()
                    )

                    failures = torch.cat((failures, output), 0)
                    total_failures = failures.sum().item()
                    total_samples_processed = failures.numel()

                    # Calculate the current probability of failure
                    pf_cpu = total_failures / total_samples_processed
                    pf_cpu_mod = (total_failures + 1) / total_samples_processed

                    pf_array = np.append(pf_array, pf_cpu)

                # Check the stopping criterion within the inner loop
                if abs(pf_cpu - pf_cpu_mod) < PF_STAB:
                    print(
                        f"Finished at iteration {i // batch_size + 1} with {total_samples_processed} samples processed. Pf: {pf_cpu}"
                    )
                    return pf_array[-1]

                # Free up the GPU memory
                del batch, output
                torch.cuda.empty_cache()

        return pf_array[-1]  # or np.mean(pf_array) if averaging is preferred

    def train_model(
        self,
        batch_size,
        num_epochs,
        loss_f_gaps,
        loss_f_slack,
        optimizer_gaps,
        optimizer_slack,
        device=DEVICE,
        display_progress_disable=False,
    ):
        """
        Train the dual network model.

        Args:
            batch_size (int): Batch size for training.
            num_epochs (int): Number of epochs to train.
            loss_f_gaps (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device to use for training (CPU or GPU).
            display_progress_disable (bool): Flag to disable tqdm progress display.

        Returns:
            None
        """
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
        self.to(device)
        best_loss = float("inf")
        best_weights = None

        # Determine tqdm function based on environment
        if otaf.common.is_running_in_notebook():
            tq = tqdm.tqdm_notebook
        else:
            tq = tqdm.tqdm

        try:
            for epoch in range(num_epochs):
                self.train()
                running_loss_gaps = 0.0
                running_loss_slack = 0.0

                with tq(
                    range(0, len(self.X_train), batch_size),
                    unit="batch",
                    disable=display_progress_disable,
                ) as bar:
                    bar.set_description(f"Epoch {epoch+1:03d}")
                    for i in bar:
                        inputs = self.X_train[i : i + batch_size].to(device)
                        gaps_targets = self.y_train_gaps[i : i + batch_size].to(device)
                        slack_targets = self.y_train_slack[i : i + batch_size].to(device)

                        optimizer_gaps.zero_grad()
                        # Step 1: Optimize the upper network (defects to gaps)
                        if (
                            self.model_defects_to_gaps.input_normalization
                        ):  # X is not in standard space
                            pred_gaps = self.model_defects_to_gaps(
                                (inputs - self.mean_def) / self.std_def
                            )
                        else:
                            pred_gaps = self.model_defects_to_gaps(inputs)
                        if self.model_defects_to_gaps.output_normalization:
                            pred_gaps = pred_gaps * self.std_gap + self.mean_gap
                        loss_gaps = loss_f_gaps(pred_gaps, gaps_targets)
                        loss_gaps.backward(retain_graph=True)
                        optimizer_gaps.step()

                        # Detach pred_gaps to avoid modifying it in-place in the next step
                        pred_gaps = pred_gaps.detach()

                        # Step 2: Optimize the combined network (defects + gaps to slack)
                        optimizer_slack.zero_grad()
                        if self.model_gap_def_to_slack.input_normalization:
                            combined_array = torch.cat(
                                (
                                    (inputs - self.mean_def) / self.std_def,
                                    (pred_gaps - self.mean_gap) / self.std_gap,
                                ),
                                dim=1,
                            )
                        else:
                            combined_array = torch.cat((inputs, pred_gaps), dim=1)
                        pred_slack = self.model_gap_def_to_slack(combined_array)
                        if self.model_gap_def_to_slack.output_normalization:
                            pred_slack = pred_slack * self.std_slack + self.mean_slack

                        loss_slack = loss_f_slack(pred_slack, slack_targets)
                        loss_slack.backward()

                        optimizer_slack.step()

                        running_loss_gaps += loss_gaps.item() * inputs.size(0)
                        running_loss_slack += loss_slack.item() * inputs.size(0)

                        bar.set_postfix(loss_gaps=loss_gaps.item(), loss_slack=loss_slack.item())

                # Calculate average training losses
                train_loss_gaps = running_loss_gaps / len(self.X_train)
                train_loss_slack = running_loss_slack / len(self.X_train)

                # Validation phase
                self.metric.reset()
                self.eval()

                val_loss_gaps = 0.0
                val_loss_slack = 0.0
                with torch.no_grad():
                    for i in range(0, len(self.X_test), batch_size):
                        inputs = self.X_test[i : i + batch_size].to(device)
                        gaps_targets = self.y_test_gaps[i : i + batch_size].to(device)
                        slack_targets = self.y_test_slack[i : i + batch_size].to(device)

                        # Forward pass through both networks
                        pred_gaps = self.model_defects_to_gaps(inputs)
                        if not self.model_defects_to_gaps.output_normalization:
                            pred_gaps = (pred_gaps - self.mean_gap) / self.std_gap
                        combined_array = torch.cat((inputs, pred_gaps), dim=1)
                        pred_slack = self.model_gap_def_to_slack(combined_array)

                        # Compute losses
                        loss_gaps = loss_f_gaps(pred_gaps, gaps_targets)
                        loss_slack = loss_f_slack(pred_slack, slack_targets)
                        val_loss_gaps += loss_gaps.item() * inputs.size(0)
                        val_loss_slack += loss_slack.item() * inputs.size(0)
                        self.metric.update(slack_targets, pred_slack)

                # Calculate average validation losses
                val_loss_gaps = val_loss_gaps / len(self.X_test)
                val_loss_slack = val_loss_slack / len(self.X_test)

                self.history_gaps.append(val_loss_gaps)
                self.history_slack.append(val_loss_slack)

                R2 = float(self.metric.compute())
                self.metric.reset()

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Train Loss Gaps: {train_loss_gaps:.4f}, Train Loss Slack: {train_loss_slack:.4f}, Val Loss Gaps: {val_loss_gaps:.4f}, Val Loss Slack: {val_loss_slack:.4f}, R2: {R2:.4f}"
                )

                # Save the best model weights
                val_loss = val_loss_gaps + val_loss_slack
                if val_loss < best_loss:
                    self.best_loss = val_loss
                    self.best_R2 = R2
                    best_weights = copy.deepcopy(self.state_dict())

        except KeyboardInterrupt:
            print("Training interrupted by user")

        print(
            f"Finished training at epoch {epoch+1} with best validation loss {self.best_loss:.6f} and R2 of {self.best_R2:.6f}"
        )

        # Restore the best model weights
        if best_weights is not None:
            self.best_weights = best_weights
            self.load_state_dict(best_weights)
        torch.cuda.empty_cache()

    def plot_results(self):
        plt.plot(self.history_gaps, label="gaps")
        plt.plot(self.history_slack, label="slack")
        plt.show()
