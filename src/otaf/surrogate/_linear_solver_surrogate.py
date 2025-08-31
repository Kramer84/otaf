# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = ["LPNeuralSurrogateTolereancing", "initialize_model_weights"]

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

import tqdm

import otaf
from ._base_models import get_custom_mlp_layers, add_gaussian_noise

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class LPNeuralSurrogateTolereancing(nn.Module):
    """
    - Compatibility equations (equality constraints) become A_eq_Def*X + A_eq_Gap*Y + K_eq = 0.
    - Interface equations (inequality constraints) become A_ub_Def*X + A_ub_Gap*Y + K_ub >= 0.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        X,
        y,
        nEq,
        nUb,
        A_eq_Def,
        A_eq_Gap,
        K_eq,
        A_ub_Def,
        A_ub_Gap,
        K_ub,
        clamping=False,
        finish_critertion_epoch=20,
        loss_finish=1e-16,
        metric_finish=0.999,
        max_epochs=100,
        batch_size=100,
        compile_model=True,
        train_size=0.7,
        save_path=None,
        input_description=None,
        display_progress_disable=True,
        noise_dims=[],  # If populated it generated noise for the selected dimensions during training.
        noise_level=0.1,
    ):
        super().__init__()

        self.register_buffer("input_dim", torch.tensor(input_dim, dtype=int, requires_grad=False))
        self.register_buffer("output_dim", torch.tensor(output_dim, dtype=int, requires_grad=False))

        self.register_buffer("nEq", torch.tensor(nEq, dtype=int, requires_grad=False))
        self.register_buffer("nUb", torch.tensor(nUb, dtype=int, requires_grad=False))

        # All the linear programming matrices.
        self.register_buffer(
            "A_eq_Def", torch.tensor(A_eq_Def, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "A_eq_Gap", torch.tensor(A_eq_Gap, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer("K_eq", torch.tensor(K_eq, dtype=torch.float32, requires_grad=False))
        self.register_buffer(
            "A_ub_Def", torch.tensor(A_ub_Def, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "A_ub_Gap", torch.tensor(A_ub_Gap, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer("K_ub", torch.tensor(K_ub, dtype=torch.float32, requires_grad=False))

        if input_dim == 1:
            X = X.reshape(-1, 1)
        if output_dim == 1:
            y = y.reshape(-1, 1)

        self.X_raw = copy.deepcopy(X)
        self.y_raw = copy.deepcopy(y)

        self.register_buffer("X_mean", torch.tensor(X.mean(axis=0), requires_grad=False))
        self.register_buffer("y_mean", torch.tensor(y.mean(axis=0), requires_grad=False))
        self.register_buffer(
            "X_std",
            torch.tensor(np.where(X.std(axis=0) == 0.0, 1, X.std(axis=0)), requires_grad=False),
        )
        self.register_buffer(
            "y_std",
            torch.tensor(np.where(y.std(axis=0) == 0.0, 1, y.std(axis=0)), requires_grad=False),
        )

        self.X = torch.tensor(self.X_raw, dtype=torch.float32)
        self.y = torch.tensor(self.y_raw, dtype=torch.float32)

        self.clamping = clamping

        # Finish criterions
        self.finish_critertion_epoch = finish_critertion_epoch
        self.loss_finish = loss_finish
        self.metric_finish = metric_finish

        self.compile_model = compile_model
        self.train_size = train_size

        self.save_path = save_path
        self.input_description = input_description
        self.display_progress_disable = display_progress_disable

        self.get_train_test_data()

        # Performance metric
        self.metric = R2Score()

        # loss function and optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

        # training parameters
        self.n_epochs = max_epochs  # number of epochs to run
        self.batch_size = batch_size  # size of each batch
        self.batch_start = torch.arange(0, len(self.X_train), self.batch_size)

        # Hold the best model
        self.best_metric = 0
        self.best_loss = np.inf  # init to infinity
        self.best_weights = copy.deepcopy(self.state_dict())
        self.history_loss = []
        self.history_metric = []

        self.noise_dims = noise_dims
        self.noise_level = noise_level

        # layers
        self.layer_eq = nn.Sequential(
            *otaf.surrogate.get_custom_mlp_layers(
                [self.nEq, self.nEq + self.output_dim // 2, self.output_dim],
                activation_class=torch.nn.CELU,
            )
        )

        self.layer_ub = nn.Sequential(
            *otaf.surrogate.get_custom_mlp_layers(
                [self.nUb, self.nUb + self.output_dim // 2, self.output_dim],
                activation_class=torch.nn.CELU,
            )
        )

        self.layer_XY = nn.Sequential(
            *otaf.surrogate.get_custom_mlp_layers(
                [self.input_dim, self.output_dim], activation_class=torch.nn.CELU
            )
        )

        self.layer_aydyx = nn.Sequential(
            *otaf.surrogate.get_custom_mlp_layers(
                [
                    self.nUb + self.output_dim + self.nEq,
                    self.output_dim + (self.nEq + self.nUb) * 2,
                    self.output_dim * 2,
                ]
            )
        )

        self.layer_x_rem = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            *otaf.surrogate.get_custom_mlp_layers(
                [self.input_dim, self.input_dim], activation_class=torch.nn.CELU
            ),
        )

        self.layer_pred = nn.Sequential(
            *otaf.surrogate.get_custom_mlp_layers(
                [self.output_dim * 2, (self.output_dim * 3) // 2, self.output_dim],
                activation_class=torch.nn.CELU,
            )
        )

        self.layer_act_y_h1 = nn.Sequential(nn.BatchNorm1d(self.nEq), nn.Tanh())
        self.layer_act_y_h2 = nn.Sequential(nn.BatchNorm1d(self.nUb), nn.Tanh())

    def forward(self, x):
        # Here everything is done in the original space. x is the vector of defects.

        Keq = self.K_eq.unsqueeze(1)  # Expand dimensions to match for broadcasting
        Kub = self.K_ub.unsqueeze(1)  # Expand dimensions to match for broadcasting

        # Perform matrix multiplications by adjusting the order of the operands
        AeqXK = (
            torch.matmul(x, self.A_eq_Def.T) + Keq.T
        )  # Transpose A_eq_Def for correct dimension alignment
        AubXK = (
            torch.matmul(x, self.A_ub_Def.T) + Kub.T
        )  # Transpose A_ub_Def for correct dimension alignment

        y1 = self.layer_eq(self.layer_act_y_h1(AeqXK))  # Transpose to match expected input shape
        y2 = self.layer_ub(self.layer_act_y_h2(AubXK))  # Transpose to match expected input shape
        y3 = self.layer_XY((x - self.X_mean) / self.X_std)

        AeqY = torch.matmul(
            y1 * self.y_std + self.y_mean, self.A_eq_Gap.T
        )  # Transpose A_eq_Gap for correct dimension alignment
        AubY = torch.matmul(
            y2 * self.y_std + self.y_mean, self.A_ub_Gap.T
        )  # Transpose A_ub_Gap for correct dimension alignment

        # yd = self.layer_norm_yd(y2 - y1)

        AyDy = torch.cat((AeqY, y3, AubY), dim=1)  # Correct concatenation dimension

        yxrem = self.layer_aydyx(AyDy)  # + self.layer_x_rem(x)

        pred = self.layer_pred(yxrem)

        return pred

    def evaluate_model_non_standard_space(self, x, batch_size=50000, return_on_gpu=False):
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

    def pf_monte_carlo_bruteforce(
        self,
        composed_distribution,
        N_MC_MAX=int(1e9),
        N_GEN_MAX=int(1e7),
        batch_size=500000,
        PF_STAB=1e-6,
        threshold=0.0,
    ):
        """
        N_MC_MAX : Monte carlo size
        N_GEN_MAX : Max sample size to generate
        PF_STAB : Max variability of probability if one new point is added.
        threshold : failure threshold
        """
        with torch.no_grad():
            N_FIN = 0  # Final size of monte carlo
            X_std, X_mean = (
                self.X_std.clone().detach().to(DEVICE),
                self.X_mean.clone().detach().to(DEVICE),
            )
            y_std, y_mean = (
                self.y_std.clone().detach().to(DEVICE),
                self.y_mean.clone().detach().to(DEVICE),
            )
            self.eval()
            self.to(DEVICE)
            means = list(composed_distribution.getMean())
            stnds = list(composed_distribution.getStandardDeviation())
            torchDist = torch.distributions.Normal(torch.Tensor(means), torch.Tensor(stnds))
            pf_array = np.array([], dtype="float32")
            failures = (
                torch.Tensor().to(torch.int8).to_sparse().to(DEVICE)
            )  # Array to store the failures
            for i in range(N_MC_MAX // N_GEN_MAX):
                sample = torchDist.sample((N_GEN_MAX,)).to("cpu")
                sample = (sample - X_mean.cpu()) / X_std.cpu()

                for j in range(0, N_GEN_MAX, batch_size):
                    batch = sample[j : j + batch_size].to(DEVICE)
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
                            f"Finished at iteration {int((i+1)*(j/batch_size+1))} with {total_samples_processed} experiments. Pf: {pf_cpu}"
                        )
                        return pf_array[-1]  # or np.mean(pf_array) to return the average Pf

        return pf_array[-1]  # or np.mean(pf_array) if averaging is preferred

    def get_train_test_data(self):
        # train-test split of the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, train_size=self.train_size, shuffle=True
        )
        self.X_train = X_train
        self.y_train = torch.atleast_2d(y_train)
        self.X_test = X_test
        self.y_test = torch.atleast_2d(y_test)

    def train_model(self):
        self.to(DEVICE)
        if otaf.common.is_running_in_notebook():
            tq = tqdm.tqdm_notebook
        else:
            tq = tqdm.tqdm

        try:
            # training loop
            y_test = self.y_test.to(DEVICE)
            X_test = self.X_test.to(DEVICE)
            for epoch in range(self.n_epochs):
                # Model training
                self.train(True)
                with tq(
                    self.batch_start,
                    unit="batch",
                    mininterval=0.05,
                    disable=self.display_progress_disable,
                ) as bar:
                    bar.set_description(f"Epoch {epoch:03d}")
                    for start in bar:
                        # take a batch
                        X_batch = self.X_train[start : start + self.batch_size]
                        y_batch = self.y_train[start : start + self.batch_size]
                        # Add noise if needed:
                        if self.noise_dims and self.noise_level > 0:
                            X_batch[:, self.noise_dims] = otaf.surrogate.add_gaussian_noise(
                                X_batch[:, self.noise_dims], self.noise_level
                            )

                        # backward pass
                        self.optimizer.zero_grad()
                        # forward pass
                        y_pred = self(X_batch.to(DEVICE))
                        loss = self.criterion(y_pred, y_batch.to(DEVICE))
                        loss.backward()
                        # update weights
                        self.optimizer.step()
                        # print progress
                        bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])

                        if self.clamping:
                            for p in self.parameters():
                                p.data.clamp_(-1.0, 1.0)

                # validation
                self.eval()
                y_pred = self(X_test)
                loss = float(self.criterion(y_pred, y_test))
                self.history_loss.append(loss)
                self.metric.update(y_test, y_pred)
                metric = float(self.metric.compute())
                self.metric.reset()
                self.history_metric.append(metric)

                print(f"Epoch {epoch + 1:03d}, Val Loss: {loss:.6f}, Val metric: {metric:.6f}")

                if loss < self.best_loss:
                    self.best_metric = metric
                    self.best_loss = loss
                    self.best_weights = copy.deepcopy(self.state_dict())

                if self.training_stopping_criterion(epoch):
                    break

                if self.scheduler:
                    self.scheduler.step()

        except KeyboardInterrupt:
            print("Training interrupted by user")

        print(
            f"Finished training at epoch {epoch+1} with best loss {self.best_loss:.6f} and metric of {self.best_metric:.6f}"
        )

        # restore self.model and return best accuracy
        self.load_state_dict(self.best_weights)
        torch.cuda.empty_cache()

    def plot_results(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)

        ax1.plot(self.history_loss, color="C0")
        ax1.set_xlabel("Epoch", color="C0")
        ax1.set_ylabel("Loss", color="C0")
        ax1.tick_params(axis="x", colors="C0")
        ax1.tick_params(axis="y", colors="C0")

        ax2.plot(self.history_metric, color="C1")
        ax2.yaxis.tick_right()
        ax2.set_ylabel("Metric", color="C1")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(axis="y", colors="C1")
        ax2.set_xticks([])

        plt.show()

    def save_model(self):
        torch.save(self.state_dict(), self.save_path)

    def load_model(self):
        self.load_state_dict(torch.load(self.save_path))
        self.best_weights = copy.deepcopy(self.state_dict())

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


def initialize_model_weights(model):
    classname = model.__class__.__name__
    # for every Linear layer in a self.model..
    if classname.find("Linear") != -1:
        # get the number of the inputs
        n = model.in_features
        y = 1.0 / np.sqrt(n)
        model.weight.data.uniform_(-y, y)
        model.bias.data.fill_(0)


def should_early_stop(val_loss_history, patience=10):
    # Implement your logic for early stopping
    if len(val_loss_history) > patience:
        # Check if no improvement in the last 'patience' epochs
        return all(
            x >= y for x, y in zip(val_loss_history[-patience:], val_loss_history[-patience + 1 :])
        )
    return False
