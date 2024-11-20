# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = ["NeuralRegressorNetwork", "initialize_model_weights"]

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

import otaf

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class NeuralRegressorNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        X,
        y,
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
        input_normalization=True,  # If off the inputs are first normalized and then the mean reaplied to keep the mean information within it.
        output_normalization=True,  # If off the mean is re-added after normalization.
        noise_dims=[],  # If populated it generated noise for the selected dimensions during training.
        noise_level=0.1,
    ):
        super().__init__()

        self.register_buffer("input_dim", torch.tensor(input_dim, dtype=int, requires_grad=False))
        self.register_buffer("output_dim", torch.tensor(output_dim, dtype=int, requires_grad=False))

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

        if input_normalization:
            self.X = (torch.tensor(self.X_raw, dtype=torch.float32) - self.X_mean) / self.X_std
        else:
            self.X = torch.tensor(self.X_raw, dtype=torch.float32)

        if output_normalization:
            self.y = (torch.tensor(self.y_raw, dtype=torch.float32) - self.y_mean) / self.y_std
        else:
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

        self.register_buffer(
            "input_normalization",
            torch.tensor(input_normalization, dtype=bool, requires_grad=False),
        )
        self.register_buffer(
            "output_normalization",
            torch.tensor(output_normalization, dtype=bool, requires_grad=False),
        )

        self.get_train_test_data()

        # Initialize the base model, output of d 1
        self.model = otaf.surrogate.get_base_relu_mlp_model(input_dim, output_dim)

        # Performance metric
        self.metric = R2Score()

        # loss function and optimizer
        self.loss_fn = torch.nn.MSELoss()
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

        if self.input_normalization:
            x = (x - self.X_mean.cpu()) / self.X_std.cpu()

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
                    # Convert the output to the original space and move it to CPU if necessary
                    if self.output_normalization:
                        output = output * self.y_std + self.y_mean

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
                    if self.output_normalization:
                        output = self(batch) * y_std + y_mean
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

    def forward(self, x):
        logits = self.model(x)
        return logits

    def train_model(self):
        self.to(DEVICE)
        self.model.to(DEVICE)
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
                        loss = self.loss_fn(y_pred, y_batch.to(DEVICE))
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
                loss = self.loss_fn(y_pred, y_test)
                loss_f = float(loss)
                self.history_loss.append(loss_f)
                self.metric.update(y_test, y_pred)
                R2 = float(self.metric.compute())
                self.history_metric.append(R2)
                self.metric.reset()

                print(f"Epoch {epoch + 1:03d}, Val Loss: {loss_f:.6f}, Val R2: {R2:.6f}")

                if loss_f < self.best_loss:
                    self.best_metric = R2
                    self.best_loss = loss_f
                    self.best_weights = copy.deepcopy(self.state_dict())

                if self.training_stopping_criterion(epoch):
                    break

                if self.scheduler:
                    self.scheduler.step()

        except KeyboardInterrupt:
            print("Training interrupted by user")

        print(
            f"Finished training at epoch {epoch+1} with best loss {self.best_loss:.6f} and R2 of {self.best_metric:.6f}"
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

    def get_model_as_openturns_function(self, batch_size=50000):
        func = lambda x: np.array(
            self.evaluate_model_non_standard_space(x, batch_size).detach().numpy()
        )
        otFunc = ot.PythonFunction(
            self.input_dim,
            self.output_dim,
            # func=func,
            func_sample=func,
            gradient=self.gradient,
            hessian=self.hessian,
        )
        otFunc.setName("OpenTURNS Python function linking to AI surrogate.")
        if self.input_description:
            otFunc.setInputDescription(self.input_description)
        return otFunc

    def gradient(self, inP):
        inP = torch.tensor(np.array(inP), dtype=torch.float32, requires_grad=True)
        y = self.evaluate_model_non_standard_space(inP)
        return np.array(jacobian(y, inP))

    def hessian(self, inP):
        inP = torch.tensor(np.array(inP), dtype=torch.float32, requires_grad=True)
        y = self.evaluate_model_non_standard_space(inP)
        return np.array(hessian(y, inP))

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


def initialize_model_weights(model, init_type="xavier_uniform", init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
    model (torch.nn.Module): PyTorch model.
    init_type (str): The name of an initialization method: 'normal', 'xavier_uniform', 'xavier_normal',
                     'kaiming_uniform', 'kaiming_normal', 'orthogonal', 'uniform'.
    init_gain (float): Scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight.data, a=0, mode="fan_in", nonlinearity="relu")
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in", nonlinearity="relu")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == "uniform":
                nn.init.uniform_(m.weight.data, -init_gain, init_gain)
            else:
                raise NotImplementedError(f"Initialization method [{init_type}] is not implemented")

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


## Custom jacobian and hessian in pytorch #########################################################


def jacobian(y, x, create_graph=False):
    """
    Compute the Jacobian matrix of y with respect to x.

    Args:
        y (torch.Tensor): Output tensor.
        x (torch.Tensor): Input tensor whose gradients are required.
        create_graph (bool): If True, constructs the graph during the computation, allowing for higher order derivatives.

    Returns:
        torch.Tensor: The Jacobian matrix.
    """
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.0
        (grad_x,) = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph
        )
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.0
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    """
    Compute the Hessian matrix of y with respect to x.

    Args:
        y (torch.Tensor): Output tensor.
        x (torch.Tensor): Input tensor whose second order gradients are required.

    Returns:
        torch.Tensor: The Hessian matrix.
    """
    return jacobian(jacobian(y, x, create_graph=True), x)
