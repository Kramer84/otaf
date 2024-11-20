# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = ["Autoencoder"]

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


########################################

########################################

#### NOT IMPORTED IN THE MODULE (__init__.py)

########################################

########################################


class Autoencoder(nn.Module):
    """General Autoencoder class.

    Parameters
    ----------
    input_dim : int
        Input shape.
    enc_dim : int
        Desired encoded shape.
    hidden_layers : list of int
        List of hidden layer sizes for the encoder (decoder will mirror these layers).
    activation : callable
        Activation function class to use (e.g., nn.ReLU).
    layer_class : callable
        Layer class to use (e.g., nn.Linear).
    layer_kwargs : dict
        Additional keyword arguments for the layer class.
    activation_kwargs : dict
        Additional keyword arguments for the activation function class.
    dropout : float
        Dropout rate for dropout layers.
    """

    def __init__(
        self,
        input_dim,
        enc_dim,
        X,
        hidden_layers=None,
        clamping=False,
        max_epochs=100,
        batch_size=100,
        compile_model=True,
        train_size=0.7,
        save_path=None,
        input_description=None,
        display_progress_disable=True,
        activation=nn.ReLU,
        layer_class=nn.Linear,
        layer_kwargs=None,
        activation_kwargs=None,
        dropout=0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.enc_dim = enc_dim
        self.hidden_layers = hidden_layers if hidden_layers is not None else [128, 64]

        self.X_raw = X
        self.X_mean = X.mean()
        self.X_std = X.std()
        self.X = (self.X_raw - self.X_mean) / self.X_std

        self.clamping = clamping
        self.compile_model = compile_model
        self.train_size = train_size

        self.save_path = save_path
        self.input_description = input_description
        self.display_progress_disable = display_progress_disable

        self.get_train_test_data()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

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
        self.best_R2 = 0
        self.best_loss = np.inf  # init to infinity
        self.best_weights = None
        self.history = []
        self.grad_loss = []  # We want to use the grad of the mse as a stopping criteria.

        self.activation = activation
        self.layer_class = layer_class
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}
        self.activation_kwargs = activation_kwargs if activation_kwargs is not None else {}
        self.dropout = dropout

        self._build_network()

    def get_train_test_data(self):
        # train-test split of the dataset
        size = self.X.shape[0]
        self.X_train = torch.tensor(self.X[..., int(size * self.train_size)], dtype=torch.float32)
        self.X_test = torch.tensor(
            self.X[..., int(size * (1 - self.train_size))], dtype=torch.float32
        )

    def _build_network(self):
        # Encoder layers
        encoder_layer_sizes = [self.input_dim] + self.hidden_layers + [self.enc_dim]
        encoder_layers = get_custom_mlp_layers(
            encoder_layer_sizes,
            self.layer_class,
            self.activation,
            self.layer_kwargs,
            self.activation_kwargs,
        )
        encoder_layers.insert(2, nn.Dropout(self.dropout))
        encoder_layers.insert(6, nn.Dropout(self.dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layer_sizes = [self.enc_dim] + self.hidden_layers[::-1] + [self.input_dim]
        decoder_layers = get_custom_mlp_layers(
            decoder_layer_sizes,
            self.layer_class,
            self.activation,
            self.layer_kwargs,
            self.activation_kwargs,
        )
        decoder_layers.insert(1, nn.BatchNorm1d(self.enc_dim))
        decoder_layers.insert(4, nn.Dropout(self.dropout))
        decoder_layers.insert(8, nn.Dropout(self.dropout))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train(self):
        self.model.to(DEVICE)
        if otaf.common.is_running_in_notebook():
            tq = tqdm.tqdm_notebook
        else:
            tq = tqdm.tqdm

        try:
            # training loop
            for epoch in range(self.n_epochs):
                # Model training
                self.model.train()
                with tq(
                    self.batch_start,
                    unit="batch",
                    mininterval=0.05,
                    disable=self.display_progress_disable,
                ) as bar:
                    bar.set_description(f"Epoch {epoch}")
                    for start in bar:
                        # take a batch
                        X_batch = self.X_train[start : start + self.batch_size]
                        y_batch = self.y_train[start : start + self.batch_size]
                        # backward pass
                        self.optimizer.zero_grad()
                        # forward pass
                        y_pred = self.forward(X_batch.to(DEVICE))
                        loss = self.criterion(y_pred, y_batch.to(DEVICE))
                        loss.backward()
                        # update weights
                        self.optimizer.step()
                        # print progress
                        bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])

                        if self.clamping:
                            for p in self.model.parameters():
                                p.data.clamp_(-1.0, 1.0)

                # Model validation
                self.model.eval()
                y_pred = self.model(self.X_test.to(DEVICE))
                loss = self.criterion(y_pred, self.y_test.to(DEVICE))
                loss_f = float(loss)
                self.history.append(loss_f)
                self.metric.update(self.y_test.to(DEVICE), y_pred)
                R2 = float(self.metric.compute())
                self.metric.reset()

                print(f"Epoch {epoch + 1}, Val Loss: {loss_f}, Val R2: {R2}")

                if loss_f < self.best_loss:
                    self.best_R2 = R2
                    self.best_loss = loss_f
                    self.best_weights = copy.deepcopy(self.model.state_dict())

                if self.scheduler:
                    self.scheduler.step()

        except KeyboardInterrupt:
            print("Training interrupted by user")

        print(
            f"Finished training at epoch {epoch+1} with best loss {round(self.best_loss,6)} and R2 of {round(np.sqrt(self.best_R2),6)} (in standard space)"
        )

        # restore self.model and return best accuracy
        self.model.load_state_dict(self.best_weights)
        torch.cuda.empty_cache()

    def train_model(self, optimizer, error, n_epochs, x):
        self.train()
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = error(output, x)
            loss.backward()
            optimizer.step()

            if epoch % int(0.1 * n_epochs) == 0:
                print(f"epoch {epoch} \t Loss: {loss.item():.4g}")

    def infer(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded
