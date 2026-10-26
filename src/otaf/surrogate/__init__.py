"""Tools for the construction of neural network surrogate models, including base architectures and utility functions."""

from __future__ import annotations

__author__ = "Kramer84"
import torch

torch._dynamo.config.suppress_errors = True
from ._base_models import (
    add_gaussian_noise,
    get_base_relu_mlp_model,
    get_base_tanh_mlp_model,
    get_custom_mlp_layers,
    get_variable_size_sequential_linear_model,
)
from ._neural_surrogate import NeuralRegressorNetwork, initialize_model_weights

__all__ = [
    "NeuralRegressorNetwork",
    "initialize_model_weights",
    "get_variable_size_sequential_linear_model",
    "get_base_relu_mlp_model",
    "get_base_tanh_mlp_model",
    "get_custom_mlp_layers",
    "add_gaussian_noise",
]
