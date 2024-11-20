# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = [
    "get_variable_size_sequential_linear_model",
    "get_base_relu_mlp_model",
    "get_base_tanh_mlp_model",
    "get_custom_mlp_layers",
    "add_gaussian_noise",
]


import numpy as np

import torch
import torch.nn as nn


def get_variable_size_sequential_linear_model(
    input_dim,
    activation=None,
    relative_max_layer_size=3.0,
    relative_max_layer_depth=1 / 4,
    relative_layer_number=1 / 2,
    min_layer_number=4,
):
    """Constructs a variable sized neural network models based on input dimensions."""
    if activation is None:
        activation = nn.ReLU

    ir = lambda x: int(round(x))  # Simplify rounding function
    d = input_dim

    n_layers = max(
        ir(d * relative_layer_number) + 1, min_layer_number + 1
    )  # In the sense of different layer sizes.
    m_layer_id = max(ir(n_layers * relative_max_layer_depth), 1)
    m_layer_id = min(m_layer_id, n_layers - 2)
    m_layer_size = ir(d * relative_max_layer_size)

    coef_lin_up = (m_layer_size - d) / (
        m_layer_id + 1
    )  # How much the size of model rises from input to max layer
    coef_lin_down = (d - m_layer_size) / (
        n_layers - m_layer_id
    )  # How much the size of model goes down from max layer to output

    # Calculate layer sizes
    lin_size_l = [d]
    for i in range(1, n_layers):
        if i < m_layer_id:
            size = ir(d + ((m_layer_size - d) / m_layer_id) * i)

        elif i > m_layer_id:
            size = ir(
                m_layer_size - (m_layer_size - 1) * (i - m_layer_id) / (n_layers - m_layer_id - 1)
            )
        else:
            size = m_layer_size
        lin_size_l.append(size)
    print(f"Linear layer sizes  : {lin_size_l}")

    # Create layers
    seq_args = []  # list of arguments for nn.Sequential
    for i in range(1, n_layers):
        seq_args.append(nn.Linear(lin_size_l[i - 1], lin_size_l[i]))
        if i < n_layers - 1:
            seq_args.append(activation())
    return nn.Sequential(*seq_args)


def get_base_relu_mlp_model(input_dim, output_dim, compile_model=True):
    layer_sizes = list(
        map(
            int,
            [
                input_dim,
                7 * (0.75 * input_dim + 0.25 * output_dim),
                5 * (0.5 * input_dim + 0.5 * output_dim),
                3 * (0.25 * input_dim + 0.75 * output_dim),
                output_dim,
            ],
        )
    )
    print(layer_sizes)
    layers = get_custom_mlp_layers(layer_sizes, activation_class=nn.ReLU)

    model = nn.Sequential(*layers)
    if compile_model:
        return torch.compile(model)
    else:
        return model


def get_base_tanh_mlp_model(input_dim, output_dim, compile_model=True):
    layer_sizes = list(
        map(
            int,
            [
                input_dim,
                7 * (0.75 * input_dim + 0.25 * output_dim),
                5 * (0.5 * input_dim + 0.5 * output_dim),
                3 * (0.25 * input_dim + 0.75 * output_dim),
                output_dim,
            ],
        )
    )
    layers = get_custom_mlp_layers(layer_sizes, activation_class=nn.Tanh)

    model = nn.Sequential(*layers)
    if compile_model:
        return torch.compile(model)
    else:
        return model


def get_custom_mlp_layers(
    layer_sizes,
    layer_class=nn.Linear,
    activation_class=nn.ReLU,
    layer_kwargs=None,
    activation_kwargs=None,
    dropout_class=None,
    dropout_kwargs=None,
):
    """
    Creates a list of layers for a multi-layer perceptron (MLP) with specified layer sizes, layer class, activation function, and optional dropout layers.

    Args:
        layer_sizes (list of int): List of layer sizes [input, hidden1, ..., hiddenN, output].
        layer_class (callable): Layer class to use (e.g., nn.Linear).
        activation_class (callable): Activation function class to use (e.g., nn.ReLU).
        layer_kwargs (dict): Dictionary of additional keyword arguments for the layer class.
        activation_kwargs (dict): Dictionary of additional keyword arguments for the activation function class.
        dropout_prob (float, optional): Dropout probability. If None, no dropout layers are added.

    Returns:
        list: List of layers for the MLP.
    """
    if layer_kwargs is None:
        layer_kwargs = {}
    if activation_kwargs is None:
        activation_kwargs = {}
    if dropout_kwargs is None:
        dropout_kwargs = {}

    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(layer_class(layer_sizes[i], layer_sizes[i + 1], **layer_kwargs))
        if i < len(layer_sizes) - 2:  # No activation after the last layer
            layers.append(activation_class(**activation_kwargs))
            if dropout_class is not None:
                layers.append(dropout_class(**dropout_kwargs))

    return layers


def add_gaussian_noise(tensor, alpha):
    """
    Add Gaussian noise to a tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, M).
        alpha (float): Scaling factor for the standard deviation.

    Returns:
        torch.Tensor: Tensor with added Gaussian noise.
    """
    # Calculate the standard deviation along axis 1
    std_dev = tensor.std(dim=1, keepdim=True)

    # Generate Gaussian noise
    noise = torch.randn_like(tensor) * std_dev * alpha

    # Add noise to the original tensor
    noisy_tensor = tensor + noise

    return noisy_tensor
