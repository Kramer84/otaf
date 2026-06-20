# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = [
    "get_variable_size_sequential_linear_model",
    "get_base_relu_mlp_model",
    "get_base_tanh_mlp_model",
    "get_custom_mlp_layers",
    "add_gaussian_noise",
]


from typing import Any, List, Optional, Type, Callable, Dict, Union
import torch
import torch.nn as nn


def get_variable_size_sequential_linear_model(
    input_dim: int,
    activation: Optional[Type[nn.Module]] = None,
    relative_max_layer_size: float = 3.0,
    relative_max_layer_depth: float = 0.25,
    relative_layer_number: float = 0.5,
    min_layer_number: int = 4,
) -> nn.Sequential:
    """
    Construct a variable-sized sequential linear neural network model.

    The network size, depth, and layer sizing profile scale dynamically based on 
    the provided input dimensions, forming a bottleneck or expansion pyramid 
    that peaks at a designated intermediate layer depth before scaling down.

    Parameters
    ----------
    input_dim : int
        The number of input features for the initial linear layer.
    activation : Optional[Type[nn.Module]], default=None
        The PyTorch activation function class to instantiate between hidden layers.
        Defaults to `nn.ReLU` if not specified.
    relative_max_layer_size : float, default=3.0
        Multiplier determining the maximum hidden layer size relative to `input_dim`.
    relative_max_layer_depth : float, default=0.25
        Fraction of the total layers at which the maximum layer size occurs.
    relative_layer_number : float, default=0.5
        Multiplier used to scale the total number of layers relative to `input_dim`.
    min_layer_number : int, default=4
        The absolute minimum number of layer sizing transitions allowed.

    Returns
    -------
    nn.Sequential
        A sequential container containing the constructed linear and activation layers.
    """
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


def get_base_relu_mlp_model(
    input_dim: int, 
    output_dim: int, 
    compile_model: bool = False
) -> Union[nn.Sequential, Any]:
    """
    Construct a baseline multi-layer perceptron with specialized progressive sizing.

    Calculate hidden layer dimensions using linear interpolations heavily biased toward
    width expansions in the initial sections before tapering down to the target dimension.

    Parameters
    ----------
    input_dim : int
        The number of input features entering the network.
    output_dim : int
        The number of output channels leaving the final linear projection.
    compile_model : bool, default=False
        Flag determining whether to apply JIT compilation optimizations via `torch.compile`.

    Returns
    -------
    Union[nn.Sequential, Any]
        The constructed sequential neural network container, wrapped via compilation if requested.
    """
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


def get_base_tanh_mlp_model(
    input_dim: int, 
    output_dim: int, 
    compile_model: bool = False
) -> Union[nn.Sequential, Any]:
    """
    Construct a baseline multi-layer perceptron utilizing Tanh activation functions.

    Calculate specific intermediate hidden layer sizes via heavily expanded linear 
    interpolations between input and output boundaries, building a smooth bottleneck topology.

    Parameters
    ----------
    input_dim : int
        The number of input dimensions entering the network.
    output_dim : int
        The number of output channels leaving the final linear projection.
    compile_model : bool, default=False
        Flag determining whether to apply JIT compilation optimizations via `torch.compile`.

    Returns
    -------
    Union[nn.Sequential, Any]
        The constructed sequential neural network container, wrapped via compilation if requested.
    """
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
    layer_sizes: List[int],
    layer_class: Callable[..., nn.Module] = nn.Linear,
    activation_class: Optional[Type[nn.Module]] = nn.ReLU,
    layer_kwargs: Optional[Dict[str, Any]] = None,
    activation_kwargs: Optional[Dict[str, Any]] = None,
    dropout_class: Optional[Type[nn.Module]] = None,
    dropout_kwargs: Optional[Dict[str, Any]] = None,
) -> List[nn.Module]:
    """
    Create a list of structural layers for building a multi-layer perceptron.

    Iterate through the specified size sequence to instantiate and chain the hidden 
    layers, activation functions, and optional regularization tracking components.

    Parameters
    ----------
    layer_sizes : List[int]
        Sequence of layer dimensions defining the structural boundaries 
        [input_dim, hidden_1, ..., hidden_n, output_dim].
    layer_class : Callable[..., nn.Module], default=nn.Linear
        The constructor or class template used to instantiate transformation blocks.
    activation_class : Optional[Type[nn.Module]], default=nn.ReLU
        The class type of the chosen non-linear activation unit. If None, drops
        activation layers from hidden transitions.
    layer_kwargs : Optional[Dict[str, Any]], default=None
        Additional keyword configurations passed directly to the `layer_class`.
    activation_kwargs : Optional[Dict[str, Any]], default=None
        Additional keyword configurations passed directly to the `activation_class`.
    dropout_class : Optional[Type[nn.Module]], default=None
        The class type of a dropout or regularization layer (e.g., `nn.Dropout`).
    dropout_kwargs : Optional[Dict[str, Any]], default=None
        Additional keyword configurations passed directly to the `dropout_class`.

    Returns
    -------
    List[nn.Module]
        An ordered collection of structural network components ready for execution.
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


def add_gaussian_noise(tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Add feature-wise Gaussian noise to an input tensor.

    Calculate the standard deviation across the feature dimension (axis 1) for 
    each sample, scale it by a distortion factor, and inject zero-mean normal 
    distribution perturbations into the original matrix.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (N, M) to be corrupted with noise.
    alpha : float
        Scaling coefficient applied to the calculated standard deviation.

    Returns
    -------
    torch.Tensor
        The corrupted tensor containing the injected Gaussian perturbations.
    """
    # Calculate the standard deviation along axis 1
    std_dev = tensor.std(dim=1, keepdim=True)

    # Generate Gaussian noise
    noise = torch.randn_like(tensor) * std_dev * alpha

    # Add noise to the original tensor
    noisy_tensor = tensor + noise

    return noisy_tensor
