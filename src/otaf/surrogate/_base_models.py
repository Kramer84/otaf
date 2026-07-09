__author__ = "Kramer84"
__all__ = [
    "get_variable_size_sequential_linear_model",
    "get_base_relu_mlp_model",
    "get_base_tanh_mlp_model",
    "get_custom_mlp_layers",
    "add_gaussian_noise",
]
from typing import Any, Callable, Dict, List, Optional, Type, Union

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
    """Construct a variable-sized sequential linear model.

    The network size, depth, and layer sizing profile scale
    dynamically based on the provided input dimensions, forming a
    bottleneck or expansion pyramid that peaks at a designated
    intermediate layer depth before scaling down.

    Parameters
    ----------
    input_dim : int
        The number of input features for the initial linear layer.
    activation : Type[nn.Module], optional
        The PyTorch activation function class to instantiate between
        hidden layers. Defaults to ``nn.ReLU`` if not specified.
    relative_max_layer_size : float, optional
        Multiplier determining the maximum hidden layer size relative
        to `input_dim`. The default is 3.0.
    relative_max_layer_depth : float, optional
        Fraction of the total layers at which the maximum layer size
        occurs. The default is 0.25.
    relative_layer_number : float, optional
        Multiplier used to scale the total number of layers relative
        to `input_dim`. The default is 0.5.
    min_layer_number : int, optional
        The absolute minimum number of layer sizing transitions
        allowed. The default is 4.

    Returns
    -------
    nn.Sequential
        A sequential container containing the constructed linear and
        activation layers.
    """
    if activation is None:
        activation = nn.ReLU
    ir = lambda x: int(round(x))
    d = input_dim
    n_layers = max(ir(d * relative_layer_number) + 1, min_layer_number + 1)
    m_layer_id = max(ir(n_layers * relative_max_layer_depth), 1)
    m_layer_id = min(m_layer_id, n_layers - 2)
    m_layer_size = ir(d * relative_max_layer_size)
    coef_lin_up = (m_layer_size - d) / (m_layer_id + 1)
    coef_lin_down = (d - m_layer_size) / (n_layers - m_layer_id)
    lin_size_l = [d]
    for i in range(1, n_layers):
        if i < m_layer_id:
            size = ir(d + (m_layer_size - d) / m_layer_id * i)
        elif i > m_layer_id:
            size = ir(
                m_layer_size
                - (m_layer_size - 1) * (i - m_layer_id) / (n_layers - m_layer_id - 1)
            )
        else:
            size = m_layer_size
        lin_size_l.append(size)
    print(f"Linear layer sizes  : {lin_size_l}")
    seq_args = []
    for i in range(1, n_layers):
        seq_args.append(nn.Linear(lin_size_l[i - 1], lin_size_l[i]))
        if i < n_layers - 1:
            seq_args.append(activation())
    return nn.Sequential(*seq_args)


def get_base_relu_mlp_model(
    input_dim: int, output_dim: int, compile_model: bool = False
) -> Union[nn.Sequential, Any]:
    """Construct a baseline multi-layer perceptron with specialized sizing.

    Calculate hidden layer dimensions using linear interpolations
    heavily biased toward width expansions in the initial sections
    before tapering down to the target dimension.

    Parameters
    ----------
    input_dim : int
        The number of input features entering the network.
    output_dim : int
        The number of output channels leaving the final linear projection.
    compile_model : bool, optional
        Flag determining whether to apply JIT compilation optimizations
        via ``torch.compile``. The default is ``False``.

    Returns
    -------
    nn.Sequential or Any
        The constructed sequential neural network container, wrapped via
        compilation if `compile_model` is ``True``.
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
    input_dim: int, output_dim: int, compile_model: bool = False
) -> Union[nn.Sequential, Any]:
    """Construct a baseline multi-layer perceptron using Tanh activations.

    Calculate specific intermediate hidden layer sizes via heavily
    expanded linear interpolations between input and output boundaries,
    building a smooth bottleneck topology.

    Parameters
    ----------
    input_dim : int
        The number of input dimensions entering the network.
    output_dim : int
        The number of output channels leaving the final linear projection.
    compile_model : bool, optional
        Flag determining whether to apply JIT compilation optimizations
        via ``torch.compile``. The default is ``False``.

    Returns
    -------
    nn.Sequential or Any
        The constructed sequential neural network container, wrapped via
        compilation if `compile_model` is ``True``.
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
    """Create a list of structural layers for a multi-layer perceptron.

    Iterate through the specified size sequence to instantiate and chain
    the hidden layers, activation functions, and optional regularization
    tracking components.

    Parameters
    ----------
    layer_sizes : list of int
        Sequence of layer dimensions defining the structural boundaries
        ``[input_dim, hidden_1, ..., hidden_n, output_dim]``.
    layer_class : Callable, optional
        The constructor or class template used to instantiate transformation
        blocks. The default is ``nn.Linear``.
    activation_class : Type[nn.Module], optional
        The class type of the chosen non-linear activation unit. If
        ``None``, drops activation layers from hidden transitions. The
        default is ``nn.ReLU``.
    layer_kwargs : dict, optional
        Additional keyword configurations passed directly to the
        `layer_class`.
    activation_kwargs : dict, optional
        Additional keyword configurations passed directly to the
        `activation_class`.
    dropout_class : Type[nn.Module], optional
        The class type of a dropout or regularization layer (e.g.,
        ``nn.Dropout``).
    dropout_kwargs : dict, optional
        Additional keyword configurations passed directly to the
        `dropout_class`.

    Returns
    -------
    list of nn.Module
        An ordered collection of structural network components ready for
        execution.
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
        if i < len(layer_sizes) - 2:
            layers.append(activation_class(**activation_kwargs))
            if dropout_class is not None:
                layers.append(dropout_class(**dropout_kwargs))
    return layers


def add_gaussian_noise(tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """Add feature-wise Gaussian noise to an input tensor.

    Calculate the standard deviation across the feature dimension (axis 1)
    for each sample, scale it by a distortion factor, and inject zero-mean
    normal distribution perturbations into the original matrix.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape ``(N, M)`` to be corrupted with noise.
    alpha : float
        Scaling coefficient applied to the calculated standard deviation.

    Returns
    -------
    torch.Tensor
        The corrupted tensor containing the injected Gaussian
        perturbations.
    """
    std_dev = tensor.std(dim=1, keepdim=True)
    noise = torch.randn_like(tensor) * std_dev * alpha
    noisy_tensor = tensor + noise
    return noisy_tensor
