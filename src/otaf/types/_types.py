# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "Int32Array1D",
    "Int32Array2D",
    "Int32Array3D",
    "Float64Array1D",
    "Float64Array2D",
    "Float64Array3D",
    "Matrix",
    "Vector",
    "Int32Array4",
    "Int32Array3x3",
]

import re
from typing import Annotated
import numpy as np
from beartype import beartype
from beartype.vale import Is


Int32Array1D = Annotated[
    np.ndarray, Is[lambda array: array.ndim == 1 and np.issubdtype(array.dtype, np.int32)]
]

Int32Array2D = Annotated[
    np.ndarray, Is[lambda array: array.ndim == 2 and np.issubdtype(array.dtype, np.int32)]
]

Int32Array3D = Annotated[
    np.ndarray, Is[lambda array: array.ndim == 3 and np.issubdtype(array.dtype, np.int32)]
]

Float64Array1D = Annotated[
    np.ndarray, Is[lambda array: array.ndim == 1 and np.issubdtype(array.dtype, np.float64)]
]

Float64Array2D = Annotated[
    np.ndarray, Is[lambda array: array.ndim == 2 and np.issubdtype(array.dtype, np.float64)]
]

Float64Array3D = Annotated[
    np.ndarray, Is[lambda array: array.ndim == 3 and np.issubdtype(array.dtype, np.float64)]
]

Matrix = Annotated[
    np.ndarray, Is[lambda array: array.ndim == 2 and np.issubdtype(array.dtype, np.float64)]
]

Vector = Annotated[
    np.ndarray, Is[lambda array: array.ndim == 1 and np.issubdtype(array.dtype, np.float64)]
]

Int32Array4 = Annotated[
    np.ndarray, Is[lambda array: array.shape == (4,) and np.issubdtype(array.dtype, np.int32)]
]

Int32Array3x3 = Annotated[
    np.ndarray, Is[lambda array: array.shape == (3, 3) and np.issubdtype(array.dtype, np.int32)]
]
