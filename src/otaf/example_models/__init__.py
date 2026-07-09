"""Function based representation of the examples in ``notebooks/``"""
from __future__ import annotations

__author__ = "Kramer84"
from . import models_2_D as models2D
from . import models_3_D as models3D
from .models_2_D import model1, model2
from .models_3_D import model3, model4

__all__ = [
    "models2D",
    "models3D",
    "model1",
    "model2",
    "model3",
    "model4",   
]
