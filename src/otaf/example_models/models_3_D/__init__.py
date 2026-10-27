"""3D reference models: ``model3`` (30 DOF), ``model4`` (50 DOF) and ``model4_auto``."""
from __future__ import annotations

__author__ = "Kramer84"
from otaf.geometry import LinearizationStrategy

from . import _model_3D_30_dof as model3
from . import _model_3D_50_dof as model4
from . import _model_3D_pins_auto as model4_auto

__all__ = [
    "LinearizationStrategy",
    "model3",
    "model4",
    "model4_auto",
]
