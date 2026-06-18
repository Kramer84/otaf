from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"

import torch
torch._dynamo.config.suppress_errors = True

from ._neural_surrogate import *
from ._base_models import *

__all__ = (
    _neural_surrogate.__all__
    + _base_models.__all__
)
