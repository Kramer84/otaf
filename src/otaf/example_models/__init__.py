# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "Kramer84"

# Option (ii): Exposes the subpackages
# This allows: otaf.example_models.models2D.model1
#         and: otaf.example_models.models3D.model3
from . import models_2_D as models2D
from . import models_3_D as models3D

# Option (i): Bubbles the modules up to the top level
# This allows: otaf.example_models.model1
#         and: otaf.example_models.model3
from .models_2_D import model1, model2
from .models_3_D import model3, model3c, model4