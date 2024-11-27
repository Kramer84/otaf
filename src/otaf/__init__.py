# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__requires__ = [
    "numpy",
    "sympy",
    "openturns",
    "scipy",
    "matplotlib",
    "joblib",
    "beartype",
    "trimesh",
]

import logging as _logging

import torch

from .assembly_modeling import (
    SystemOfConstraintsAssemblyModel,
    AssemblyDataProcessor,
    CompatibilityLoopHandling,
    DeviationMatrix,
    TransformationMatrix,
    FirstOrderMatrixExpansion,
    GapMatrix,
    InterfaceLoopHandling,
)

from . import geometry
from . import constants
from . import plotting
from . import exceptions
from . import common
from . import uncertainty
from . import surrogate
from . import sensitivity
from . import optimization
from . import sampling
from . import distribution
from . import capabilities
from . import example_models

# Remove assembly_modeling from the namespace
del assembly_modeling

__all__ = [
    "SystemOfConstraintsAssemblyModel",
    "AssemblyDataProcessor",
    "CompatibilityLoopHandling",
    "DeviationMatrix",
    "TransformationMatrix",
    "FirstOrderMatrixExpansion",
    "GapMatrix",
    "InterfaceLoopHandling",
    "geometry",
    "constants",
    "plotting",
    "exceptions",
    "common",
    "uncertainty",
    "surrogate",
    "sensitivity",
    "optimization",
    "sampling",
    "distribution",
    "capabilities",
    "example_models"
]

torch._dynamo.config.suppress_errors = True

# Define a custom log record factory to inject class name into log records
_old_factory = _logging.getLogRecordFactory()


def _record_factory(*args, **kwargs):
    import inspect # Lazy import out of namespace
    record = _old_factory(*args, **kwargs)

    # Try to get the class name, if we're inside a class method
    frame = inspect.currentframe().f_back.f_back
    if "self" in frame.f_locals:
        record.class_name = frame.f_locals["self"].__class__.__name__
    else:
        # Attempt to check if it's inside a class by inspecting the stack further up
        frame = inspect.currentframe().f_back.f_back.f_back
        if "self" in frame.f_locals:
            record.class_name = frame.f_locals["self"].__class__.__name__
        else:
            record.class_name = "RootLogger"  # Default to RootLogger if class not found

    return record


# Set the new factory
_logging.setLogRecordFactory(_record_factory)

# Configure _logging to include class name in the log message
_logging.basicConfig(
    filename="otaf_tmp.log",
    filemode="w",
    level=_logging.INFO,
    format="[%(asctime)s] {%(class_name)s %(funcName)s:%(lineno)d} %(levelname)s - %(message)s",
)

_logging.info("Initializing open (mechanical) tolerance analysis framework")
