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

import logging
import inspect

from collections import defaultdict

import torch

from .denavitTolerancingBasicObjects import *
from .matrixPolynomeTaylorExpansion import *
from .toleranceAnalysisMatrixPreparer import *
from .systemDataAugmented import *
from .compatibilityLoopHandling import *
from .interfaceLoopHandling import *

import otaf.geometry
import otaf.constants
import otaf.plotting
import otaf.exceptions
import otaf.types
import otaf.common
import otaf.uncertainty
import otaf.surrogate
import otaf.sensitivity
import otaf.optimization
import otaf.sampling
import otaf.distribution
import otaf.capabilities


torch._dynamo.config.suppress_errors = True

tree = lambda: defaultdict(tree)  # Special dictionary

# Define a custom log record factory to inject class name into log records
old_factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)

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
logging.setLogRecordFactory(record_factory)

# Configure logging to include class name in the log message
logging.basicConfig(
    filename="otaf_tmp.log",
    filemode="w",
    level=logging.INFO,
    format="[%(asctime)s] {%(class_name)s %(funcName)s:%(lineno)d} %(levelname)s - %(message)s",
)

logging.info("Initializing open (mechanical) tolerance analysis framework")


__all__ = (
    denavitTolerancingBasicObjects.__all__
    + matrixPolynomeTaylorExpansion.__all__
    + toleranceAnalysisMatrixPreparer.__all__
    + systemDataAugmented.__all__
    + compatibilityLoopHandling.__all__
    + interfaceLoopHandling.__all__
)
