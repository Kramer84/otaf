from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__requires__ = ["numpy", "sympy", "openturns", "scipy", "matplotlib",
                "joblib", "beartype", "trimesh"]

import logging as _logging
import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING

# --- Eager core (small, foundational) ----------------------------------------
from . import constants, exceptions, common, geometry

# --- Lazy submodules ---------------------------------------------------------
_lazy_submodules = {
    "plotting": f"{__name__}.plotting",
    "sampling": f"{__name__}.sampling",
    "distribution": f"{__name__}.distribution",
    "sensitivity": f"{__name__}.sensitivity",
    "capabilities": f"{__name__}.capabilities",
    "uncertainty": f"{__name__}.uncertainty",
    "surrogate": f"{__name__}.surrogate",
    "optimization": f"{__name__}.optimization",
    "tolerances": f"{__name__}.tolerances",
    "example_models": f"{__name__}.example_models",
}

# --- Lazy re-exports from _assembly_modeling ---------------------------------
_reexports = {
    "SystemOfConstraintsAssemblyModel": "_assembly_modeling",
    "AssemblyDataProcessor": "_assembly_modeling",
    "CompatibilityLoopHandling": "_assembly_modeling",
    "DeviationMatrix": "_assembly_modeling",
    "TransformationMatrix": "_assembly_modeling",
    "FirstOrderMatrixExpansion": "_assembly_modeling",
    "GapMatrix": "_assembly_modeling",
    "InterfaceLoopHandling": "_assembly_modeling",
    "I4": "_assembly_modeling",
    "J4": "_assembly_modeling",
}

def __getattr__(name: str):
    # lazy submodule?
    target = _lazy_submodules.get(name)
    if target is not None:
        mod = _importlib.import_module(target)
        globals()[name] = mod
        return mod

    # lazy re-export?
    modname = _reexports.get(name)
    if modname is not None:
        mod = _importlib.import_module(f".{modname}", __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    # advertise lazy names so IDEs/dir() see them
    return sorted(set(list(globals().keys()) + list(_lazy_submodules) + list(_reexports)))

# For static type checkers only
if _TYPE_CHECKING:
    from . import sampling, distribution, sensitivity, capabilities
    from . import uncertainty, surrogate, optimization, tolerances, example_models
    from ._assembly_modeling import (  # type: ignore[F401]
        SystemOfConstraintsAssemblyModel, AssemblyDataProcessor, CompatibilityLoopHandling,
        DeviationMatrix, TransformationMatrix, FirstOrderMatrixExpansion,
        GapMatrix, InterfaceLoopHandling, I4, J4
    )
    # if you moved plotting to lazy:
    # from . import plotting

__all__ = [
    # re-exported classes (resolve lazily at first access)
    "SystemOfConstraintsAssemblyModel", "AssemblyDataProcessor", "CompatibilityLoopHandling",
    "DeviationMatrix", "TransformationMatrix", "FirstOrderMatrixExpansion",
    "GapMatrix", "InterfaceLoopHandling", "I4", "J4",
    # subpackages
    "geometry", "constants", "plotting", "exceptions", "common",
    "uncertainty", "surrogate", "sensitivity", "optimization",
    "sampling", "distribution", "capabilities", "example_models", "tolerances",
]


# Logging setup function
def setup_logging(filename="otaf_tmp.log", level=_logging.INFO):
    """
    Set up logging for the module.

    Parameters
    ----------
    filename : str, optional
        Name of the log file. Default is "otaf_tmp.log".
    level : int, optional
        Logging level. Default is logging.INFO.
    """
    # Define a custom log record factory to inject class name into log records
    _old_factory = _logging.getLogRecordFactory()

    def _record_factory(*args, **kwargs):
        import inspect  # Lazy import out of namespace
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
        filename=filename,
        filemode="w",
        level=level,
        format="[%(asctime)s] {%(class_name)s %(funcName)s:%(lineno)d} %(levelname)s - %(message)s",
    )

    _logging.info("Logging initialized for open (mechanical) tolerance analysis framework")


# Notify users that logging is optional
_logging.info("To enable logging, call setup_logging()")
