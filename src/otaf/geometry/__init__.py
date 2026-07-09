"""Geometric primitives and spatial analysis utilities for the OTAF project."""
from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"

from ._geometry import *
from ._meshes import *

__all__ = _geometry.__all__ + _meshes.__all__
