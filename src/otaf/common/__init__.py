from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"

from collections import defaultdict

from ._common import *

__all__ = _common.__all__

tree = lambda: defaultdict(tree)  # Special dictionary
