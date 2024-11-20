__all__ = ["SobolKarhunenLoeveFieldSensitivityAlgorithm"]


import openturns as ot
from collections.abc import Iterable, Sequence
from collections import UserList
from copy import copy, deepcopy
from numbers import Complex, Integral, Real, Rational, Number
from math import isnan
import re


def all_same(items=None):
    # Checks if all items of a list are the same
    return all(x == items[0] for x in items)


def atLeastList(elem=None):
    if isinstance(elem, (Iterable, Sequence, list)) and not isinstance(elem, (str, bytes)):
        return list(elem)
    else:
        return [elem]


def list_(*args):
    return list(args)


def zip_(*args):
    return map(list_, *args)


def checkIfNanInSample(sample):
    return isnan(sum(sample.computeMean()))


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def noLogInNotebook(func):
    def inner(*args, **kwargs):
        if isnotebook():
            ot.Log.Show(ot.Log.NONE)
        results = func(*args, **kwargs)
        if isnotebook():
            ot.Log.Show(ot.Log.DEFAULT)
        return results

    return inner
