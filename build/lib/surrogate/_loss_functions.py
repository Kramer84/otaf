# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = ["LimitSpaceFocusedLoss", "PositiveLimitSpaceFocusedLoss"]


import os
import logging
import re
import copy

from time import time

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import tqdm
import openturns as ot

from sklearn.model_selection import train_test_split
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import R2Score

import otaf


class LimitSpaceFocusedLoss(nn.Module):
    """
    Custom loss function which amplifies the penalty based on the sign disagreement between the model's output and the target.

    Attributes:
        a (float): A small constant added for numerical stability in computing the exponential function.
        b (float): Scaling factor that adjusts the steepness of the exponential function.
        square (bool): If True, the squared difference is used; if False, the absolute difference is used.

    Args:
        a (float): Small constant to avoid division by zero. Default: 0.04.
        b (float): Scaling factor for the exponential part of the loss. Default: 3.0.
        square (bool): Indicator of whether to use square error or absolute error. Default: True.
    """

    def __init__(self, a=0.04, b=3.0, square=True):
        # If square is false use abs function istead of square.
        super().__init__()
        self.a = a
        self.b = b
        self.square = square

    def forward(self, output, target):
        """Calculate the forward pass of the loss function.

        Args:
            output (torch.Tensor): The predictions from the model.
            target (torch.Tensor): The ground truth values.

        Returns:
            torch.Tensor: The computed mean loss.
        """
        if self.square:
            mse_mae = torch.square
        else:
            mse_mae = torch.abs
        outtar = output * target
        sign_diff_loss = 1 / torch.exp(
            self.b * (torch.exp(outtar / (torch.abs(outtar) + self.a)) - 1)
        )
        MSE_torsor = mse_mae(output - target) * (1 + sign_diff_loss)
        return MSE_torsor.mean()


class PositiveLimitSpaceFocusedLoss(nn.Module):
    """
    Extends LimitSpaceFocusedLoss by specifically penalizing positive discrepancies between model output and target more heavily.

    Attributes:
        a (float): A small constant added for numerical stability in computing the exponential function.
        b (float): Scaling factor that adjusts the steepness of the exponential function.
        c (int): Factor to amplify the loss when the target is positive.
        square (bool): If True, the squared difference is used; if False, the absolute difference is used.

    Args:
        a (float): Small constant to avoid division by zero. Default: 0.04.
        b (float): Scaling factor for the exponential part of the loss. Default: 3.0.
        c (int): Amplification factor for positive target values. Default: 4.
        square (bool): Indicator of whether to use square error or absolute error. Default: True.
    """

    def __init__(self, a=0.04, b=3.0, c=4, square=True):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.square = square

    def forward(self, output, target):
        """Calculate the forward pass of the loss function, with added penalties for positive discrepancies.

        Args:
            output (torch.Tensor): The predictions from the model.
            target (torch.Tensor): The ground truth values.

        Returns:
            torch.Tensor: The computed mean loss.
        """
        if self.square:
            mse_mae = torch.square
        else:
            mse_mae = torch.abs
        diff = output - target
        diffsquared_postive_penalized = torch.where(
            target >= 0, self.c * mse_mae(diff), mse_mae(diff)
        )
        prod = output * target
        f_prod = prod / (torch.abs(prod) + self.a)
        g_f_prod = self.b * (torch.exp(f_prod) - 1)
        MSE_torsor = diffsquared_postive_penalized * (1 + 1 / torch.exp(g_f_prod))
        return MSE_torsor.mean()
