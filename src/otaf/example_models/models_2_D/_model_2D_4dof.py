from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"

import copy
import logging
import numpy as np
import sympy as sp
import openturns as ot
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional

import otaf

# From the notebook 2D_Model_4DOF_Auto_GLD


def get_assembly_data(X1=99.8, X2=100.0, X3=10.0):
    # Global coordinate system
    R0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x_, y_, z_ = R0[0], R0[1], R0[2]
    # Important points
    # Part 1 (male)
    P1A0, P1A1, P1A2 = (
        np.array((0, X3 / 2, 0.0)),
        np.array((0, X3, 0.0)),
        np.array((0, 0, 0.0)),
    )
    P1B0, P1B1, P1B2 = (
        np.array((X1, X3 / 2, 0.0)),
        np.array((X1, X3, 0.0)),
        np.array((X1, 0, 0.0)),
    )
    P1C0, P1C1, P1C2 = (
        np.array((X1 / 2, 0, 0.0)),
        np.array((0, 0, 0.0)),
        np.array((X1, 0, 0.0)),
    )
    # Part 2 (female)
    P2A0, P2A1, P2A2 = (
        np.array((0, X3 / 2, 0.0)),
        np.array((0, X3, 0.0)),
        np.array((0, 0, 0.0)),
    )
    P2B0, P2B1, P2B2 = (
        np.array((X2, X3 / 2, 0.0)),
        np.array((X2, X3, 0.0)),
        np.array((X2, 0, 0.0)),
    )
    P2C0, P2C1, P2C2 = (
        np.array((X2 / 2, 0, 0.0)),
        np.array((0, 0, 0.0)),
        np.array((X2, 0, 0.0)),
    )

    # Local coordinate systems
    # Part1
    RP1a = np.array([-1 * x_, -1 * y_, z_])
    RP1b = R0
    RP1c = np.array([-y_, x_, z_])

    # Part2
    RP2a = R0
    RP2b = np.array([-1 * x_, -1 * y_, z_])
    RP2c = np.array([y_, -1 * x_, z_])

    # High level representation of the assembly:

    system_data = {
        "PARTS" : {
            '1' : {
                "a" : {
                    "FRAME": RP1a,
                    "POINTS": {'A0' : P1A0, 'A1' : P1A1, 'A2' : P1A2},
                    "TYPE": "plane",
                    "INTERACTIONS": ['P2a'],
                    "CONSTRAINTS_D": ["PERFECT"], # In this modelization, only defects on the right side
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "b" : {
                    "FRAME": RP1b,
                    "POINTS": {'B0' : P1B0, 'B1' : P1B1, 'B2' : P1B2},
                    "TYPE": "plane",
                    "INTERACTIONS": ['P2b'],
                    "CONSTRAINTS_D": ["NONE"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "c" : {
                    "FRAME": RP1c,
                    "POINTS": {'C0' : P1C0, 'C1' : P1C1, 'C2' : P1C2},
                    "TYPE": "plane",
                    "INTERACTIONS": ['P2c'],
                    "CONSTRAINTS_D": ["PERFECT"],
                    "CONSTRAINTS_G": ["SLIDING"],
                },
            },
            '2' : {
                "a" : {
                    "FRAME": RP2a,
                    "POINTS": {'A0' : P2A0, 'A1' : P2A1, 'A2' : P2A2},
                    "TYPE": "plane",
                    "INTERACTIONS": ['P1a'],
                    "CONSTRAINTS_D": ["PERFECT"], # In this modelization, only defects on the right side
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "b" : {
                    "FRAME": RP2b,
                    "POINTS": {'B0' : P2B0, 'B1' : P2B1, 'B2' : P2B2},
                    "TYPE": "plane",
                    "INTERACTIONS": ['P1b'],
                    "CONSTRAINTS_D": ["NONE"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "c" : {
                    "FRAME": RP2c,
                    "POINTS": {'C0' : P2C0, 'C1' : P2C1, 'C2' : P2C2},
                    "TYPE": "plane",
                    "INTERACTIONS": ['P1c'],
                    "CONSTRAINTS_D": ["PERFECT"],
                    "CONSTRAINTS_G": ["SLIDING"],
                },
            }
        },
        "LOOPS": {
            "COMPATIBILITY": {
                "L0": "P1cC0 -> P2cC0 -> P2aA0 -> P1aA0",
                "L1": "P1cC0 -> P2cC0 -> P2bB0 -> P1bB0",
            },
        },
        "GLOBAL_CONSTRAINTS": "2D_NZ",
    }
    return system_data

#OLD OUTPUTS
#str(SDA.compatibility_loops_expanded)
SDA_compatibility_loops_expanded = "{'L0': 'D1c -> GP1cC0P2cC0 -> Di2c -> TP2cC0aA0 -> D2a -> GP2aA0P1aA0 -> Di1a -> TP1aA0cC0', 'L1': 'D1c -> GP1cC0P2cC0 -> Di2c -> TP2cC0bB0 -> D2b -> GP2bB0P1bB0 -> Di1b -> TP1bB0cC0'}"
#str(CLH.get_compatibility_expression_from_FO_matrices())
CLH_get_compatibility_expression_from_FO_matrices = '[-gamma_g_1, 499*gamma_g_1/10 + v_g_1, -5*gamma_g_1 - u_g_1 + v_g_0, -gamma_d_4 + gamma_d_5 - gamma_g_2, -499*gamma_d_4/10 + 499*gamma_d_5/10 - 499*gamma_g_2/10 - v_g_2, -5*gamma_d_4 + 5*gamma_d_5 - 5*gamma_g_2 + u_d_4 + u_d_5 + u_g_2 + v_g_0 - 1/5]'

def getAssemblyDataProcessorObject(system_data=None):
    SDA = otaf.AssemblyDataProcessor(system_data)
    SDA.generate_expanded_loops()
    return SDA

def getCompatibilityLoopHandlingObject(SDA=None):
    SDA = SDA if SDA is not None else getAssemblyDataProcessorObject(get_assembly_data())
    CLH = otaf.CompatibilityLoopHandling(SDA)
    #compatibility_expressions = CLH.get_compatibility_expression_from_FO_matrices()
    return CLH

def getInterfaceLoopHandlingObject(SDA=None,CLH=None):
    if not SDA or not CLH:
        SDA = getAssemblyDataProcessorObject(get_assembly_data())
        CLH = getCompatibilityLoopHandlingObject(SDA)
    ILH = otaf.InterfaceLoopHandling(SDA, CLH, circle_resolution=20)
    return ILH

def getSystemOfConstraintsAssemblyModel(X1=99.8, X2=100.0, X3=10.0):
    SDA = getAssemblyDataProcessorObject(get_assembly_data(X1, X2, X3))
    CLH = getCompatibilityLoopHandlingObject(SDA)
    ILH = getInterfaceLoopHandlingObject(SDA, CLH)
    compatibility_expressions = CLH.get_compatibility_expression_from_FO_matrices()
    interface_constraints = ILH.get_interface_loop_expressions()
    SOCAM = otaf.SystemOfConstraintsAssemblyModel(
        compatibility_expressions, interface_constraints)
    SOCAM.embedOptimizationVariable()
    return SOCAM


def getDistributionParams(tol=0.28, capa=1.0, X3=10.0):
    deviation_symbols = sp.symbols('u_d_4 gamma_d_4 u_d_5 gamma_d_5')
    sigma_translation = tol / (6*capa)
    rotation_max = tol / X3
    sigma_rotation = (2*rotation_max) / (6*capa)
    RandDeviationVect = otaf.distribution.get_composed_normal_defect_distribution(
        defect_names=deviation_symbols,
        sigma_dict = {"alpha":sigma_rotation, 
                    "beta":sigma_rotation,
                    "gamma":sigma_rotation, 
                    "u":sigma_translation, 
                    "v":sigma_translation, 
                    "w":sigma_translation,},
        mu_dict= {"alpha":0.0, 
                "beta":0.0,
                "gamma":0.0, 
                "u":0.0, 
                "v":0.0, 
                "w":0.0})
    max_std_vect = np.array([sigma_translation, sigma_rotation, sigma_translation, sigma_rotation])
    return RandDeviationVect, deviation_symbols, max_std_vect, np.array([0.0]*4)