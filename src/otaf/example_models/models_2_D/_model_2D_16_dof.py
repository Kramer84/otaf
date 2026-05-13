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
from otaf.tolerances import sigma_delta_3D_plane


# From the notebook 2D_Model_16DOF_Auto_GLD

def get_assembly_data(hM = 10, hF = 10.2, L1 = 30, L2 = 70, L3 = 30, lM = 10, lF = 10.2):
    # Pièce 1 (male)
    P1A0, P1A1, P1A2 = (
        np.array((L1 - lM / 2, hM / 2, 0.0)),
        np.array((L1 - lM / 2, 0, 0.0)),
        np.array((L1 - lM / 2, hM, 0.0)),
    )
    P1B0, P1B1, P1B2 = (
        np.array((L1 + lM / 2, hM / 2, 0.0)),
        np.array((L1 + lM / 2, 0, 0.0)),
        np.array((L1 + lM / 2, hM, 0.0)),
    )
    P1C0, P1C1, P1C2 = (
        np.array((L1 + L2 - lM / 2, hM / 2, 0.0)),
        np.array((L1 + L2 - lM / 2, 0, 0.0)),
        np.array((L1 + L2 - lM / 2, hM, 0.0)),
    )
    P1D0, P1D1, P1D2 = (
        np.array((L1 + L2 + lM / 2, hM / 2, 0.0)),
        np.array((L1 + L2 + lM / 2, 0, 0.0)),
        np.array((L1 + L2 + lM / 2, hM, 0.0)),
    )
    P1E0, P1E1 = np.array((0.0, 0.0, 0.0)), np.array((L1 + L2 + L3, 0.0, 0.0))

    # Pièce 2 (femelle)  # On met les points à hM et pas hF pour qu'ils soient bien oposées! (Besoin??)
    P2A0, P2A1, P2A2 = (
        np.array((L1 - lF / 2, hF / 2, 0.0)),
        np.array((L1 - lF / 2, 0, 0.0)),
        np.array((L1 - lF / 2, hF, 0.0)),
    )
    P2B0, P2B1, P2B2 = (
        np.array((L1 + lF / 2, hF / 2, 0.0)),
        np.array((L1 + lF / 2, 0, 0.0)),
        np.array((L1 + lF / 2, hF, 0.0)),
    )
    P2C0, P2C1, P2C2 = (
        np.array((L1 + L2 - lF / 2, hF / 2, 0.0)),
        np.array((L1 + L2 - lF / 2, 0, 0.0)),
        np.array((L1 + L2 - lF / 2, hF, 0.0)),
    )
    P2D0, P2D1, P2D2 = (
        np.array((L1 + L2 + lF / 2, hF / 2, 0.0)),
        np.array((L1 + L2 + lF / 2, 0, 0.0)),
        np.array((L1 + L2 + lF / 2, hF, 0.0)),
    )
    P2E0, P2E1 = np.array((0.0, 0.0, 0.0)), np.array((L1 + L2 + L3, 0.0, 0.0))

    R0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x_, y_, z_ = R0[0], R0[1], R0[2]

    # Pièce1
    RP1b = RP1d = R0
    RP1a = RP1c = np.array([-x_, -y_, z_]).T
    RP1e = np.array([y_, -x_, z_]).T

    # Pièce2
    RP2a = RP2c = R0
    RP2b = RP2d = np.array([-x_, -y_, z_]).T
    RP2e = np.array([-y_, x_, z_]).T

    TP1cC0, TP1cC1, TP1cC2 = (
        otaf.geometry.tfrt(RP1c, P1C0),
        otaf.geometry.tfrt(RP1c, P1C1),
        otaf.geometry.tfrt(RP1c, P1C2),
    )
    TP1aA0, TP1aA1, TP1aA2 = (
        otaf.geometry.tfrt(RP1a, P1A0),
        otaf.geometry.tfrt(RP1a, P1A1),
        otaf.geometry.tfrt(RP1a, P1A2),
    )
    TP1bB0, TP1bB1, TP1bB2 = (
        otaf.geometry.tfrt(RP1b, P1B0),
        otaf.geometry.tfrt(RP1b, P1B1),
        otaf.geometry.tfrt(RP1b, P1B2),
    )
    TP1dD0, TP1dD1, TP1dD2 = (
        otaf.geometry.tfrt(RP1d, P1D0),
        otaf.geometry.tfrt(RP1d, P1D1),
        otaf.geometry.tfrt(RP1d, P1D2),
    )
    TP1eE0, TP1eE1 = otaf.geometry.tfrt(RP1e, P1E0), otaf.geometry.tfrt(RP1e, P1E1)
    TP2aA0, TP2aA1, TP2aA2 = (
        otaf.geometry.tfrt(RP2a, P2A0),
        otaf.geometry.tfrt(RP2a, P2A1),
        otaf.geometry.tfrt(RP2a, P2A2),
    )
    TP2cC0, TP2cC1, TP2cC2 = (
        otaf.geometry.tfrt(RP2c, P2C0),
        otaf.geometry.tfrt(RP2c, P2C1),
        otaf.geometry.tfrt(RP2c, P2C2),
    )
    TP2bB0, TP2bB1, TP2bB2 = (
        otaf.geometry.tfrt(RP2b, P2B0),
        otaf.geometry.tfrt(RP2b, P2B1),
        otaf.geometry.tfrt(RP2b, P2B2),
    )
    TP2dD0, TP2dD1, TP2dD2 = (
        otaf.geometry.tfrt(RP2d, P2D0),
        otaf.geometry.tfrt(RP2d, P2D1),
        otaf.geometry.tfrt(RP2d, P2D2),
    )
    TP2eE0, TP2eE1 = otaf.geometry.tfrt(RP2e, P2E0), otaf.geometry.tfrt(RP2e, P2E1)

    system_data = {
        "PARTS": {
            "1": {
                "a": {
                    "FRAME": RP1a,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["NONE"],
                    "POINTS": {"A0": P1A0, "A1": P1A1, "A2": P1A2},
                    "INTERACTIONS": ["P2a"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "b": {
                    "FRAME": RP1b,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["NONE"],
                    "POINTS": {"B0": P1B0, "B1": P1B1, "B2": P1B2},
                    "INTERACTIONS": ["P2b"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "c": {
                    "FRAME": RP1c,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["NONE"],
                    "POINTS": {"C0": P1C0, "C1": P1C1, "C2": P1C2},
                    "INTERACTIONS": ["P2c"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "d": {
                    "FRAME": RP1d,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["NONE"],
                    "POINTS": {"D0": P1D0, "D1": P1D1, "D2": P1D2},
                    "INTERACTIONS": ["P2d"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "e": {
                    "FRAME": RP1e,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["PERFECT"],
                    "POINTS": {"E0": P1E0, "E1": P1E1},
                    "INTERACTIONS": ["P2e"],
                    "CONSTRAINTS_G": ["SLIDING"],
                },
            },
            "2": {
                "a": {
                    "FRAME": RP2a,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["NONE"],
                    "POINTS": {"A0": P2A0, "A1": P2A1, "A2": P2A2},
                    "INTERACTIONS": ["P1a"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "b": {
                    "FRAME": RP2b,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["NONE"],
                    "POINTS": {"B0": P2B0, "B1": P2B1, "B2": P2B2},
                    "INTERACTIONS": ["P1b"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "c": {
                    "FRAME": RP2c,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["NONE"],
                    "POINTS": {"C0": P2C0, "C1": P2C1, "C2": P2C2},
                    "INTERACTIONS": ["P1c"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "d": {
                    "FRAME": RP2d,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["NONE"],
                    "POINTS": {"D0": P2D0, "D1": P2D1, "D2": P2D2},
                    "INTERACTIONS": ["P1d"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "e": {
                    "FRAME": RP2e,
                    "TYPE": "plane",
                    "CONSTRAINTS_D": ["PERFECT"],
                    "POINTS": {"E0": P2E0, "E1": P2E1},
                    "INTERACTIONS": ["P1e"],
                    "CONSTRAINTS_G": ["SLIDING"],
                },
            },
        },
        "LOOPS": {
            "COMPATIBILITY": {
                "L0": "P1eE0 -> P2eE0 -> P2aA0 -> P1aA0",
                "L1": "P1eE0 -> P2eE0 -> P2dD0 -> P1dD0",
                "L2": "P1cC0 -> P2cC0 -> P2bB0 -> P1bB0",
                "L3": "P1bB0 -> P2bB0 -> P2aA0 -> P1aA0",
            },
        },
        "GLOBAL_CONSTRAINTS": "2D_NZ",
    }
    return system_data

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

def getSystemOfConstraintsAssemblyModel(hM = 10, hF = 10.2, L1 = 30, L2 = 70, L3 = 30, lM = 10, lF = 10.2):
    SDA = getAssemblyDataProcessorObject(get_assembly_data(hM, hF, L1, L2, L3, lM, lF))
    CLH = getCompatibilityLoopHandlingObject(SDA)
    ILH = getInterfaceLoopHandlingObject(SDA, CLH)
    compatibility_expressions = CLH.get_compatibility_expression_from_FO_matrices()
    interface_constraints = ILH.get_interface_loop_expressions()
    SOCAM = otaf.SystemOfConstraintsAssemblyModel(
        compatibility_expressions, interface_constraints)
    SOCAM.embedOptimizationVariable()
    return SOCAM


def getDistributionParams(tol=0.16, capa=1.0, hM=10, hF=10.2):
    deviation_symbols = list(sp.symbols(
        'u_d_2, gamma_d_2 u_d_3 gamma_d_3 u_d_4 gamma_d_4 u_d_5 gamma_d_5 u_d_6 gamma_d_6 u_d_7 gamma_d_7 u_d_8 gamma_d_8 u_d_9 gamma_d_9'))

    sigma_e_pos = tol / (6 * capa)
    theta_max_m = tol / hM
    theta_max_f = tol / hF

    sigma_e_theta_m = (2 * theta_max_m) / (6 * capa)
    sigma_e_theta_f = (2 * theta_max_f) / (6 * capa)

    RandDeviationVect = otaf.distribution.get_composed_normal_defect_distribution(
        defect_names=deviation_symbols,
        sigma_dict = {"gamma_d_2":sigma_e_theta_f, 
                    "gamma_d_3":sigma_e_theta_m,
                    "gamma_d_4":sigma_e_theta_f,
                    "gamma_d_5":sigma_e_theta_m,
                    "gamma_d_6":sigma_e_theta_m,
                    "gamma_d_7":sigma_e_theta_f,
                    "gamma_d_8":sigma_e_theta_f,
                    "gamma_d_9":sigma_e_theta_m,
                    "u":sigma_e_pos})
    max_std_vect = np.array([
        sigma_e_pos, sigma_e_theta_f,
        sigma_e_pos, sigma_e_theta_m,
        sigma_e_pos, sigma_e_theta_f,
        sigma_e_pos, sigma_e_theta_m,
        sigma_e_pos, sigma_e_theta_m,
        sigma_e_pos, sigma_e_theta_f,
        sigma_e_pos, sigma_e_theta_f,
        sigma_e_pos, sigma_e_theta_m
    ])
    return RandDeviationVect, deviation_symbols,  max_std_vect, np.array([0.0]*16)

dim=16
sample_multiplier = np.eye(dim)
no_tol = False

# Let's define the credal sets of admissible standard deviations
def evalCredalSetConstraints(x_std, tol=0.16, capa=1.0, hM=10, hF=10.2):
    """
    x_std is the vector of standard deviations of the defects, in the order [u_d_4, gamma_d_4, u_d_5, gamma_d_5]
    """
    target = tol / (6*capa)

    constraint1 = (sigma_delta_3D_plane(hF/2, 0, x_std[0], 0,  x_std[1]) - target)/target
    constraint2 = (sigma_delta_3D_plane(hM/2, 0, x_std[2], 0,  x_std[3]) - target)/target
    constraint3 = (sigma_delta_3D_plane(hF/2, 0, x_std[4], 0,  x_std[5]) - target)/target
    constraint4 = (sigma_delta_3D_plane(hM/2, 0, x_std[6], 0,  x_std[7]) - target)/target
    constraint5 = (sigma_delta_3D_plane(hM/2, 0, x_std[8], 0,  x_std[9]) - target)/target
    constraint6 = (sigma_delta_3D_plane(hF/2, 0, x_std[10], 0,  x_std[11]) - target)/target
    constraint7 = (sigma_delta_3D_plane(hF/2, 0, x_std[12], 0,  x_std[13]) - target)/target
    constraint8 = (sigma_delta_3D_plane(hM/2, 0, x_std[14], 0,  x_std[15]) - target)/target
    return [constraint1, constraint2, constraint3, constraint4, constraint5, constraint6, constraint7, constraint8]

def evalScaledCredalSetConstraints(x_scaled, max_std_vect, tracker=None, experiment_key=None, tol=0.16, capa=1.0, hM=10, hF=10.2):
    # Unscale back to real physical dimensions
    x_real = x_scaled * max_std_vect
    # Evaluate the aggregated manual constraints with real values
    constraint_array = evalCredalSetConstraints(x_real, tol=tol, capa=capa, hM=hM, hF=hF)
    if tracker:
        tracker.update_constraint_data(
            exp_key=experiment_key,
            x=x_scaled,
            constraints=constraint_array
        )
    return constraint_array

def getScaledCredalSetConstraintsFunction(max_std_vect, tracker=None, experiment_key=None, tol=0.16, capa=1.0, hM=10, hF=10.2):
    return lambda x_scaled : evalScaledCredalSetConstraints(x_scaled, max_std_vect, tracker, experiment_key, tol=tol, capa=capa, hM=hM, hF=hF)