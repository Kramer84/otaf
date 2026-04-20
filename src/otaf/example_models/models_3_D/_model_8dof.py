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


### Different measures of our problem
NX = 2 ## Number of holes on x axis
NY = 1 ## Number of holes on y axis
Dext = 20 ## Diameter of holes in mm
Dint = 19.8 ## Diameter of pins in mm
EH = 50 ## Distance between the hole axises
LB = 25 # Distance between border holes axis and edge.
hPlate = 30 #Height of the plates in mm
hPin = 60 #Height of the pins in mm

CIRCLE_RESOLUTION = 8 # NUmber of points to model the contour of the outer holes

# Global coordinate system
N_PARTS = NX * NY + 2 #Number of pins plus the 2 holed plates
LX = (NX - 1) * EH + 2*LB
LY = (NY - 1) * EH + 2*LB

contour_points = np.array([[0,0,0],[LX,0,0],[LX,LY,0],[0,LY,0]])

R0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
x_, y_, z_ = R0[0], R0[1], R0[2]

Frame1 = np.array([z_,y_,-x_])
Frame2 = np.array([-z_,y_,x_])

system_data = {
    "PARTS" : {
        '0' : {
            "a" : {
                "FRAME": Frame1,
                "POINTS": {'A0' : np.array([0,0,0]),
                           'A1' : np.array([LX,0,0]),
                           'A2' : np.array([LX,LY,0]),
                           'A3' : np.array([0,LY,0]),
                        },
                "TYPE": "plane",
                "INTERACTIONS": ['P1a'],
                "CONSTRAINTS_D": ["PERFECT"],
                "CONSTRAINTS_G": ["SLIDING"],
            }
        },
        '1' : {
            "a" : {
                "FRAME": Frame2,
                "POINTS": {'A0' : np.array([0,0,0]),
                           'A1' : np.array([LX,0,0]),
                           'A2' : np.array([LX,LY,0]),
                           'A3' : np.array([0,LY,0]),
                        },
                "TYPE": "plane",
                "INTERACTIONS": ['P0a'],
                "CONSTRAINTS_D": ["PERFECT"],
                "CONSTRAINTS_G": ["SLIDING"],
            }
        }
    },
    "LOOPS": {
        "COMPATIBILITY": {
        },
    },
    "GLOBAL_CONSTRAINTS": "3D",
}

alpha_gen = otaf.common.alphabet_generator()
next(alpha_gen) # skipping 'a' as it has already been used above
part_id = 2 # Start part index for pins
for i in range(NX):
    for j in range(NY):
        pcor = np.array([LB+i*EH, LB+j*EH, 0]) # Point coordinate for hole / pins
        slab = next(alpha_gen) # Surface label, same for each mating pin so its easeir to track
        # Creating pin
        system_data["PARTS"][str(part_id)] = {}
        system_data["PARTS"][str(part_id)][slab] = {
            "FRAME": Frame1, # Frame doesn't really matter, as long as x is aligned on the axis
            "ORIGIN": pcor,
            "TYPE": "cylinder",
            "RADIUS": Dint / 2,
            "EXTENT_LOCAL": {"x_max": hPin/2, "x_min": -hPin/2},
            "INTERACTIONS": [f"P0{slab}", f"P1{slab}"],
            "SURFACE_DIRECTION": "centrifugal",
            "CONSTRAINTS_D": ["PERFECT"], # No defects on the pins
            "BLOCK_ROTATIONS_G": 'x', # The pins do not rotate around their axis
            "BLOCK_TRANSLATIONS_G": 'x', # The pins do not slide along their axis
        }
        # Adding hole to part 0
        system_data["PARTS"]["0"][slab] = {
            "FRAME": Frame1,
            "ORIGIN": pcor + np.array([0,0,-hPlate/2]), # The cylinder modeling the hole feature is below the planar feature
            "TYPE": "cylinder",
            "RADIUS": Dext / 2,
            "EXTENT_LOCAL": {"x_max": hPlate/2, "x_min": -hPlate/2},
            "INTERACTIONS": [f"P{part_id}{slab}"],
            "SURFACE_DIRECTION": "centripetal",
        }
        # Adding hole to part 1
        system_data["PARTS"]["1"][slab] = {
            "FRAME": Frame2,
            "ORIGIN": pcor  + np.array([0,0,hPlate/2]), # The cylinder modeling the hole feature is above the planar
            "TYPE": "cylinder",
            "RADIUS": Dext / 2,
            "EXTENT_LOCAL": {"x_max": hPlate/2, "x_min": -hPlate/2},
            "INTERACTIONS": [f"P{part_id}{slab}"],
            "SURFACE_DIRECTION": "centripetal",
        }
        # Construct Compatibility loop
        loop_id = f"L{part_id-1}"
        formater = lambda i,l : f"P{i}{l}{l.upper()}0"
        system_data["LOOPS"]["COMPATIBILITY"][loop_id] = f"P0aA0 -> {formater(0,slab)} -> {formater(part_id,slab)} -> {formater(1,slab)} -> P1aA0"
        part_id += 1

#OLD OUTPUTS
#str(SDA.compatibility_loops_expanded)
SDA_compatibility_loops_expanded = "{'L1': 'TP0aA0bB0 -> D0b -> GP0bB0P2bB0 -> Di2b -> D2b -> GP2bB0P1bB0 -> Di1b -> TP1bB0aA0 -> D1a -> GP1aA0P0aA0 -> Di0a', 'L2': 'TP0aA0cC0 -> D0c -> GP0cC0P3cC0 -> Di3c -> D3c -> GP3cC0P1cC0 -> Di1c -> TP1cC0aA0 -> D1a -> GP1aA0P0aA0 -> Di0a'}"
#str(CLH.get_compatibility_expression_from_FO_matrices())
CLH_get_compatibility_expression_from_FO_matrices = '[-gamma_d_0 - gamma_d_2 - gamma_g_0 - gamma_g_1, beta_d_0 - beta_d_2 + beta_g_0 + beta_g_1, alpha_g_2, -25*beta_d_0 + 25*beta_d_2 - 25*beta_g_0 - 25*beta_g_1 + 25*gamma_d_0 + 25*gamma_d_2 + 25*gamma_g_0 + 25*gamma_g_1 + 30, 15*gamma_d_0 + 15*gamma_d_2 + 15*gamma_g_0 + 15*gamma_g_1 + v_d_0 - v_d_2 + v_g_0 + v_g_1 + v_g_2, -15*beta_d_0 + 15*beta_d_2 - 15*beta_g_0 - 15*beta_g_1 + w_d_0 + w_d_2 + w_g_0 + w_g_1 - w_g_2, -gamma_d_5 - gamma_d_7 - gamma_g_3 - gamma_g_4, beta_d_5 - beta_d_7 + beta_g_3 + beta_g_4, alpha_g_2, -75*beta_d_5 + 75*beta_d_7 - 75*beta_g_3 - 75*beta_g_4 + 25*gamma_d_5 + 25*gamma_d_7 + 25*gamma_g_3 + 25*gamma_g_4 + 30, 15*gamma_d_5 + 15*gamma_d_7 + 15*gamma_g_3 + 15*gamma_g_4 + v_d_5 - v_d_7 + v_g_2 + v_g_3 + v_g_4, -15*beta_d_5 + 15*beta_d_7 - 15*beta_g_3 - 15*beta_g_4 + w_d_5 + w_d_7 - w_g_2 + w_g_3 + w_g_4]'

def getAssemblyDataProcessorObject():
    SDA = otaf.AssemblyDataProcessor(system_data)
    SDA.generate_expanded_loops()
    return SDA

def getCompatibilityLoopHandlingObject(SDA=None):
    SDA = SDA if SDA is not None else getAssemblyDataProcessorObject()
    CLH = otaf.CompatibilityLoopHandling(SDA)
    #compatibility_expressions = CLH.get_compatibility_expression_from_FO_matrices()
    return CLH

def getInterfaceLoopHandlingObject(SDA=None,CLH=None):
    if not SDA or not CLH:
        SDA = getAssemblyDataProcessorObject()
        CLH = getCompatibilityLoopHandlingObject(SDA)
    ILH = otaf.InterfaceLoopHandling(SDA, CLH, circle_resolution=20)
    return ILH
