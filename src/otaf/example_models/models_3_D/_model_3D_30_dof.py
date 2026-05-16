"""
constraint_matrix_builder.py
============================
Builds the linear constraint matrices for the assembly tolerance analysis problem,
ready to be passed to SystemOfConstraintsAssemblyModel(matrices=...).

Convention (matching SystemOfConstraintsAssemblyModel.__call__):
    Equality   : A_eq_Def @ x + A_eq_Gap @ g + K_eq  = 0      (14 rows)
    Inequality : A_ub_Def @ x + A_ub_Gap @ g + K_ub >= 0      (4·Nd rows)

Variable layout
---------------
x  (x_full, size 30) – deviations + diameters, see x_full_labels below
g  (gaps,   size 17) – clearance torsor components,  see g_labels

Linearization reference
-----------------------
Quadratic interface constraint Ci(u,v) = u² + v² − ΔR² ≤ 0
is approximated by Nd linear half-planes (polygon strategy):

    Circumscribed : u·cos θk + v·sin θk ≤ ΔR
    Inscribed     : u·cos θk + v·sin θk ≤ ΔR·cos(π/Nd)
    Mean          : u·cos θk + v·sin θk ≤ ΔR·(1 + cos(π/Nd)) / 2

with θk = 2πk/Nd,  k = 1 … Nd.
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import Tuple

import sympy as sp
import otaf
from otaf.tolerances import sigma_delta_circular_feature


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class LinearizationStrategy(Enum):
    """Polygon strategy used to linearize the quadratic interface constraints."""
    INSCRIBED      = "inscribed"       # conservative  – polygon inside the circle
    MEAN           = "mean"            # balanced       – average of inscribed / circumscribed
    CIRCUMSCRIBED  = "circumscribed"   # optimistic     – polygon outside the circle


# Convenient 6-tuple alias
MatrixBundle = Tuple[
    np.ndarray,  # A_eq_Def  (nC  × nX)
    np.ndarray,  # A_eq_Gap  (nC  × nG)
    np.ndarray,  # K_eq      (nC,)
    np.ndarray,  # A_ub_Def  (nI  × nX)
    np.ndarray,  # A_ub_Gap  (nI  × nG)
    np.ndarray,  # K_ub      (nI,)
]


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
NX = 30   # length of x_full
NG = 17   # length of g  (without the optional slack variable)
NC = 14   # number of compatibility (equality) equations


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_constraint_matrices(
    L: np.ndarray,
    Nd: int = 32,
    strategy: LinearizationStrategy = LinearizationStrategy.CIRCUMSCRIBED,
) -> MatrixBundle:
    """
    Build all six constraint matrices for the assembly tolerance LP.

    Parameters
    ----------
    L : array-like, shape (11,)
        Nominal lengths  L[0]=l1 … L[10]=l11.
    Nd : int
        Number of polygon facets per quadratic constraint (≥ 3).
        Typical values: 8 (fast), 16 (accurate), 32 (high-fidelity).
    strategy : LinearizationStrategy
        Circumscribed (default, outer bound) → feasible set ⊆ true set.
        Inscribed  (inner bound)             → feasible set ⊇ true set.
        Mean       (balanced)                → between the two.

    Returns
    -------
    (A_eq_Def, A_eq_Gap, K_eq, A_ub_Def, A_ub_Gap, K_ub)
        Ready for ``SystemOfConstraintsAssemblyModel(matrices=...)``

    Notes
    -----
    * Diameters x[22..25] appear in **A_ub_Def** (not K_ub) so that each
      sample in a Monte Carlo study automatically gets the correct gap bound.
    * K_eq and K_ub are identically zero for this problem – they are included
      for API completeness and forward compatibility.
    """
    L = np.asarray(L, dtype=float)
    if L.shape != (11,):
        raise ValueError(f"L must have shape (11,), got {L.shape}")
    if Nd < 3:
        raise ValueError(f"Nd must be >= 3, got {Nd}")

    NI = 4 * Nd   # total linearised inequality rows

    # -----------------------------------------------------------------------
    # 1.  EQUALITY CONSTRAINTS  (A_eq_Def @ x + A_eq_Gap @ g + K_eq = 0)
    # -----------------------------------------------------------------------
    A_eq_Def = np.zeros((NC, NX))
    A_eq_Gap = np.zeros((NC, NG))
    K_eq     = np.zeros(NC)

    # --- Loop 1 ---
    # Cc[0]: -x[8] - g[0] + x[12] - x[4] + x[1] = 0
    A_eq_Def[0, [1, 4, 8, 12]] = [+1, -1, -1, +1]
    A_eq_Gap[0, 0] = -1

    # Cc[1]: -x[9] - g[1] + x[13] - x[5] + x[2] = 0
    A_eq_Def[1, [2, 5, 9, 13]] = [+1, -1, -1, +1]
    A_eq_Gap[1, 1] = -1

    # Cc[2]: -g[2] - g[6] = 0
    A_eq_Gap[2, [2, 6]] = [-1, -1]

    # Cc[3]: -x[6] - g[3] + x[10] - g[7] = 0
    A_eq_Def[3, [6, 10]] = [-1, +1]
    A_eq_Gap[3, [3, 7]] = [-1, -1]

    # Cc[4]: -x[7] - g[4] + x[11] - g[8] = 0
    A_eq_Def[4, [7, 11]] = [-1, +1]
    A_eq_Gap[4, [4, 8]] = [-1, -1]

    # Cc[5]: -g[5] - x[3] + x[0] = 0
    A_eq_Def[5, [0, 3]] = [+1, -1]
    A_eq_Gap[5, 5] = -1

    # --- Loop 2 ---
    # Cc[6]: -x[8] - g[0] + x[12] - x[20] + g[9] + x[16] = 0
    A_eq_Def[6, [8, 12, 16, 20]] = [-1, +1, +1, -1]
    A_eq_Gap[6, [0, 9]] = [-1, +1]

    # Cc[7]: -x[9] - g[1] + x[13] - x[21] + g[10] + x[17] = 0
    A_eq_Def[7, [9, 13, 17, 21]] = [-1, +1, +1, -1]
    A_eq_Gap[7, [1, 10]] = [-1, +1]

    # Cc[8]: -g[2] + g[11] = 0
    A_eq_Gap[8, [2, 11]] = [-1, +1]

    # Cc[9]: -x[6] - g[3] + x[10] - x[18] + g[12] + L[1]*g[11] + x[14] = 0
    A_eq_Def[9, [6, 10, 14, 18]] = [-1, +1, +1, -1]
    A_eq_Gap[9, [3, 11, 12]] = [-1, L[1], +1]

    # Cc[10]: -x[7] - g[4] + x[11] - x[19] + g[13] - L[0]*g[11] + x[15] = 0
    A_eq_Def[10, [7, 11, 15, 19]] = [-1, +1, +1, -1]
    A_eq_Gap[10, [4, 11, 13]] = [-1, -L[0], +1]

    # Cc[11]: -g[5] - L[0]*x[21] + L[1]*x[20] + g[14] + L[0]*g[10] - L[1]*g[9]
    #         + L[0]*x[17] - L[1]*x[16] = 0
    A_eq_Def[11, [16, 17, 20, 21]] = [-L[1], +L[0], +L[1], -L[0]]
    A_eq_Gap[11, [5, 9, 10, 14]] = [-1, -L[1], +L[0], +1]

    # --- Loop 3 (functional condition) ---
    # Cc[12]: -x[6] - L[8]*x[9] - g[3] + L[7]*g[2] - L[8]*g[1]
    #         + x[10] + L[8]*x[13] - x[28] + g[15] + x[26] = 0
    A_eq_Def[12, [6, 9, 10, 13, 26, 28]] = [-1, -L[8], +1, +L[8], +1, -1]
    A_eq_Gap[12, [1, 2, 3, 15]] = [-L[8], +L[7], -1, +1]

    # Cc[13]: -x[7] + L[8]*x[8] - g[4] + L[8]*g[0] - L[6]*g[2]
    #         + x[11] - L[8]*x[12] - x[29] + g[16] + x[27] = 0
    A_eq_Def[13, [7, 8, 11, 12, 27, 29]] = [-1, +L[8], +1, -L[8], +1, -1]
    A_eq_Gap[13, [0, 2, 4, 16]] = [+L[8], -L[6], -1, +1]

    # -----------------------------------------------------------------------
    # 2.  INEQUALITY CONSTRAINTS (linearized)
    #     Original quadratic:  u² + v² ≤ ΔR²
    #     Linearised (k-th facet, negated to ≥ form):
    #         -u·cos θk  -v·sin θk  + ΔR·rf  ≥  0
    #     where rf = radius_factor (strategy-dependent).
    #
    #     ΔR = gap = (d_hole - d_pin) / 2  →  appears in A_ub_Def via x[22..25]
    # -----------------------------------------------------------------------
    A_ub_Def = np.zeros((NI, NX))
    A_ub_Gap = np.zeros((NI, NG))
    K_ub     = np.zeros(NI)

    # --- Polygon angle samples ---
    theta   = 2.0 * np.pi * np.arange(1, Nd + 1) / Nd    # θk, k=1..Nd
    cos_t   = np.cos(theta)
    sin_t   = np.sin(theta)

    # --- Radius factor (common to all k for a given strategy) ---
    theta1 = 2.0 * np.pi / Nd
    if strategy == LinearizationStrategy.INSCRIBED:
        rf = np.cos(theta1 / 2.0)
    elif strategy == LinearizationStrategy.MEAN:
        rf = (1.0 + np.cos(theta1 / 2.0)) / 2.0
    else:   # CIRCUMSCRIBED (default)
        rf = 1.0

    half_rf = 0.5 * rf   # coefficient for x[22]-x[23] or x[24]-x[25]

    for k in range(Nd):
        ck, sk = cos_t[k], sin_t[k]

        # --- Ci[0]: u = g[3],            v = g[4],   ΔR = gap_b ---
        # -ck·g[3] - sk·g[4] + half_rf·x[22] - half_rf·x[23] >= 0
        r = k
        A_ub_Def[r, [22, 23]] = [+half_rf, -half_rf]
        A_ub_Gap[r, [3, 4]]   = [-ck, -sk]

        # --- Ci[1]: u = g[3]+L[2]·g[1],  v = g[4]-L[2]·g[0],  ΔR = gap_b ---
        # Expanding the negation:
        #   L[2]·sk·g[0] - L[2]·ck·g[1] - ck·g[3] - sk·g[4]
        #   + half_rf·x[22] - half_rf·x[23] >= 0
        r = Nd + k
        A_ub_Def[r, [22, 23]]    = [+half_rf, -half_rf]
        A_ub_Gap[r, [0, 1, 3, 4]] = [+L[2]*sk, -L[2]*ck, -ck, -sk]

        # --- Ci[2]: u = g[12],           v = g[13],  ΔR = gap_c ---
        # -ck·g[12] - sk·g[13] + half_rf·x[24] - half_rf·x[25] >= 0
        r = 2*Nd + k
        A_ub_Def[r, [24, 25]] = [+half_rf, -half_rf]
        A_ub_Gap[r, [12, 13]] = [-ck, -sk]

        # --- Ci[3]: u = g[12]+L[3]·g[10], v = g[13]-L[3]·g[9], ΔR = gap_c ---
        # L[3]·sk·g[9] - L[3]·ck·g[10] - ck·g[12] - sk·g[13]
        # + half_rf·x[24] - half_rf·x[25] >= 0
        r = 3*Nd + k
        A_ub_Def[r, [24, 25]]       = [+half_rf, -half_rf]
        A_ub_Gap[r, [9, 10, 12, 13]] = [+L[3]*sk, -L[3]*ck, -ck, -sk]

    return A_eq_Def, A_eq_Gap, K_eq, A_ub_Def, A_ub_Gap, K_ub


x_full_labels = [
    # Planar Features
    "w_1a1", "alpha_1a1", "beta_1a1",      # Plane 1a
    "w_2a2", "alpha_2a2", "beta_2a2",      # Plane 2a

    # Cylindrical Features
    "u_1b1", "v_1b1", "alpha_1b1", "beta_1b1", # Cylinder 1b
    "u_2b2", "v_2b2", "alpha_2b2", "beta_2b2", # Cylinder 2b
    "u_1c1", "v_1c1", "alpha_1c1", "beta_1c1", # Cylinder 1c
    "u_2c2", "v_2c2", "alpha_2c2", "beta_2c2", # Cylinder 2c

    # Diameters
    "d_1b", "d_3b", "d_1c", "d_4c",

    # Functional Surfaces
    "u_1g1", "v_1g1",                      # Plane 1g
    "u_2g2", "v_2g2"                       # Plane 2g
]

g_labels = [
    "u_1a2a", "v_1a2a", "gamma_1a2a",      # Plane-to-plane clearance
    "u_3b1b", "v_3b1b", "w_3b1b",          # Pin 3 to Hole 1b clearance
    "alpha_3b1b", "beta_3b1b", "gamma_3b1b",
    "u_4c1c", "v_4c1c", "w_4c1c",          # Pin 4 to Hole 1c clearance
    "alpha_4c1c", "beta_4c1c", "gamma_4c1c",
    "u_2g1g_1", "u_2g1g_2"                 # Functional surface clearances
]

# We rempa to the naming convention used in otaf.
x_full_labels_mapping = {
"w_1a1" : "u_d_0", "alpha_1a1" : "beta_d_0", "beta_1a1" : "gamma_d_0",
"w_2a2" : "u_d_1", "alpha_2a2" : "beta_d_1", "beta_2a2" : "gamma_d_1",
"u_1b1" : "v_d_2", "v_1b1" : "w_d_2", "alpha_1b1" : "beta_d_2", "beta_1b1" : "gamma_d_2",
"u_2b2" : "v_d_3", "v_2b2" : "w_d_3", "alpha_2b2" : "beta_d_3", "beta_2b2" : "gamma_d_3",
"u_1c1" : "v_d_4", "v_1c1" : "w_d_4", "alpha_1c1" : "beta_d_4", "beta_1c1" : "gamma_d_4",
"u_2c2" : "v_d_5", "v_2c2" : "w_d_5", "alpha_2c2" : "beta_d_5", "beta_2c2" : "gamma_d_5",
"d_1b" : "d_d_2", "d_3b" : "d_d_6", "d_1c" : "d_d_4", "d_4c" : "d_d_7",
"u_1g1" : "v_d_8", "v_1g1" : "w_d_8", "u_2g2" : "beta_d_8", "v_2g2" : "gamma_d_8"
}

g_labels_mapping = {
"u_1a2a":"v_g_0", "v_1a2a":"w_g_0", "gamma_1a2a":"alpha_g_0",
"u_3b1b":"u_g_1", "v_3b1b":"v_g_1", "w_3b1b":"w_g_1",
"alpha_3b1b":"alpha_g_1", "beta_3b1b":"beta_g_1", "gamma_3b1b":"gamma_g_1",
"u_4c1c":"u_g_2", "v_4c1c":"v_g_2", "w_4c1c":"w_g_2",
"alpha_4c1c":"alpha_g_2", "beta_4c1c":"beta_g_2", "gamma_4c1c":"gamma_g_2",
"u_2g1g_1":"u_g_3",
"u_2g1g_2":"u_g_4"
}

x_mp_labels = [
    # Plane 1a points (z-displacements)
    "w_1a1", "w_1a1_C", "w_1a1_H",
    # Plane 2a points (z-displacements)
    "w_2a2", "w_2a2_C", "w_2a2_H",

    # Cylinder 1b points (base and top radial displacements)
    "u_1b1", "v_1b1", "u_1b1_B", "v_1b1_B",
    # Cylinder 2b points (base and bottom radial displacements)
    "u_2b2", "v_2b2", "u_2b2_E", "v_2b2_E",

    # Cylinder 1c points (base and top radial displacements)
    "u_1c1", "v_1c1", "u_1c1_D", "v_1c1_D",
    # Cylinder 2c points (base and bottom radial displacements)
    "u_2c2", "v_2c2", "u_2c2_F", "v_2c2_F",

    # Diameters (Intrinsic)
    "d_1b", "d_3b", "d_1c", "d_4c",

    # Functional surfaces (translations)
    "u_1g1", "v_1g1", "u_2g2", "v_2g2"
]

def get_mp_to_xfull_transformation_matrix(L=[100, 40, 30, 30, 20, 20, 120, 50, 40, 50, -30]):
    """
    Returns a 30x30 matrix T such that: X_full = T @ X_mp
    L must be a list or array of lengths l1 through l11, mapped to indices 0 to 10.
    """
        
    T = np.zeros((30, 30))
    
    # Precompute common denominator for planes
    D = L[0] * L[10] - L[1] * L[9]  # l1*l11 - l2*l10
    
    # ---------------------------------------------------------
    # Planes 1a and 2a
    # ---------------------------------------------------------
    # w_1a1
    T[0, 0] = 1.0
    # alpha_1a1
    T[1, 0] = (L[9] - L[0]) / D   # w_1a1
    T[1, 1] = -L[9] / D           # w_1a1_C
    T[1, 2] = L[0] / D            # w_1a1_H
    # beta_1a1
    T[2, 0] = (L[10] - L[1]) / D  # w_1a1
    T[2, 1] = -L[10] / D          # w_1a1_C
    T[2, 2] = L[1] / D            # w_1a1_H

    # w_2a2
    T[3, 3] = 1.0
    # alpha_2a2
    T[4, 3] = (L[9] - L[0]) / D   # w_2a2
    T[4, 4] = -L[9] / D           # w_2a2_C
    T[4, 5] = L[0] / D            # w_2a2_H
    # beta_2a2
    T[5, 3] = (L[10] - L[1]) / D  # w_2a2
    T[5, 4] = -L[10] / D          # w_2a2_C
    T[5, 5] = L[1] / D            # w_2a2_H

    # ---------------------------------------------------------
    # Cylinders 1b and 2b
    # ---------------------------------------------------------
    # u_1b1, v_1b1
    T[6, 6] = 1.0
    T[7, 7] = 1.0
    # alpha_1b1 = (v_1b1 - v_1b1_B) / l3
    T[8, 7] = 1.0 / L[2]
    T[8, 9] = -1.0 / L[2]
    # beta_1b1 = (u_1b1_B - u_1b1) / l3
    T[9, 8] = 1.0 / L[2]
    T[9, 6] = -1.0 / L[2]

    # u_2b2, v_2b2
    T[10, 10] = 1.0
    T[11, 11] = 1.0
    # alpha_2b2 = (-v_2b2 + v_2b2_E) / l5
    T[12, 11] = -1.0 / L[4]
    T[12, 13] = 1.0 / L[4]
    # beta_2b2 = (-u_2b2_E + u_2b2) / l5
    T[13, 10] = 1.0 / L[4]
    T[13, 12] = -1.0 / L[4]

    # ---------------------------------------------------------
    # Cylinders 1c and 2c
    # ---------------------------------------------------------
    # u_1c1, v_1c1
    T[14, 14] = 1.0
    T[15, 15] = 1.0
    # alpha_1c1 = (v_1c1 - v_1c1_D) / l4
    T[16, 15] = 1.0 / L[3]
    T[16, 17] = -1.0 / L[3]
    # beta_1c1 = (u_1c1_D - u_1c1) / l4
    T[17, 16] = 1.0 / L[3]
    T[17, 14] = -1.0 / L[3]

    # u_2c2, v_2c2
    T[18, 18] = 1.0
    T[19, 19] = 1.0
    # alpha_2c2 = (-v_2c2 + v_2c2_F) / l6
    T[20, 19] = -1.0 / L[5]
    T[20, 21] = 1.0 / L[5]
    # beta_2c2 = (-u_2c2_F + u_2c2) / l6
    T[21, 18] = 1.0 / L[5]
    T[21, 20] = -1.0 / L[5]

    # ---------------------------------------------------------
    # Diameters and Functional Surfaces (Direct Passthrough)
    # ---------------------------------------------------------
    for i in range(22, 30):
        T[i, i] = 1.0

    return T


def getSystemOfConstraintsAssemblyModel(L = [100, 40, 30, 30, 20, 20, 120, 50, 40, 50, -30], Nd=64, strategy=LinearizationStrategy.CIRCUMSCRIBED):
    mats = build_constraint_matrices(L, Nd, strategy)
    SOCAM = otaf.SystemOfConstraintsAssemblyModel(matrices=list(mats))
    d_labels = [sp.Symbol(x_full_labels_mapping[lab]) for lab in x_full_labels]
    g_labels_loc = [sp.Symbol(g_labels_mapping[lab]) for lab in g_labels]
    SOCAM.deviation_symbols = d_labels
    SOCAM.gap_symbols = g_labels_loc
    SOCAM.embedOptimizationVariable()
    return SOCAM

def getDistributionParams(tol=None, capa=None, param_set=1):
    """
    Returns the normal distributions directly for the measured points.
    param_set matches the values in Table C.2.
    """
    # Table C.2 Values
    if param_set == 1:
        mu_d_ext, sigma_d_ext = 20.0, 0.06
        mu_d_int, sigma_d_int = 19.8, 0.06
        mu_trans, sigma_trans = 0.0, 0.01
    elif param_set == 2:
        mu_d_ext, sigma_d_ext = 20.0, 0.03
        mu_d_int, sigma_d_int = 19.8, 0.03
        mu_trans, sigma_trans = 0.0, 0.01
    else: # param_set == 3
        mu_d_ext, sigma_d_ext = 20.0, 0.02
        mu_d_int, sigma_d_int = 19.8, 0.02
        mu_trans, sigma_trans = 0.0, 0.01

    # Indices 0 to 21 are all translational measured points
    mu_list = [mu_trans] * 22
    sigma_list = [sigma_trans] * 22

    # Indices 22 to 25 are the intrinsic diameters (1b, 3b, 1c, 4c)
    mu_list.extend([mu_d_ext, mu_d_int, mu_d_ext, mu_d_int])
    sigma_list.extend([sigma_d_ext, sigma_d_int, sigma_d_ext, sigma_d_int])

    # Indices 26 to 29 are the functional translations
    mu_list.extend([mu_trans] * 4)
    sigma_list.extend([sigma_trans] * 4)

    # Convert to arrays
    mu_arr = np.array(mu_list)
    sigma_arr = np.array(sigma_list)

    RandDeviationVect = otaf.distribution.get_composed_normal_defect_distribution(
        defect_names=x_mp_labels,
        mu_list=mu_list,
        sigma_list=sigma_list
    )
    
    return RandDeviationVect, x_mp_labels, sigma_arr, mu_arr

dim=30
sample_multiplier = get_mp_to_xfull_transformation_matrix()
no_tol = True

# Let's define the credal sets of admissible standard deviations
def evalCredalSetConstraints(x_std, tol=None, capa=None, param_set=1):
    """
    x_std is the vector of standard deviations of the defects, in the order:
    [0, 1, 2]        w_1a1 w_1a1_C w_1a1_H
    [3, 4, 5]        w_2a2 w_2a2_C w_2a2_H
    [6, 7, 8, 9]     u_1b1 v_1b1   u_1b1_B v_1b1_B
    [10, 11, 12, 13] u_2b2 v_2b2   u_2b2_E v_2b2_E
    [14, 15, 16, 17] u_1c1 v_1c1   u_1c1_D v_1c1_D
    [18, 19, 20, 21] u_2c2 v_2c2   u_2c2_F v_2c2_F
    [22, 23, 24, 25] d_1b  d_3b    d_1c    d_4c
    [26, 27, 28, 29] u_1g1 v_1g1   u_2g2   v_2g2
    """
    # Table C.2 Values
    if param_set == 1:
        mu_d_ext, sigma_d_ext = 20.0, 0.06
        mu_d_int, sigma_d_int = 19.8, 0.06
        mu_trans, sigma_trans = 0.0, 0.01
    elif param_set == 2:
        mu_d_ext, sigma_d_ext = 20.0, 0.03
        mu_d_int, sigma_d_int = 19.8, 0.03
        mu_trans, sigma_trans = 0.0, 0.01
    else: # param_set == 3
        mu_d_ext, sigma_d_ext = 20.0, 0.02
        mu_d_int, sigma_d_int = 19.8, 0.02
        mu_trans, sigma_trans = 0.0, 0.01
        
    target0 = sigma_trans
    target1 = sigma_delta_circular_feature(0, sigma_d_ext/2, sigma_trans, sigma_trans)

    # Dividing by 2 cause std of radius is expected.
    # --- Helper function to streamline circular feature evaluation ---
    def eval_circ(d_idx, u_base, v_base, u_top, v_top):
        devs = [
            sigma_delta_circular_feature(0,       x_std[d_idx]/2, x_std[u_base], x_std[v_base]),
            sigma_delta_circular_feature(np.pi/2, x_std[d_idx]/2, x_std[u_base], x_std[v_base]),
            sigma_delta_circular_feature(0,       x_std[d_idx]/2, x_std[u_top],  x_std[v_top]),
            sigma_delta_circular_feature(np.pi/2, x_std[d_idx]/2, x_std[u_top],  x_std[v_top])
        ]
        return (np.max(devs) - target1) / target1

    # --- Constraints Evaluation ---
    
    # Planar and general translation features (using np.max for array slices)
    constraint1 = (np.max(x_std[0:3]) - target0) / target0    # Part 1 planar a1
    constraint2 = (np.max(x_std[3:6]) - target0) / target0    # Part 2 planar a2
    constraint7 = (np.max(x_std[26:28]) - target0) / target0  # Translation features g1
    constraint8 = (np.max(x_std[28:30]) - target0) / target0  # Translation features g2


    # Circular Features: eval_circ(diam, u_base, v_base, u_top, v_top)
    constraint3 = eval_circ(22, 6, 7, 8, 9)       # Part 1 hole b (1b1)
    constraint4 = eval_circ(23, 10, 11, 12, 13)   # Part 2 pin b (2b2)
    constraint5 = eval_circ(24, 14, 15, 16, 17)   # Part 1 hole c (1c1)
    constraint6 = eval_circ(25, 18, 19, 20, 21)   # Part 2 pin c (2c2)

    return np.array([
        constraint1, 
        constraint2, 
        constraint3, 
        constraint4, 
        constraint5, 
        constraint6, 
        constraint7,
        constraint8
    ])

def evalScaledCredalSetConstraints(x_scaled, max_std_vect, tracker=None, experiment_key=None, tol=None, capa=None, param_set=1):
    # Unscale back to real physical dimensions
    x_real = x_scaled * max_std_vect
    # Evaluate the aggregated manual constraints with real values
    constraint_array = evalCredalSetConstraints(x_real, tol=tol, capa=capa, param_set=param_set)
    if tracker:
        tracker.update_constraint_data(
            exp_key=experiment_key,
            x=x_scaled,
            constraints=constraint_array
        )
    return constraint_array

def getScaledCredalSetConstraintsFunction(max_std_vect, tracker=None, experiment_key=None, tol=None, capa=None, param_set=1):
    return lambda x_scaled : evalScaledCredalSetConstraints(x_scaled, max_std_vect, tracker, experiment_key, tol=tol, capa=capa,  param_set=param_set)