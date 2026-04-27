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
x  (x_full, size 30) – deviations + diameters, see x_full_labels in the Cython file
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
