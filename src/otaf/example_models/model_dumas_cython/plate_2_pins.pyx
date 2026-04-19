import cython
import numpy as np
cimport numpy as np

# =============================================================================
# 1. CORE C-LEVEL MATH FUNCTIONS (NOGIL)
# These execute at pure C speed without Python overhead.
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _compute_Cc(const double[:] x, const double[:] g, const double[:] L, double[:] Cc) nogil:
    """
    Computes the 14 equality constraints (Cc = 0).
    x is the standard base vector of 26 components (translations + rotations).
    g is the 17-component vector of optimization variables (jeux).
    """
    # Boucle 1: (1)/(2)/(3) au point A
    Cc[0] = -x[0] - g[0] + x[1] - x[2] + x[3]
    Cc[1] = -x[4] - g[1] + x[5] - x[6] + x[7]
    Cc[2] = -g[2] - g[6]
    Cc[3] = -x[14] - g[3] + x[15] - g[7]
    Cc[4] = -x[18] - g[4] + x[19] - g[8]
    Cc[5] = -g[5] - x[8] + x[9]

    # Boucle 2: (1)/(3)/(2)/(4) au point A
    Cc[6] = -x[0] - g[0] + x[1] - x[10] + g[9] + x[11]
    Cc[7] = -x[4] - g[1] + x[5] - x[12] + g[10] + x[13]
    Cc[8] = -g[2] + g[11]
    Cc[9] = -x[14] - g[3] + x[15] - x[16] + g[12] + L[1] * g[11] + x[17]
    Cc[10] = -x[18] - g[4] + x[19] - x[20] + g[13] - L[0] * g[11] + x[21]
    Cc[11] = -g[5] - L[0] * x[12] + L[1] * x[10] + g[14] + L[0] * g[10] - L[1] * g[9] + L[0] * x[13] - L[1] * x[11]

    # Boucle 3: Condition fonctionnelle au point G
    Cc[12] = -x[14] - L[8] * x[4] - g[3] + L[7] * g[2] - L[8] * g[1] + x[15] + L[8] * x[5] - x[22] + g[15] + x[23]
    Cc[13] = -x[18] + L[8] * x[0] - g[4] + L[8] * g[0] - L[6] * g[2] + x[19] - L[8] * x[1] - x[24] + g[16] + x[25]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _compute_Ci(const double[:] g, const double[:] L, double gap_b, double gap_c, double[:] Ci) nogil:
    """
    Computes the 4 interface inequality constraints (Ci <= 0).
    Using manual multiplication instead of powers for C-level optimization.
    """
    cdef double temp_1, temp_2

    Ci[0] = (g[3] * g[3]) + (g[4] * g[4]) - (gap_b * gap_b)

    temp_1 = g[3] + L[2] * g[1]  # u_3b1b + l3 * b_3b1b
    temp_2 = g[4] - L[2] * g[0]  # v_3b1b - l3 * a_3b1b
    Ci[1] = (temp_1 * temp_1) + (temp_2 * temp_2) - (gap_b * gap_b)

    Ci[2] = (g[12] * g[12]) + (g[13] * g[13]) - (gap_c * gap_c)

    temp_1 = g[12] + L[3] * g[10] # u_4c1c + l4 * b_4c1c
    temp_2 = g[13] - L[3] * g[9]  # v_4c1c - l4 * a_4c1c
    Ci[3] = (temp_1 * temp_1) + (temp_2 * temp_2) - (gap_c * gap_c)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _convert_points_to_base(const double[:] xp, const double[:] L, double[:] x_base) nogil:
    """
    Converts the 26-component vector containing physical point coordinates (top/bottom)
    into the 26-component base vector containing rotations via Equations (C.15).
    """
    cdef double denom = L[0]*L[10] - L[1]*L[9]  # l1*l11 - l2*l10

    # 1. Calculate Rotations (Eq C.15)
    # xp mapping for a1, a2: 0: w1a1_H, 1: w1a1, 2: w1a1_C | 3: w2a2_H, 4: w2a2, 5: w2a2_C
    x_base[3] = (L[0]*xp[0] + (L[9]-L[0])*xp[1] - L[9]*xp[2]) / denom        # alpha_1a1
    x_base[7] = (L[1]*xp[0] + (L[10]-L[1])*xp[1] - L[10]*xp[2]) / denom      # beta_1a1
    x_base[2] = (L[0]*xp[3] + (L[9]-L[0])*xp[4] - L[9]*xp[5]) / denom        # alpha_2a2
    x_base[6] = (L[1]*xp[3] + (L[10]-L[1])*xp[4] - L[10]*xp[5]) / denom      # beta_2a2

    # xp mapping for b1, b2: 6:u1b1_B, 7:u1b1, 8:v1b1_B, 9:v1b1 | 10:u2b2_E, 11:u2b2, 12:v2b2_E, 13:v2b2
    x_base[4] = (xp[6] - xp[7]) / L[2]     # beta_1b1
    x_base[0] = (xp[9] - xp[8]) / L[2]     # alpha_1b1
    x_base[5] = (-xp[10] + xp[11]) / L[4]  # beta_2b2
    x_base[1] = (-xp[13] + xp[12]) / L[4]  # alpha_2b2

    # xp mapping for c1, c2: 14:u1c1_D, 15:u1c1, 16:v1c1_D, 17:v1c1 | 18:u2c2_F, 19:u2c2, 20:v2c2_F, 21:v2c2
    x_base[13] = (xp[14] - xp[15]) / L[3]  # beta_1c1
    x_base[11] = (xp[17] - xp[16]) / L[3]  # alpha_1c1
    x_base[12] = (-xp[18] + xp[19]) / L[5] # beta_2c2
    x_base[10] = (-xp[21] + xp[20]) / L[5] # alpha_2c2

    # 2. Transfer standard translations directly
    x_base[9] = xp[1]   # w_1a1
    x_base[8] = xp[4]   # w_2a2
    x_base[14] = xp[7]  # u_1b1
    x_base[18] = xp[9]  # v_1b1
    x_base[15] = xp[11] # u_2b2
    x_base[19] = xp[13] # v_2b2
    x_base[17] = xp[15] # u_1c1
    x_base[21] = xp[17] # v_1c1
    x_base[16] = xp[19] # u_2c2
    x_base[20] = xp[21] # v_2c2

    # 3. Transfer remaining gap translations
    x_base[22] = xp[22] # u_2g2
    x_base[23] = xp[23] # u_1g1
    x_base[24] = xp[24] # v_2g2
    x_base[25] = xp[25] # v_1g1


# =============================================================================
# 2. SCIPY OPTIMIZER INTERFACES (PYTHON FACING)
# These allocate Numpy arrays and format outputs for scipy.optimize.minimize
# =============================================================================

def objective_function(g, d_max):
    """
    Evaluates the functional condition Cf (Objective to minimize/check).
    Returns a scalar float.
    g[15]: u_2g1g, g[16]: v_2g1g
    """
    return d_max - (g[15] + g[16])


def constraints_eq_base(g, x_base, L):
    """
    Equality constraints (Cc = 0) using the DIRECT rotations vector (x_base).
    Returns a numpy array of size 14.
    """
    cdef np.ndarray[np.float64_t, ndim=1] Cc = np.empty(14, dtype=np.float64)
    _compute_Cc(x_base, g, L, Cc)
    return Cc


def constraints_eq_points(g, x_points, L):
    """
    Equality constraints (Cc = 0) using the 2D TOP/BOTTOM HOLE points vector (x_points).
    Dynamically calculates rotations and returns a numpy array of size 14.
    """
    cdef np.ndarray[np.float64_t, ndim=1] x_base = np.empty(26, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] Cc = np.empty(14, dtype=np.float64)

    _convert_points_to_base(x_points, L, x_base)
    _compute_Cc(x_base, g, L, Cc)

    return Cc


def constraints_ineq(g, L, d_1b, d_3b, d_1c, d_4c):
    """
    Inequality constraints (Ci <= 0).
    Returns a numpy array of size 4.
    Note: For scipy.optimize, 'ineq' constraints are generally defined as C(x) >= 0.
    If you use SciPy's default SLSQP, you should return `-Ci` in your wrapper definition.
    """
    cdef np.ndarray[np.float64_t, ndim=1] Ci = np.empty(4, dtype=np.float64)
    cdef double gap_b = (d_1b - d_3b) / 2.0
    cdef double gap_c = (d_1c - d_4c) / 2.0

    _compute_Ci(g, L, gap_b, gap_c, Ci)

    return Ci
