import cython
import numpy as np
cimport numpy as np

# =============================================================================
# 1. CORE C-LEVEL MATH FUNCTIONS (NOGIL)
# These execute at pure C speed without Python overhead.
# =============================================================================

"""
Parameters:
-----------
x_full : const double[:] (Size: 30) - Full parameter array (Deviations + Diameters)

    -- PLANAR FEATURES (2 planes * 3 variables = 6 variables) --
    Plane 1a (Cover base):
    x_full[0]: w_1a1      x_full[1]: alpha_1a1  x_full[2]: beta_1a1
    Plane 2a (Socle base):
    x_full[3]: w_2a2      x_full[4]: alpha_2a2  x_full[5]: beta_2a2
    -- CYLINDRICAL FEATURES (4 cylinders * 4 variables = 16 variables) --
    Cylinder 1b (Hole in cover):
    x_full[6]: u_1b1      x_full[7]: v_1b1      x_full[8]: alpha_1b1  x_full[9]: beta_1b1
    Cylinder 2b (Pin 3, modeled on socle 2):
    x_full[10]: u_2b2     x_full[11]: v_2b2     x_full[12]: alpha_2b2 x_full[13]: beta_2b2
    Cylinder 1c (Hole in cover):
    x_full[14]: u_1c1     x_full[15]: v_1c1     x_full[16]: alpha_1c1 x_full[17]: beta_1c1
    Cylinder 2c (Pin 4, modeled on socle 2):
    x_full[18]: u_2c2     x_full[19]: v_2c2     x_full[20]: alpha_2c2 x_full[21]: beta_2c2

    -- DIAMETERS (4 variables) --
    x_full[22]: d_1b (Diameter of hole 1b)
    x_full[23]: d_3b (Diameter of pin 3)
    x_full[24]: d_1c (Diameter of hole 1c)
    x_full[25]: d_4c (Diameter of pin 4)

    -- FUNCTIONAL SURFACES (4 variables)
    Plane 1g (surface cover):
    x_full[26]: u_1g1      x_full[27]: v_1g1
    Plane 2g (Functional surface cover):
    x_full[28]: u_2g2      x_full[29]: v_2g2

g : const double[:] (Size: 17) - Clearance torsor components (jeux)
    g[0] : u_1a2a, g[1] : v_1a2a, g[2] : gamma_1a2a
    g[3] : u_3b1b, g[4] : v_3b1b, g[5] : w_3b1b
    g[6] : alpha_3b1b, g[7] : beta_3b1b, g[8] : gamma_3b1b
    g[9] : u_4c1c, g[10] : v_4c1c, g[11] : w_4c1c
    g[12] : alpha_4c1c, g[13] : beta_4c1c, g[14] : gamma_4c1c
    g[15] : u_2g1g, g[16] : u_2g1g

L : const double[:] (Size: 11) - Nominal lengths (L[0] = l1 ... L[10] = l11)
d_max : double - Maximum functional displacement
"""
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

L_labels = [f"l{i}" for i in range(1, 12)]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _compute_compatibility_constraints(
    const double[:] x_full,
    const double[:] g,
    const double[:] L,
    double d_max,
    double[:] Cc,
) nogil:
    """
    Computes the equality constraints (Cc = 0)

    Outputs (Mutated in-place):
    ---------------------------
    Cc : double[:] (Size: 14) - Compatibility equations
    """
    # Loop 1: (1)/(2)/(3) written at point A
    Cc[0] = -x_full[8] - g[0] + x_full[12] - x_full[4] + x_full[1]
    Cc[1] = -x_full[9] - g[1] + x_full[13] - x_full[5] + x_full[2]
    Cc[2] = -g[2] - g[6]
    Cc[3] = -x_full[6] - g[3] + x_full[10] - g[7]
    Cc[4] = -x_full[7] - g[4] + x_full[11] - g[8]
    Cc[5] = -g[5] - x_full[3] + x_full[0]

    # Loop 2: (1)/(3)/(2)/(4) written at point A
    Cc[6] = -x_full[8] - g[0] + x_full[12] - x_full[20] + g[9] + x_full[16]
    Cc[7] = -x_full[9] - g[1] + x_full[13] - x_full[21] + g[10] + x_full[17]
    Cc[8] = -g[2] + g[11]
    Cc[9] = -x_full[6] - g[3] + x_full[10] - x_full[18] + g[12] + L[1] * g[11] + x_full[14]
    Cc[10] = -x_full[7] - g[4] + x_full[11] - x_full[19] + g[13] - L[0] * g[11] + x_full[15]
    Cc[11] = -g[5] - L[0] * x_full[21] + L[1] * x_full[20] + g[14] + L[0] * g[10] - L[1] * g[9] + L[0] * x_full[17] - L[1] * x_full[16]

    # Loop 3: Function condition (1)/(3)/(2)/Cf written at point G
    Cc[12] = -x_full[6] - L[8] * x_full[9] - g[3] + L[7] * g[2] - L[8] * g[1] + x_full[10] + L[8] * x_full[13] - x_full[28] + g[15] + x_full[26]
    Cc[13] = -x_full[7] + L[8] * x_full[8] - g[4] + L[8] * g[0] - L[6] * g[2] + x_full[11] - L[8] * x_full[12] - x_full[29] + g[16] + x_full[27]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _compute_interface_constraints(
    const double[:] x_full,
    const double[:] g,
    const double[:] L,
    double d_max,
    double[:] Cc,
    double[:] Ci
) nogil:
    """
    Computes the interface constraints (Ci <= 0).

    Outputs (Mutated in-place):
    ---------------------------
    Ci : double[:] (Size: 4)  - Interface constraints
    """

    # 1. Unpack Diameters and compute gaps
    cdef double gap_b = (x_full[22] - x_full[23]) / 2.0
    cdef double gap_c = (x_full[24] - x_full[25]) / 2.0

    # 2. Inequality Constraints (Ci)
    cdef double temp_1, temp_2

    Ci[0] = (g[3] * g[3]) + (g[4] * g[4]) - (gap_b * gap_b)

    temp_1 = g[3] + L[2] * g[1]
    temp_2 = g[4] - L[2] * g[0]
    Ci[1] = (temp_1 * temp_1) + (temp_2 * temp_2) - (gap_b * gap_b)

    Ci[2] = (g[12] * g[12]) + (g[13] * g[13]) - (gap_c * gap_c)

    temp_1 = g[12] + L[3] * g[10]
    temp_2 = g[13] - L[3] * g[9]
    Ci[3] = (temp_1 * temp_1) + (temp_2 * temp_2) - (gap_c * gap_c)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _convert_points_to_base(const double[:] xp, const double[:] L, double[:] x_full) nogil:
    """
    Converts the 30-component vector containing physical point coordinates (top/bottom)
    into the 30-component base vector containing rotations via Equations (C.15).
    """
    cdef double denom = L[0]*L[10] - L[1]*L[9]  # l1*l11 - l2*l10

    # 1. Calculate Rotations (Eq C.15)
    # Plane 1a: xp[0]=w1a1_H, xp[1]=w1a1, xp[2]=w1a1_C
    x_full[1] = (L[0]*xp[0] + (L[9]-L[0])*xp[1] - L[9]*xp[2]) / denom        # alpha_1a1
    x_full[2] = (L[1]*xp[0] + (L[10]-L[1])*xp[1] - L[10]*xp[2]) / denom      # beta_1a1

    # Plane 2a: xp[3]=w2a2_H, xp[4]=w2a2, xp[5]=w2a2_C
    x_full[4] = (L[0]*xp[3] + (L[9]-L[0])*xp[4] - L[9]*xp[5]) / denom        # alpha_2a2
    x_full[5] = (L[1]*xp[3] + (L[10]-L[1])*xp[4] - L[10]*xp[5]) / denom      # beta_2a2

    # Cyl 1b: xp[6]=u1b1_B, xp[7]=u1b1, xp[8]=v1b1_B, xp[9]=v1b1
    x_full[8] = (xp[9] - xp[8]) / L[2]     # alpha_1b1
    x_full[9] = (xp[6] - xp[7]) / L[2]     # beta_1b1

    # Cyl 2b: xp[10]=u2b2_E, xp[11]=u2b2, xp[12]=v2b2_E, xp[13]=v2b2
    x_full[12] = (-xp[13] + xp[12]) / L[4] # alpha_2b2
    x_full[13] = (-xp[10] + xp[11]) / L[4] # beta_2b2

    # Cyl 1c: xp[14]=u1c1_D, xp[15]=u1c1, xp[16]=v1c1_D, xp[17]=v1c1
    x_full[16] = (xp[17] - xp[16]) / L[3]  # alpha_1c1
    x_full[17] = (xp[14] - xp[15]) / L[3]  # beta_1c1

    # Cyl 2c: xp[18]=u2c2_F, xp[19]=u2c2, xp[20]=v2c2_F, xp[21]=v2c2
    x_full[20] = (-xp[21] + xp[20]) / L[5] # alpha_2c2
    x_full[21] = (-xp[18] + xp[19]) / L[5] # beta_2c2

    # 2. Transfer standard translations directly
    x_full[0] = xp[1]   # w_1a1
    x_full[3] = xp[4]   # w_2a2
    x_full[6] = xp[7]   # u_1b1
    x_full[7] = xp[9]   # v_1b1
    x_full[10] = xp[11] # u_2b2
    x_full[11] = xp[13] # v_2b2
    x_full[14] = xp[15] # u_1c1
    x_full[15] = xp[17] # v_1c1
    x_full[18] = xp[19] # u_2c2
    x_full[19] = xp[21] # v_2c2

    # 3. Transfer diameters and functional surfaces directly
    x_full[22] = xp[22] # d_1b
    x_full[23] = xp[23] # d_3b
    x_full[24] = xp[24] # d_1c
    x_full[25] = xp[25] # d_4c
    x_full[26] = xp[26] # u_1g1
    x_full[27] = xp[27] # v_1g1
    x_full[28] = xp[28] # u_2g2
    x_full[29] = xp[29] # v_2g2


# =============================================================================
# 2. SCIPY OPTIMIZER INTERFACES (PYTHON FACING)
# =============================================================================

def objective_function(g, d_max):
    """
    Evaluates the functional condition Cf (Objective to minimize/check).
    Returns a scalar float.
    g[15]: u_2g1g, g[16]: v_2g1g
    """
    return d_max - (g[15] + g[16])


def constraints_eq_base(g, x_full, L, d_max):
    """
    Equality constraints (Cc = 0) using the DIRECT rotations vector (x_full).
    Returns a numpy array of size 14.
    """
    cdef np.ndarray[np.float64_t, ndim=1] Cc = np.empty(14, dtype=np.float64)
    _compute_compatibility_constraints(x_full, g, L, d_max, Cc)
    return Cc


def constraints_eq_points(g, x_points, L, d_max):
    """
    Equality constraints (Cc = 0) using the 2D TOP/BOTTOM HOLE points vector (x_points).
    Dynamically calculates rotations and returns a numpy array of size 14.
    """
    cdef np.ndarray[np.float64_t, ndim=1] x_full = np.empty(30, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] Cc = np.empty(14, dtype=np.float64)

    _convert_points_to_base(x_points, L, x_full)
    _compute_compatibility_constraints(x_full, g, L, d_max, Cc)

    return Cc


def constraints_ineq(g, x_full, L, d_max):
    """
    Inequality constraints (Ci <= 0).
    Returns a numpy array of size 4.
    Note: For scipy.optimize, 'ineq' constraints are generally defined as C(x) >= 0.
    If you use SciPy's default SLSQP, you should return `-Ci` in your wrapper definition.
    """
    cdef np.ndarray[np.float64_t, ndim=1] Ci = np.empty(4, dtype=np.float64)

    # Note: Passed a dummy Cc array to match the _compute_interface_constraints
    # signature provided in your snippet, even though it isn't mutated there.
    cdef np.ndarray[np.float64_t, ndim=1] dummy_Cc = np.empty(1, dtype=np.float64)

    _compute_interface_constraints(x_full, g, L, d_max, dummy_Cc, Ci)

    return Ci
