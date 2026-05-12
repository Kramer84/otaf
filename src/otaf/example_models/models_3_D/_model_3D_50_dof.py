from __future__ import annotations

import re
import copy
import numpy as np
from enum import Enum
from typing import Tuple, Sequence

import otaf
import sympy as sp

# Compiled once at module level — free at call time
_VAR_RE = re.compile(r'^([A-Za-z]+)_([A-Za-z0-9]+)$')

_PREFIX_MAP: dict[str, str] = { #For use with otaf
    'w': 'u',  'alpha': 'beta',  'beta': 'gamma',
    'u': 'v',  'v': 'w',        'gamma': 'alpha',
    'd': 'd',
}

class LinearizationStrategy(Enum):
    """Polygon strategy used to linearize the quadratic interface constraints."""
    INSCRIBED      = "inscribed"       # conservative  – polygon inside the circle
    MEAN           = "mean"            # balanced       – average of inscribed / circumscribed
    CIRCUMSCRIBED  = "circumscribed"   # optimistic     – polygon outside the circle

class SmallDispTorsor:
    def __init__(self, name, omega, v, point, point_name):
        """
        omega: 3x1 SymPy Matrix (rotation)
        v: 3x1 SymPy Matrix (translation)
        point: 3x1 SymPy Matrix (coordinates of the reduction point)
        """
        self.name = name
        self.omega = omega
        self.v = v
        self.point = point
        self.point_name = point_name

    def move_to(self, target_point, target_name):
        """
        Applies Varignon's theorem: V_target = V_current + (Target -> Current) x Omega
        """
        # Vector from target point to current point
        target_to_current = self.point - target_point
        v_new = self.v + target_to_current.cross(self.omega)
        return SmallDispTorsor(f"{self.name}_{target_name}", self.omega, v_new, target_point, target_name)

    def __add__(self, other):
        if not self.point.equals(other.point):
            raise ValueError(
                f"Cannot add torsors at different points: "
                f"'{self.point_name}' vs '{other.point_name}'. "
                f"Use .move_to() first."
            )
        return SmallDispTorsor(f"({self.name}+{other.name})",
                               self.omega + other.omega,
                               self.v + other.v,
                               self.point, self.point_name)

    def __neg__(self):
        return SmallDispTorsor(f"(-{self.name})", -self.omega, -self.v, self.point, self.point_name)

    def __sub__(self, other):
        return self + (-other)

class CylindricalInterfaceConstraint:
    def __init__(self, name, gap_torsor, d_hole, d_pin, length, Nd=8,
                 strategy=LinearizationStrategy.CIRCUMSCRIBED, z_eval_points=None):
        """
        Generates linear non-interpenetration constraints using polygon approximation.

        Nd: Number of sides of the linearizing polygon.
        strategy: LinearizationStrategy enum defining the polygon bounding method.
        z_eval_points: List of z-coordinates (relative to the torsor origin) where the
                       2D radial constraint is evaluated. If None, defaults to
                       [-length/2, length/2], evaluating at both bore faces
                       relative to the torsor origin at the hole centre.
        """
        self.name = name
        self.gap_torsor = gap_torsor
        self.d_hole = d_hole
        self.d_pin = d_pin
        self.length = length
        self.Nd = Nd
        self.strategy = strategy

        # Dumas strictly evaluates at the functional extremities.
        # If the torsor is at one end of the pin, the evaluation points are 0 and L.
        self.z_eval_points = z_eval_points if z_eval_points is not None else [-self.length/2, self.length/2]

    def generate_equations(self):
        """
        Returns a list of linear expressions. For the mechanism to assemble,
        ALL returned expressions must be <= 0.
        """
        constraints = []

        # Delta R as defined in Equation (2.28)
        delta_R = (self.d_hole - self.d_pin) / 2

        # Base angle for scaling calculations
        theta1 = 2.0 * sp.pi / self.Nd

        # Calculate the radius correction factor based on the chosen strategy
        if self.strategy == LinearizationStrategy.INSCRIBED:
            rf = sp.cos(theta1 / 2.0)
        elif self.strategy == LinearizationStrategy.MEAN:
            rf = (1.0 + sp.cos(theta1 / 2.0)) / 2.0
        else: # CIRCUMSCRIBED
            rf = 1.0

        u_center = self.gap_torsor.v[0]
        v_center = self.gap_torsor.v[1]
        alpha = self.gap_torsor.omega[0]
        beta = self.gap_torsor.omega[1]

        # Evaluate the 2D polygon constraint at each specified Z point
        for point_idx, z in enumerate(self.z_eval_points):
            # Varignon transport to the specific cross-section
            u_local = u_center + z * beta
            v_local = v_center - z * alpha

            # Generate an inequality for each facet of the polygon (Equation 2.24 generalized)
            for k in range(1, self.Nd + 1):
                theta_k = 2.0 * sp.pi * k / self.Nd
                ck = sp.cos(theta_k)
                sk = sp.sin(theta_k)

                # Linear constraint: u*cos(theta) + v*sin(theta) - (Delta_R * rf) <= 0
                expr = sp.expand(u_local * ck + v_local * sk - delta_R * rf)

                constraints.append({
                    'eval_point_index': point_idx + 1,
                    'z_offset': z,
                    'facet_k': k,
                    'expression': expr
                })

        return constraints


# ==========================================
# 2. DYNAMIC MAPPING GENERATOR
# ==========================================


def create_dynamic_mapping(
    defect_vars: Sequence,
    clearance_vars: Sequence,
) -> dict:
    """
    Extracts (prefix, suffix) pairs from sympy variable names via regex,
    assigns compact indices, and returns a substitution mapping dict.

    Naming convention expected: `<prefix>_<suffix>`, e.g. `w_1a2b`.
    Handles coordinate permutations (via PREFIX_MAP) and reversible
    gap directions (4-char suffix palindrome detection).
    """
    mapping: dict = {}
    # ── Defect variables ─────────────────────────────────────────────
    # Single pass: collect ordered-unique suffixes AND build mapping.
    defect_index: dict[str, int] = {}
    for var in defect_vars:
        m = _VAR_RE.match(var.name)
        if not m:
            continue
        prefix, suffix = m.group(1), m.group(2)
        idx = defect_index.setdefault(suffix, len(defect_index))
        mapping[var] = sp.Symbol(f"{_PREFIX_MAP.get(prefix, prefix)}_d_{idx}")
    # ── Clearance variables ───────────────────────────────────────────
    clearance_index: dict[str, int] = {}   # canonical suffix → gap index
    g_counter = 0
    for var in clearance_vars:
        m = _VAR_RE.match(var.name)
        if not m:
            continue
        prefix, suffix = m.group(1), m.group(2)
        # Check forward match first
        if suffix in clearance_index:
            c_idx, sign = clearance_index[suffix], 1
        # Check reversed 4-char pair  (e.g. "1b2b" ↔ "2b1b")
        elif len(suffix) == 4 and (rev := suffix[2:] + suffix[:2]) in clearance_index:
            c_idx, sign = clearance_index[rev], -1
        # New canonical suffix
        else:
            clearance_index[suffix] = g_counter
            c_idx, sign = g_counter, 1
            g_counter += 1
        new_sym = sp.Symbol(f"{_PREFIX_MAP.get(prefix, prefix)}_g_{c_idx}")
        mapping[var] = new_sym if sign == 1 else -new_sym
    return mapping

def extract_linear_matrices(expressions, clearance_vars, defect_vars, length_vars=None, length_subs=None):
    """
    Takes a list of linear expressions and extracts matrices A, B, and C such that:
    [A] * X_clearance + [B] * X_defect + [C] = 0

    Parameters:
    - expressions: List of SymPy linear expressions.
    - clearance_vars: List of SymPy symbols for the gap unknowns.
    - defect_vars: List of SymPy symbols for the defect parameters.
    - length_vars: (Optional) List of SymPy symbols for the mechanism lengths.
    - length_subs: (Optional) Dictionary mapping length symbols to numerical values
                   e.g., {l0: 100.0, l1: 150.0}.
    """

    # 1. Substitute lengths if numerical values are provided
    if length_vars and length_subs:
        # Ensure we only substitute the specified length variables
        valid_subs = {k: v for k, v in length_subs.items() if k in length_vars}
        expressions = [expr.subs(valid_subs) for expr in expressions]

    # 2. Extract [A] matrix (Clearances)
    # SymPy's linear_eq_to_matrix solves for A * X = b.
    # So: A * X_clearance = rhs1  <=>  A * X_clearance - rhs1 = 0
    A, rhs1 = sp.linear_eq_to_matrix(expressions, *clearance_vars)

    # 3. Extract [B] matrix (Defects)
    # B * X_defect = rhs2  <=>  B * X_defect - rhs2 = -rhs1
    B, rhs2 = sp.linear_eq_to_matrix(-rhs1, *defect_vars)

    # 4. Extract [C] matrix (Constants)
    # The leftover (-rhs2) represents the constant terms [C].
    # Therefore: [A]*X_clearance + [B]*X_defect + [C] = 0
    C = -rhs2

    # 5. Validation Check
    # If lengths were substituted, A, B, and C should ideally be free of symbols
    if length_subs:
        remaining_symbols = A.free_symbols | B.free_symbols | C.free_symbols
        if remaining_symbols:
            print(f"Warning: Matrices still contain unbound symbols: {remaining_symbols}")

    return A, B, C

# 1. Distance Definitions
l0, l1, l2, l3 = sp.symbols('l0 l1 l2 l3')

# 2. Coordinate Definitions
# Assuming B3 is the origin [0, 0, 0] based on the directional vectors
B3 = sp.Matrix([0, 0, 0])
B0 = sp.Matrix([0, l1, 0])
B1 = sp.Matrix([l0, l1, 0])
B2 = sp.Matrix([l0, 0, 0])

# Central reduction point
R = sp.Matrix([l0/2, l1/2, 0])

# Centers of the upper plate holes (A_i')
A_prime_0 = B0 + sp.Matrix([0, 0, l2/2])
A_prime_1 = B1 + sp.Matrix([0, 0, l2/2])
A_prime_2 = B2 + sp.Matrix([0, 0, l2/2])
A_prime_3 = B3 + sp.Matrix([0, 0, l2/2])

# Centers of the lower plate holes (C_i')
C_prime_0 = B0 + sp.Matrix([0, 0, -l3/2])
C_prime_1 = B1 + sp.Matrix([0, 0, -l3/2])
C_prime_2 = B2 + sp.Matrix([0, 0, -l3/2])
C_prime_3 = B3 + sp.Matrix([0, 0, -l3/2])


# 3. Variable Definitions
# Defect variables (Plates)
a1b1, b1b1, w1b1 = sp.symbols('alpha_1b1 beta_1b1 w_1b1')
a2b2, b2b2, w2b2 = sp.symbols('alpha_2b2 beta_2b2 w_2b2')

# Defect variables (Holes in plates, no defect for cylinders)
a1d1, b1d1, u1d1, v1d1 = sp.symbols('alpha_1d1 beta_1d1 u_1d1 v_1d1')
a2d2, b2d2, u2d2, v2d2 = sp.symbols('alpha_2d2 beta_2d2 u_2d2 v_2d2')

a1e1, b1e1, u1e1, v1e1 = sp.symbols('alpha_1e1 beta_1e1 u_1e1 v_1e1')
a2e2, b2e2, u2e2, v2e2 = sp.symbols('alpha_2e2 beta_2e2 u_2e2 v_2e2')

a1f1, b1f1, u1f1, v1f1 = sp.symbols('alpha_1f1 beta_1f1 u_1f1 v_1f1')
a2f2, b2f2, u2f2, v2f2 = sp.symbols('alpha_2f2 beta_2f2 u_2f2 v_2f2')

a1g1, b1g1, u1g1, v1g1 = sp.symbols('alpha_1g1 beta_1g1 u_1g1 v_1g1')
a2g2, b2g2, u2g2, v2g2 = sp.symbols('alpha_2g2 beta_2g2 u_2g2 v_2g2')

#Size defects (diameters)
# Plate 1 holes
d1d, d1e, d1f, d1g = sp.symbols('d_1d d_1e d_1f d_1g')
# Plate 2 holes
d2d, d2e, d2f, d2g = sp.symbols('d_2d d_2e d_2f d_2g')
# Pins
d3d, d4e, d5f, d6g = sp.symbols('d_3d d_4e d_5f d_6g')

# Clearance variables (Planar contact)
u1b2b, v1b2b, g1b2b = sp.symbols('u_1b2b v_1b2b gamma_1b2b')

# Clearance variables (Pins to Plates)
a3d1d, b3d1d, g3d1d, u3d1d, v3d1d, w3d1d = sp.symbols('alpha_3d1d beta_3d1d gamma_3d1d u_3d1d v_3d1d w_3d1d')
a3d2d, b3d2d, u3d2d, v3d2d = sp.symbols('alpha_3d2d beta_3d2d u_3d2d v_3d2d')

a4e1e, b4e1e, g4e1e, u4e1e, v4e1e, w4e1e = sp.symbols('alpha_4e1e beta_4e1e gamma_4e1e u_4e1e v_4e1e w_4e1e')
a4e2e, b4e2e, u4e2e, v4e2e = sp.symbols('alpha_4e2e beta_4e2e u_4e2e v_4e2e')

a5f1f, b5f1f, g5f1f, u5f1f, v5f1f, w5f1f = sp.symbols('alpha_5f1f beta_5f1f gamma_5f1f u_5f1f v_5f1f w_5f1f')
a5f2f, b5f2f, u5f2f, v5f2f = sp.symbols('alpha_5f2f beta_5f2f u_5f2f v_5f2f')

a6g1g, b6g1g, g6g1g, u6g1g, v6g1g, w6g1g = sp.symbols('alpha_6g1g beta_6g1g gamma_6g1g u_6g1g v_6g1g w_6g1g')
a6g2g, b6g2g, u6g2g, v6g2g = sp.symbols('alpha_6g2g beta_6g2g u_6g2g v_6g2g')


# 4. Defect Torsors Definitions
T_1b_1 = SmallDispTorsor("T_{1b/1}", sp.Matrix([a1b1, b1b1, 0]), sp.Matrix([0, 0, w1b1]), R, "R")
T_2b_2 = SmallDispTorsor("T_{2b/2}", sp.Matrix([a2b2, b2b2, 0]), sp.Matrix([0, 0, w2b2]), R, "R")

T_1d_1 = SmallDispTorsor("T_{1d/1}", sp.Matrix([a1d1, b1d1, 0]), sp.Matrix([u1d1, v1d1, 0]), A_prime_0, "A'_0")
T_2d_2 = SmallDispTorsor("T_{2d/2}", sp.Matrix([a2d2, b2d2, 0]), sp.Matrix([u2d2, v2d2, 0]), C_prime_0, "C'_0")

T_1e_1 = SmallDispTorsor("T_{1e/1}", sp.Matrix([a1e1, b1e1, 0]), sp.Matrix([u1e1, v1e1, 0]), A_prime_1, "A'_1")
T_2e_2 = SmallDispTorsor("T_{2e/2}", sp.Matrix([a2e2, b2e2, 0]), sp.Matrix([u2e2, v2e2, 0]), C_prime_1, "C'_1")

T_1f_1 = SmallDispTorsor("T_{1f/1}", sp.Matrix([a1f1, b1f1, 0]), sp.Matrix([u1f1, v1f1, 0]), A_prime_2, "A'_2")
T_2f_2 = SmallDispTorsor("T_{2f/2}", sp.Matrix([a2f2, b2f2, 0]), sp.Matrix([u2f2, v2f2, 0]), C_prime_2, "C'_2")

T_1g_1 = SmallDispTorsor("T_{1g/1}", sp.Matrix([a1g1, b1g1, 0]), sp.Matrix([u1g1, v1g1, 0]), A_prime_3, "A'_3")
T_2g_2 = SmallDispTorsor("T_{2g/2}", sp.Matrix([a2g2, b2g2, 0]), sp.Matrix([u2g2, v2g2, 0]), C_prime_3, "C'_3")


# 5. Clearance (Gap) Torsors Definitions
G_1b_2b = SmallDispTorsor("G_{1b/2b}", sp.Matrix([0, 0, g1b2b]), sp.Matrix([u1b2b, v1b2b, 0]), R, "R")

G_3d_1d = SmallDispTorsor("G_{3d/1d}", sp.Matrix([a3d1d, b3d1d, g3d1d]), sp.Matrix([u3d1d, v3d1d, w3d1d]), A_prime_0, "A'_0")
G_3d_2d = SmallDispTorsor("G_{3d/2d}", sp.Matrix([a3d2d, b3d2d, 0]), sp.Matrix([u3d2d, v3d2d, 0]), C_prime_0, "C'_0")

G_4e_1e = SmallDispTorsor("G_{4e/1e}", sp.Matrix([a4e1e, b4e1e, g4e1e]), sp.Matrix([u4e1e, v4e1e, w4e1e]), A_prime_1, "A'_1")
G_4e_2e = SmallDispTorsor("G_{4e/2e}", sp.Matrix([a4e2e, b4e2e, 0]), sp.Matrix([u4e2e, v4e2e, 0]), C_prime_1, "C'_1")

G_5f_1f = SmallDispTorsor("G_{5f/1f}", sp.Matrix([a5f1f, b5f1f, g5f1f]), sp.Matrix([u5f1f, v5f1f, w5f1f]), A_prime_2, "A'_2")
G_5f_2f = SmallDispTorsor("G_{5f/2f}", sp.Matrix([a5f2f, b5f2f, 0]), sp.Matrix([u5f2f, v5f2f, 0]), C_prime_2, "C'_2")

G_6g_1g = SmallDispTorsor("G_{6g/1g}", sp.Matrix([a6g1g, b6g1g, g6g1g]), sp.Matrix([u6g1g, v6g1g, w6g1g]), A_prime_3, "A'_3")
G_6g_2g = SmallDispTorsor("G_{6g/2g}", sp.Matrix([a6g2g, b6g2g, 0]), sp.Matrix([u6g2g, v6g2g, 0]), C_prime_3, "C'_3")




# (Assuming l0, l1, l2, l3, and points R, A_prime_i, C_prime_i are defined as before)
# (Assuming SmallDispTorsor and CylindricalInterfaceConstraint classes are defined as before)

# ==========================================
# 1. VARIABLE AGGREGATION
# ==========================================

# Group all defect variables into a single list
defect_vars = [
    a1b1, b1b1, w1b1, a2b2, b2b2, w2b2,  # Planar 0 - 5
    a1d1, b1d1, u1d1, v1d1, a2d2, b2d2, u2d2, v2d2, # Hole d 6-13
    a1e1, b1e1, u1e1, v1e1, a2e2, b2e2, u2e2, v2e2, # Hole e 14-21
    a1f1, b1f1, u1f1, v1f1, a2f2, b2f2, u2f2, v2f2, # Hole f 22-29
    a1g1, b1g1, u1g1, v1g1, a2g2, b2g2, u2g2, v2g2, # Hole g 30-37
    d1d, d1e, d1f, d1g, d2d, d2e, d2f, d2g, d3d, d4e, d5f, d6g # Diameters 38-49
]

# Group all clearance/gap variables into a single list
clearance_vars = [
    u1b2b, v1b2b, g1b2b, # Planar gap
    a3d1d, b3d1d, g3d1d, u3d1d, v3d1d, w3d1d, # Pin 3 floating
    a3d2d, b3d2d, u3d2d, v3d2d,               # Pin 3 seated
    a4e1e, b4e1e, g4e1e, u4e1e, v4e1e, w4e1e, # Pin 4 floating
    a4e2e, b4e2e, u4e2e, v4e2e,               # Pin 4 seated
    a5f1f, b5f1f, g5f1f, u5f1f, v5f1f, w5f1f, # Pin 5 floating
    a5f2f, b5f2f, u5f2f, v5f2f,               # Pin 5 seated
    a6g1g, b6g1g, g6g1g, u6g1g, v6g1g, w6g1g, # Pin 6 floating
    a6g2g, b6g2g, u6g2g, v6g2g                # Pin 6 seated
]

# (Assuming Torsors T_1b_1, G_1b_2b, etc., are instantiated here as before)


# ==========================================
# 2. COMPUTATION FUNCTIONS (Data Extraction)
# ==========================================

def get_kinematic_loop_expressions():
    """
    Computes the loops and returns a flat list of 24 scalar expressions,
    along with the lists of defect and clearance variables.
    """
    T_common = T_1b_1 - G_1b_2b - T_2b_2

    L1 = T_common + T_2d_2.move_to(R, "R") + G_3d_2d.move_to(R, "R") - G_3d_1d.move_to(R, "R") - T_1d_1.move_to(R, "R")
    L2 = T_common + T_2e_2.move_to(R, "R") + G_4e_2e.move_to(R, "R") - G_4e_1e.move_to(R, "R") - T_1e_1.move_to(R, "R")
    L3 = T_common + T_2f_2.move_to(R, "R") + G_5f_2f.move_to(R, "R") - G_5f_1f.move_to(R, "R") - T_1f_1.move_to(R, "R")
    L4 = T_common + T_2g_2.move_to(R, "R") + G_6g_2g.move_to(R, "R") - G_6g_1g.move_to(R, "R") - T_1g_1.move_to(R, "R")

    loops = [L1, L2, L3, L4]
    expressions = []

    # Flatten the 6 equations from each of the 4 loops
    for loop in loops:
        for i in range(3):
            expressions.append(sp.expand(loop.omega[i]))
        for i in range(3):
            expressions.append(sp.expand(loop.v[i]))

    return expressions, defect_vars, clearance_vars


def get_interface_constraint_expressions(num_points=16, strategy=LinearizationStrategy.CIRCUMSCRIBED, z_eval_points=None):
    """
    Evaluates interface constraints and returns a flat list of expressions.
    (Requires expressions <= 0 for non-interpenetration).
    """
    # Define the interface objects
    interfaces = [
        CylindricalInterfaceConstraint("1d/3d", G_3d_1d, d1d, d3d, l2, Nd=num_points, strategy=strategy, z_eval_points=z_eval_points),
        CylindricalInterfaceConstraint("1e/4e", G_4e_1e, d1e, d4e, l2, Nd=num_points, strategy=strategy, z_eval_points=z_eval_points),
        CylindricalInterfaceConstraint("1f/5f", G_5f_1f, d1f, d5f, l2, Nd=num_points, strategy=strategy, z_eval_points=z_eval_points),
        CylindricalInterfaceConstraint("1g/6g", G_6g_1g, d1g, d6g, l2, Nd=num_points, strategy=strategy, z_eval_points=z_eval_points),
        CylindricalInterfaceConstraint("2d/3d", G_3d_2d, d2d, d3d, l3, Nd=num_points, strategy=strategy, z_eval_points=z_eval_points),
        CylindricalInterfaceConstraint("2e/4e", G_4e_2e, d2e, d4e, l3, Nd=num_points, strategy=strategy, z_eval_points=z_eval_points),
        CylindricalInterfaceConstraint("2f/5f", G_5f_2f, d2f, d5f, l3, Nd=num_points, strategy=strategy, z_eval_points=z_eval_points),
        CylindricalInterfaceConstraint("2g/6g", G_6g_2g, d2g, d6g, l3, Nd=num_points, strategy=strategy, z_eval_points=z_eval_points)
    ]

    expressions = []
    for interface in interfaces:
        constraints = interface.generate_equations()
        for c in constraints:
            expressions.append(c['expression'])

    return expressions


# ==========================================
# 3. TRANSFORMATION HELPER
# ==========================================

def rename_variables(expressions, mapping_dict):
    """
    Takes a list of SymPy expressions and a dictionary mapping old symbols to new ones.
    Returns a new list of substituted expressions.

    Example mapping_dict: {a1b1: sp.Symbol('alpha_new')}
    """
    return [expr.subs(mapping_dict) for expr in expressions]


# ==========================================
# 4. PRINTING FUNCTIONS (View generation)
# ==========================================

def print_kinematic_loops(expressions):
    """
    Prints the flattened list of kinematic loop expressions.
    Assumes blocks of 6 correspond to one complete torsor loop.
    """
    axis_names = ['Rotation x', 'Rotation y', 'Rotation z',
                  'Translation x', 'Translation y', 'Translation z']

    for idx, expr in enumerate(expressions):
        loop_num = (idx // 6) + 1
        eq_num = (idx % 6) + 1

        if eq_num == 1:
            print(f"\n=== Loop {loop_num} Equations ===")

        print(f"Eq {eq_num} [{axis_names[eq_num-1]}]:\n{expr} = 0")


def print_interface_constraints(expressions):
    """
    Prints the flattened list of interface constraint expressions.
    """
    print("\n=== Interface Constraints (Eq <= 0) ===")
    for idx, expr in enumerate(expressions):
        print(f"Constraint {idx+1}:\n{expr} <= 0")


def getSystemOfConstraintsAssemblyModel(
        L=[50, 50, 30, 30],
        CIRCLE_RESOLUTION=64, 
        strategy=LinearizationStrategy.CIRCUMSCRIBED):
    assert len(L) == 4, "Length list L must contain exactly 4 elements corresponding to l0, l1, l2, l3."
    l0, l1, l2, l3 = sp.symbols('l0 l1 l2 l3')
    length_subs = {l0: L[0], l1: L[1], l2: L[2], l3: L[3]}
    loop_exprs, d_vars, c_vars = get_kinematic_loop_expressions()
    interf_exprs = get_interface_constraint_expressions(
        num_points=CIRCLE_RESOLUTION, 
        strategy=strategy, 
        z_eval_points=[-L[2]/2, L[2]/2])

    otaf_mapping = create_dynamic_mapping(defect_vars, clearance_vars)
    mapped_loop_exprs = rename_variables(loop_exprs, otaf_mapping)
    mapped_interf_exprs = rename_variables(interf_exprs, otaf_mapping)
    mapped_clearances = [otaf_mapping[var] for var in clearance_vars if var in otaf_mapping]
    mapped_defects = [otaf_mapping[var] for var in defect_vars if var in otaf_mapping]


    comp_mats = extract_linear_matrices(mapped_loop_exprs, 
                                        mapped_clearances, 
                                        mapped_defects, 
                                        [l0,l1,l2,l3], 
                                        length_subs)
    interf_mats = extract_linear_matrices(mapped_interf_exprs, 
                                         mapped_clearances,
                                        mapped_defects, 
                                        [l0,l1,l2,l3], 
                                        length_subs)
    # 1. Extract matrices as numpy arrays (ensure float dtype)
    # The coefficient matrices must remain 2D, but the constant vectors MUST be 1D
    A_eq_gap = np.array(comp_mats[0], dtype=float)
    A_eq_def = np.array(comp_mats[1], dtype=float)
    K_eq     = np.array(comp_mats[2], dtype=float).flatten()  # <--- Flatten to 1D

    A_ub_gap = np.array(interf_mats[0], dtype=float)
    A_ub_def = np.array(interf_mats[1], dtype=float)
    K_ub     = np.array(interf_mats[2], dtype=float).flatten()  # <--- Flatten to 1D

    # 2. Repack them in the exact order SOCAM expects: (Defect, Gap, Constant)
    # We multiply the inequality matrices by -1 to convert from (<= 0) to (>= 0)
    matrices_for_socam = [
        A_eq_def, 
        A_eq_gap, 
        K_eq,
        -A_ub_def,  
        -A_ub_gap,  
        -K_ub       
    ]

    SOCAM =otaf.SystemOfConstraintsAssemblyModel(matrices=matrices_for_socam)
    SOCAM.deviation_symbols = copy.deepcopy(mapped_defects)
    SOCAM.gap_symbols = copy.deepcopy(mapped_clearances)
    SOCAM.embedOptimizationVariable()
    return SOCAM


def getDistributionParams(
        tol=0.25, capa=1.0, hPlate=30.0, 
        EH=50.0, LB=25.0, Dext=20.0, Dint=19.8):
    # Defining the uncertainties on the position and orientation of the holes
    sigma_e_pos = tol / (6 * capa)
    theta_max = tol / hPlate
    sigma_e_theta = (2 * theta_max) / (6 * capa)
    # Defining the uncertainties on the position and orientation of the plates
    sigma_e_pos_plate = tol / (6 * capa)
    sigma_e_theta_plate = (2 * (tol/(EH+LB))) / (6 * capa)
    #Defining uncertaintis on the diameters.
    sigma_diam = sigma_e_pos
    otaf_mapping = create_dynamic_mapping(defect_vars, clearance_vars)
    mapped_defects = [otaf_mapping[var] for var in defect_vars if var in otaf_mapping]
    mu_vect = np.array([0.0]*38+[Dext]*8+[Dint]*4)
    std_vect = np.array([sigma_e_theta_plate, sigma_e_theta_plate, sigma_e_pos_plate]*2
                    +[sigma_e_theta, sigma_e_theta, sigma_e_pos, sigma_e_pos]*8
                    +[sigma_diam]*12)

    RandDeviationVect = otaf.distribution.get_composed_normal_defect_distribution(
    mapped_defects, mu_list = mu_vect.tolist(), sigma_list = std_vect.tolist())

    return RandDeviationVect, mapped_defects, std_vect, mu_vect 

dim=50