from __future__ import annotations

import copy
import re
from enum import Enum
from typing import Sequence

import numpy as np
import sympy as sp

import otaf
from otaf.tolerances import sigma_delta_3D_plane, sigma_delta_cylindrical_feature

_VAR_RE = re.compile("^([A-Za-z]+)_([A-Za-z0-9]+)$")
_PREFIX_MAP: dict[str, str] = {
    "w": "u",
    "alpha": "beta",
    "beta": "gamma",
    "u": "v",
    "v": "w",
    "gamma": "alpha",
    "d": "d",
}


class LinearizationStrategy(Enum):
    INSCRIBED = "inscribed"
    MEAN = "mean"
    CIRCUMSCRIBED = "circumscribed"


class SmallDispTorsor:
    def __init__(self, name, omega, v, point, point_name):
        self.name = name
        self.omega = omega
        self.v = v
        self.point = point
        self.point_name = point_name

    def move_to(self, target_point, target_name):
        target_to_current = self.point - target_point
        v_new = self.v + target_to_current.cross(self.omega)
        return SmallDispTorsor(
            f"{self.name}_{target_name}", self.omega, v_new, target_point, target_name
        )

    def __add__(self, other):
        if not self.point.equals(other.point):
            raise ValueError(
                f"Cannot add torsors at different points: '{self.point_name}' vs '{other.point_name}'. Use .move_to() first."
            )
        return SmallDispTorsor(
            f"({self.name}+{other.name})",
            self.omega + other.omega,
            self.v + other.v,
            self.point,
            self.point_name,
        )

    def __neg__(self):
        return SmallDispTorsor(
            f"(-{self.name})", -self.omega, -self.v, self.point, self.point_name
        )

    def __sub__(self, other):
        return self + -other


class CylindricalInterfaceConstraint:
    def __init__(
        self,
        name,
        gap_torsor,
        d_hole,
        d_pin,
        length,
        Nd=8,
        strategy=LinearizationStrategy.CIRCUMSCRIBED,
        z_eval_points=None,
    ):
        self.name = name
        self.gap_torsor = gap_torsor
        self.d_hole = d_hole
        self.d_pin = d_pin
        self.length = length
        self.Nd = Nd
        self.strategy = strategy
        self.z_eval_points = (
            z_eval_points
            if z_eval_points is not None
            else [-self.length / 2, self.length / 2]
        )

    def generate_equations(self):
        constraints = []
        delta_R = (self.d_hole - self.d_pin) / 2
        theta1 = 2.0 * sp.pi / self.Nd
        if self.strategy == LinearizationStrategy.INSCRIBED:
            rf = sp.cos(theta1 / 2.0)
        elif self.strategy == LinearizationStrategy.MEAN:
            rf = (1.0 + sp.cos(theta1 / 2.0)) / 2.0
        else:
            rf = 1.0
        u_center = self.gap_torsor.v[0]
        v_center = self.gap_torsor.v[1]
        alpha = self.gap_torsor.omega[0]
        beta = self.gap_torsor.omega[1]
        for point_idx, z in enumerate(self.z_eval_points):
            u_local = u_center + z * beta
            v_local = v_center - z * alpha
            for k in range(1, self.Nd + 1):
                theta_k = 2.0 * sp.pi * k / self.Nd
                ck = sp.cos(theta_k)
                sk = sp.sin(theta_k)
                expr = sp.expand(u_local * ck + v_local * sk - delta_R * rf)
                constraints.append(
                    {
                        "eval_point_index": point_idx + 1,
                        "z_offset": z,
                        "facet_k": k,
                        "expression": expr,
                    }
                )
        return constraints


def create_dynamic_mapping(defect_vars: Sequence, clearance_vars: Sequence) -> dict:
    mapping: dict = {}
    defect_index: dict[str, int] = {}
    for var in defect_vars:
        m = _VAR_RE.match(var.name)
        if not m:
            continue
        prefix, suffix = (m.group(1), m.group(2))
        idx = defect_index.setdefault(suffix, len(defect_index))
        mapping[var] = sp.Symbol(f"{_PREFIX_MAP.get(prefix, prefix)}_d_{idx}")
    clearance_index: dict[str, int] = {}
    g_counter = 0
    for var in clearance_vars:
        m = _VAR_RE.match(var.name)
        if not m:
            continue
        prefix, suffix = (m.group(1), m.group(2))
        if suffix in clearance_index:
            c_idx, sign = (clearance_index[suffix], 1)
        elif len(suffix) == 4 and (rev := (suffix[2:] + suffix[:2])) in clearance_index:
            c_idx, sign = (clearance_index[rev], -1)
        else:
            clearance_index[suffix] = g_counter
            c_idx, sign = (g_counter, 1)
            g_counter += 1
        new_sym = sp.Symbol(f"{_PREFIX_MAP.get(prefix, prefix)}_g_{c_idx}")
        mapping[var] = new_sym if sign == 1 else -new_sym
    return mapping


def extract_linear_matrices(
    expressions, clearance_vars, defect_vars, length_vars=None, length_subs=None
):
    if length_vars and length_subs:
        valid_subs = {k: v for k, v in length_subs.items() if k in length_vars}
        expressions = [expr.subs(valid_subs) for expr in expressions]
    A, rhs1 = sp.linear_eq_to_matrix(expressions, *clearance_vars)
    B, rhs2 = sp.linear_eq_to_matrix(-rhs1, *defect_vars)
    C = -rhs2
    if length_subs:
        remaining_symbols = A.free_symbols | B.free_symbols | C.free_symbols
        if remaining_symbols:
            print(
                f"Warning: Matrices still contain unbound symbols: {remaining_symbols}"
            )
    return (A, B, C)


l0, l1, l2, l3 = sp.symbols("l0 l1 l2 l3")
B3 = sp.Matrix([0, 0, 0])
B0 = sp.Matrix([0, l1, 0])
B1 = sp.Matrix([l0, l1, 0])
B2 = sp.Matrix([l0, 0, 0])
R = sp.Matrix([l0 / 2, l1 / 2, 0])
A_prime_0 = B0 + sp.Matrix([0, 0, l2 / 2])
A_prime_1 = B1 + sp.Matrix([0, 0, l2 / 2])
A_prime_2 = B2 + sp.Matrix([0, 0, l2 / 2])
A_prime_3 = B3 + sp.Matrix([0, 0, l2 / 2])
C_prime_0 = B0 + sp.Matrix([0, 0, -l3 / 2])
C_prime_1 = B1 + sp.Matrix([0, 0, -l3 / 2])
C_prime_2 = B2 + sp.Matrix([0, 0, -l3 / 2])
C_prime_3 = B3 + sp.Matrix([0, 0, -l3 / 2])
a1b1, b1b1, w1b1 = sp.symbols("alpha_1b1 beta_1b1 w_1b1")
a2b2, b2b2, w2b2 = sp.symbols("alpha_2b2 beta_2b2 w_2b2")
a1d1, b1d1, u1d1, v1d1 = sp.symbols("alpha_1d1 beta_1d1 u_1d1 v_1d1")
a2d2, b2d2, u2d2, v2d2 = sp.symbols("alpha_2d2 beta_2d2 u_2d2 v_2d2")
a1e1, b1e1, u1e1, v1e1 = sp.symbols("alpha_1e1 beta_1e1 u_1e1 v_1e1")
a2e2, b2e2, u2e2, v2e2 = sp.symbols("alpha_2e2 beta_2e2 u_2e2 v_2e2")
a1f1, b1f1, u1f1, v1f1 = sp.symbols("alpha_1f1 beta_1f1 u_1f1 v_1f1")
a2f2, b2f2, u2f2, v2f2 = sp.symbols("alpha_2f2 beta_2f2 u_2f2 v_2f2")
a1g1, b1g1, u1g1, v1g1 = sp.symbols("alpha_1g1 beta_1g1 u_1g1 v_1g1")
a2g2, b2g2, u2g2, v2g2 = sp.symbols("alpha_2g2 beta_2g2 u_2g2 v_2g2")
d1d, d1e, d1f, d1g = sp.symbols("d_1d d_1e d_1f d_1g")
d2d, d2e, d2f, d2g = sp.symbols("d_2d d_2e d_2f d_2g")
d3d, d4e, d5f, d6g = sp.symbols("d_3d d_4e d_5f d_6g")
u1b2b, v1b2b, g1b2b = sp.symbols("u_1b2b v_1b2b gamma_1b2b")
a3d1d, b3d1d, g3d1d, u3d1d, v3d1d, w3d1d = sp.symbols(
    "alpha_3d1d beta_3d1d gamma_3d1d u_3d1d v_3d1d w_3d1d"
)
a3d2d, b3d2d, u3d2d, v3d2d = sp.symbols("alpha_3d2d beta_3d2d u_3d2d v_3d2d")
a4e1e, b4e1e, g4e1e, u4e1e, v4e1e, w4e1e = sp.symbols(
    "alpha_4e1e beta_4e1e gamma_4e1e u_4e1e v_4e1e w_4e1e"
)
a4e2e, b4e2e, u4e2e, v4e2e = sp.symbols("alpha_4e2e beta_4e2e u_4e2e v_4e2e")
a5f1f, b5f1f, g5f1f, u5f1f, v5f1f, w5f1f = sp.symbols(
    "alpha_5f1f beta_5f1f gamma_5f1f u_5f1f v_5f1f w_5f1f"
)
a5f2f, b5f2f, u5f2f, v5f2f = sp.symbols("alpha_5f2f beta_5f2f u_5f2f v_5f2f")
a6g1g, b6g1g, g6g1g, u6g1g, v6g1g, w6g1g = sp.symbols(
    "alpha_6g1g beta_6g1g gamma_6g1g u_6g1g v_6g1g w_6g1g"
)
a6g2g, b6g2g, u6g2g, v6g2g = sp.symbols("alpha_6g2g beta_6g2g u_6g2g v_6g2g")
T_1b_1 = SmallDispTorsor(
    "T_{1b/1}", sp.Matrix([a1b1, b1b1, 0]), sp.Matrix([0, 0, w1b1]), R, "R"
)
T_2b_2 = SmallDispTorsor(
    "T_{2b/2}", sp.Matrix([a2b2, b2b2, 0]), sp.Matrix([0, 0, w2b2]), R, "R"
)
T_1d_1 = SmallDispTorsor(
    "T_{1d/1}",
    sp.Matrix([a1d1, b1d1, 0]),
    sp.Matrix([u1d1, v1d1, 0]),
    A_prime_0,
    "A'_0",
)
T_2d_2 = SmallDispTorsor(
    "T_{2d/2}",
    sp.Matrix([a2d2, b2d2, 0]),
    sp.Matrix([u2d2, v2d2, 0]),
    C_prime_0,
    "C'_0",
)
T_1e_1 = SmallDispTorsor(
    "T_{1e/1}",
    sp.Matrix([a1e1, b1e1, 0]),
    sp.Matrix([u1e1, v1e1, 0]),
    A_prime_1,
    "A'_1",
)
T_2e_2 = SmallDispTorsor(
    "T_{2e/2}",
    sp.Matrix([a2e2, b2e2, 0]),
    sp.Matrix([u2e2, v2e2, 0]),
    C_prime_1,
    "C'_1",
)
T_1f_1 = SmallDispTorsor(
    "T_{1f/1}",
    sp.Matrix([a1f1, b1f1, 0]),
    sp.Matrix([u1f1, v1f1, 0]),
    A_prime_2,
    "A'_2",
)
T_2f_2 = SmallDispTorsor(
    "T_{2f/2}",
    sp.Matrix([a2f2, b2f2, 0]),
    sp.Matrix([u2f2, v2f2, 0]),
    C_prime_2,
    "C'_2",
)
T_1g_1 = SmallDispTorsor(
    "T_{1g/1}",
    sp.Matrix([a1g1, b1g1, 0]),
    sp.Matrix([u1g1, v1g1, 0]),
    A_prime_3,
    "A'_3",
)
T_2g_2 = SmallDispTorsor(
    "T_{2g/2}",
    sp.Matrix([a2g2, b2g2, 0]),
    sp.Matrix([u2g2, v2g2, 0]),
    C_prime_3,
    "C'_3",
)
G_1b_2b = SmallDispTorsor(
    "G_{1b/2b}", sp.Matrix([0, 0, g1b2b]), sp.Matrix([u1b2b, v1b2b, 0]), R, "R"
)
G_3d_1d = SmallDispTorsor(
    "G_{3d/1d}",
    sp.Matrix([a3d1d, b3d1d, g3d1d]),
    sp.Matrix([u3d1d, v3d1d, w3d1d]),
    A_prime_0,
    "A'_0",
)
G_3d_2d = SmallDispTorsor(
    "G_{3d/2d}",
    sp.Matrix([a3d2d, b3d2d, 0]),
    sp.Matrix([u3d2d, v3d2d, 0]),
    C_prime_0,
    "C'_0",
)
G_4e_1e = SmallDispTorsor(
    "G_{4e/1e}",
    sp.Matrix([a4e1e, b4e1e, g4e1e]),
    sp.Matrix([u4e1e, v4e1e, w4e1e]),
    A_prime_1,
    "A'_1",
)
G_4e_2e = SmallDispTorsor(
    "G_{4e/2e}",
    sp.Matrix([a4e2e, b4e2e, 0]),
    sp.Matrix([u4e2e, v4e2e, 0]),
    C_prime_1,
    "C'_1",
)
G_5f_1f = SmallDispTorsor(
    "G_{5f/1f}",
    sp.Matrix([a5f1f, b5f1f, g5f1f]),
    sp.Matrix([u5f1f, v5f1f, w5f1f]),
    A_prime_2,
    "A'_2",
)
G_5f_2f = SmallDispTorsor(
    "G_{5f/2f}",
    sp.Matrix([a5f2f, b5f2f, 0]),
    sp.Matrix([u5f2f, v5f2f, 0]),
    C_prime_2,
    "C'_2",
)
G_6g_1g = SmallDispTorsor(
    "G_{6g/1g}",
    sp.Matrix([a6g1g, b6g1g, g6g1g]),
    sp.Matrix([u6g1g, v6g1g, w6g1g]),
    A_prime_3,
    "A'_3",
)
G_6g_2g = SmallDispTorsor(
    "G_{6g/2g}",
    sp.Matrix([a6g2g, b6g2g, 0]),
    sp.Matrix([u6g2g, v6g2g, 0]),
    C_prime_3,
    "C'_3",
)
defect_vars = [
    a1b1,
    b1b1,
    w1b1,
    a2b2,
    b2b2,
    w2b2,
    a1d1,
    b1d1,
    u1d1,
    v1d1,
    a2d2,
    b2d2,
    u2d2,
    v2d2,
    a1e1,
    b1e1,
    u1e1,
    v1e1,
    a2e2,
    b2e2,
    u2e2,
    v2e2,
    a1f1,
    b1f1,
    u1f1,
    v1f1,
    a2f2,
    b2f2,
    u2f2,
    v2f2,
    a1g1,
    b1g1,
    u1g1,
    v1g1,
    a2g2,
    b2g2,
    u2g2,
    v2g2,
    d1d,
    d1e,
    d1f,
    d1g,
    d2d,
    d2e,
    d2f,
    d2g,
    d3d,
    d4e,
    d5f,
    d6g,
]
clearance_vars = [
    u1b2b,
    v1b2b,
    g1b2b,
    a3d1d,
    b3d1d,
    g3d1d,
    u3d1d,
    v3d1d,
    w3d1d,
    a3d2d,
    b3d2d,
    u3d2d,
    v3d2d,
    a4e1e,
    b4e1e,
    g4e1e,
    u4e1e,
    v4e1e,
    w4e1e,
    a4e2e,
    b4e2e,
    u4e2e,
    v4e2e,
    a5f1f,
    b5f1f,
    g5f1f,
    u5f1f,
    v5f1f,
    w5f1f,
    a5f2f,
    b5f2f,
    u5f2f,
    v5f2f,
    a6g1g,
    b6g1g,
    g6g1g,
    u6g1g,
    v6g1g,
    w6g1g,
    a6g2g,
    b6g2g,
    u6g2g,
    v6g2g,
]


def get_kinematic_loop_expressions():
    T_common = T_1b_1 - G_1b_2b - T_2b_2
    L1 = (
        T_common
        + T_2d_2.move_to(R, "R")
        + G_3d_2d.move_to(R, "R")
        - G_3d_1d.move_to(R, "R")
        - T_1d_1.move_to(R, "R")
    )
    L2 = (
        T_common
        + T_2e_2.move_to(R, "R")
        + G_4e_2e.move_to(R, "R")
        - G_4e_1e.move_to(R, "R")
        - T_1e_1.move_to(R, "R")
    )
    L3 = (
        T_common
        + T_2f_2.move_to(R, "R")
        + G_5f_2f.move_to(R, "R")
        - G_5f_1f.move_to(R, "R")
        - T_1f_1.move_to(R, "R")
    )
    L4 = (
        T_common
        + T_2g_2.move_to(R, "R")
        + G_6g_2g.move_to(R, "R")
        - G_6g_1g.move_to(R, "R")
        - T_1g_1.move_to(R, "R")
    )
    loops = [L1, L2, L3, L4]
    expressions = []
    for loop in loops:
        for i in range(3):
            expressions.append(sp.expand(loop.omega[i]))
        for i in range(3):
            expressions.append(sp.expand(loop.v[i]))
    return (expressions, defect_vars, clearance_vars)


def get_interface_constraint_expressions(
    num_points=16, strategy=LinearizationStrategy.CIRCUMSCRIBED, z_eval_points=None
):
    interfaces = [
        CylindricalInterfaceConstraint(
            "1d/3d",
            G_3d_1d,
            d1d,
            d3d,
            l2,
            Nd=num_points,
            strategy=strategy,
            z_eval_points=z_eval_points,
        ),
        CylindricalInterfaceConstraint(
            "1e/4e",
            G_4e_1e,
            d1e,
            d4e,
            l2,
            Nd=num_points,
            strategy=strategy,
            z_eval_points=z_eval_points,
        ),
        CylindricalInterfaceConstraint(
            "1f/5f",
            G_5f_1f,
            d1f,
            d5f,
            l2,
            Nd=num_points,
            strategy=strategy,
            z_eval_points=z_eval_points,
        ),
        CylindricalInterfaceConstraint(
            "1g/6g",
            G_6g_1g,
            d1g,
            d6g,
            l2,
            Nd=num_points,
            strategy=strategy,
            z_eval_points=z_eval_points,
        ),
        CylindricalInterfaceConstraint(
            "2d/3d",
            G_3d_2d,
            d2d,
            d3d,
            l3,
            Nd=num_points,
            strategy=strategy,
            z_eval_points=z_eval_points,
        ),
        CylindricalInterfaceConstraint(
            "2e/4e",
            G_4e_2e,
            d2e,
            d4e,
            l3,
            Nd=num_points,
            strategy=strategy,
            z_eval_points=z_eval_points,
        ),
        CylindricalInterfaceConstraint(
            "2f/5f",
            G_5f_2f,
            d2f,
            d5f,
            l3,
            Nd=num_points,
            strategy=strategy,
            z_eval_points=z_eval_points,
        ),
        CylindricalInterfaceConstraint(
            "2g/6g",
            G_6g_2g,
            d2g,
            d6g,
            l3,
            Nd=num_points,
            strategy=strategy,
            z_eval_points=z_eval_points,
        ),
    ]
    expressions = []
    for interface in interfaces:
        constraints = interface.generate_equations()
        for c in constraints:
            expressions.append(c["expression"])
    return expressions


def rename_variables(expressions, mapping_dict):
    return [expr.subs(mapping_dict) for expr in expressions]


def print_kinematic_loops(expressions):
    axis_names = [
        "Rotation x",
        "Rotation y",
        "Rotation z",
        "Translation x",
        "Translation y",
        "Translation z",
    ]
    for idx, expr in enumerate(expressions):
        loop_num = idx // 6 + 1
        eq_num = idx % 6 + 1
        if eq_num == 1:
            print(f"\n=== Loop {loop_num} Equations ===")
        print(f"Eq {eq_num} [{axis_names[eq_num - 1]}]:\n{expr} = 0")


def print_interface_constraints(expressions):
    print("\n=== Interface Constraints (Eq <= 0) ===")
    for idx, expr in enumerate(expressions):
        print(f"Constraint {idx + 1}:\n{expr} <= 0")


def getSystemOfConstraintsAssemblyModel(
    L=[50, 50, 30, 30],
    CIRCLE_RESOLUTION=64,
    strategy=LinearizationStrategy.CIRCUMSCRIBED,
):
    assert len(L) == 4, (
        "Length list L must contain exactly 4 elements corresponding to l0, l1, l2, l3."
    )
    l0, l1, l2, l3 = sp.symbols("l0 l1 l2 l3")
    length_subs = {l0: L[0], l1: L[1], l2: L[2], l3: L[3]}
    loop_exprs, d_vars, c_vars = get_kinematic_loop_expressions()
    interf_exprs = get_interface_constraint_expressions(
        num_points=CIRCLE_RESOLUTION,
        strategy=strategy,
        z_eval_points=[-L[2] / 2, L[2] / 2],
    )
    otaf_mapping = create_dynamic_mapping(defect_vars, clearance_vars)
    mapped_loop_exprs = rename_variables(loop_exprs, otaf_mapping)
    mapped_interf_exprs = rename_variables(interf_exprs, otaf_mapping)
    mapped_clearances = [
        otaf_mapping[var] for var in clearance_vars if var in otaf_mapping
    ]
    mapped_defects = [otaf_mapping[var] for var in defect_vars if var in otaf_mapping]
    comp_mats = extract_linear_matrices(
        mapped_loop_exprs,
        mapped_clearances,
        mapped_defects,
        [l0, l1, l2, l3],
        length_subs,
    )
    interf_mats = extract_linear_matrices(
        mapped_interf_exprs,
        mapped_clearances,
        mapped_defects,
        [l0, l1, l2, l3],
        length_subs,
    )
    A_eq_gap = np.array(comp_mats[0], dtype=float)
    A_eq_def = np.array(comp_mats[1], dtype=float)
    K_eq = np.array(comp_mats[2], dtype=float).flatten()
    A_ub_gap = np.array(interf_mats[0], dtype=float)
    A_ub_def = np.array(interf_mats[1], dtype=float)
    K_ub = np.array(interf_mats[2], dtype=float).flatten()
    matrices_for_socam = [A_eq_def, A_eq_gap, K_eq, -A_ub_def, -A_ub_gap, -K_ub]
    SOCAM = otaf.SystemOfConstraintsAssemblyModel(matrices=matrices_for_socam)
    SOCAM.deviation_symbols = copy.deepcopy(mapped_defects)
    SOCAM.gap_symbols = copy.deepcopy(mapped_clearances)
    SOCAM.embedOptimizationVariable()
    return SOCAM


def getDistributionParams(
    tol=0.21, capa=1.0, hPlate=30.0, EH=50.0, LB=25.0, Dext=20.0, Dint=19.8
):
    sigma_e_pos = tol / (6 * capa)
    theta_max = tol / hPlate
    sigma_e_theta = 2 * theta_max / (6 * capa)
    sigma_e_pos_plate = tol / (6 * capa)
    sigma_e_theta_plate = 2 * (tol / (EH + LB)) / (6 * capa)
    sigma_diam = sigma_e_pos
    otaf_mapping = create_dynamic_mapping(defect_vars, clearance_vars)
    mapped_defects = [otaf_mapping[var] for var in defect_vars if var in otaf_mapping]
    mu_vect = np.array([0.0] * 38 + [Dext] * 8 + [Dint] * 4)
    std_vect = np.array(
        [sigma_e_theta_plate, sigma_e_theta_plate, sigma_e_pos_plate] * 2
        + [sigma_e_theta, sigma_e_theta, sigma_e_pos, sigma_e_pos] * 8
        + [sigma_diam] * 12
    )
    RandDeviationVect = otaf.distribution.get_composed_normal_defect_distribution(
        mapped_defects, mu_list=mu_vect.tolist(), sigma_list=std_vect.tolist()
    )
    return (RandDeviationVect, mapped_defects, std_vect, mu_vect)


dim = 50
sample_multiplier = np.eye(dim)
no_tol = False


def evalCredalSetConstraints(x_std, tol=0.21, capa=1.0, hPlate=30.0):
    target = tol / (6 * capa)
    cons0 = lambda x: (sigma_delta_3D_plane(50, 50, x[2], x[0], x[1]) - target) / target
    cons1 = lambda x: (sigma_delta_3D_plane(50, 50, x[5], x[3], x[4]) - target) / target
    cons2 = lambda x: (
        (
            np.maximum(
                sigma_delta_cylindrical_feature(
                    hPlate / 2, 0, x[38], x[8], x[6], x[9], x[7]
                ),
                sigma_delta_cylindrical_feature(
                    hPlate / 2, np.pi / 2, x[38], x[8], x[6], x[9], x[7]
                ),
            )
            - target
        )
        / target
    )
    cons3 = lambda x: (
        (
            np.maximum(
                sigma_delta_cylindrical_feature(
                    hPlate / 2, 0, x[39], x[16], x[14], x[17], x[15]
                ),
                sigma_delta_cylindrical_feature(
                    hPlate / 2, np.pi / 2, x[39], x[16], x[14], x[17], x[15]
                ),
            )
            - target
        )
        / target
    )
    cons4 = lambda x: (
        (
            np.maximum(
                sigma_delta_cylindrical_feature(
                    hPlate / 2, 0, x[40], x[24], x[22], x[25], x[23]
                ),
                sigma_delta_cylindrical_feature(
                    hPlate / 2, np.pi / 2, x[40], x[24], x[22], x[25], x[23]
                ),
            )
            - target
        )
        / target
    )
    cons5 = lambda x: (
        (
            np.maximum(
                sigma_delta_cylindrical_feature(
                    hPlate / 2, 0, x[41], x[32], x[30], x[33], x[31]
                ),
                sigma_delta_cylindrical_feature(
                    hPlate / 2, np.pi / 2, x[41], x[32], x[30], x[33], x[31]
                ),
            )
            - target
        )
        / target
    )
    cons6 = lambda x: (
        (
            np.maximum(
                sigma_delta_cylindrical_feature(
                    hPlate / 2, 0, x[42], x[12], x[10], x[13], x[11]
                ),
                sigma_delta_cylindrical_feature(
                    hPlate / 2, np.pi / 2, x[42], x[12], x[10], x[13], x[11]
                ),
            )
            - target
        )
        / target
    )
    cons7 = lambda x: (
        (
            np.maximum(
                sigma_delta_cylindrical_feature(
                    hPlate / 2, 0, x[43], x[20], x[18], x[21], x[19]
                ),
                sigma_delta_cylindrical_feature(
                    hPlate / 2, np.pi / 2, x[43], x[20], x[18], x[21], x[19]
                ),
            )
            - target
        )
        / target
    )
    cons8 = lambda x: (
        (
            np.maximum(
                sigma_delta_cylindrical_feature(
                    hPlate / 2, 0, x[44], x[28], x[26], x[29], x[27]
                ),
                sigma_delta_cylindrical_feature(
                    hPlate / 2, np.pi / 2, x[44], x[28], x[26], x[29], x[27]
                ),
            )
            - target
        )
        / target
    )
    cons9 = lambda x: (
        (
            np.maximum(
                sigma_delta_cylindrical_feature(
                    hPlate / 2, 0, x[45], x[36], x[34], x[37], x[35]
                ),
                sigma_delta_cylindrical_feature(
                    hPlate / 2, np.pi / 2, x[45], x[36], x[34], x[37], x[35]
                ),
            )
            - target
        )
        / target
    )
    cons10 = lambda x: (x[46] - target) / target
    cons11 = lambda x: (x[47] - target) / target
    cons12 = lambda x: (x[48] - target) / target
    cons13 = lambda x: (x[49] - target) / target
    return np.array(
        [
            cons0(x_std),
            cons1(x_std),
            cons2(x_std),
            cons3(x_std),
            cons4(x_std),
            cons5(x_std),
            cons6(x_std),
            cons7(x_std),
            cons8(x_std),
            cons9(x_std),
            cons10(x_std),
            cons11(x_std),
            cons12(x_std),
            cons13(x_std),
        ]
    )


def evalScaledCredalSetConstraints(
    x_scaled,
    max_std_vect,
    tracker=None,
    experiment_key=None,
    tol=0.21,
    capa=1.0,
    hPlate=30.0,
):
    x_real = x_scaled * max_std_vect
    constraint_array = evalCredalSetConstraints(
        x_real, tol=tol, capa=capa, hPlate=hPlate
    )
    if tracker:
        tracker.update_constraint_data(
            exp_key=experiment_key, x=x_scaled, constraints=constraint_array
        )
    return constraint_array


def getScaledCredalSetConstraintsFunction(
    max_std_vect, tracker=None, experiment_key=None, tol=0.21, capa=1.0, hPlate=30.0
):
    return lambda x_scaled: evalScaledCredalSetConstraints(
        x_scaled,
        max_std_vect,
        tracker,
        experiment_key,
        tol=tol,
        capa=capa,
        hPlate=hPlate,
    )
