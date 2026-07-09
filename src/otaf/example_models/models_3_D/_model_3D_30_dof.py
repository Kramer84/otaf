from __future__ import annotations

__author__ = "Kramer84"
__all__ = [
    "get_system_of_constraints_assembly_model",
    "get_distribution_params",
    "eval_credal_set_constraints",
    "eval_scaled_credal_set_constraints",
    "get_scaled_credal_set_constraints_function"
    "dim",
    "sample_multiplier",
    "no_tol", 
    ]

from enum import Enum
from typing import Tuple, Callable, Any

import numpy as np
import sympy as sp

import otaf
from otaf.tolerances import sigma_delta_circular_feature


class LinearizationStrategy(Enum):
    INSCRIBED = "inscribed"
    MEAN = "mean"
    CIRCUMSCRIBED = "circumscribed"


MatrixBundle = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
NX = 30 # Number of defect variables
NG = 17 # Number of clearance variables
NC = 14 # Number of compatibility equations


def build_constraint_matrices(
    L: np.ndarray | list[float],
    Nd: int = 32,
    strategy: LinearizationStrategy = LinearizationStrategy.CIRCUMSCRIBED,
) -> MatrixBundle:
    """Build the equality and inequality constraint matrices.

    Parameters
    ----------
    L : array_like
        An 11-element array containing geometric length parameters.
    Nd : int, default 32
        The number of discretization points used for circular features.
        Must be greater than or equal to 3.
    strategy : LinearizationStrategy, optional
        The linearization strategy applied to the circular constraints.
        Default is ``LinearizationStrategy.CIRCUMSCRIBED``.

    Returns
    -------
    MatrixBundle
        A tuple of six NumPy arrays: (A_eq_Def, A_eq_Gap, K_eq,
        A_ub_Def, A_ub_Gap, K_ub).

    Raises
    ------
    ValueError
        If `L` does not have shape (11,) or if `Nd` is less than 3.
    """
    L = np.asarray(L, dtype=float)
    if L.shape != (11,):
        raise ValueError(f"L must have shape (11,), got {L.shape}")
    if Nd < 3:
        raise ValueError(f"Nd must be >= 3, got {Nd}")
    NI = 4 * Nd
    A_eq_Def = np.zeros((NC, NX))
    A_eq_Gap = np.zeros((NC, NG))
    K_eq = np.zeros(NC)
    A_eq_Def[0, [1, 4, 8, 12]] = [+1, -1, -1, +1]
    A_eq_Gap[0, 0] = -1
    A_eq_Def[1, [2, 5, 9, 13]] = [+1, -1, -1, +1]
    A_eq_Gap[1, 1] = -1
    A_eq_Gap[2, [2, 6]] = [-1, -1]
    A_eq_Def[3, [6, 10]] = [-1, +1]
    A_eq_Gap[3, [3, 7]] = [-1, -1]
    A_eq_Def[4, [7, 11]] = [-1, +1]
    A_eq_Gap[4, [4, 8]] = [-1, -1]
    A_eq_Def[5, [0, 3]] = [+1, -1]
    A_eq_Gap[5, 5] = -1
    A_eq_Def[6, [8, 12, 16, 20]] = [-1, +1, +1, -1]
    A_eq_Gap[6, [0, 9]] = [-1, +1]
    A_eq_Def[7, [9, 13, 17, 21]] = [-1, +1, +1, -1]
    A_eq_Gap[7, [1, 10]] = [-1, +1]
    A_eq_Gap[8, [2, 11]] = [-1, +1]
    A_eq_Def[9, [6, 10, 14, 18]] = [-1, +1, +1, -1]
    A_eq_Gap[9, [3, 11, 12]] = [-1, L[1], +1]
    A_eq_Def[10, [7, 11, 15, 19]] = [-1, +1, +1, -1]
    A_eq_Gap[10, [4, 11, 13]] = [-1, -L[0], +1]
    A_eq_Def[11, [16, 17, 20, 21]] = [-L[1], +L[0], +L[1], -L[0]]
    A_eq_Gap[11, [5, 9, 10, 14]] = [-1, -L[1], +L[0], +1]
    A_eq_Def[12, [6, 9, 10, 13, 26, 28]] = [-1, -L[8], +1, +L[8], +1, -1]
    A_eq_Gap[12, [1, 2, 3, 15]] = [-L[8], +L[7], -1, +1]
    A_eq_Def[13, [7, 8, 11, 12, 27, 29]] = [-1, +L[8], +1, -L[8], +1, -1]
    A_eq_Gap[13, [0, 2, 4, 16]] = [+L[8], -L[6], -1, +1]
    A_ub_Def = np.zeros((NI, NX))
    A_ub_Gap = np.zeros((NI, NG))
    K_ub = np.zeros(NI)
    theta = 2.0 * np.pi * np.arange(1, Nd + 1) / Nd
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    theta1 = 2.0 * np.pi / Nd
    if strategy == LinearizationStrategy.INSCRIBED:
        rf = np.cos(theta1 / 2.0)
    elif strategy == LinearizationStrategy.MEAN:
        rf = (1.0 + np.cos(theta1 / 2.0)) / 2.0
    else:
        rf = 1.0
    half_rf = 0.5 * rf
    for k in range(Nd):
        ck, sk = (cos_t[k], sin_t[k])
        r = k
        A_ub_Def[r, [22, 23]] = [+half_rf, -half_rf]
        A_ub_Gap[r, [3, 4]] = [-ck, -sk]
        r = Nd + k
        A_ub_Def[r, [22, 23]] = [+half_rf, -half_rf]
        A_ub_Gap[r, [0, 1, 3, 4]] = [+L[2] * sk, -L[2] * ck, -ck, -sk]
        r = 2 * Nd + k
        A_ub_Def[r, [24, 25]] = [+half_rf, -half_rf]
        A_ub_Gap[r, [12, 13]] = [-ck, -sk]
        r = 3 * Nd + k
        A_ub_Def[r, [24, 25]] = [+half_rf, -half_rf]
        A_ub_Gap[r, [9, 10, 12, 13]] = [+L[3] * sk, -L[3] * ck, -ck, -sk]
    return (A_eq_Def, A_eq_Gap, K_eq, A_ub_Def, A_ub_Gap, K_ub)


x_full_labels = [
    "w_1a1",
    "alpha_1a1",
    "beta_1a1",
    "w_2a2",
    "alpha_2a2",
    "beta_2a2",
    "u_1b1",
    "v_1b1",
    "alpha_1b1",
    "beta_1b1",
    "u_2b2",
    "v_2b2",
    "alpha_2b2",
    "beta_2b2",
    "u_1c1",
    "v_1c1",
    "alpha_1c1",
    "beta_1c1",
    "u_2c2",
    "v_2c2",
    "alpha_2c2",
    "beta_2c2",
    "d_1b",
    "d_3b",
    "d_1c",
    "d_4c",
    "u_1g1",
    "v_1g1",
    "u_2g2",
    "v_2g2",
]
g_labels = [
    "u_1a2a",
    "v_1a2a",
    "gamma_1a2a",
    "u_3b1b",
    "v_3b1b",
    "w_3b1b",
    "alpha_3b1b",
    "beta_3b1b",
    "gamma_3b1b",
    "u_4c1c",
    "v_4c1c",
    "w_4c1c",
    "alpha_4c1c",
    "beta_4c1c",
    "gamma_4c1c",
    "u_2g1g_1",
    "u_2g1g_2",
]
x_full_labels_mapping = {
    "w_1a1": "u_d_0",
    "alpha_1a1": "beta_d_0",
    "beta_1a1": "gamma_d_0",
    "w_2a2": "u_d_1",
    "alpha_2a2": "beta_d_1",
    "beta_2a2": "gamma_d_1",
    "u_1b1": "v_d_2",
    "v_1b1": "w_d_2",
    "alpha_1b1": "beta_d_2",
    "beta_1b1": "gamma_d_2",
    "u_2b2": "v_d_3",
    "v_2b2": "w_d_3",
    "alpha_2b2": "beta_d_3",
    "beta_2b2": "gamma_d_3",
    "u_1c1": "v_d_4",
    "v_1c1": "w_d_4",
    "alpha_1c1": "beta_d_4",
    "beta_1c1": "gamma_d_4",
    "u_2c2": "v_d_5",
    "v_2c2": "w_d_5",
    "alpha_2c2": "beta_d_5",
    "beta_2c2": "gamma_d_5",
    "d_1b": "d_d_2",
    "d_3b": "d_d_6",
    "d_1c": "d_d_4",
    "d_4c": "d_d_7",
    "u_1g1": "v_d_8",
    "v_1g1": "w_d_8",
    "u_2g2": "beta_d_8",
    "v_2g2": "gamma_d_8",
}
g_labels_mapping = {
    "u_1a2a": "v_g_0",
    "v_1a2a": "w_g_0",
    "gamma_1a2a": "alpha_g_0",
    "u_3b1b": "u_g_1",
    "v_3b1b": "v_g_1",
    "w_3b1b": "w_g_1",
    "alpha_3b1b": "alpha_g_1",
    "beta_3b1b": "beta_g_1",
    "gamma_3b1b": "gamma_g_1",
    "u_4c1c": "u_g_2",
    "v_4c1c": "v_g_2",
    "w_4c1c": "w_g_2",
    "alpha_4c1c": "alpha_g_2",
    "beta_4c1c": "beta_g_2",
    "gamma_4c1c": "gamma_g_2",
    "u_2g1g_1": "u_g_3",
    "u_2g1g_2": "u_g_4",
}
x_mp_labels = [
    "w_1a1",
    "w_1a1_C",
    "w_1a1_H",
    "w_2a2",
    "w_2a2_C",
    "w_2a2_H",
    "u_1b1",
    "v_1b1",
    "u_1b1_B",
    "v_1b1_B",
    "u_2b2",
    "v_2b2",
    "u_2b2_E",
    "v_2b2_E",
    "u_1c1",
    "v_1c1",
    "u_1c1_D",
    "v_1c1_D",
    "u_2c2",
    "v_2c2",
    "u_2c2_F",
    "v_2c2_F",
    "d_1b",
    "d_3b",
    "d_1c",
    "d_4c",
    "u_1g1",
    "v_1g1",
    "u_2g2",
    "v_2g2",
]


def get_mp_to_xfull_transformation_matrix(
    L: np.ndarray | list[float] = [
        100,
        40,
        30,
        30,
        20,
        20,
        120,
        50,
        40,
        50,
        -30,
    ],
) -> np.ndarray:
    """Compute the transformation matrix from MP to full defect space.

    Parameters
    ----------
    L : array_like, optional
        Geometric dimensions and parameters of the assembly.
        The default is [100, 40, 30, 30, 20, 20, 120, 50, 40, 50, -30].

    Returns
    -------
    np.ndarray
        A 30x30 transformation matrix.
    """
    T = np.zeros((30, 30))
    D = L[0] * L[10] - L[1] * L[9]
    T[0, 0] = 1.0
    T[1, 0] = (L[9] - L[0]) / D
    T[1, 1] = -L[9] / D
    T[1, 2] = L[0] / D
    T[2, 0] = (L[10] - L[1]) / D
    T[2, 1] = -L[10] / D
    T[2, 2] = L[1] / D
    T[3, 3] = 1.0
    T[4, 3] = (L[9] - L[0]) / D
    T[4, 4] = -L[9] / D
    T[4, 5] = L[0] / D
    T[5, 3] = (L[10] - L[1]) / D
    T[5, 4] = -L[10] / D
    T[5, 5] = L[1] / D
    T[6, 6] = 1.0
    T[7, 7] = 1.0
    T[8, 7] = 1.0 / L[2]
    T[8, 9] = -1.0 / L[2]
    T[9, 8] = 1.0 / L[2]
    T[9, 6] = -1.0 / L[2]
    T[10, 10] = 1.0
    T[11, 11] = 1.0
    T[12, 11] = -1.0 / L[4]
    T[12, 13] = 1.0 / L[4]
    T[13, 10] = 1.0 / L[4]
    T[13, 12] = -1.0 / L[4]
    T[14, 14] = 1.0
    T[15, 15] = 1.0
    T[16, 15] = 1.0 / L[3]
    T[16, 17] = -1.0 / L[3]
    T[17, 16] = 1.0 / L[3]
    T[17, 14] = -1.0 / L[3]
    T[18, 18] = 1.0
    T[19, 19] = 1.0
    T[20, 19] = -1.0 / L[5]
    T[20, 21] = 1.0 / L[5]
    T[21, 18] = 1.0 / L[5]
    T[21, 20] = -1.0 / L[5]
    for i in range(22, 30):
        T[i, i] = 1.0
    return T


def get_system_of_constraints_assembly_model(
    L: np.ndarray | list[float] = [
        100,
        40,
        30,
        30,
        20,
        20,
        120,
        50,
        40,
        50,
        -30,
    ],
    Nd: int = 64,
    strategy: LinearizationStrategy = LinearizationStrategy.CIRCUMSCRIBED,
) -> otaf.SystemOfConstraintsAssemblyModel:
    """Construct the system of constraints assembly model.

    Parameters
    ----------
    L : array_like, optional
        Geometric dimensions and parameters of the assembly.
        The default is [100, 40, 30, 30, 20, 20, 120, 50, 40, 50, -30].
    Nd : int, default 64
        The number of discretization points for linearization.
    strategy : LinearizationStrategy, optional
        The linearization strategy applied to circular constraints.
        Default is ``LinearizationStrategy.CIRCUMSCRIBED``.

    Returns
    -------
    otaf.SystemOfConstraintsAssemblyModel
        The initialized assembly model with embedded optimization
        variables.
    """
    mats = build_constraint_matrices(L, Nd, strategy)
    SOCAM = otaf.SystemOfConstraintsAssemblyModel(matrices=list(mats))
    d_labels = [sp.Symbol(x_full_labels_mapping[lab]) for lab in x_full_labels]
    g_labels_loc = [sp.Symbol(g_labels_mapping[lab]) for lab in g_labels]
    SOCAM.deviation_symbols = d_labels
    SOCAM.gap_symbols = g_labels_loc
    SOCAM.embedOptimizationVariable()
    return SOCAM


def get_distribution_params(
    tol: float | None = None, capa: float | None = None, param_set: int = 1
) -> tuple[Any, list[str], np.ndarray, np.ndarray]:
    """Compute defect distribution parameters based on the parameter set.

    Parameters
    ----------
    tol : float, optional
        Tolerance parameter (unused in this configuration).
    capa : float, optional
        Process capability index (unused in this configuration).
    param_set : int, default 1
        The parameter set choice determining mean and variance shifts.
        Options are 1, 2, or 3.

    Returns
    -------
    RandDeviationVect : otaf.distribution.ComposedDistribution
        The joint normal defect distribution model.
    list of str
        The descriptive tracking labels for mid-point variables.
    sigma_arr : np.ndarray
        A 1D array of calculated standard deviations.
    mu_arr : np.ndarray
        A 1D array of calculated mean parameter offsets.
    """
    if param_set == 1:
        mu_d_ext, sigma_d_ext = (20.0, 0.06)
        mu_d_int, sigma_d_int = (19.8, 0.06)
        mu_trans, sigma_trans = (0.0, 0.01)
    elif param_set == 2:
        mu_d_ext, sigma_d_ext = (20.0, 0.03)
        mu_d_int, sigma_d_int = (19.8, 0.03)
        mu_trans, sigma_trans = (0.0, 0.01)
    else:
        mu_d_ext, sigma_d_ext = (20.0, 0.02)
        mu_d_int, sigma_d_int = (19.8, 0.02)
        mu_trans, sigma_trans = (0.0, 0.01)
    mu_list = [mu_trans] * 22
    sigma_list = [sigma_trans] * 22
    mu_list.extend([mu_d_ext, mu_d_int, mu_d_ext, mu_d_int])
    sigma_list.extend([sigma_d_ext, sigma_d_int, sigma_d_ext, sigma_d_int])
    mu_list.extend([mu_trans] * 4)
    sigma_list.extend([sigma_trans] * 4)
    mu_arr = np.array(mu_list)
    sigma_arr = np.array(sigma_list)
    RandDeviationVect = otaf.distribution.get_composed_normal_defect_distribution(
        defect_names=x_mp_labels, mu_list=mu_list, sigma_list=sigma_list
    )
    return (RandDeviationVect, x_mp_labels, sigma_arr, mu_arr)


dim = 30
sample_multiplier = get_mp_to_xfull_transformation_matrix()
no_tol = True


def eval_credal_set_constraints(
    x_std: np.ndarray,
    tol: float | None = None,
    capa: float | None = None,
    param_set: int = 1,
) -> np.ndarray:
    """Evaluate the normalized credal set boundary conditions.

    Parameters
    ----------
    x_std : np.ndarray
        A 1D array containing standard deviation vector values.
    tol : float, optional
        Tolerance parameter (unused in this configuration).
    capa : float, optional
        Process capability index (unused in this configuration).
    param_set : int, default 1
        The active parameter configuration variant index.

    Returns
    -------
    np.ndarray
        An array containing the evaluated constraint metrics.
    """
    if param_set == 1:
        mu_d_ext, sigma_d_ext = (20.0, 0.06)
        mu_d_int, sigma_d_int = (19.8, 0.06)
        mu_trans, sigma_trans = (0.0, 0.01)
    elif param_set == 2:
        mu_d_ext, sigma_d_ext = (20.0, 0.03)
        mu_d_int, sigma_d_int = (19.8, 0.03)
        mu_trans, sigma_trans = (0.0, 0.01)
    else:
        mu_d_ext, sigma_d_ext = (20.0, 0.02)
        mu_d_int, sigma_d_int = (19.8, 0.02)
        mu_trans, sigma_trans = (0.0, 0.01)
    target0 = sigma_trans
    target1 = sigma_delta_circular_feature(0, sigma_d_ext / 2, sigma_trans, sigma_trans)

    def eval_circ(d_idx, u_base, v_base, u_top, v_top):
        devs = [
            sigma_delta_circular_feature(
                0, x_std[d_idx] / 2, x_std[u_base], x_std[v_base]
            ),
            sigma_delta_circular_feature(
                np.pi / 2, x_std[d_idx] / 2, x_std[u_base], x_std[v_base]
            ),
            sigma_delta_circular_feature(
                0, x_std[d_idx] / 2, x_std[u_top], x_std[v_top]
            ),
            sigma_delta_circular_feature(
                np.pi / 2, x_std[d_idx] / 2, x_std[u_top], x_std[v_top]
            ),
        ]
        return (np.max(devs) - target1) / target1

    constraint1 = (np.max(x_std[0:3]) - target0) / target0
    constraint2 = (np.max(x_std[3:6]) - target0) / target0
    constraint7 = (np.max(x_std[26:28]) - target0) / target0
    constraint8 = (np.max(x_std[28:30]) - target0) / target0
    constraint3 = eval_circ(22, 6, 7, 8, 9)
    constraint4 = eval_circ(23, 10, 11, 12, 13)
    constraint5 = eval_circ(24, 14, 15, 16, 17)
    constraint6 = eval_circ(25, 18, 19, 20, 21)
    return np.array(
        [
            constraint1,
            constraint2,
            constraint3,
            constraint4,
            constraint5,
            constraint6,
            constraint7,
            constraint8,
        ]
    )


def eval_scaled_credal_set_constraints(
    x_scaled: np.ndarray,
    max_std_vect: np.ndarray,
    tracker: Any | None = None,
    experiment_key: Any | None = None,
    tol: float | None = None,
    capa: float | None = None,
    param_set: int = 1,
) -> np.ndarray:
    """Map scaled deviations to real values and evaluate constraints.

    Parameters
    ----------
    x_scaled : np.ndarray
        The scaled standard deviation vector inputs.
    max_std_vect : np.ndarray
        The upper-bound limits for standard deviation mapping.
    tracker : Any, optional
        Data logging tracker instance. Default is None.
    experiment_key : Any, optional
        Unique identifier key for tracking logs. Default is None.
    tol : float, optional
        Tolerance parameter (unused in this configuration).
    capa : float, optional
        Process capability index (unused in this configuration).
    param_set : int, default 1
        The active parameter configuration variant index.

    Returns
    -------
    np.ndarray
        The calculated constraint evaluation bounds array.
    """
    x_real = x_scaled * max_std_vect
    constraint_array = eval_credal_set_constraints(
        x_real, tol=tol, capa=capa, param_set=param_set
    )
    if tracker:
        tracker.update_constraint_data(
            exp_key=experiment_key, x=x_scaled, constraints=constraint_array
        )
    return constraint_array


def get_scaled_credal_set_constraints_function(
    max_std_vect: np.ndarray,
    tracker: Any | None = None,
    experiment_key: Any | None = None,
    tol: float | None = None,
    capa: float | None = None,
    param_set: int = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a wrapped lambda function for scaled constraints.

    Parameters
    ----------
    max_std_vect : np.ndarray
        The upper-bound limits for standard deviation mapping.
    tracker : Any, optional
        Data logging tracker instance. Default is None.
    experiment_key : Any, optional
        Unique identifier key for tracking logs. Default is None.
    tol : float, optional
        Tolerance parameter (unused in this configuration).
    capa : float, optional
        Process capability index (unused in this configuration).
    param_set : int, default 1
        The active parameter configuration variant index.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A single-argument function mapping `x_scaled` to its
        evaluated constraint array.
    """
    return lambda x_scaled: eval_scaled_credal_set_constraints(
        x_scaled,
        max_std_vect,
        tracker,
        experiment_key,
        tol=tol,
        capa=capa,
        param_set=param_set,
    )
