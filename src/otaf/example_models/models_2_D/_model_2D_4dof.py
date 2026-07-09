from __future__ import annotations

__author__ = "Kramer84"
__all__ = [
    "get_system_of_constraints_assembly_model",
    "get_distribution_params",
    "eval_credal_set_constraints",
    "eval_scaled_credal_set_constraints",
    "get_scaled_credal_set_constraints_function",
    "dim",
    "sample_multiplier",
    "no_tol",    
]
from typing import Any, Callable

import numpy as np
import sympy as sp

import otaf
from otaf.tolerances import sigma_delta_3D_plane


def get_assembly_data(
    X1: float = 99.8, X2: float = 100.0, X3: float = 10.0
) -> dict[str, dict[str, Any]]:
    """Generate the system configuration data for an assembly.

    Parameters
    ----------
    X1 : float, default 99.8
        Dimension parameter X1.
    X2 : float, default 100.0
        Dimension parameter X2.
    X3 : float, default 10.0
        Dimension parameter X3.

    Returns
    -------
    dict
        A dictionary containing the structural parts, interactions,
        kinematic constraints, loop topologies, and global constraints.
    """
    R0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x_, y_, z_ = (R0[0], R0[1], R0[2])
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
    RP1a = np.array([-1 * x_, -1 * y_, z_])
    RP1b = R0
    RP1c = np.array([-y_, x_, z_])
    RP2a = R0
    RP2b = np.array([-1 * x_, -1 * y_, z_])
    RP2c = np.array([y_, -1 * x_, z_])
    system_data = {
        "PARTS": {
            "1": {
                "a": {
                    "FRAME": RP1a,
                    "POINTS": {"A0": P1A0, "A1": P1A1, "A2": P1A2},
                    "TYPE": "plane",
                    "INTERACTIONS": ["P2a"],
                    "CONSTRAINTS_D": ["PERFECT"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "b": {
                    "FRAME": RP1b,
                    "POINTS": {"B0": P1B0, "B1": P1B1, "B2": P1B2},
                    "TYPE": "plane",
                    "INTERACTIONS": ["P2b"],
                    "CONSTRAINTS_D": ["NONE"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "c": {
                    "FRAME": RP1c,
                    "POINTS": {"C0": P1C0, "C1": P1C1, "C2": P1C2},
                    "TYPE": "plane",
                    "INTERACTIONS": ["P2c"],
                    "CONSTRAINTS_D": ["PERFECT"],
                    "CONSTRAINTS_G": ["SLIDING"],
                },
            },
            "2": {
                "a": {
                    "FRAME": RP2a,
                    "POINTS": {"A0": P2A0, "A1": P2A1, "A2": P2A2},
                    "TYPE": "plane",
                    "INTERACTIONS": ["P1a"],
                    "CONSTRAINTS_D": ["PERFECT"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "b": {
                    "FRAME": RP2b,
                    "POINTS": {"B0": P2B0, "B1": P2B1, "B2": P2B2},
                    "TYPE": "plane",
                    "INTERACTIONS": ["P1b"],
                    "CONSTRAINTS_D": ["NONE"],
                    "CONSTRAINTS_G": ["FLOATING"],
                },
                "c": {
                    "FRAME": RP2c,
                    "POINTS": {"C0": P2C0, "C1": P2C1, "C2": P2C2},
                    "TYPE": "plane",
                    "INTERACTIONS": ["P1c"],
                    "CONSTRAINTS_D": ["PERFECT"],
                    "CONSTRAINTS_G": ["SLIDING"],
                },
            },
        },
        "LOOPS": {
            "COMPATIBILITY": {
                "L0": "P1cC0 -> P2cC0 -> P2aA0 -> P1aA0",
                "L1": "P1cC0 -> P2cC0 -> P2bB0 -> P1bB0",
            }
        },
        "GLOBAL_CONSTRAINTS": "2D_NZ",
    }
    return system_data


SDA_compatibility_loops_expanded = "{'L0': 'D1c -> GP1cC0P2cC0 -> Di2c -> TP2cC0aA0 -> D2a -> GP2aA0P1aA0 -> Di1a -> TP1aA0cC0', 'L1': 'D1c -> GP1cC0P2cC0 -> Di2c -> TP2cC0bB0 -> D2b -> GP2bB0P1bB0 -> Di1b -> TP1bB0cC0'}"
CLH_get_compatibility_expression_from_FO_matrices = "[-gamma_g_1, 499*gamma_g_1/10 + v_g_1, -5*gamma_g_1 - u_g_1 + v_g_0, -gamma_d_4 + gamma_d_5 - gamma_g_2, -499*gamma_d_4/10 + 499*gamma_d_5/10 - 499*gamma_g_2/10 - v_g_2, -5*gamma_d_4 + 5*gamma_d_5 - 5*gamma_g_2 + u_d_4 + u_d_5 + u_g_2 + v_g_0 - 1/5]"


def get_assembly_data_processor_object(
    system_data: dict[str, Any] | None = None,
) -> otaf.AssemblyDataProcessor:
    """Initialize and prepare an AssemblyDataProcessor instance.

    Parameters
    ----------
    system_data : dict, optional
        The configuration dictionary containing assembly definitions.
        If None, default assembly data is generated.

    Returns
    -------
    otaf.AssemblyDataProcessor
        The data processor with generated expanded loops.
    """
    SDA = otaf.AssemblyDataProcessor(system_data)
    SDA.generate_expanded_loops()
    return SDA


def get_compatibility_loop_handling_object(
    SDA: otaf.AssemblyDataProcessor | None = None,
) -> otaf.CompatibilityLoopHandling:
    """Create a CompatibilityLoopHandling instance for assembly loops.

    Parameters
    ----------
    SDA : otaf.AssemblyDataProcessor, optional
        The assembly data processor. If None, a default instance
        is instantiated.

    Returns
    -------
    otaf.CompatibilityLoopHandling
        The configured compatibility loop handler object.
    """
    SDA = (
        SDA if SDA is not None else get_assembly_data_processor_object(get_assembly_data())
    )
    CLH = otaf.CompatibilityLoopHandling(SDA)
    return CLH


def get_interface_loop_handling_object(
    SDA: otaf.AssemblyDataProcessor | None = None,
    CLH: otaf.CompatibilityLoopHandling | None = None,
) -> otaf.InterfaceLoopHandling:
    """Create an InterfaceLoopHandling instance with set resolution.

    Parameters
    ----------
    SDA : otaf.AssemblyDataProcessor, optional
        The assembly data processor. If None, a default hierarchy
        is created along with `CLH`.
    CLH : otaf.CompatibilityLoopHandling, optional
        The compatibility loop handler. If None, a default hierarchy
        is created along with `SDA`.

    Returns
    -------
    otaf.InterfaceLoopHandling
        The initialized interface loop handler object.
    """
    if not SDA or not CLH:
        SDA = get_assembly_data_processor_object(get_assembly_data())
        CLH = get_compatibility_loop_handling_object(SDA)
    ILH = otaf.InterfaceLoopHandling(SDA, CLH, circle_resolution=20)
    return ILH


def get_system_of_constraints_assembly_model(
    X1: float = 99.8, X2: float = 100.0, X3: float = 10.0
) -> otaf.SystemOfConstraintsAssemblyModel:
    """Construct the complete system of constraints assembly model.

    Parameters
    ----------
    X1 : float, default 99.8
        Dimension parameter X1 passed to the assembly data.
    X2 : float, default 100.0
        Dimension parameter X2 passed to the assembly data.
    X3 : float, default 10.0
        Dimension parameter X3 passed to the assembly data.

    Returns
    -------
    otaf.SystemOfConstraintsAssemblyModel
        The initialized assembly model with embedded optimization
        variables.
    """
    SDA = get_assembly_data_processor_object(get_assembly_data(X1, X2, X3))
    CLH = get_compatibility_loop_handling_object(SDA)
    ILH = get_interface_loop_handling_object(SDA, CLH)
    compatibility_expressions = CLH.get_compatibility_expression_from_FO_matrices()
    interface_constraints = ILH.get_interface_loop_expressions()
    SOCAM = otaf.SystemOfConstraintsAssemblyModel(
        compatibility_expressions, interface_constraints
    )
    SOCAM.embedOptimizationVariable()
    return SOCAM


def get_distribution_params(
    tol: float = 0.31, capa: float = 1.0, X3: float = 10.0
) -> tuple[Any, list[sp.Symbol], np.ndarray, np.ndarray]:
    """Compute defect distribution parameters and variance vectors.

    Parameters
    ----------
    tol : float, default 0.31
        Tolerance limit value used to compute standard deviations.
    capa : float, default 1.0
        Process capability index factor.
    X3 : float, default 10.0
        Geometric dimension component used for rotation limits.

    Returns
    -------
    RandDeviationVect : otaf.distribution.ComposedDistribution
        The joint normal defect distribution model.
    deviation_symbols : list of sympy.Symbol
        The symbolic tracking parameters for spatial deviations.
    max_std_vect : np.ndarray
        A 1D array of calculated maximum standard deviations.
    np.ndarray
        A 1D array of zero-initialized mean parameter offsets.
    """
    deviation_symbols = list(sp.symbols("u_d_4 gamma_d_4 u_d_5 gamma_d_5"))
    sigma_translation = tol / (6 * capa)
    rotation_max = tol / X3
    sigma_rotation = 2 * rotation_max / (6 * capa)
    RandDeviationVect = otaf.distribution.get_composed_normal_defect_distribution(
        defect_names=deviation_symbols,
        sigma_dict={
            "alpha": sigma_rotation,
            "beta": sigma_rotation,
            "gamma": sigma_rotation,
            "u": sigma_translation,
            "v": sigma_translation,
            "w": sigma_translation,
        },
        mu_dict={"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "u": 0.0, "v": 0.0, "w": 0.0},
    )
    max_std_vect = np.array(
        [sigma_translation, sigma_rotation, sigma_translation, sigma_rotation]
    )
    return (RandDeviationVect, deviation_symbols, max_std_vect, np.array([0.0] * 4))


dim = 4
sample_multiplier = np.eye(dim)
no_tol = False


def eval_credal_set_constraints(
    x_std: np.ndarray,
    tol: float = 0.31,
    capa: float = 1.0,
    X3: float = 10.0,
) -> np.ndarray:
    """Evaluate the normalized credal set boundary conditions.

    Parameters
    ----------
    x_std : np.ndarray
        A 1D array containing standard deviation vector values.
    tol : float, default 0.31
        The baseline design tolerance.
    capa : float, default 1.0
        The process capability standard multiplier.
    X3 : float, default 10.0
        Geometric dimension value for plane evaluation.

    Returns
    -------
    np.ndarray
        An array containing the two normalized constraint scaling
        metrics.
    """
    target = tol / (6 * capa)
    constraint1 = (
        sigma_delta_3D_plane(X3 / 2, 0, x_std[0], 0, x_std[1]) - target
    ) / target
    constraint2 = (
        sigma_delta_3D_plane(X3 / 2, 0, x_std[2], 0, x_std[3]) - target
    ) / target
    return np.array([constraint1, constraint2])


def eval_scaled_credal_set_constraints(
    x_scaled: np.ndarray,
    max_std_vect: np.ndarray,
    tracker: Any | None = None,
    experiment_key: Any | None = None,
    tol: float = 0.31,
    capa: float = 1.0,
    X3: float = 10.0,
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
    tol : float, default 0.31
        The baseline design tolerance.
    capa : float, default 1.0
        The process capability standard multiplier.
    X3 : float, default 10.0
        Geometric dimension value for plane evaluation.

    Returns
    -------
    np.ndarray
        The calculated constraint evaluation bounds array.
    """
    x_real = x_scaled * max_std_vect
    constraint_array = eval_credal_set_constraints(x_real, tol=tol, capa=capa, X3=X3)
    if tracker:
        tracker.update_constraint_data(
            exp_key=experiment_key, x=x_scaled, constraints=constraint_array
        )
    return constraint_array


def get_scaled_credal_set_constraints_function(
    max_std_vect: np.ndarray,
    tracker: Any | None = None,
    experiment_key: Any | None = None,
    tol: float = 0.31,
    capa: float = 1.0,
    X3: float = 10.0,
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
    tol : float, default 0.31
        The baseline design tolerance.
    capa : float, default 1.0
        The process capability index multiplier.
    X3 : float, default 10.0
        Geometric dimension value for plane evaluation.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A single-argument function mapping `x_scaled` to its
        evaluated constraint array.
    """
    return lambda x_scaled: eval_scaled_credal_set_constraints(
        x_scaled, max_std_vect, tracker, experiment_key, tol=tol, capa=capa, X3=X3
    )
