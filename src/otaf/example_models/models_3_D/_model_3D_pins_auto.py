"""Two plates joined by pins, built by the automatic pipeline.

This is the assembly of ``model4`` (two plates, ``NX x NY`` pins through both,
a planar contact between the plates), described as an input dictionary and
built by :class:`otaf.AssemblyDataProcessor`, :class:`otaf.CompatibilityLoopHandling`
and :class:`otaf.InterfaceLoopHandling` instead of hand-written loop equations.
With the default dimensions and perfect plates, it gives the same slack as
``model4`` sample by sample, for every linearization strategy
(``tests/test_cylinder_interfaces.py``).

Modelling choices, as in ``model4``:

- the planar contact between the plates is perfect and sliding;
- the pins are perfect cylinders (only their diameters vary): a pin is crossed
  by the compatibility loops at a single point, so it has no position or
  orientation deviation;
- each pin is blocked axially and in rotation about its axis in the lower
  plate (part ``"0"``) only. Blocking it in both plates would also forbid the
  relative rotation of the plates about the pin axes, which is only limited by
  the radial clearances;
- the deviations of the hole diameters and of the pin diameters are variables
  (``d_d_<k>``) when ``diameter_deviation`` is true.
"""

from __future__ import annotations

__author__ = "Kramer84"
__all__ = [
    "get_assembly_data",
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
from otaf.tolerances import sigma_delta_cylindrical_feature


def get_assembly_data(
    NX: int = 2,
    NY: int = 2,
    Dext: float = 20.0,
    Dint: float = 19.8,
    EH: float = 50.0,
    LB: float = 25.0,
    hPlate: float = 30.0,
    hPin: float = 60.0,
    diameter_deviation: bool = True,
) -> dict[str, Any]:
    """Input dictionary of the assembly.

    Parameters
    ----------
    NX, NY : int, default 2
        Number of pins along x and y.
    Dext : float, default 20.0
        Nominal hole diameter.
    Dint : float, default 19.8
        Nominal pin diameter.
    EH : float, default 50.0
        Distance between neighbouring pin axes.
    LB : float, default 25.0
        Distance between the outer pin axes and the plate edges.
    hPlate : float, default 30.0
        Plate thickness (length of the holes).
    hPin : float, default 60.0
        Pin length.
    diameter_deviation : bool, default True
        Whether the hole and pin diameters are deviation variables.

    Returns
    -------
    dict
        The system data: plate ``"0"`` (below z = 0), plate ``"1"`` (above), and
        one part per pin (``"2"``, ``"3"``, ...), with one compatibility loop per pin.
    """
    LX = (NX - 1) * EH + 2 * LB
    LY = (NY - 1) * EH + 2 * LB
    x_, y_, z_ = np.eye(3)
    # Columns are the local axes; local x is the cylinder axis / plane normal.
    frame_down = np.array([z_, y_, -x_])  # local x = -z
    frame_up = np.array([-z_, y_, x_])  # local x = +z
    contour = {
        "A0": np.array([0.0, 0.0, 0.0]),
        "A1": np.array([LX, 0.0, 0.0]),
        "A2": np.array([LX, LY, 0.0]),
        "A3": np.array([0.0, LY, 0.0]),
    }

    def plate(frame, other):
        return {
            "a": {
                "FRAME": frame,
                "POINTS": {k: v.copy() for k, v in contour.items()},
                "TYPE": "plane",
                "INTERACTIONS": [f"P{other}a"],
                "CONSTRAINTS_D": ["PERFECT"],
                "CONSTRAINTS_G": ["SLIDING"],
            }
        }

    system_data = {
        "PARTS": {"0": plate(frame_down, "1"), "1": plate(frame_up, "0")},
        "LOOPS": {"COMPATIBILITY": {}},
        "GLOBAL_CONSTRAINTS": "3D",
    }
    labels = otaf.common.alphabet_generator()
    next(labels)  # "a" is the planar contact
    part_id = 2
    for i in range(NX):
        for j in range(NY):
            axis_point = np.array([LB + i * EH, LB + j * EH, 0.0])
            s = next(labels)
            system_data["PARTS"][str(part_id)] = {
                s: {
                    "FRAME": frame_down,
                    "ORIGIN": axis_point,
                    "TYPE": "cylinder",
                    "RADIUS": Dint / 2,
                    "EXTENT_LOCAL": {"x_max": hPin / 2, "x_min": -hPin / 2},
                    "INTERACTIONS": [f"P0{s}", f"P1{s}"],
                    "SURFACE_DIRECTION": "centrifugal",
                    "CONSTRAINTS_D": ["PERFECT"],
                    "DIAMETER_DEVIATION": diameter_deviation,
                }
            }
            system_data["PARTS"]["0"][s] = {
                "FRAME": frame_down,
                "ORIGIN": axis_point + np.array([0.0, 0.0, -hPlate / 2]),
                "TYPE": "cylinder",
                "RADIUS": Dext / 2,
                "EXTENT_LOCAL": {"x_max": hPlate / 2, "x_min": -hPlate / 2},
                "INTERACTIONS": [f"P{part_id}{s}"],
                "SURFACE_DIRECTION": "centripetal",
                "DIAMETER_DEVIATION": diameter_deviation,
                "BLOCK_ROTATIONS_G": "x",
                "BLOCK_TRANSLATIONS_G": "x",
            }
            system_data["PARTS"]["1"][s] = {
                "FRAME": frame_up,
                "ORIGIN": axis_point + np.array([0.0, 0.0, hPlate / 2]),
                "TYPE": "cylinder",
                "RADIUS": Dext / 2,
                "EXTENT_LOCAL": {"x_max": hPlate / 2, "x_min": -hPlate / 2},
                "INTERACTIONS": [f"P{part_id}{s}"],
                "SURFACE_DIRECTION": "centripetal",
                "DIAMETER_DEVIATION": diameter_deviation,
            }
            origin = lambda part, label: f"P{part}{label}{label.upper()}0"  # noqa: E731
            system_data["LOOPS"]["COMPATIBILITY"][f"L{part_id - 1}"] = (
                f"P0aA0 -> {origin(0, s)} -> {origin(part_id, s)} -> {origin(1, s)} -> P1aA0"
            )
            part_id += 1
    return system_data


def get_system_of_constraints_assembly_model(
    circle_resolution: int = 64,
    strategy: otaf.geometry.LinearizationStrategy | str = otaf.geometry.LinearizationStrategy.CIRCUMSCRIBED,
    **assembly_kwargs: Any,
) -> otaf.SystemOfConstraintsAssemblyModel:
    """Build the assembly model with the automatic pipeline.

    Parameters
    ----------
    circle_resolution : int, default 64
        Number of facets of the polygon replacing each circular clearance zone.
    strategy : otaf.geometry.LinearizationStrategy or str, optional
        Circumscribed (default), inscribed or mean polygon.
    **assembly_kwargs
        Passed to :func:`get_assembly_data`.

    Returns
    -------
    otaf.SystemOfConstraintsAssemblyModel
        The model, with the slack variable ``s`` embedded.
    """
    ADP = otaf.AssemblyDataProcessor(get_assembly_data(**assembly_kwargs))
    ADP.generate_expanded_loops()
    CLH = otaf.CompatibilityLoopHandling(ADP)
    compatibility_expressions = CLH.get_compatibility_expressions()
    ILH = otaf.InterfaceLoopHandling(
        ADP,
        CLH,
        circle_resolution=circle_resolution,
        linearization=otaf.geometry.LinearizationStrategy(strategy),
    )
    SOCAM = otaf.SystemOfConstraintsAssemblyModel(
        compatibility_expressions, ILH.get_interface_expressions()
    )
    SOCAM.embedOptimizationVariable()
    return SOCAM


def get_distribution_params(
    tol: float = 0.21,
    capa: float = 1.0,
    hPlate: float = 30.0,
    diameter_deviation: bool = True,
    NX: int = 2,
    NY: int = 2,
) -> tuple[Any, list[sp.Symbol], np.ndarray, np.ndarray]:
    """Independent centred normal deviations, with the standard deviations of ``model4``.

    Position deviations of the holes have ``tol / (6 capa)``, orientation
    deviations ``2 (tol / hPlate) / (6 capa)`` and diameter deviations
    ``tol / (6 capa)``.

    Returns
    -------
    RandDeviationVect : openturns.JointDistribution
        The joint distribution, in the order of the model's deviation symbols.
    deviation_symbols : list of sympy.Symbol
    std_vect : np.ndarray
    mu_vect : np.ndarray
    """
    sigma_e_pos = tol / (6 * capa)
    sigma_e_theta = 2 * (tol / hPlate) / (6 * capa)
    deviation_symbols = _deviation_symbols(NX * NY, diameter_deviation)
    sigma = {"v": sigma_e_pos, "w": sigma_e_pos, "beta": sigma_e_theta, "gamma": sigma_e_theta, "d": sigma_e_pos}
    std_vect = np.array([sigma[str(s).split("_")[0]] for s in deviation_symbols])
    mu_vect = np.zeros(len(deviation_symbols))
    RandDeviationVect = otaf.distribution.get_composed_normal_defect_distribution(
        deviation_symbols, mu_list=mu_vect.tolist(), sigma_list=std_vect.tolist()
    )
    return (RandDeviationVect, deviation_symbols, std_vect, mu_vect)


def _deviation_symbols(n_pins: int, diameter_deviation: bool) -> list[sp.Symbol]:
    """Deviation symbols of the model, in the order of its columns (without building it).

    Feature indices follow the order in which the compatibility loops create the
    deviation matrices: the loop of the first pin creates the lower hole (0), the
    upper hole (1) and the two planar faces (2, 3; perfect, so without variables);
    each further loop creates its lower and upper holes. Pins have no deviation
    matrix and get their diameter indices after all of them.
    ``tests/test_cylinder_interfaces.py`` checks this against the built model.
    """
    hole_indices = [0, 1] + [4 + k for k in range(2 * (n_pins - 1))]
    symbols = []
    for k in hole_indices:
        names = ["v", "w", "beta", "gamma"] + (["d"] if diameter_deviation else [])
        symbols += [sp.Symbol(f"{n}_d_{k}") for n in names]
    if diameter_deviation:
        first_pin = max(hole_indices) + 1
        symbols += [sp.Symbol(f"d_d_{first_pin + i}") for i in range(n_pins)]
    return symbols


def eval_credal_set_constraints(
    x_std: np.ndarray,
    tol: float = 0.21,
    capa: float = 1.0,
    hPlate: float = 30.0,
    diameter_deviation: bool = True,
    NX: int = 2,
    NY: int = 2,
) -> np.ndarray:
    """Capability constraints of the features, normalized by the target standard deviation.

    The credal set is the set of standard deviations for which the largest standard
    deviation of the measured defect over each feature equals ``tol / (6 capa)``:
    the constraint value is 0 on it, negative inside it.

    - Hole: the radial deviation of the axis, in the directions of the local y and z
      axes, at the hole ends (``hPlate / 2`` from its centre), with the diameter
      deviation as radial component (as in ``model4``):
      ``sqrt(s_d^2 + s_v^2 + (hPlate/2)^2 s_gamma^2)`` and
      ``sqrt(s_d^2 + s_w^2 + (hPlate/2)^2 s_beta^2)``.
    - Pin: its diameter deviation alone.

    Parameters
    ----------
    x_std : np.ndarray
        Standard deviations, in the order of the model's deviation symbols.

    Returns
    -------
    np.ndarray
        One value per hole (in symbol order), then one per pin.
    """
    names = [str(s) for s in _deviation_symbols(NX * NY, diameter_deviation)]
    std = dict(zip(names, np.asarray(x_std, dtype=float)))
    target = tol / (6 * capa)
    z = hPlate / 2
    holes = [n.split("_")[-1] for n in names if n.startswith("v_d_")]
    pins = [n.split("_")[-1] for n in names if n.startswith("d_d_") and f"v_d_{n.split('_')[-1]}" not in std]
    constraints = []
    for k in holes:
        sr = std.get(f"d_d_{k}", 0.0)
        worst = max(
            sigma_delta_cylindrical_feature(z, 0.0, sr, std[f"v_d_{k}"], std[f"beta_d_{k}"], std[f"w_d_{k}"], std[f"gamma_d_{k}"]),
            sigma_delta_cylindrical_feature(z, np.pi / 2, sr, std[f"v_d_{k}"], std[f"beta_d_{k}"], std[f"w_d_{k}"], std[f"gamma_d_{k}"]),
        )
        constraints.append((worst - target) / target)
    for k in pins:
        constraints.append((std[f"d_d_{k}"] - target) / target)
    return np.array(constraints)


def eval_scaled_credal_set_constraints(
    x_scaled: np.ndarray,
    max_std_vect: np.ndarray,
    tracker: Any | None = None,
    experiment_key: Any | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """:func:`eval_credal_set_constraints` of ``x_scaled * max_std_vect``.

    ``x_scaled`` holds the ratios to the largest admissible standard deviations
    (the normalized vector of the thesis, in ``[0, 1]``). ``kwargs`` are passed to
    :func:`eval_credal_set_constraints`. The values are logged in ``tracker`` if given.
    """
    constraint_array = eval_credal_set_constraints(np.asarray(x_scaled) * max_std_vect, **kwargs)
    if tracker:
        tracker.update_constraint_data(exp_key=experiment_key, x=x_scaled, constraints=constraint_array)
    return constraint_array


def get_scaled_credal_set_constraints_function(
    max_std_vect: np.ndarray,
    tracker: Any | None = None,
    experiment_key: Any | None = None,
    **kwargs: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """:func:`eval_scaled_credal_set_constraints` as a function of ``x_scaled`` only."""
    return lambda x_scaled: eval_scaled_credal_set_constraints(
        x_scaled, max_std_vect, tracker, experiment_key, **kwargs
    )


dim = len(_deviation_symbols(4, True))
sample_multiplier = np.eye(dim)
no_tol = False
