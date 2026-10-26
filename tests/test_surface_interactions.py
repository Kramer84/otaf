"""Facing-point pairing in SurfaceInteractionManager.

Contact constraints are generated between paired points of two interacting
surfaces. Each pair must sit at the same location along the contact (the ends
of the overlap of the two surfaces), otherwise lever arms are wrong or
constraints are placed where no contact can occur.
"""

import logging

import numpy as np
import pytest

import otaf
from otaf._assembly_modeling._interface_constraints._surfaceInteractionManager import (
    SurfaceInteractionManager,
)


def _paired_coordinates(data, axis):
    """Return the set of paired point coordinates ``(x_a, x_b)`` along ``axis``, origin pair excluded."""
    logging.disable(logging.WARNING)
    try:
        adp = otaf.AssemblyDataProcessor(data)
        manager = SurfaceInteractionManager(adp)
        manager.get_facing_point_dictionary()
    finally:
        logging.disable(logging.NOTSET)

    (part_a, surf_a), (part_b, surf_b) = [(p, next(iter(s))) for p, s in data["PARTS"].items()]
    coords = {
        part_a: {k: float(v[axis]) for k, v in adp["PARTS"][part_a][surf_a]["POINTS"].items()},
        part_b: {k: float(v[axis]) for k, v in adp["PARTS"][part_b][surf_b]["POINTS"].items()},
    }
    origin_pair = tuple(sorted((coords[part_a][f"{surf_a.upper()}0"], coords[part_b][f"{surf_b.upper()}0"])))
    pairs = set()
    for part, surf, other in ((part_a, surf_a, part_b), (part_b, surf_b, part_a)):
        for point, facing in manager.facingPointDict[part][surf].items():
            for label in facing:
                other_point = label[len(other) + 1 :]  # "<part><surf><POINT>"
                pair = tuple(sorted((round(coords[part][point], 9), round(coords[other][other_point], 9))))
                if pair != tuple(round(x, 9) for x in origin_pair):
                    pairs.add(pair)
    return pairs


# --- Cylinders: axial overlap ----------------------------------------------------------


def _axis_frame(sign):
    return np.array([[sign, 0.0, 0.0], [0.0, sign, 0.0], [0.0, 0.0, 1.0]])


def _two_cylinders(origin_pin, extent_pin, origin_hole, extent_hole, hole_axis_sign):
    """Pin (part 1, axis +x) and hole (part 2, axis ±x), coaxial on the global x axis."""

    def cylinder(origin, extent, sign, radius, other):
        return {
            "TYPE": "cylinder",
            "FRAME": _axis_frame(sign),
            "ORIGIN": np.array([origin, 0.0, 0.0]),
            "RADIUS": radius,
            "EXTENT_LOCAL": {"x_min": extent[0], "x_max": extent[1]},
            "INTERACTIONS": [other],
        }

    return {
        "PARTS": {
            "1": {"a": cylinder(origin_pin, extent_pin, 1.0, 5.0, "P2a")},
            "2": {"a": cylinder(origin_hole, extent_hole, hole_axis_sign, 5.1, "P1a")},
        },
        "LOOPS": {"COMPATIBILITY": {}},
        "GLOBAL_CONSTRAINTS": "3D",
    }


# The pin always spans global x in [0, 30]; its origin is at x = 10 unless stated.
CYLINDER_CASES = {
    # name: (pin origin, pin local extent, hole origin, hole local extent, hole axis sign, overlap)
    "same axis, hole longer": (10, (-10, 20), 1, (-6, 34), 1, (0, 30)),
    "same axis, hole shorter": (10, (-10, 20), 1, (4, 24), 1, (5, 25)),
    "same axis, staggered": (12, (-12, 18), 1, (9, 39), 1, (10, 30)),
    "opposed axes, hole longer": (10, (-10, 20), 1, (-34, 6), -1, (0, 30)),
    "opposed axes, hole shorter": (10, (-10, 20), 1, (-24, -4), -1, (5, 25)),
    "opposed axes, staggered": (12, (-12, 18), 1, (-39, -9), -1, (10, 30)),
    "opposed axes, staggered other side": (10, (-10, 20), 1, (-19, 11), -1, (0, 20)),
}

# An overlap end that lands on an already existing point (here the pin's origin at x = 10)
# is skipped as a duplicate, and the pairing is skipped with it.
OVERLAP_END_ON_EXISTING_POINT = {
    "same axis, overlap end on pin origin": (10, (-10, 20), 1, (9, 39), 1, (10, 30)),
    "opposed axes, overlap end on pin origin": (10, (-10, 20), 1, (-39, -9), -1, (10, 30)),
}


@pytest.mark.parametrize("case", CYLINDER_CASES)
def test_cylinder_contact_points_are_paired_at_overlap_ends(case):
    *geometry, (start, end) = CYLINDER_CASES[case]
    assert _paired_coordinates(_two_cylinders(*geometry), axis=0) == {(start, start), (end, end)}


@pytest.mark.xfail(strict=True, reason="open bug: overlap end coinciding with an existing point is not paired")
@pytest.mark.parametrize("case", OVERLAP_END_ON_EXISTING_POINT)
def test_cylinder_overlap_end_on_existing_point_is_still_paired(case):
    *geometry, (start, end) = OVERLAP_END_ON_EXISTING_POINT[case]
    assert _paired_coordinates(_two_cylinders(*geometry), axis=0) == {(start, start), (end, end)}


# --- Planes: overlap of facing faces ------------------------------------------------------


def _two_vertical_faces(height_male, height_female, clearance=0.1):
    """Two facing vertical faces in 2D, both starting at y = 0 (as P1b/P2b of model2)."""

    def face(x, height, normal, other):
        return {
            "TYPE": "plane",
            "FRAME": np.array([[normal, 0.0, 0.0], [0.0, normal, 0.0], [0.0, 0.0, 1.0]]),
            "POINTS": {
                "B0": np.array([x, height / 2, 0.0]),
                "B1": np.array([x, 0.0, 0.0]),
                "B2": np.array([x, height, 0.0]),
            },
            "INTERACTIONS": [other],
        }

    return {
        "PARTS": {
            "1": {"b": face(35.0, height_male, 1.0, "P2b")},
            "2": {"b": face(35.0 + clearance, height_female, -1.0, "P1b")},
        },
        "LOOPS": {"COMPATIBILITY": {}},
        "GLOBAL_CONSTRAINTS": "2D_NZ",
    }


def test_plane_contact_points_paired_when_faces_are_equal():
    assert _paired_coordinates(_two_vertical_faces(10.0, 10.0), axis=1) == {(0.0, 0.0), (10.0, 10.0)}


@pytest.mark.xfail(
    strict=True,
    reason="open bug: plane points are paired by proximity (0.25 mm) instead of projecting the "
    "overlap of the two faces; the female top (y=10.2) is paired with the male top (y=10)",
)
def test_plane_contact_points_are_paired_at_overlap_ends():
    assert _paired_coordinates(_two_vertical_faces(10.0, 10.2), axis=1) == {(0.0, 0.0), (10.0, 10.0)}


@pytest.mark.xfail(
    strict=True,
    reason="open bug: faces whose ends differ by more than the 0.25 mm search radius lose the contact",
)
def test_plane_contact_survives_larger_height_difference():
    assert _paired_coordinates(_two_vertical_faces(10.0, 11.0), axis=1) == {(0.0, 0.0), (10.0, 10.0)}
