"""Contact-point pairing in SurfaceInteractionManager.

Contact constraints are generated between paired points of two interacting
surfaces. Each pair must sit at the same location along the contact, at the
vertices (ends) of the overlap of the two surfaces: otherwise lever arms are
wrong, or constraints are placed where no contact can occur, or contacts are
lost.
"""

import logging

import numpy as np
import pytest

import otaf
from otaf._assembly_modeling._interface_constraints._surfaceInteractionManager import (
    SurfaceInteractionManager,
)


def _contact_pairs(data, axes):
    """Coordinates ``((a...), (b...))`` along `axes` of every contact pair (one direction each)."""
    logging.disable(logging.WARNING)
    try:
        adp = otaf.AssemblyDataProcessor(data)
        manager = SurfaceInteractionManager(adp)
        manager.get_facing_point_dictionary()
    finally:
        logging.disable(logging.NOTSET)

    (part_a, surf_a), (part_b, surf_b) = [(p, next(iter(s))) for p, s in data["PARTS"].items()]
    points = {
        (part_a, surf_a): adp["PARTS"][part_a][surf_a]["POINTS"],
        (part_b, surf_b): adp["PARTS"][part_b][surf_b]["POINTS"],
    }
    prefix = f"GP{part_a}{surf_a}"
    pairs = set()
    for gap in manager.contactGapNames:
        if not gap.startswith(prefix):
            continue
        label_a, rest = gap[len(prefix) :].split("P")
        label_b = rest[len(part_b) + len(surf_b) :]
        coords_a = tuple(round(float(points[(part_a, surf_a)][label_a][i]), 9) for i in axes)
        coords_b = tuple(round(float(points[(part_b, surf_b)][label_b][i]), 9) for i in axes)
        pairs.add((coords_a, coords_b))
    return pairs


def _coincident(*locations):
    return {((*loc,), (*loc,)) for loc in locations}


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


# The pin always spans global x in [0, 30].
CYLINDER_CASES = {
    # name: (pin origin, pin local extent, hole origin, hole local extent, hole axis sign, overlap)
    "same axis, hole longer": (10, (-10, 20), 1, (-6, 34), 1, (0, 30)),
    "same axis, hole shorter": (10, (-10, 20), 1, (4, 24), 1, (5, 25)),
    "same axis, staggered": (12, (-12, 18), 1, (9, 39), 1, (10, 30)),
    "opposed axes, hole longer": (10, (-10, 20), 1, (-34, 6), -1, (0, 30)),
    "opposed axes, hole shorter": (10, (-10, 20), 1, (-24, -4), -1, (5, 25)),
    "opposed axes, staggered": (12, (-12, 18), 1, (-39, -9), -1, (10, 30)),
    "opposed axes, staggered other side": (10, (-10, 20), 1, (-19, 11), -1, (0, 20)),
    # Overlap ends landing on an existing point (the pin origin) used to lose the contact.
    "same axis, overlap end on pin origin": (10, (-10, 20), 1, (9, 39), 1, (10, 30)),
    "opposed axes, overlap end on pin origin": (10, (-10, 20), 1, (-39, -9), -1, (10, 30)),
    # Origins at a cylinder end used to be rejected as duplicate coordinates.
    "pin origin at its end": (0, (0, 30), 1, (-6, 34), 1, (0, 30)),
    "both origins at the shared end": (0, (0, 30), 0, (0, 30), 1, (0, 30)),
    "hole origin at its end, opposed axes": (10, (-10, 20), 30, (0, 30), -1, (0, 30)),
}


@pytest.mark.parametrize("case", CYLINDER_CASES)
def test_cylinder_contact_points_are_paired_at_overlap_ends(case):
    *geometry, (start, end) = CYLINDER_CASES[case]
    assert _contact_pairs(_two_cylinders(*geometry), axes=(0,)) == _coincident((start,), (end,))


def test_cylinders_without_axial_overlap_raise():
    with pytest.raises(otaf.exceptions.NoAxialOverlapException):
        _contact_pairs(_two_cylinders(10, (-10, 20), 50, (-5, 5), 1), axes=(0,))


# --- Planes: overlap of facing faces ------------------------------------------------------


def _two_vertical_faces(male_points_y, female_points_y, clearance=0.1):
    """Two facing vertical faces in 2D (as P1b/P2b of model2); the first point is the origin."""

    def face(x, ys, normal, other):
        return {
            "TYPE": "plane",
            "FRAME": np.array([[normal, 0.0, 0.0], [0.0, normal, 0.0], [0.0, 0.0, 1.0]]),
            "POINTS": {f"B{i}": np.array([x, y, 0.0]) for i, y in enumerate(ys)},
            "INTERACTIONS": [other],
        }

    return {
        "PARTS": {
            "1": {"b": face(35.0, male_points_y, 1.0, "P2b")},
            "2": {"b": face(35.0 + clearance, female_points_y, -1.0, "P1b")},
        },
        "LOOPS": {"COMPATIBILITY": {}},
        "GLOBAL_CONSTRAINTS": "2D_NZ",
    }


PLANE_2D_CASES = {
    # name: (male point heights, female point heights, overlap); first height is the origin
    "equal faces": ((5.0, 0.0, 10.0), (5.0, 0.0, 10.0), (0.0, 10.0)),
    "female 0.2 higher (model2)": ((5.0, 0.0, 10.0), (5.1, 0.0, 10.2), (0.0, 10.0)),
    "female 1 mm higher (beyond the old 0.25 mm radius)": ((5.0, 0.0, 10.0), (5.5, 0.0, 11.0), (0.0, 10.0)),
    "female shorter and shifted": ((5.0, 0.0, 10.0), (6.0, 3.0, 9.0), (3.0, 9.0)),
    "staggered faces": ((5.0, 0.0, 10.0), (9.0, 4.0, 14.0), (4.0, 10.0)),
    "origins at the bottom end": ((0.0, 10.0), (0.0, 10.2), (0.0, 10.0)),
}


@pytest.mark.parametrize("case", PLANE_2D_CASES)
def test_plane_contact_points_are_paired_at_overlap_ends(case):
    male, female, (start, end) = PLANE_2D_CASES[case]
    assert _contact_pairs(_two_vertical_faces(male, female), axes=(1,)) == _coincident((start,), (end,))


def test_plane_contact_points_lie_on_their_own_face():
    data = _two_vertical_faces((5.0, 0.0, 10.0), (5.1, 0.0, 10.2))
    for (y_male,), (y_female,) in _contact_pairs(data, axes=(1,)):
        assert 0.0 <= y_male <= 10.0 and 0.0 <= y_female <= 10.2


def test_planes_without_overlap_raise():
    with pytest.raises(otaf.exceptions.NoPlanarOverlapException):
        _contact_pairs(_two_vertical_faces((5.0, 0.0, 10.0), (25.0, 20.0, 30.0)), axes=(1,))


def _two_rectangular_faces(rect_a, rect_b, clearance=0.1):
    """Facing rectangles in 3D, normal along x; ``rect = (y0, z0, width, height)``, origin at the center."""

    def face(x, rect, normal, other):
        y0, z0, w, h = rect
        corners = [(y0, z0), (y0 + w, z0), (y0 + w, z0 + h), (y0, z0 + h)]
        points = {"A0": np.array([x, y0 + w / 2, z0 + h / 2])}
        points.update({f"A{i + 1}": np.array([x, y, z]) for i, (y, z) in enumerate(corners)})
        return {
            "TYPE": "plane",
            "FRAME": np.array([[normal, 0.0, 0.0], [0.0, normal, 0.0], [0.0, 0.0, 1.0]]),
            "POINTS": points,
            "INTERACTIONS": [other],
        }

    return {
        "PARTS": {
            "1": {"a": face(0.0, rect_a, 1.0, "P2a")},
            "2": {"a": face(clearance, rect_b, -1.0, "P1a")},
        },
        "LOOPS": {"COMPATIBILITY": {}},
        "GLOBAL_CONSTRAINTS": "3D",
    }


def test_3d_plane_contact_points_are_the_overlap_rectangle():
    pairs = _contact_pairs(_two_rectangular_faces((0, 0, 10, 6), (4, 2, 10, 10)), axes=(1, 2))
    assert pairs == _coincident((4.0, 2.0), (10.0, 2.0), (10.0, 6.0), (4.0, 6.0))


def test_3d_plane_contained_face():
    pairs = _contact_pairs(_two_rectangular_faces((0, 0, 10, 10), (2, 3, 4, 5)), axes=(1, 2))
    assert pairs == _coincident((2.0, 3.0), (6.0, 3.0), (6.0, 8.0), (2.0, 8.0))


def test_plane_cylinder_interaction_is_not_silently_ignored():
    data = _two_vertical_faces((5.0, 0.0, 10.0), (5.0, 0.0, 10.0))
    data["PARTS"]["2"]["b"] = {
        "TYPE": "cylinder",
        "FRAME": _axis_frame(1.0),
        "ORIGIN": np.array([40.0, 5.0, 0.0]),
        "RADIUS": 5.0,
        "EXTENT_LOCAL": {"x_min": -1.0, "x_max": 1.0},
        "INTERACTIONS": ["P1b"],
    }
    with pytest.raises(NotImplementedError):
        _contact_pairs(data, axes=(1,))
