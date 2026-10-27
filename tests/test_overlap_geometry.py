import numpy as np
import pytest

from otaf.geometry import interval_overlap, planar_overlap_vertices


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ((0.0, 10.0), (0.0, 10.2), (0.0, 10.0)),
        ((10.0, 0.0), (5.0, 20.0), (5.0, 10.0)),  # bounds in any order
        ((0.0, 10.0), (2.0, 3.0), (2.0, 3.0)),
        ((0.0, 1.0), (1.0, 2.0), None),  # touching only
        ((0.0, 1.0), (2.0, 3.0), None),
    ],
)
def test_interval_overlap(a, b, expected):
    assert interval_overlap(a, b) == expected


def _rectangle(x0, y0, w, h):
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], dtype=float)


def _as_set(vertices):
    return {tuple(np.round(v, 9)) for v in vertices}


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (_rectangle(0, 0, 10, 6), _rectangle(4, 2, 10, 10), _rectangle(4, 2, 6, 4)),
        (_rectangle(0, 0, 10, 10), _rectangle(2, 2, 3, 3), _rectangle(2, 2, 3, 3)),
        (_rectangle(0, 0, 4, 2), _rectangle(0, 0, 4, 2), _rectangle(0, 0, 4, 2)),
        (np.array([[0, 0], [6, 0], [0, 6.0]]), _rectangle(1, 1, 10, 10), np.array([[1, 1], [5, 1], [1, 5.0]])),
    ],
)
def test_planar_overlap_of_polygons(a, b, expected):
    assert _as_set(planar_overlap_vertices(a, b)) == _as_set(expected)
    assert _as_set(planar_overlap_vertices(b, a)) == _as_set(expected)


def test_planar_overlap_ignores_interior_points():
    a = np.vstack([_rectangle(0, 0, 10, 10), [[5.0, 5.0], [2.0, 7.0]]])
    assert _as_set(planar_overlap_vertices(a, _rectangle(5, 5, 10, 10))) == _as_set(_rectangle(5, 5, 5, 5))


def test_planar_overlap_of_collinear_faces_is_a_segment():
    male = np.array([[5.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    female = np.array([[5.1, 0.0], [0.0, 0.0], [10.2, 0.0]])
    np.testing.assert_array_equal(planar_overlap_vertices(male, female), [[0.0, 0.0], [10.0, 0.0]])
    # Vertical segments, exact values (no round-off from the projection).
    np.testing.assert_array_equal(
        planar_overlap_vertices(male[:, ::-1], female[:, ::-1]), [[0.0, 0.0], [0.0, 10.0]]
    )


@pytest.mark.parametrize(
    "a, b",
    [
        (_rectangle(0, 0, 10, 10), _rectangle(10, 0, 5, 5)),  # shared edge only
        (_rectangle(0, 0, 1, 1), _rectangle(5, 5, 1, 1)),
        (np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[2.0, 0.0], [3.0, 0.0]])),
    ],
)
def test_planar_overlap_empty(a, b):
    assert planar_overlap_vertices(a, b).shape == (0, 2)


def test_planar_overlap_rejects_segment_against_polygon():
    with pytest.raises(ValueError):
        planar_overlap_vertices(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), _rectangle(0, 0, 3, 3))
