import unittest
import numpy as np
from scipy.spatial import ConvexHull
import otaf

ar = np.array


class TestGeometryFunctions(unittest.TestCase):
    def test_points_in_cylinder(self):
        self.assertTrue(
            otaf.points_in_cylinder(ar([0, 0, 0]), ar([0, 0, 1]), 1, ar([0, 0, 0.5]))
        )
        self.assertFalse(
            otaf.points_in_cylinder(ar([0, 0, 0]), ar([0, 0, 1]), 1, ar([0, 1.1, 0.5]))
        )

    def test_points_in_cylinder_vect(self):
        self.assertTrue(
            otaf.points_in_cylinder_vect(
                ar([0, 0, 0]),
                ar([0, 0, 1]),
                1,
                ar([[0, 0, 0.5], [-0.3, -0.3, 0.5], [0.3, 0.3, 0.5]]),
            ).all()
        )
        self.assertFalse(
            otaf.points_in_cylinder_vect(
                ar([0, 0, 0]),
                ar([0, 0, 1]),
                1,
                ar([[0, 0, -0.1], [-0.7, 0.9, 0.5], [1.1, 0, 0.5]]),
            ).any()
        )

    def test_euclidean_distance(self):
        self.assertAlmostEqual(otaf.euclidean_distance(ar([0, 0]), ar([3, 4])), 5)

    def test_angle_between_vectors(self):
        self.assertAlmostEqual(
            otaf.angle_between_vectors(ar([1, 0]), ar([0, 1])), np.pi / 2
        )

    def test_point_in_hull(self):
        hull = ar([[0, 0], [1, 0], [0, 1]])
        self.assertTrue(otaf.point_in_hull(ar([0.5, 0.5]), hull))
        self.assertFalse(otaf.point_in_hull(ar([1.5, 1.5]), hull))

    def test_line_plane_intersection(self):
        plane_normal = ar([0, 0, 1])
        plane_point = ar([0, 0, 1])
        line_point = ar([0, 0, 0])
        line_direction = ar([0, 0, 1])
        np.testing.assert_almost_equal(
            otaf.line_plane_intersection(
                plane_normal, plane_point, line_point, line_direction
            ),
            ar([0, 0, 1]),
        )

    def test_closest_point_on_plane(self):
        np.testing.assert_almost_equal(
            otaf.closest_point_on_plane(ar([0, 0, 1]), ar([0, 0, 1]), ar([1, 1, 1])),
            ar([1, 1, 1]),
        )

    def test_closest_point_on_plane(self):
        np.testing.assert_almost_equal(
            otaf.closest_point_on_plane(ar([0, 0, 1]), ar([0, 0, 1]), ar([1, 1, 1])),
            ar([1, 1, 1]),
        )

    def test_closest_point_on_line(self):
        np.testing.assert_almost_equal(
            otaf.closest_point_on_line(ar([0, 0]), ar([1, 1]), ar([0, 2])), ar([1, 1])
        )

    def test_boxes_overlap(self):
        self.assertTrue(
            otaf.boxes_overlap(
                ar([0, 0, 0]), ar([1, 1, 1]), ar([0.5, 0.5, 0.5]), ar([1.5, 1.5, 1.5])
            )
        )
        self.assertFalse(
            otaf.boxes_overlap(
                ar([0, 0, 0]), ar([1, 1, 1]), ar([2, 2, 2]), ar([3, 3, 3])
            )
        )

    def test_point_plane_distance(self):
        self.assertAlmostEqual(
            otaf.point_plane_distance(ar([0, 0, 1]), ar([0, 0, 0]), ar([0, 0, 1])), 1
        )

    def test_point_to_segment_distance(self):
        self.assertAlmostEqual(
            otaf.point_to_segment_distance(ar([0, 2]), ar([0, 0]), ar([1, 1])),
            np.sqrt(2),
        )

    def test_are_planes_coincident(self):
        self.assertTrue(
            otaf.are_planes_coincident(
                ar([1, 0, 0]), ar([0, 0, 0]), ar([1, 0, 0]), ar([0, 0, 0])
            )
        )
        self.assertFalse(
            otaf.are_planes_coincident(
                ar([1, 0, 0]), ar([0, 0, 0]), ar([0, 1, 0]), ar([0, 0, 0])
            )
        )

    def test_are_planes_parallel(self):
        self.assertTrue(otaf.are_planes_parallel(ar([1, 0, 0]), ar([1, 0, 0])))
        self.assertFalse(otaf.are_planes_parallel(ar([1, 0, 0]), ar([0, 1, 0])))

    def test_are_planes_perpendicular(self):
        self.assertTrue(otaf.are_planes_perpendicular(ar([1, 0, 0]), ar([0, 1, 0])))
        self.assertFalse(otaf.are_planes_perpendicular(ar([1, 0, 0]), ar([1, 0, 0])))

    def test_angle_between_planes(self):
        self.assertAlmostEqual(
            otaf.angle_between_planes(ar([1, 0, 0]), ar([0, 1, 0])), np.pi / 2
        )

    def test_distance_between_planes(self):
        self.assertAlmostEqual(
            otaf.distance_between_planes(
                ar([0, 0, 1]), ar([0, 0, 0]), ar([0, 0, 1]), ar([0, 0, 1])
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
