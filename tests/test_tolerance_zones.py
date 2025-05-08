import unittest
import numpy as np
from otaf.tolerances import MiSdofToleranceZones, points_within_bounds, bound_distance

s2 = np.sqrt(2)

class TestMiSdofToleranceZones(unittest.TestCase):

    def test_circular_zone(self):
        X = np.array([[0, 0], [s2/4, s2/4], [1, 1]])
        t = 1
        valid, dist_func = MiSdofToleranceZones.circular_zone(X, t)
        expected_valid = np.array([1, 1, 0])
        np.testing.assert_array_equal(valid, expected_valid)
        self.assertAlmostEqual(dist_func[1], t/2)

    def test_two_concentric_circles(self):
        X = np.array([[0, 0], [s2/4, s2/4], [1, 1]])
        t = 1
        valid, dist_func = MiSdofToleranceZones.two_concentric_circles(X, t)
        expected_valid = np.array([1, 1, 0])
        np.testing.assert_array_equal(valid, expected_valid)
        self.assertAlmostEqual(dist_func[1], t/2)

    def test_two_parallel_straight_lines(self):
        X = np.array([[0, 0], [0.5, 0.0], [0.4, 0.1], [1, 1]])
        t = 1
        L = 2
        valid, dist_func = MiSdofToleranceZones.two_parallel_straight_lines(X, t, L)
        expected_valid = np.array([1, 1, 1, 0])
        np.testing.assert_array_equal(valid, expected_valid)

    def test_spherical_zone(self):
        X = np.array([[0, 0, 0], [0.28, 0.28, 0.28], [1, 1, 1]])
        t = 1
        valid, dist_func = MiSdofToleranceZones.spherical_zone(X, t)
        expected_valid = np.array([1, 1, 0])
        np.testing.assert_array_equal(valid, expected_valid)

    def test_cylindrical_zone(self):
        X = np.array([[0, 0, 0, 0], [0.125, 0.2, 0.125, 0.2], [1, 1, 1, 1]])
        t = 1
        L = 2
        valid, dist_func = MiSdofToleranceZones.cylindrical_zone(X, t, L)
        expected_valid = np.array([1, 1, 0])
        np.testing.assert_array_equal(valid, expected_valid)

    def test_parallelepiped_zone(self):
        X = np.array([[0, 0, 0, 0], [0.125, 0.2, 0.125, 0.2], [1, 1, 1, 1]])
        t1, t2, L = 1, 1, 2
        valid, dist_funcs = MiSdofToleranceZones.parallelepiped_zone(X, t1, t2, L)
        expected_valid = np.array([1, 1, 0])
        np.testing.assert_array_equal(valid, expected_valid)

    def test_two_parallel_planes(self):
        X = np.array([[0, 0, 0], [0.3, 0.1, 0.1], [0.5, 0, 0], [1, 1, 1]])
        t = 1
        Lx, Ly = 2, 2
        valid, dist_func = MiSdofToleranceZones.two_parallel_planes(X, t, Lx, Ly)
        expected_valid = np.array([1, 1, 1, 0])
        np.testing.assert_array_equal(valid, expected_valid)

class TestUtilityFunctions(unittest.TestCase):

    def test_bound_distance(self):
        self.assertEqual(bound_distance(5, 3, 7), 2)
        self.assertEqual(bound_distance(5, 5, 7), 0)
        self.assertEqual(bound_distance(8, 3, 7), 1)

    def test_points_within_bounds(self):
        sample = np.array([[0, 0], [0.5, 0.5], [1, 1], [0.5,-0.5], [1, 0], [0,1]])
        bounds = np.array([[-0.5, -0.5], [0.5, 0.5]])
        expected_valid = np.array([1, 1, 0, 1, 0, 0])
        valid = points_within_bounds(sample, bounds)
        np.testing.assert_array_equal(valid, expected_valid)

if __name__ == "__main__":
    unittest.main()
