import unittest
import numpy as np
import sympy as sp
import logging
import otaf
from otaf import (
    I4,
    J4,
    ROTATION_MATRIX_180_Z_AXIS,
    GET_I4,
    inverse_mstring,
    is_affine_transformation_matrix,
)


class TestMatrixOperations(unittest.TestCase):
    def test_I4(self):
        i4 = I4()
        self.assertEqual(i4.TYPE, "I4")
        result = i4.get_matrix()
        expected = GET_I4()
        self.assertTrue(sp.Matrix(result).equals(expected))

    def test_J4(self):
        j4 = J4()
        self.assertEqual(j4.TYPE, "J4")
        result = j4.get_matrix()
        expected = ROTATION_MATRIX_180_Z_AXIS()
        self.assertTrue(sp.Matrix(result).equals(expected))

    def test_ROTATION_MATRIX_180_Z_AXIS(self):
        result = ROTATION_MATRIX_180_Z_AXIS()
        expected = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertTrue(sp.Matrix(result).equals(sp.Matrix(expected)))

    def test_GET_I4(self):
        result = GET_I4()
        expected = np.identity(4)
        self.assertTrue(sp.Matrix(result).equals(sp.Matrix(expected)))

    def test_inverse_mstring(self):
        self.assertEqual(
            inverse_mstring("TP1234abcdABCD5678efghEFGH3456"),
            "TP1234efghEFGH3456abcdABCD5678",
        )
        self.assertEqual(
            inverse_mstring("GP1234abcdABCD5678P9012efghEFGH3456"),
            "GP9012efghEFGH3456P1234abcdABCD5678",
        )
        with self.assertRaises(ValueError):
            inverse_mstring("TP1234abcd5678")

    def test_is_affine_transformation_matrix(self):
        valid_matrix = np.array(
            [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        )
        invalid_matrix = np.array(
            [[1, 1, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        )
        self.assertTrue(is_affine_transformation_matrix(valid_matrix))
        self.assertFalse(is_affine_transformation_matrix(invalid_matrix))


class TestTransformationMatrix(unittest.TestCase):
    def test_change_of_basis_matrix(self):
        tr = otaf.TransformationMatrix()

        initial = np.identity(4)
        final = np.array([[1, 0, 0, 2], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

        change_matrix = tr.get_change_of_basis_matrix(initial, final)

        # Check if matrix is 4x4
        self.assertEqual(change_matrix.shape, (4, 4))

        # Validate the translation part (4th column)
        np.testing.assert_almost_equal(
            change_matrix[:, 3], np.atleast_2d(np.array([2, 1, 1, 1])).T
        )

    def test_get_matrix(self):
        tr = otaf.TransformationMatrix()

        initial = np.identity(4)
        final = np.array([[1, 0, 0, 2], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

        tr._initial = initial
        tr._final = final

        matrix = tr.get_matrix()

        # Check if matrix is 4x4
        self.assertEqual(matrix.shape, (4, 4))

        # Validate the translation part (4th column)
        np.testing.assert_almost_equal(
            matrix[:, 3], np.atleast_2d(np.array([2, 1, 1, 1])).T
        )

    def test_get_inverse(self):
        tr = otaf.TransformationMatrix()

        initial = np.identity(4)
        final = np.array([[1, 0, 0, 2], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

        tr._initial = initial
        tr._final = final

        inverse_matrix = tr.get_inverse().get_matrix()

        # Check if matrix is 4x4
        self.assertEqual(inverse_matrix.shape, (4, 4))

        # Validate the translation part (4th column)
        np.testing.assert_almost_equal(
            inverse_matrix[:, 3], np.atleast_2d(np.array([-2, -1, -1, 1])).T
        )

    def test_non_global_orientation_same(self):
        tr = otaf.TransformationMatrix()

        initial = np.array([[0, 1, 0, 3], [1, 0, 0, 2], [0, 0, -1, 1], [0, 0, 0, 1]])

        final = initial.copy()
        final[:, 3] = [5, 3, 0, 1]

        change_matrix = tr.get_change_of_basis_matrix(initial, final)

        # Validate the translation part (4th column)
        np.testing.assert_almost_equal(
            change_matrix[:, 3], np.atleast_2d(np.array([1, 2, 1, 1])).T
        )

    def test_different_rotation(self):
        tr = otaf.TransformationMatrix()

        initial = np.array([[0, 1, 0, 3], [1, 0, 0, 2], [0, 0, -1, 1], [0, 0, 0, 1]])

        final = np.array([[0, 0, -1, 5], [0, 1, 0, 3], [1, 0, 0, 0], [0, 0, 0, 1]])

        change_matrix = tr.get_change_of_basis_matrix(initial, final)

        # Validate the translation part (4th column)
        np.testing.assert_almost_equal(
            change_matrix[:, 3], np.atleast_2d(np.array([1, 2, 1, 1])).T
        )

        # Validate the rotation part (3x3 sub-matrix)
        expected_rot = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        np.testing.assert_almost_equal(change_matrix[:3, :3], expected_rot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
