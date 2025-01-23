import unittest
import numpy as np
from otaf.tolerances import (
    MiSdofToleranceZones,
    FeatureLevelStatisticalConstraint,
    ComposedAssemblyLevelStatisticalConstraint,
    NormalizedAssemblyLevelConstraint,
)


class TestPracticalUsage(unittest.TestCase):
    def setUp(self):
        # Initialize parameters
        self.tol = 0.01
        self.hPlate = 0.02
        self.sigma_e_pos = 0.005
        self.sigma_e_theta = 0.002
        self.midof_funcs = MiSdofToleranceZones()

        # Feature-level constraints
        self.feature_constraint_list = []
        for _ in range(8):  # 8 identical features
            fconst = FeatureLevelStatisticalConstraint(
                self.midof_funcs.cylindrical_zone,
                mif_args=(self.tol, self.hPlate),
                n_dof=4,
                n_sample=10000,  # Reduced for faster tests
                target="std",
                target_val=self.sigma_e_pos * np.sqrt(1 - (2 / np.pi)),
                isNormal=True,
                normalizeOutput=True,
            )
            self.feature_constraint_list.append(fconst)

        # Composed assembly constraint
        self.composed_assembly_constraint = ComposedAssemblyLevelStatisticalConstraint(
            self.feature_constraint_list
        )

        # Parameter bounds for a single feature
        self.param_bounds_one_feature = [
            [0.0, 0.0], [1e-8, self.sigma_e_pos],  # u mean, std
            [0.0, 0.0], [1e-8, self.sigma_e_pos],  # v mean, std
            [0.0, 0.0], [1e-8, self.sigma_e_theta],  # alpha mean, std
            [0.0, 0.0], [1e-8, self.sigma_e_theta],  # beta mean, std
        ]
        self.param_bounds = [self.param_bounds_one_feature] * 8  # 8 identical features

        # Normalized assembly constraint
        self.normalized_assembly_constraint = NormalizedAssemblyLevelConstraint(
            self.composed_assembly_constraint,
            param_val_bounds=self.param_bounds,
        )

    def test_feature_level_constraint(self):
        # Test one feature constraint
        params = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]  # Valid normalized parameters
        result = self.feature_constraint_list[0](params)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_composed_assembly_constraint(self):
        # Test composed assembly constraint
        composed_params = [[0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]] * 8  # 8 features, 4DOF per feature, 2 params per DFO
        result = self.composed_assembly_constraint(composed_params)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (8,))

    def test_normalized_assembly_constraint(self):
        # Test normalized assembly constraint
        normalized_params = [[0.5] * 8] * 8  # Valid normalized parameters
        result = self.normalized_assembly_constraint(normalized_params)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (8,))

    def test_invalid_param_size(self):
        # Test for invalid parameter sizes
        with self.assertRaises(ValueError):
            invalid_params = [[0.5, 0.5, 0.5]] * 8  # Incorrect size
            self.composed_assembly_constraint(invalid_params)

    def test_bounds_check(self):
        # Test parameter bounds enforcement
        invalid_normalized_params = [[1.5] * 8] * 8  # Exceeds bounds
        result = self.normalized_assembly_constraint(invalid_normalized_params)
        self.assertTrue(np.all(result >= 0))  # Ensure result is processed properly


if __name__ == "__main__":
    unittest.main()
