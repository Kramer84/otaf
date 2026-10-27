"""Every random draw in OTAF can be made reproducible with a seed."""

import numpy as np
import openturns as ot

import otaf


def test_set_seed_fixes_numpy_and_openturns():
    draws = []
    for _ in range(2):
        otaf.common.set_seed(7)
        draws.append((np.random.rand(3), np.array(ot.Normal(2).getSample(3))))
    np.testing.assert_array_equal(draws[0][0], draws[1][0])
    np.testing.assert_array_equal(draws[0][1], draws[1][1])


def test_correlated_samples_seed():
    a = otaf.distribution.generate_correlated_samples(corr=0.5, N=4, seed=3)
    b = otaf.distribution.generate_correlated_samples(corr=0.5, N=4, seed=3)
    np.testing.assert_array_equal(a, b)


def test_sobol_design_seed():
    dist = ot.JointDistribution([ot.Normal()] * 2)

    def design(method):
        exp = otaf.sensitivity.SobolIndicesExperimentWithComposedDistribution(dist, 5)
        return np.array(exp.generate(method=method, seed=11))

    for method in ("MonteCarlo", "LHS"):
        np.testing.assert_array_equal(design(method), design(method))


def test_surface_sampler_seed():
    def sample():
        sampler = otaf.sampling.UniformSurfaceSampler(
            dim=2,
            g=lambda x: float(x @ x - 1.0),
            bounds=[(-2.0, 2.0)] * 2,
            starting_points=np.array([[1.0, 0.0]]),
            seed=5,
        )
        return sampler.sample_surface(4)

    np.testing.assert_array_equal(sample(), sample())
