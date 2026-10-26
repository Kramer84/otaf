"""Fast checks on the public API of the example models (no model construction)."""

import numpy as np
import pytest

import otaf

EXAMPLE_MODELS = {
    "model1": otaf.example_models.model1,
    "model2": otaf.example_models.model2,
    "model3": otaf.example_models.model3,
    "model4": otaf.example_models.model4,
}


@pytest.mark.parametrize("name", EXAMPLE_MODELS)
def test_all_exports_exist(name):
    # Guards against implicit string concatenation in __all__ (missing comma).
    module = EXAMPLE_MODELS[name]
    missing = [attr for attr in module.__all__ if not hasattr(module, attr)]
    assert missing == []


def test_model2_credal_set_constraints_at_zero_std():
    # With zero standard deviation every constraint sits at (0 - target) / target = -1.
    model = otaf.example_models.model2
    constraints = model.eval_credal_set_constraints(np.zeros(model.dim))
    np.testing.assert_allclose(constraints, -np.ones(8))


def test_model2_scaled_constraints_match_unscaled():
    model = otaf.example_models.model2
    rng = np.random.default_rng(0)
    max_std = np.full(model.dim, 0.02)
    x_scaled = rng.uniform(0.0, 1.0, model.dim)

    expected = model.eval_credal_set_constraints(x_scaled * max_std)
    direct = model.eval_scaled_credal_set_constraints(x_scaled, max_std)
    wrapped = model.get_scaled_credal_set_constraints_function(max_std)(x_scaled)

    np.testing.assert_allclose(direct, expected)
    np.testing.assert_allclose(wrapped, expected)
