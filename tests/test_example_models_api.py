"""Fast checks on the public API of the example models (no model construction)."""

import numpy as np
import pytest

import otaf

EXAMPLE_MODELS = {
    "model1": otaf.example_models.model1,
    "model2": otaf.example_models.model2,
    "model3": otaf.example_models.model3,
    "model4": otaf.example_models.model4,
    "model4_auto": otaf.example_models.model4_auto,
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


def test_model3_sample_multiplier_maps_measured_points_to_their_own_feature():
    # model3 samples defects measured at points (Dumas, 2014) and maps them to feature
    # defects by position (sample_multiplier). With the labels the mapping is checked by
    # name: every entry links a measured defect to a defect of the same feature
    # ("1a1" in w_1a1_C and alpha_1a1), and the defects measured at the feature origin
    # are passed through unchanged.
    model = otaf.example_models.model3
    rows, _, columns = model.sample_multiplier_labels()
    _, labels, _, _ = model.get_distribution_params()
    assert columns == [str(x) for x in labels]

    def feature(label):
        return label.split("_")[1]

    T = np.asarray(model.sample_multiplier)
    assert T.shape == (len(rows), len(columns))
    for i, j in zip(*np.nonzero(T), strict=True):
        assert feature(rows[i]) == feature(columns[j]), (rows[i], columns[j])
    for i, row in enumerate(rows):
        if row in columns:
            np.testing.assert_array_equal(T[i], np.eye(len(columns))[columns.index(row)], err_msg=row)
