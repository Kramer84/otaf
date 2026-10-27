"""The examples of the input-format guide build the models they claim to."""

import re
from pathlib import Path

import numpy as np
import pytest

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide" / "input_format.md"


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and not isinstance(a, np.ndarray):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (np.ndarray, float, int)) and not isinstance(a, bool):
        return np.allclose(a, b)
    return a == b


@pytest.fixture(scope="module")
def guide_namespace():
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), flags=re.S)
    assert len(blocks) >= 4
    namespace = {}
    for block in blocks:
        exec(compile(block, str(GUIDE), "exec"), namespace)
    return namespace


@pytest.mark.slow
def test_2d_example_is_model1(guide_namespace):
    from otaf.example_models.models_2_D._model_2D_4dof import get_assembly_data

    assert _same(guide_namespace["system_data"], get_assembly_data())
    assert guide_namespace["slack"] == pytest.approx((100.0 - 99.8) / 2, abs=1e-7)


@pytest.mark.slow
def test_pin_example_is_model4_auto(guide_namespace):
    from otaf.example_models.models_3_D._model_3D_pins_auto import get_assembly_data

    assert _same(guide_namespace["pin_system_data"], get_assembly_data(NX=1, NY=1))
    assert guide_namespace["pin_slack"] == pytest.approx((20.0 - 19.8) / 2, abs=1e-7)


def test_validation_example(guide_namespace):
    from otaf.schema import SystemData

    assert isinstance(guide_namespace["model"], SystemData)
    assert guide_namespace["json_schema"]["title"] == "SystemData"
    assert guide_namespace["adp_from_model"].system_data.keys() == guide_namespace["system_data"].keys()
