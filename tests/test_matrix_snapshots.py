"""Golden-master tests: the constraint matrices of the example models must not change.

Any change to the modelling pipeline that alters these matrices is either a bug
or an intended change of the model. In the latter case, regenerate the
snapshots and explain why in the commit message:

    OTAF_UPDATE_SNAPSHOTS=1 pytest tests/test_matrix_snapshots.py
"""

import os
from pathlib import Path

import numpy as np
import pytest

from matrix_utils import MATRIX_NAMES, assert_same_constraint_system, system_from_model

SNAPSHOT_DIR = Path(__file__).parent / "data" / "matrix_snapshots"
UPDATE = os.environ.get("OTAF_UPDATE_SNAPSHOTS") == "1"

MODELS = [
    "model1",
    "model2",
    "model3",
    pytest.param("model4", marks=pytest.mark.slow),  # ~60 s, mostly SymPy
]


@pytest.mark.parametrize("name", MODELS)
def test_constraint_system_matches_snapshot(name, soc_model):
    actual = system_from_model(soc_model(name))
    path = SNAPSHOT_DIR / f"{name}.npz"
    if UPDATE:
        np.savez_compressed(path, **actual)
        pytest.skip(f"snapshot {path.name} regenerated")
    expected = dict(np.load(path))

    # Same constraints on the same named variables (order-independent) ...
    assert_same_constraint_system(actual, expected)
    # ... and the same layout, so that row/column indices stay meaningful.
    for key in ("deviation_symbols", "gap_symbols"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    for key in MATRIX_NAMES:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12, err_msg=key)


_DUMP_SCRIPT = """
import hashlib, sys, numpy as np, otaf
sys.path.insert(0, {tests_dir!r})
from matrix_utils import system_from_model
h = hashlib.sha256()
for name in ("model1", "model2"):
    system = system_from_model(getattr(otaf.example_models, name).get_system_of_constraints_assembly_model())
    for key in sorted(system):
        h.update(key.encode()); h.update(np.ascontiguousarray(system[key]).tobytes())
print(h.hexdigest())
"""


def test_model_construction_does_not_depend_on_hash_seed():
    # Set iteration order over strings changes with PYTHONHASHSEED. Seeds 8 and 9 used
    # to flip which direction of a gap matrix was kept in model2, changing its constraints.
    import subprocess
    import sys

    script = _DUMP_SCRIPT.format(tests_dir=str(Path(__file__).parent))
    digests = set()
    for seed in ("1", "8", "9"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        run = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        digests.add(run.stdout.strip().splitlines()[-1])
    assert len(digests) == 1
