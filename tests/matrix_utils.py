"""Helpers to compare systems of constraints independently of their layout.

Two systems are considered equal when they contain the same constraints on the
same named variables, regardless of the order of the variables (columns) or
of the constraints (rows).
"""

import numpy as np

MATRIX_NAMES = ("A_eq_Def", "A_eq_Gap", "K_eq", "A_ub_Def", "A_ub_Gap", "K_ub")


def system_from_model(soc):
    """Extract matrices and variable names from a SystemOfConstraintsAssemblyModel."""
    system = {name: np.asarray(getattr(soc, name), dtype=float) for name in MATRIX_NAMES}
    system["deviation_symbols"] = np.array([str(s) for s in soc.deviation_symbols])
    system["gap_symbols"] = np.array([str(s) for s in soc.gap_symbols])
    return system


def canonical_block(A_def, A_gap, K, deviation_names, gap_names, decimals=10):
    """Return ``(names, rows)`` with columns sorted by name and rows sorted lexicographically.

    The constant term ``K`` is kept as the last column.
    """
    names = np.concatenate([np.asarray(deviation_names), np.asarray(gap_names)])
    order = np.argsort(names, kind="stable")
    full = np.hstack([A_def, A_gap])[:, order]
    rows = np.round(np.hstack([full, np.asarray(K).reshape(-1, 1)]), decimals) + 0.0  # drop -0.0
    rows = rows[np.lexsort(rows.T[::-1])] if rows.size else rows
    return names[order], rows


def assert_same_constraint_system(actual, expected, decimals=10):
    """Assert two systems (dicts as returned by :func:`system_from_model`) are equivalent."""
    for kind in ("eq", "ub"):
        names_a, rows_a = canonical_block(
            actual[f"A_{kind}_Def"], actual[f"A_{kind}_Gap"], actual[f"K_{kind}"],
            actual["deviation_symbols"], actual["gap_symbols"], decimals,
        )
        names_e, rows_e = canonical_block(
            expected[f"A_{kind}_Def"], expected[f"A_{kind}_Gap"], expected[f"K_{kind}"],
            expected["deviation_symbols"], expected["gap_symbols"], decimals,
        )
        np.testing.assert_array_equal(names_a, names_e, err_msg=f"{kind}: variable names differ")
        assert rows_a.shape == rows_e.shape, f"{kind}: {rows_a.shape} != {rows_e.shape}"
        np.testing.assert_allclose(rows_a, rows_e, atol=10.0 ** -decimals, err_msg=f"{kind}: rows differ")
