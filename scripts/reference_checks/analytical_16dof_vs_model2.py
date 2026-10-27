"""Check OTAF's model2 against the analytical 16-DOF reference.

Same model as notebooks/AnalyticalModels/Analytical_2D_Reference_Assembly_16DOF.ipynb
(geometry, tolerance and parametrization of the thesis, Ch. 4, second example): the
assembly as 7 compatibility equations on the 8 face clearances, whose slack (largest
common clearance, the quantity OTAF maximizes) has a closed form. The script checks
the closed form against a linear program, then compares the slack distribution with
model2's on independent samples of the same distribution.

Usage: python scripts/reference_checks/analytical_16dof_vs_model2.py [n_samples]
"""
import logging
import sys

import numpy as np
import openturns as ot
from scipy.optimize import milp, Bounds, LinearConstraint

# Geometry of the thesis (Ch. 4, over-constrained 2D assembly)
hM, hF = 10.0, 10.2           # heights of the male bosses and of the female pockets
L1, L2 = 30.0, 70.0           # pocket / boss centres at x = L1 and x = L1 + L2
lM, lF = 10.0, 10.2           # widths of the bosses and of the pockets
TOL, CAPA = 0.16, 1.0

# Vertical faces: X = female (pocket) faces, Y = male (boss) faces, left to right.
X_POS = np.array([L1 - lF / 2, L1 + lF / 2, L1 + L2 - lF / 2, L1 + L2 + lF / 2])
Y_POS = np.array([L1 - lM / 2, L1 + lM / 2, L1 + L2 - lM / 2, L1 + L2 + lM / 2])
X_INV = np.array([False, True, False, True])  # outward normal along -x
Y_INV = np.array([True, False, True, False])
# Each face rotates about its centre (half its height); the clearances are measured at
# the ends of the overlap of the faces: bottom (y = 0) and top (y = hM).
Y_BOTTOM, Y_TOP = 0.0, hM

# 7 compatibility equations A J + b = 0 on the 8 clearances
# J = (J1b, J2b, J3b, J4b, J1h, J2h, J3h, J4h), b = bottom, h = top.
A_EQ = np.array([
    [-1, 0, 0, -1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, -1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [-1, 0, 0, 0, 1, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, -1, 0, 0],
], dtype=float)
A_PINV = np.linalg.pinv(A_EQ)
NULL = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=float)  # rigid sliding of the male part along x
assert np.allclose(A_EQ @ NULL, 0)

VARIABLES = ["u_X1", "g_X1", "u_X2", "g_X2", "u_X3", "g_X3", "u_X4", "g_X4",
             "u_Y1", "g_Y1", "u_Y2", "g_Y2", "u_Y3", "g_Y3", "u_Y4", "g_Y4"]


def face_positions(H, u, g, centre, inv):
    """x of a face at the bottom and top contact sections (translation u, rotation g)."""
    sign = np.where(inv, -1.0, 1.0)
    bottom = H + sign * u - g * (Y_BOTTOM - centre)
    top = H + sign * u - g * (Y_TOP - centre)
    return bottom, top


def b_eq(d):
    """Constant terms of the 7 equations, for deviations d of shape (n, 16)."""
    d = np.atleast_2d(d)
    u, g = d[:, 0::2], d[:, 1::2]
    Xb, Xh = face_positions(X_POS, u[:, :4], g[:, :4], hF / 2, X_INV)
    Yb, Yh = face_positions(Y_POS, u[:, 4:], g[:, 4:], hM / 2, Y_INV)
    i = lambda a, k: a[:, k]
    return np.stack([
        i(Yb, 0) - i(Xb, 0) + i(Xb, 3) - i(Yb, 3),
        i(Yb, 1) - i(Xb, 1) + i(Xb, 2) - i(Yb, 2),
        i(Yh, 0) - i(Xh, 0) + i(Xh, 3) - i(Yh, 3),
        i(Yh, 1) - i(Xh, 1) + i(Xh, 2) - i(Yh, 2),
        i(Yb, 0) - i(Xb, 0) + i(Xh, 0) - i(Yh, 0),
        i(Yb, 0) - i(Xb, 0) + i(Xb, 1) - i(Yb, 1),
        i(Yh, 0) - i(Xh, 0) + i(Xh, 1) - i(Yh, 1),
    ], axis=1)


def slack(d):
    """Largest common clearance s (all J_i >= s), in closed form.

    The solutions of A J + b = 0 are J_p + t NULL. The smallest clearance of the faces
    that open when the male part slides (+) grows with t, the other one (-) decreases:
    the best position makes them equal, s = (min_+ J_p + min_- J_p) / 2.
    """
    Jp = -(A_PINV @ b_eq(d).T).T
    return 0.5 * (Jp[:, NULL > 0].min(axis=1) + Jp[:, NULL < 0].min(axis=1))


def slack_lp(d):
    """Same slack with an LP (check of the closed form): maximize s, A J + b = 0, J >= s."""
    out = []
    for b in b_eq(d):
        r = milp(np.r_[np.zeros(8), -1.0],
                 constraints=[LinearConstraint(np.hstack([A_EQ, np.zeros((7, 1))]), -b, -b),
                              LinearConstraint(np.hstack([np.eye(8), -np.ones((8, 1))]), 0, np.inf)],
                 bounds=Bounds(-np.inf, np.inf))
        out.append(r.x[-1] if r.status == 0 else np.nan)
    return np.array(out)


TARGET = TOL / (6 * CAPA)  # standard deviation of the measured defect at the capability target
MAX_STD = np.array([TARGET, TARGET / (hF / 2)] * 4 + [TARGET, TARGET / (hM / 2)] * 4)


def credal_constraints(lam):
    """Thesis constraint per face, normalized: sqrt(s_u^2 + (h/2)^2 s_g^2) / target - 1."""
    s = np.asarray(lam) * MAX_STD
    h = np.array([hF] * 4 + [hM] * 4)
    return np.sqrt(s[0::2] ** 2 + (h / 2) ** 2 * s[1::2] ** 2) / TARGET - 1


def boundary_lambda(phi):
    """Point of the credal-set boundary: lambda_u = cos(phi_i), lambda_g = sin(phi_i) per face."""
    phi = np.asarray(phi)
    return np.ravel(np.column_stack([np.cos(phi), np.sin(phi)]))


def standard_sample(n, seed):
    ot.RandomGenerator.SetSeed(seed)
    return np.array(ot.Normal(16).getSample(n))


def pf(lam, z, threshold=0.0):
    return np.mean(slack(z * (np.asarray(lam) * MAX_STD)) < threshold)


if __name__ == "__main__":
    from scipy.stats import ks_2samp

    import otaf

    logging.disable(logging.WARNING)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    z = standard_sample(n, seed=3)
    print("nominal slack:", slack(np.zeros(16))[0])
    print("closed form vs LP, max |diff|:", np.abs(slack(z[:300] * MAX_STD) - slack_lp(z[:300] * MAX_STD)).max())
    s_ref = slack(z * MAX_STD)
    soc = otaf.example_models.model2.get_system_of_constraints_assembly_model(hM=hM, hF=hF, L1=L1, L2=L2, lM=lM, lF=lF)
    dist, *_ = otaf.example_models.model2.get_distribution_params(tol=TOL, capa=CAPA, hM=hM, hF=hF)
    otaf.common.set_seed(4)
    s_otaf = otaf.uncertainty.compute_gap_optimizations_on_sample_batch(
        soc, np.array(dist.getSample(n)), n_cpu=1, progress_bar=False)[:, -1]
    print(f"Pf at lambda = 1: analytical {np.mean(s_ref < 0):.4f} | OTAF {np.mean(s_otaf < 0):.4f}  (n={n} each)")
    print("quantiles 1/5/50/95 %:", np.quantile(s_ref, [.01, .05, .5, .95]).round(5), np.quantile(s_otaf, [.01, .05, .5, .95]).round(5))
    print("two-sample Kolmogorov-Smirnov p-value:", ks_2samp(s_ref, s_otaf).pvalue)
