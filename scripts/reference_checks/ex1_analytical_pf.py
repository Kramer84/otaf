"""Ex1 (pocket, 4 DOF): exact Pf over the credal set, no sampling and no surrogate.

gap = (L2-L1) - (u4+u5) - y_max |g4-g5| with independent centred normals, so
Pf(alpha) = P(U + |V| > 0.2 - 2 alpha), U~N(0,a^2), V~N(0,b^2), reduced to a 1-D integral:
Pf = int phi_b(v) Q((c - y|v|)/a) dv.  Used (i) to validate the surrogate grid
(Ex1 against analytical) and (ii) to give the classical choices of the allocation
(pure translation / pure rotation / equal share, all 3^2 vertices) next to the interval.
Usage: python scripts/reference_checks/ex1_analytical_pf.py [grid.npz]
"""
import itertools, sys
import numpy as np
from scipy import integrate, stats

L1, L2, H, T, CP = 99.8, 100.0, 10.0, 0.31, 1.0
Y = H / 2
SU = T / (6 * CP); SG = SU / Y
C0 = L2 - L1

def pf(l4, l5, c=C0):
    """l = (lambda_u, lambda_gamma) of each feature; failure when gap < 0 (c = clearance - 2*alpha)."""
    a = SU * np.hypot(l4[0], l5[0])
    b = SG * np.hypot(l4[1], l5[1])
    if b == 0:
        return float(stats.norm.sf(c / a)) if a > 0 else float(c < 0)
    f = lambda v: 2 * stats.norm.pdf(v, scale=b) * (stats.norm.sf((c - Y * v) / a) if a > 0 else float(Y * v > c))
    val, _ = integrate.quad(f, 0, 12 * b, epsabs=1e-13, epsrel=1e-10, limit=200)
    return val

def lam(kind):
    """lambda=(u,g) of one feature: 'T' translation only, 'R' rotation only, 'E' equal share."""
    return {"T": (1.0, 0.0), "R": (0.0, 1.0), "E": (2 ** -.5, 2 ** -.5)}[kind]

if __name__ == "__main__":
    print(f"sigma_u,max={SU:.5f}  sigma_g,max={SG:.5f}")
    for alpha in (0.0, 0.025, 0.05, 0.075):
        c = C0 - 2 * alpha
        print(f"\nalpha={alpha}: failure when gap < {2*alpha}")
        for k4, k5 in itertools.product("TRE", repeat=2):
            print(f"  feature4={k4} feature5={k5}: Pf={pf(lam(k4), lam(k5), c):.5e}")
        us = np.linspace(0, 1, 41)
        g = np.array([[pf((u4, np.sqrt(1 - u4**2)), (u5, np.sqrt(1 - u5**2)), c) for u5 in us] for u4 in us])
        i = np.unravel_index(g.argmax(), g.shape); j = np.unravel_index(g.argmin(), g.shape)
        print(f"  credal-set range (41x41): min={g.min():.5e} at u4={us[j[0]]:.3f},u5={us[j[1]]:.3f}; max={g.max():.5e} at u4={us[i[0]]:.3f},u5={us[i[1]]:.3f}")
