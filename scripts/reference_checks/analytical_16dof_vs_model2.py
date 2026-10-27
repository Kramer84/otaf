"""Check the 16-DOF analytical notebook against OTAF's model2 (hF = 10).

The notebook (notebooks/AnalyticalModels/Analytical_2D_Reference_Assembly_16DOF.ipynb)
writes the assembly as 7 equalities on 8 clearances (no relative rotation, as the
sliding planar contact of model2 imposes) and only tests feasibility. Here the
same equalities are solved for the largest common clearance s (J_i >= s), which
is the slack that OTAF maximises, so that both slack distributions can be
compared on independent samples of the same distribution (all features have
the same standard deviations, so the distribution does not depend on how the
variables of the two models correspond).

Usage: python scripts/reference_checks/analytical_16dof_vs_model2.py [n_samples]
"""
import logging
import sys
import numpy as np, openturns as ot, otaf
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.stats import ks_2samp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from baseline_cases import solve_slack
logging.disable(logging.WARNING)
h = 10.0
X = dict(X1=24.9, X2=35.1, X3=94.9, X4=105.1); Y = dict(Y1=25.0, Y2=35.0, Y3=95.0, Y4=105.0)
A_eq = np.array([[-1,0,0,-1,0,0,0,0],[0,1,1,0,0,0,0,0],[0,0,0,0,-1,0,0,-1],[0,0,0,0,0,1,1,0],
                 [-1,0,0,0,1,0,0,0],[-1,-1,0,0,0,0,0,0],[0,0,0,0,-1,-1,0,0]], float)
def pos(H, u, a, inv=False):
    return (H - u - a*h/2, H - u + a*h/2) if inv else (H + u - a*h/2, H + u + a*h/2)
def b_eq(u1,a1,u2,a2,u3,a3,u4,a4,v1,b1,v2,b2,v3,b3,v4,b4):
    X1h,X1b = pos(X["X1"],u1,a1); X2h,X2b = pos(X["X2"],u2,a2,True); X3h,X3b = pos(X["X3"],u3,a3); X4h,X4b = pos(X["X4"],u4,a4,True)
    Y1h,Y1b = pos(Y["Y1"],v1,b1,True); Y2h,Y2b = pos(Y["Y2"],v2,b2); Y3h,Y3b = pos(Y["Y3"],v3,b3,True); Y4h,Y4b = pos(Y["Y4"],v4,b4)
    return np.array([Y1b-X1b+X4b-Y4b, Y2b-X2b+X3b-Y3b, Y1h-X1h+X4h-Y4h, Y2h-X2h+X3h-Y3h,
                     Y1b-X1b+X1h-Y1h, Y1b-X1b+X2b-Y2b, Y1h-X1h+X2h-Y2h])
def analytical_s(x):
    # variables J (8) and s: A J + b = 0, J - s >= 0, maximise s.
    A = np.hstack([A_eq, np.zeros((7,1))]); G = np.hstack([np.eye(8), -np.ones((8,1))])
    c = np.r_[np.zeros(8), -1.0]
    r = milp(c, constraints=[LinearConstraint(A, -b_eq(*x), -b_eq(*x)), LinearConstraint(G, 0, np.inf)],
             bounds=Bounds(np.r_[np.full(8,-np.inf), -np.inf], np.inf))
    return r.x[-1] if r.status == 0 else np.nan
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
tol, capa = 0.16, 1.0
sp_, st_ = tol/(6*capa), 2*(tol/h)/(6*capa)
ot.RandomGenerator.SetSeed(3)
xa = np.array(ot.Normal(16).getSample(n)) * np.tile([sp_, st_], 8)
sa = np.array([analytical_s(x) for x in xa])
soc = otaf.example_models.model2.get_system_of_constraints_assembly_model(hF=10.0)
dist, *_ = otaf.example_models.model2.get_distribution_params(tol=tol, capa=capa, hF=10.0)
ot.RandomGenerator.SetSeed(4)
so, _ = solve_slack(soc, np.array(dist.getSample(n)))
print(f"nominal: analytical {analytical_s(np.zeros(16)):.6f}, otaf {solve_slack(soc, np.zeros((1,16)))[0][0]:.6f}")
print(f"Pf analytical {np.mean(sa<0):.4f} | otaf {np.mean(so<0):.4f}  (n={n} each)")
print("quantiles 1/5/50/95%:", np.round(np.nanquantile(sa,[.01,.05,.5,.95]),5), np.round(np.nanquantile(so,[.01,.05,.5,.95]),5))
print("KS two-sample p-value:", ks_2samp(sa, so).pvalue)
