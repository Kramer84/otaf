"""Markdown tables of REPORT.md from results/. Usage: python make_tables.py"""
import json, glob, re, collections
from pathlib import Path
import numpy as np
R = Path(__file__).parent / "results"
M = json.loads((R / "_slack_metrics.json").read_text())
J = {f.stem: json.loads(f.read_text()) for f in R.glob("*.json") if not f.stem.startswith("_")}
gp = lambda r, *p: __import__("functools").reduce(lambda a, b: a[b], p, r)
Q = [("Overall", ("credal_overall", "all", "rmse")), ("Near kink", ("credal_overall", "near_kink", "rmse")), ("Near s=0", ("credal_overall", "near_s0", "rmse")), ("TT", ("TT", "all", "rmse")), ("RR", ("RR", "all", "rmse")), ("EE", ("EE", "all", "rmse"))]
def fmt(v):
    v = np.array(v); return f"{np.median(v):.1e}" if len(v) == 1 else f"{np.median(v):.1e} ({v.min():.1e} to {v.max():.1e})"
def rows(names, q=Q):
    for lab, ks in names:
        print(f"| {lab} | {len(ks)} | " + " | ".join(fmt([gp(M[k], *p) for k in ks]) for _, p in q) + " |")
def pct(ks, a):
    v = np.array([100 * J[k][a]["paired_diff"] / J[k][a]["pf_cf"] for k in ks])
    return f"{v[0]:+.2f}" if len(v) == 1 else f"{v.mean():+.2f} ± {v.std(ddof=1):.2f}"
F = lambda p: sorted(k for k in M if k.startswith(p))
main = [("Baseline recipe retrained: GELU 16-8-4, inflated x1.35, 200k", F("B1_")), ("ReLU 16-8-4, inflated, 200k", F("F_relu16_inflated")), ("**ReLU 16-8-4, credal mixture, 200k**", F("F_relu16_credal_200k")), ("ReLU 16-8-4, credal mixture, 1M", F("F_relu16_credal_1M")),
        ("GELU 16-8-4, credal mixture, 1M", F("F_gelu16")), ("GELU 64-32-16, inflated, 200k", F("F_gelu64_infl")), ("GELU 64-32-16, credal mixture, 1M", F("F_gelu64_credal"))]
print("## T1\n| Configuration | Seeds | " + " | ".join(l for l, _ in Q) + " |\n|---|---|" + "---|" * len(Q)); rows(main)
print("\n## T2\n| Configuration | Seeds | Near pure translation, lambda_g = 0.01 | lambda_g = 0.1 | Near pure rotation, lambda_u = 0.01 | lambda_u = 0.1 | Mixed vertex T/R | Worst of 21x21 credal map | Allocation x1.35, lambda = 1 | Box +-4.5 sigma |\n|---|---|" + "---|" * 8)
rows(main, [("", ("near_TT_lambda_g=0.01", "all", "rmse")), ("", ("near_TT_lambda_g=0.1", "all", "rmse")), ("", ("near_RR_lambda_u=0.01", "all", "rmse")), ("", ("near_RR_lambda_u=0.1", "all", "rmse")), ("", ("vertex_T_R", "all", "rmse")), ("", ("coverage21", "rmse_max")), ("", ("lambda1_x1.35", "all", "rmse")), ("", ("box_4.5sigma", "all", "rmse"))])
print("\n## T3\n| Configuration | Seeds | TT | RR | EE |\n|---|---|---|---|---|")
for lab, ks in main:
    print(f"| {lab} | {len(ks)} | {pct(ks,'TT')} | {pct(ks,'RR')} | {pct(ks,'EE')} |")
print("\n## T4\n| Candidate (one seed) | Params | Overall | Near kink | Near s=0 | TT | RR | TT Pf % |\n|---|---|---|---|---|---|---|---|")
for k in sorted(M, key=lambda z: z):
    if re.match(r"(A\d|B_|C_|D_|E_)", k):
        print(f"| {k[:-3]} | {J[k]['train']['n_params']} | " + " | ".join(f"{gp(M[k], *p):.1e}" for _, p in Q[:5]) + f" | {pct([k],'TT')} |")
