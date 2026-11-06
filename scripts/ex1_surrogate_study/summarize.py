"""Print result tables from results/*.json. Usage: python summarize.py [name-prefix ...]"""
import json, sys
from pathlib import Path
import numpy as np
import ex1lib as L
R = Path(__file__).parent / "results"
pre = sys.argv[1:]
rows = []; CF = {}
for f in sorted(R.glob("*.json"), key=lambda p: p.stat().st_mtime):
    if pre and not any(f.stem.startswith(p) for p in pre):
        continue
    rows.append((f.stem, json.loads(f.read_text())))
print(f"{'candidate':34s} {'par':>5s} {'val rmse':>8s} | {'rmse crd':>8s} {'b.band T':>9s} {'rmseb T':>8s} {'b.band R':>9s} {'rmseb R':>8s} {'b.band E':>9s} | {'PfTT':>9s} {'dT%':>6s} {'PfRR':>9s} {'dR%':>6s} {'PfEE':>9s} {'dE%':>6s} | {'med|e|':>6s} {'p5':>6s} {'p95':>6s} {'min':>6s} {'max':>6s} {'minTT':>5s} {'plIoU':>5s} {'worst95':>7s} {'pair med':>8s} {'pair max':>8s} {'pair w95':>8s}")
for k, r in rows:
    s = r["surface"]["0.0"]
    t = r.get("train", {})
    w95 = max(max(abs(v["p5"]), abs(v["p95"])) for v in r["surface"].values())
    d = lambda a: 100 * r[a]["paired_diff"] / r[a]["pf_cf"]
    P = np.load(R / f"{k}_surface.npy"); n = r["surface_n"]
    cf = CF.get(n)
    if cf is None:
        cf = CF[n] = np.load(R / f"_cf_surface_n{n}.npy")
    e = P / cf - 1
    pair = (100 * np.median(np.abs(e[0])), 100 * np.abs(e[0]).max(), 100 * max(np.percentile(np.abs(e[a]), 95) for a in range(5)))
    print(f"{k:34s} {t.get('n_params', r.get('n_params', 0)):5d} {t.get('val_rmse_mm', 0):8.1e} | {r['credal']['rmse']:8.1e} {r['TT']['bias_band']:+9.1e} {r['TT']['rmse_band']:8.1e} {r['RR']['bias_band']:+9.1e} {r['RR']['rmse_band']:8.1e} {r['EE']['bias_band']:+9.1e} | "
          f"{r['TT']['pf']:9.3e} {d('TT'):+6.2f} {r['RR']['pf']:9.3e} {d('RR'):+6.2f} {r['EE']['pf']:9.3e} {d('EE'):+6.2f} | {100*s['med_abs']:6.2f} {100*s['p5']:+6.1f} {100*s['p95']:+6.1f} {100*s['min']:+6.1f} {100*s['max']:+6.1f} {str(s['min_at_TT'])[0]:>5s} {s['plateau_iou']:5.2f} {100*w95:7.1f} {pair[0]:8.2f} {pair[1]:8.2f} {pair[2]:8.2f}")
print("dX% = paired difference with the closed form on the same 5e6 sample, %; surface columns: % relative error against quadrature, alpha=0; worst95 = largest |p5|,|p95| over the 5 alphas (%). pair = same surface against sampling on the closed form on the same points: med|e| and max|e| at alpha=0, largest 95th pct of |e| over alphas (%).")
