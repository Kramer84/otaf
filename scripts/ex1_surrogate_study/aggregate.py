"""Mean, standard deviation and range per configuration over the seeds. Usage: python aggregate.py [prefix...]"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
R = Path(__file__).parent / "results"
G = defaultdict(list)
for f in sorted(R.glob("*.json")):
    if f.stem.startswith("_"):
        continue
    if sys.argv[1:] and not any(f.stem.startswith(p) for p in sys.argv[1:]):
        continue
    G[re.sub(r"_s\d+$", "", f.stem)].append(json.loads(f.read_text()))
cf = {}
hdr = f"{'config':32s} {'k':>2s} {'surfN':>5s} | {'TT%':>16s} {'TT range':>15s} {'RR%':>14s} {'EE%':>14s} | {'slack rmse cred.':>16s} {'TTband bias':>12s} {'RRband bias':>12s} | {'surf pair w95%':>14s} {'max|pair|%':>10s} {'minTT':>5s} {'plIoU':>5s} {'s/1e6 gpu':>9s} {'s/1e6 cpu':>9s} {'train s':>7s}"
print(hdr)
for k, rs in G.items():
    n = rs[0]["surface_n"]
    if n not in cf:
        cf[n] = np.load(R / f"_cf_surface_n{n}.npy")
    dp = {a: np.array([100 * r[a]["paired_diff"] / r[a]["pf_cf"] for r in rs]) for a in ("TT", "RR", "EE")}
    b = lambda a, m: np.array([r[a][m] for r in rs])
    pw, pm = [], []
    for r in rs:
        tag = f"{r['name']}_s{r['seed']}" if "seed" in r and not r["name"].startswith("B") else (f"B1_baseline_retrain_s{r['seed']}" if "seed" in r else r["name"])
        P = np.load(R / f"{tag}_surface.npy"); e = np.abs(P / cf[n] - 1)
        pw.append(100 * max(np.percentile(e[i], 95) for i in range(5))); pm.append(100 * e.max())
    mt = sum(r["surface"]["0.0"]["min_at_TT"] for r in rs); pl = np.mean([r["surface"]["0.0"]["plateau_iou"] for r in rs])
    tg = np.mean([r.get("timing", {}).get("s_per_1e6_gpu", np.nan) for r in rs]); tc = np.mean([r.get("timing", {}).get("s_per_1e6_cpu2", np.nan) for r in rs])
    tt = np.mean([r.get("train", {}).get("train_time_s", r.get("train_time_s", np.nan)) for r in rs])
    print(f"{k:32s} {len(rs):2d} {n/1e6:5.0f} | {dp['TT'].mean():+7.2f}±{dp['TT'].std(ddof=1) if len(rs)>1 else 0:6.2f}  [{dp['TT'].min():+6.2f},{dp['TT'].max():+6.2f}] {dp['RR'].mean():+6.2f}±{dp['RR'].std(ddof=1) if len(rs)>1 else 0:5.2f} {dp['EE'].mean():+6.2f}±{dp['EE'].std(ddof=1) if len(rs)>1 else 0:5.2f} | {b('credal','rmse').mean():16.1e} {b('TT','bias_band').mean():+12.1e} {b('RR','bias_band').mean():+12.1e} | {np.mean(pw):14.3f} {np.max(pm):10.2f} {mt:2d}/{len(rs):2d} {pl:5.2f} {tg:9.4f} {tc:9.3f} {tt:7.0f}")
print("TT%,RR%,EE%: paired Pf difference with the closed form on the same 5e6 test sample, mean ± sd over seeds; surface: pair w95 = largest 95th percentile of |relative error| over 5 alphas against the closed form on the same points, averaged over seeds; max|pair| over all seeds, alphas and grid points")
