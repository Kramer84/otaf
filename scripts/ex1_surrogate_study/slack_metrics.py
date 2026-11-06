"""Geometric question: surrogate slack against closed form, on the fixed test sample (TEST_SEED), for every saved checkpoint with a complete result.
Regions: overall on the credal set (per-sample allocation uniform on [0,1]^2), near the kink 5|g4-g5| < 0.002, near s = 0 (|s| < 0.02), TT/RR/EE,
near-zero translation and rotation, the mixed vertices, a 21x21 coverage map of the credal set, and outside the credal test: lambda = 1 with stds x1.35, box +-4.5 sigma."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import ex1lib as L
torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(0.3)
tb = L.TestBed()
N = 2_000_000
z = tb.z32[:N].double()
def stats(e, ref, kink):
    b = torch.abs(ref) < 0.02; k = kink < 0.002
    f = lambda m: dict(n=int(m.sum()), bias=float(e[m].mean()), rmse=float((e[m]**2).mean()**.5), max_abs=float(e[m].abs().max())) if m.any() else None
    return dict(all=f(torch.ones_like(b)), near_s0=f(b), near_kink=f(k), wrong_sign=float(((ref + e < 0) != (ref < 0)).double().mean()))
def run(fn):
    out = {}
    def at(lam, name):
        l = torch.as_tensor(lam, device=L.DEV, dtype=torch.float64)
        x = z * l; ref = L.closed_form(x); kink = 5 * torch.abs(x[:, 1] - x[:, 3])
        with torch.no_grad():
            s = torch.cat([fn(x[i:i + (1 << 19)].float()) for i in range(0, N, 1 << 19)]).double()
        out[name] = stats(s - ref, ref, kink)
    x = (tb.z32[:5_000_000] * tb.lam32[:5_000_000]).double(); ref = L.closed_form(x); kink = 5 * torch.abs(x[:, 1] - x[:, 3])
    with torch.no_grad():
        s = torch.cat([fn(x[i:i + (1 << 19)].float()) for i in range(0, len(x), 1 << 19)]).double()
    out["credal_overall"] = stats(s - ref, ref, kink); del x, ref, kink, s
    for k, v in L.LAM.items():
        at(L.lam_vec(*v), k)
    for v in (.01, .02, .05, .1):
        at(L.lam_vec(v, v), f"near_RR_lambda_u={v}")                       # almost no translation
        d = (1 - v**2)**.5
        at(L.lam_vec(d, d), f"near_TT_lambda_g={v}")                       # almost no rotation
    at(L.lam_vec(1., 0.), "vertex_T_R"); at(L.lam_vec(0., 1.), "vertex_R_T")
    at(np.ones(4) * 1.35, "lambda1_x1.35")
    box = torch.as_tensor(np.random.default_rng(L.TEST_SEED + 1).uniform(-1, 1, (N, 4)) * L.SIGMA * 4.5, device=L.DEV)
    ref = L.closed_form(box); kink = 5 * torch.abs(box[:, 1] - box[:, 3])
    with torch.no_grad():
        s = torch.cat([fn(box[i:i + (1 << 19)].float()) for i in range(0, N, 1 << 19)]).double()
    out["box_4.5sigma"] = stats(s - ref, ref, kink)
    grid = np.linspace(0, 1, 21); rm = np.zeros((21, 21)); bi = np.zeros((21, 21)); mx = np.zeros((21, 21))
    zz = z[:200_000]
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            x = zz * torch.as_tensor(L.lam_vec(a, b), device=L.DEV, dtype=torch.float64)
            with torch.no_grad():
                e = fn(x.float()).double() - L.closed_form(x)
            rm[i, j], bi[i, j], mx[i, j] = float((e**2).mean()**.5), float(e.mean()), float(e.abs().max())
    out["coverage21"] = dict(rmse_median=np.median(rm), rmse_max=rm.max(), bias_abs_max=np.abs(bi).max(), max_abs=mx.max(), rmse_argmax=[grid[v] for v in np.unravel_index(rm.argmax(), rm.shape)], max_abs_argmax=[grid[v] for v in np.unravel_index(mx.argmax(), mx.shape)])
    return out
res = {"control_closed_form": run(lambda x: L.closed_form(x))}
ck = {p.stem: p for p in sorted((L.HERE / "ckpt").glob("*.pth")) if (L.HERE / "results" / f"{p.stem}.json").exists()}
ck["B0_nov_checkpoint"] = L.SCRIPTS / "artifacts/model1_4_dof_surrogate.pth"; ck["B0_may_checkpoint"] = L.SCRIPTS / "artifacts/old_may2026/model1_4_dof_surrogate.pth"
for tag, p in ck.items():
    net = L.load_ckpt(p); res[tag] = run(net); del net; torch.cuda.empty_cache()
    r = res[tag]; print(f"{tag:38s} overall {r['credal_overall']['all']['rmse']:.1e} kink {r['credal_overall']['near_kink']['rmse']:.1e} s0 {r['credal_overall']['near_s0']['rmse']:.1e} TT {r['TT']['all']['rmse']:.1e} RR {r['RR']['all']['rmse']:.1e} cov.max {r['coverage21']['rmse_max']:.1e} max|e| {r['coverage21']['max_abs']:.1e} x1.35 {r['lambda1_x1.35']['all']['rmse']:.1e} box {r['box_4.5sigma']['all']['rmse']:.1e}", flush=True)
(L.HERE / "results" / "_slack_metrics.json").write_text(json.dumps(L.jsonable(res), indent=1))
