"""Train ReLU surrogates on a sample that covers the credal set (recipe of scripts/ex1_surrogate_study/REPORT.md).

Sample: 60 % credal mixture (per feature, a random subset of components or a random direction, scaled so that the capability
constraint is active), 20 % reduced credal mixture (features outside the reduced model at zero, as in --reduced-mode of
imprecise_bounds_analysis.py), 20 % all lambda = 1, all inflated by `mult` as in train_neural_networks.py. Labels: the linear program.
Training: AdamW, MSE, batch 4096, 40000 steps, lr 2e-3 cosine to 1e-6, best validation weights, 60/40 split. Lowest validation over seeds kept.
Usage (from scripts/): python train_surrogates_relu.py --models model2_16_dof model3_30_dof model4_50_dof [--seeds 0 1 2] [--out-dir artifacts/relu_nov2026]
"""
import argparse, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import otaf

CFG = {"model2_16_dof": dict(mod="model2", mult=1.21, n=200_000, hidden=[32, 16, 8], reduced=None),
       "model3_30_dof": dict(mod="model3", mult=1.26, n=200_000, hidden=[64, 32, 16, 8], reduced=list(range(26))),
       "model4_50_dof": dict(mod="model4", mult=1.16, n=300_000, hidden=[128, 64, 32, 16], reduced=list(range(6, 38)))}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NCPU = 7


def feature_groups(cons, dim):
    """Constraint i -> components it depends on, found numerically (each constraint is one feature)."""
    base = np.asarray(cons(np.full(dim, .5)))
    groups = [[] for _ in base]
    for j in range(dim):
        x = np.full(dim, .5); x[j] = .9
        for i, v in enumerate(np.asarray(cons(x))):
            if abs(v - base[i]) > 1e-12:
                groups[i].append(j)
    return groups


def credal_lambda(rng, cons, groups, dim, active=None):
    """One lambda on the boundary: each constraint value is std/target - 1 and the std is homogeneous of degree one in lambda."""
    lam = np.zeros(dim)
    for g in groups:
        g_a = [j for j in g if active is None or j in active]
        if not g_a:
            continue
        if rng.random() < .5:
            k = rng.integers(1, len(g_a) + 1)
            lam[rng.choice(g_a, k, replace=False)] = 1.0
        else:
            lam[g_a] = np.abs(rng.standard_normal(len(g_a)))
    c = np.asarray(cons(lam))
    for i, g in enumerate(groups):
        if lam[g].any():
            lam[g] /= (c[i] + 1.0)
    return np.minimum(lam, 1.0)


def make_data(name, seed=420):
    cfg = CFG[name]
    model = getattr(otaf.example_models, cfg["mod"])
    dist, _, ms, mu = model.get_distribution_params()
    cons = model.get_scaled_credal_set_constraints_function(ms)
    dim = model.dim
    groups = feature_groups(cons, dim)
    otaf.common.set_seed(seed)
    z = np.array(dist.getSample(cfg["n"])) - mu
    rng = np.random.default_rng(seed)
    n = cfg["n"]
    n_red = int(.2 * n) if cfg["reduced"] else 0
    n_full = int(.2 * n)
    lam = np.ones((n, dim))
    active = set(cfg["reduced"]) if cfg["reduced"] else None
    for i in range(n - n_full):
        lam[i] = credal_lambda(rng, cons, groups, dim, active if i < n_red else None)
    x = z * lam * cfg["mult"] + mu
    x = x[rng.permutation(n)]
    print(f"[{name}] sample ready, {len(groups)} features, lambda mean {lam.mean():.3f}", flush=True)
    xt = x @ np.asarray(model.sample_multiplier, dtype=float).T
    y = otaf.uncertainty.compute_gap_optimizations_on_sample_batch(
        model.get_system_of_constraints_assembly_model(), xt.astype("float32"), bounds=None, n_cpu=NCPU,
        progress_bar=False, batch_size=500, dtype="float32")[:, -1]
    ok = np.isfinite(y)
    print(f"[{name}] labels done: finite {ok.mean():.5f}, failed share {(y[ok] < 0).mean():.4f}", flush=True)
    return x[ok].astype("float32"), y[ok].astype("float32")


def train(x, y, hidden, seed, steps=40_000):
    torch.manual_seed(seed)
    X, Y = torch.as_tensor(x, device=DEV), torch.as_tensor(y, device=DEV)
    xm, xs, ym, ys = X.mean(0), X.std(0), Y.mean(), Y.std()
    X, Y = (X - xm) / xs, ((Y - ym) / ys).unsqueeze(1)
    sizes = [x.shape[1]] + hidden + [1]
    lay = []
    for i in range(len(sizes) - 1):
        lay.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            lay.append(nn.ReLU())
    net = nn.Sequential(*lay).to(DEV)
    ntr = int(.6 * len(X))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-6)
    best, best_w, pos, perm, hist = float("inf"), None, ntr, None, []
    for step in range(1, steps + 1):
        if pos >= ntr:
            perm, pos = torch.randperm(ntr, device=DEV), 0
        idx = perm[pos:pos + 4096]; pos += 4096
        loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if step % 100 == 0:
            with torch.no_grad():
                v = float(((net(X[ntr:]) - Y[ntr:]) ** 2).mean())
            hist.append((step, float(loss), v))
            if v < best:
                best, best_w = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_w)
    return net, best, float(ys), (xm.cpu(), xs.cpu(), ym.cpu().reshape(1), ys.cpu().reshape(1)), hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default="artifacts/relu_nov2026")
    ap.add_argument("--hidden", nargs="+", type=int, default=None, help="Override the hidden layer widths")
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--extra-labels", type=int, default=0, help="Draw this many labelled points, save them to --chunk-out, and stop (see label_chunks_ex4.py)")
    ap.add_argument("--data-seed", type=int, default=421)
    ap.add_argument("--chunk-out", default=None)
    ap.add_argument("--n-cpu", type=int, default=7)
    ap.add_argument("--data-dir", default=None, help="Folder holding an existing <model>_train.npz")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        cfg, cache = dict(CFG[name]), Path(a.data_dir or out) / f"{name}_train.npz"
        cfg["hidden"] = a.hidden or cfg["hidden"]
        if a.extra_labels:
            CFG[name]["n"] = a.extra_labels
            NCPU = a.n_cpu
            x2, y2 = make_data(name, seed=a.data_seed)
            np.savez(a.chunk_out, x=x2, y=y2)
            continue
        if cache.exists():
            d = np.load(cache); x, y = d["x"], d["y"]
            print(f"[{name}] {len(x)} labelled points from {cache}", flush=True)
        else:
            t = time.time(); x, y = make_data(name); np.savez(cache, x=x, y=y)
            print(f"[{name}] data in {time.time()-t:.0f}s", flush=True)
        res = []
        for s in a.seeds:
            t = time.time(); net, v, ys, norm, hist = train(x, y, cfg["hidden"], s, a.steps)
            print(f"[{name}] seed {s}: val rmse {v**.5*ys:.3e} mm, last train MSE {hist[-1][1]:.3e} (normalised), {time.time()-t:.0f}s", flush=True)
            res.append((v, s, net, ys, norm, hist))
        v, s, net, ys, (xm, xs, ym, yst), hist = min(res, key=lambda r: r[0])
        arch = ["dim"] + [str(h) for h in cfg["hidden"]] + ["1"]
        torch.save(dict(model_state_dict={k: w.cpu() for k, w in net.state_dict().items()}, architecture=arch, input_dim=x.shape[1], activation="relu",
                        model_name=name, input_normalization=True, output_normalization=True, seed=s, val_rmse_mm=v**.5*ys, history=hist,
                        normalization_metadata=dict(X_mean=xm, y_mean=ym, X_std=xs, y_std=yst)), out / f"{name}_surrogate.pth")
        print(f"[{name}] kept seed {s}, val rmse {v**.5*ys:.3e} mm", flush=True)
