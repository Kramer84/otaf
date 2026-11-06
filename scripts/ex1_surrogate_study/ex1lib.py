"""Shared code of the Ex1 surrogate study: closed form, training domains, training loop, evaluation harness.

Slack: s = (0.2 - (u4+u5) - 5|g4-g5|)/2, variable order x = (u4, g4, u5, g5), centred.
No file outside this folder is written. Evaluation sample: fixed NumPy stream TEST_SEED, independent of any training seed.
"""
from __future__ import annotations
import copy, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SU = 0.31 / 6.0            # sigma_u at lambda = 1 (ex1_analytical_pf.py)
SG = SU / 5.0              # sigma_g at lambda = 1
SIGMA = np.array([SU, SG, SU, SG])
ALPHAS = np.array([0.0, 0.01, 0.025, 0.05, 0.075])
TEST_SEED = 20261106
N_TEST = 5_000_000
ACT = {"gelu": nn.GELU, "relu": nn.ReLU, "leaky": lambda: nn.LeakyReLU(0.01)}
LAM = {"TT": (1.0, 1.0), "RR": (0.0, 0.0), "EE": (2**-.5, 2**-.5)}


def closed_form(x):
    """Works on numpy arrays and torch tensors, x = centred (u4, g4, u5, g5)."""
    a = x[:, 1] - x[:, 3]
    ab = torch.abs(a) if isinstance(x, torch.Tensor) else np.abs(a)
    return (0.2 - (x[:, 0] + x[:, 2]) - 5.0 * ab) / 2.0


def lam_vec(l1, l2):
    return np.array([l1, (1 - l1**2) ** .5, l2, (1 - l2**2) ** .5])


# ---------------------------------------------------------------- training domains
def credal_lams(rng, n, p_vertex=0.25):
    """Per feature, lambda_d = 1 with prob p_vertex, 0 with prob p_vertex, Uniform(0,1) otherwise."""
    def one():
        u = rng.random(n)
        l = rng.random(n)
        l[u < p_vertex] = 1.0
        l[(u >= p_vertex) & (u < 2 * p_vertex)] = 0.0
        return l
    l1, l2 = one(), one()
    return np.stack([l1, np.sqrt(1 - l1**2), l2, np.sqrt(1 - l2**2)], axis=1)


def draw_inflated(rng, n, m=1.35):
    return rng.standard_normal((n, 4)) * SIGMA * m


def draw_credal(rng, n, m=1.35):
    return rng.standard_normal((n, 4)) * SIGMA * m * credal_lams(rng, n)


def draw_uniform(rng, n, half_width=4.5):
    return rng.uniform(-1, 1, (n, 4)) * SIGMA * half_width


def draw_limit(rng, n, m=1.6, band=0.02):
    """Rejection on the labels only: credal mixture with wider stds, keep |s| < band. Needs labels, not the formula."""
    out, have = [], 0
    while have < n:
        x = draw_credal(rng, max(4 * n, 100_000), m)
        x = x[np.abs(closed_form(x)) < band]
        out.append(x); have += len(x)
    return np.concatenate(out)[:n]


def draw_kink(rng, n, m=1.35, eps=0.05 * SG):
    """Uses the location of the kink (closed-form knowledge): g5 = g4 + small offset, u and g4 from the credal mixture."""
    x = draw_credal(rng, n, m)
    x[:, 3] = x[:, 1] + rng.standard_normal(n) * eps
    return x


def make_domain(name, n, seed, m=1.35):
    rng = np.random.default_rng([seed, 7919])
    if name == "inflated":
        return draw_inflated(rng, n, m)
    if name == "credal":
        return draw_credal(rng, n, m)
    if name == "uniform":
        return draw_uniform(rng, n)
    parts = {"credal+limit": (.6, .4, 0), "credal+kink": (.6, 0, .4), "credal+limit+kink": (.4, .3, .3)}[name]
    a, b = int(parts[0] * n), int(parts[1] * n)
    x = [draw_credal(rng, a, m)]
    if b:
        x.append(draw_limit(rng, b))
    if n - a - b:
        x.append(draw_kink(rng, n - a - b, m))
    x = np.concatenate(x)
    return x[rng.permutation(len(x))]


# ---------------------------------------------------------------- model
class Surrogate(nn.Module):
    def __init__(self, hidden, act, xm, xs, ym, ys, abs_feature=False):
        super().__init__()
        self.abs_feature = abs_feature
        n_in = 5 if abs_feature else 4
        sizes = [n_in] + list(hidden) + [1]
        lay = []
        for i in range(len(sizes) - 1):
            lay.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                lay.append(ACT[act]())
        self.model = nn.Sequential(*lay)
        for k, v in dict(xm=xm, xs=xs, ym=ym, ys=ys).items():
            self.register_buffer(k, torch.as_tensor(np.asarray(v), dtype=torch.float32))

    def add_feature(self, x):
        if not self.abs_feature:
            return x
        return torch.cat([x, torch.abs(x[:, 1] - x[:, 3]).unsqueeze(1)], 1)

    def forward(self, x):                                    # physical x -> physical s
        return (self.model((self.add_feature(x) - self.xm) / self.xs) * self.ys + self.ym).squeeze(1)


def n_params(net):
    return sum(p.numel() for p in net.model.parameters())


def train(cfg, seed, log=print):
    """cfg keys: hidden, act, n, domain, m, batch, steps, lr, lr_end, sched(cos|growth), gamma, wd, clamp, init(default|xavier002),
    shuffle, stop_r2, epochs_max, loss(mse|wmse|huber), wk, abs_feature, train_frac. Selection of the best weights: validation MSE."""
    t0 = time.time()
    torch.manual_seed(seed)
    x = make_domain(cfg["domain"], cfg["n"], seed, cfg.get("m", 1.35))
    y = closed_form(x)
    fail = float((y < 0).mean())
    Xr = torch.as_tensor(x, dtype=torch.float32, device=DEV)
    Yr = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    del x
    abs_f = cfg.get("abs_feature", False)
    Xf = torch.cat([Xr, torch.abs(Xr[:, 1] - Xr[:, 3]).unsqueeze(1)], 1) if abs_f else Xr
    xm, xs, ym, ys = Xf.mean(0), Xf.std(0), Yr.mean(), Yr.std()
    net = Surrogate(cfg["hidden"], cfg["act"], xm.cpu(), xs.cpu(), ym.cpu(), ys.cpu(), abs_f).to(DEV)
    if cfg.get("init", "default") == "xavier002":
        for m in net.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02); nn.init.zeros_(m.bias)
    X = (Xf - xm) / xs; Y = ((Yr - ym) / ys).unsqueeze(1)
    S = Yr
    ntr = int(cfg.get("train_frac", .6) * len(X))         # the sample is already random: first part train, rest validation
    Xt, Yt, St, Xv, Yv = X[:ntr], Y[:ntr], S[:ntr], X[ntr:], Y[ntr:]
    opt = torch.optim.AdamW(net.model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.0))
    bs = min(cfg["batch"], ntr)
    per_epoch = max(1, math.ceil(ntr / bs))
    steps = cfg.get("steps") or cfg["epochs_max"] * per_epoch
    if cfg.get("sched", "cos") == "cos":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=cfg.get("lr_end", 1e-6))
        step_each = "step"
    else:
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, cfg.get("gamma", 1.0001))
        step_each = "epoch"
    wk, lossname = cfg.get("wk", 9.0), cfg.get("loss", "mse")
    clamp, shuffle = cfg.get("clamp", False), cfg.get("shuffle", True)
    stop_r2 = cfg.get("stop_r2")
    val_every = cfg.get("val_every") or max(per_epoch, min(steps // 400, 500))
    best, best_w, hist, step, perm, stopped = float("inf"), None, [], 0, None, None
    pos = ntr
    while step < steps:
        if pos >= ntr:
            perm = torch.randperm(ntr, device=DEV) if shuffle else None
            pos = 0
        idx = perm[pos:pos + bs] if shuffle else torch.arange(pos, min(pos + bs, ntr), device=DEV)
        pos += bs
        pred = net.model(Xt[idx])
        err = pred - Yt[idx]
        if lossname == "mse":
            loss = (err**2).mean()
        elif lossname == "wmse":
            w = 1 + wk * torch.exp(-(St[idx] / 0.02) ** 2)
            loss = (w.unsqueeze(1) * err**2).sum() / w.sum()
        else:
            raise ValueError(lossname)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if clamp:
            with torch.no_grad():
                for p in net.model.parameters():
                    p.clamp_(-1.0, 1.0)
        step += 1
        if step_each == "step":
            sched.step()
        elif pos >= ntr:
            sched.step()
        if step % val_every == 0 or step == steps:
            with torch.no_grad():
                v = sum(float(((net.model(Xv[i:i + (1 << 20)]) - Yv[i:i + (1 << 20)]) ** 2).sum()) for i in range(0, len(Xv), 1 << 20)) / len(Xv)
            r2 = 1 - v / float(Yv.var())
            hist.append((step, v, r2))
            if v < best:
                best, best_w = v, copy.deepcopy(net.state_dict())
            if len(hist) % 50 == 0:
                log(f"   step {step}/{steps} val_mse_norm {v:.3e} best {best:.3e} r2 {r2:.7f} {time.time()-t0:.0f}s")
            if stop_r2 and step / per_epoch > 5 and 1 - best / float(Yv.var()) >= stop_r2:
                stopped = step
                break
    net.load_state_dict(best_w)
    net.eval()
    info = dict(fail_share=fail, train_time_s=time.time() - t0, steps_done=step, stopped_early_at=stopped, val_mse_norm=best,
                val_rmse_mm=best**.5 * float(ys), n_params=n_params(net), y_std=float(ys), x_std=[float(v) for v in xs.cpu()])
    return net, info, hist


# ---------------------------------------------------------------- evaluation
class TestBed:
    """Fixed test sample, closed-form references on it (float64) and exact quadrature."""
    def __init__(self):
        rng = np.random.default_rng(TEST_SEED)
        self.zr = rng.standard_normal((N_TEST, 4))                      # standard normal, float64
        self.mix_l = credal_uniform(rng, N_TEST)                        # per-sample allocation, uniform on [0,1]^2
        self.z32 = torch.as_tensor(self.zr * SIGMA, dtype=torch.float32, device=DEV)   # x at lambda=1
        d = np.load(SCRIPTS / "ex1_grid_5M/exact_quadrature.npz")
        self.exact, self.scales, self.alphas = d["pf"], d["scales"], d["alphas"]
        assert np.allclose(self.alphas, ALPHAS)
        self.cf = {k: closed_form(self.zr * SIGMA * lam_vec(*v)) for k, v in LAM.items()}
        self.cf["credal"] = closed_form(self.zr * SIGMA * self.mix_l)
        self.lam32 = torch.as_tensor(self.mix_l, dtype=torch.float32, device=DEV)
        self.cfg = {}

    def predict(self, fn, lam, n=N_TEST, bs=1 << 19):
        out = []
        lam = torch.as_tensor(lam, dtype=torch.float32, device=DEV)
        with torch.no_grad():
            for i in range(0, n, bs):
                out.append(fn(self.z32[i:min(i + bs, n)] * (lam if lam.dim() == 1 else lam[i:min(i + bs, n)])))
        return torch.cat(out)

    def cf_surface(self, n):
        """Direct sampling on the closed form over the grid with the same n: the noise floor of the surface comparison."""
        if n in self.cfg:
            return self.cfg[n]
        f = HERE / "results" / f"_cf_surface_n{n}.npy"
        if f.exists():
            self.cfg[n] = np.load(f)
            return self.cfg[n]
        z = torch.as_tensor(self.zr[:n] * SIGMA, dtype=torch.float64, device=DEV)
        P = np.zeros((5, 50, 50))
        for i, l2 in enumerate(self.scales):
            for j, l1 in enumerate(self.scales):
                s = closed_form(z * torch.as_tensor(lam_vec(l1, l2), device=DEV))
                P[:, i, j] = [(s < a).double().mean().item() for a in ALPHAS]
        del z
        np.save(f, P); self.cfg[n] = P
        return P


def credal_uniform(rng, n):
    l = rng.random((n, 2))
    return np.stack([l[:, 0], np.sqrt(1 - l[:, 0]**2), l[:, 1], np.sqrt(1 - l[:, 1]**2)], 1)


def slack_stats(s, cf):
    e = s.astype(np.float64) - cf
    b = np.abs(cf) < 0.02
    return dict(bias=e.mean(), rmse=(e**2).mean()**.5, bias_band=e[b].mean(), rmse_band=(e[b]**2).mean()**.5, n_band=int(b.sum()),
                wrong_sign=float(((s < 0) != (cf < 0)).mean()), max_abs=float(np.abs(e).max()))


def surface(tb, fn, n):
    P = torch.zeros((5, 50, 50), dtype=torch.float64)
    al = torch.as_tensor(ALPHAS, dtype=torch.float32, device=DEV)
    for i, l2 in enumerate(tb.scales):
        for j, l1 in enumerate(tb.scales):
            s = tb.predict(fn, lam_vec(l1, l2), n)
            P[:, i, j] = (s.unsqueeze(1) < al).float().mean(0).double().cpu()
    return P.numpy()


def surface_summary(P, Q):
    out = {}
    for k, a in enumerate(ALPHAS):
        e = P[k] / Q[k] - 1
        i = np.unravel_index(P[k].argmin(), Q[k].shape)
        iq = np.unravel_index(Q[k].argmin(), Q[k].shape)
        pl = Q[k] >= 0.98 * Q[k].max()                                  # plateau: within 2 % of the exact maximum
        plp = P[k] >= 0.98 * P[k].max()
        out[f"{a}"] = dict(med=np.median(e), p5=np.percentile(e, 5), p95=np.percentile(e, 95), min=e.min(), max=e.max(), med_abs=np.median(np.abs(e)),
                           argmin=[int(i[0]), int(i[1])], argmin_exact=[int(iq[0]), int(iq[1])], min_at_TT=bool(i == (49, 49)),
                           P_min=P[k].min(), P_max=P[k].max(), Q_min=Q[k].min(), Q_max=Q[k].max(),
                           plateau_share_exact=pl.mean(), plateau_share=plp.mean(), plateau_iou=(pl & plp).sum() / max(1, (pl | plp).sum()),
                           plateau_med_err=np.median(e[pl]))
    return out


def evaluate(tb, fn, n_surf=1_000_000, net_for_timing=None):
    r = {}
    Q = tb.exact
    exact = {"TT": Q[0, 49, 49], "RR": Q[0, 0, 0], "EE": None}
    from scipy import integrate, stats
    a = SU; b = SG                                                     # exact EE by quadrature (not on the 50-point grid)
    aE, bE = SU * (0.5 + 0.5) ** .5, SG * (0.5 + 0.5) ** .5
    exact["EE"] = integrate.quad(lambda v: 2 * stats.norm.pdf(v, scale=bE) * stats.norm.sf((0.2 - 5 * v) / aE), 0, 12 * bE, epsabs=1e-13, epsrel=1e-10, limit=200)[0]
    for k, v in LAM.items():
        s = tb.predict(fn, lam_vec(*v)).cpu().numpy().astype(np.float32)
        cf = tb.cf[k]
        st = slack_stats(s, cf)
        p = (s < 0).mean(); pc = (cf < 0).mean()
        disc_b = ((s < 0) & ~(cf < 0)).sum(); disc_c = (~(s < 0) & (cf < 0)).sum()
        var = ((disc_b + disc_c) - (float(disc_b) - float(disc_c))**2 / N_TEST) / N_TEST**2
        st.update(pf=float(p), pf_se=float((p * (1 - p) / N_TEST)**.5), pf_cf=float(pc), pf_exact=float(exact[k]),
                  paired_diff=float(p - pc), paired_se=float(var**.5), rel_err=float(p / exact[k] - 1))
        r[k] = st
    s = tb.predict(fn, tb.lam32).cpu().numpy().astype(np.float32)
    r["credal"] = slack_stats(s, tb.cf["credal"])
    t = time.time()
    P = surface(tb, fn, n_surf)
    r["surface_time_s"] = time.time() - t
    r["surface_n"] = n_surf
    r["surface"] = surface_summary(P, Q)
    return r, P


def timing(net):
    """Seconds per 1e6 points on the GPU and on 2 CPU threads."""
    x = torch.randn(1_000_000, 4) * torch.as_tensor(SIGMA, dtype=torch.float32)
    net = net.to(DEV).eval(); xd = x.to(DEV)
    with torch.no_grad():
        net(xd[:1000])
        if DEV == "cuda":
            torch.cuda.synchronize()
        t = time.time(); net(xd); torch.cuda.synchronize() if DEV == "cuda" else None; g = time.time() - t
        cn = copy.deepcopy(net).cpu()
        t = time.time(); cn(x); c = time.time() - t
    return dict(s_per_1e6_gpu=g, s_per_1e6_cpu2=c)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def save_ckpt(net, cfg, path):
    """Same keys as train_neural_networks.py, plus the activation (from_checkpoint assumes GELU and cannot load ReLU as is)."""
    arch = ["dim"] + [str(h) for h in cfg["hidden"]] + ["1"]
    torch.save(dict(model_state_dict=net.model.state_dict(), architecture=arch, input_dim=4, activation=cfg["act"], abs_feature=cfg.get("abs_feature", False),
                    model_name="model1_4_dof", input_normalization=True, output_normalization=True, config=cfg,
                    normalization_metadata=dict(X_mean=net.xm.cpu(), y_mean=net.ym.cpu().reshape(1), X_std=net.xs.cpu(), y_std=net.ys.cpu().reshape(1))), path)


def load_ckpt(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    nm = c["normalization_metadata"]
    hidden = [int(a) for a in c["architecture"][1:-1]]
    act = c.get("activation", "gelu")
    net = Surrogate(hidden, act, nm["X_mean"], nm["X_std"], nm["y_mean"].reshape(()), nm["y_std"].reshape(()), c.get("abs_feature", False))
    net.model.load_state_dict(c["model_state_dict"])
    return net.to(DEV).eval()
