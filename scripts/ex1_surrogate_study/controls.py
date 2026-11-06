"""Controls and baselines evaluated with the same harness: closed form (noise floor), hand-built 3-neuron ReLU net (exact representation),
existing Nov. and May checkpoints, and the baseline recipe retrained through the library class otaf.surrogate.NeuralRegressorNetwork
(same options as train_neural_networks.py, mult 1.35, tol 0.31; only the labels come from the closed form)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import ex1lib as L
import otaf
from otaf.example_models import model1
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.6)
tb = L.TestBed()
def record(tag, fn, n_surf=5_000_000, **extra):
    f = L.HERE / "results" / f"{tag}.json"
    if f.exists():
        print("skip", tag); return
    res, P = L.evaluate(tb, fn, n_surf)
    np.save(L.HERE / "results" / f"{tag}_surface.npy", P)
    res.update(name=tag, **extra)
    if isinstance(fn, torch.nn.Module):
        res["timing"] = L.timing(fn)
    f.write_text(json.dumps(L.jsonable(res), indent=1))
    print(f"--- {tag} Pf TT {res['TT']['pf']:.4e} RR {res['RR']['pf']:.4e} EE {res['EE']['pf']:.4e}; band bias TT {res['TT']['bias_band']:+.2e} RR {res['RR']['bias_band']:+.2e}", flush=True)

record("control_closed_form", lambda x: L.closed_form(x))
hb = L.Surrogate([3], "relu", np.zeros(4), np.ones(4), 0.0, 1.0)
with torch.no_grad():
    C = 2.0
    hb.model[0].weight.copy_(torch.tensor([[0, 1, 0, -1], [0, -1, 0, 1], [-1, 0, -1, 0]], dtype=torch.float32)); hb.model[0].bias.copy_(torch.tensor([0, 0, C], dtype=torch.float32))
    hb.model[2].weight.copy_(torch.tensor([[-2.5, -2.5, 0.5]], dtype=torch.float32)); hb.model[2].bias.copy_(torch.tensor([(0.2 - C) / 2], dtype=torch.float32))
hb = hb.to(L.DEV).eval()
record("control_handbuilt_relu3", hb, n_params=L.n_params(hb))
for tag, p in {"B0_nov_checkpoint": "artifacts/model1_4_dof_surrogate.pth", "B0_may_checkpoint": "artifacts/old_may2026/model1_4_dof_surrogate.pth"}.items():
    net = L.load_ckpt(L.SCRIPTS / p)
    print(tag, "X_std/sigma", (net.xs.cpu().numpy() / L.SIGMA).round(3), "params", L.n_params(net), flush=True)
    record(tag, net, n_params=L.n_params(net))

base = model1.get_distribution_params(tol=0.31, capa=1.0)[0]
dist = otaf.distribution.multiply_composed_distribution_standard_with_constants(base, [1.35] * 4)
for seed in [420, 1, 2]:
    tag = f"B1_baseline_retrain_s{seed}"
    if (L.HERE / "results" / f"{tag}.json").exists():
        continue
    otaf.common.set_seed(seed)
    X = np.array(dist.getSample(200_000), dtype="float32")
    y = np.asarray(L.closed_form(X.astype(np.float64)), dtype="float32")
    t = time.time()
    nn_ = otaf.surrogate.NeuralRegressorNetwork(input_dim=4, output_dim=1, X=X, y=y, clamping=True, finish_critertion_epoch=5, loss_finish=1e-6, metric_finish=0.9998,
                                                max_epochs=2500, batch_size=30000, compile_model=False, train_size=0.6, save_path=None, display_progress_disable=True, seed=seed)
    nn_.model = torch.nn.Sequential(*otaf.surrogate.get_custom_mlp_layers([4, 16, 8, 4, 1], activation_class=torch.nn.GELU))
    nn_.optimizer = torch.optim.AdamW(nn_.parameters(), lr=0.003, weight_decay=1e-4)
    otaf.surrogate.initialize_model_weights(nn_)
    nn_.scheduler = torch.optim.lr_scheduler.ExponentialLR(nn_.optimizer, 1.0001)
    nn_.loss_fn = torch.nn.MSELoss()
    nn_.train_model()
    print(tag, "epochs", len(nn_.history_loss), "best R2", nn_.best_metric, "failure share", (y < 0).mean(), "time", time.time() - t, flush=True)
    cp = dict(model_state_dict=nn_.model.state_dict(), architecture=["dim", "16", "8", "4", "1"], input_dim=4, model_name="model1_4_dof", input_normalization=True, output_normalization=True,
              normalization_metadata=dict(X_mean=nn_.X_mean.cpu(), y_mean=nn_.y_mean.cpu(), X_std=nn_.X_std.cpu(), y_std=nn_.y_std.cpu()))
    torch.save(cp, L.HERE / "ckpt" / f"{tag}.pth")
    net = L.load_ckpt(L.HERE / "ckpt" / f"{tag}.pth")
    record(tag, net, n_params=L.n_params(net), seed=seed, name_group="B1", epochs=len(nn_.history_loss), best_r2=nn_.best_metric, train_time_s=time.time() - t)
    del nn_, net; torch.cuda.empty_cache()
