"""Train and evaluate named candidates. Results: results/<name>_s<seed>.json, surfaces .npy, ckpt/<name>_s<seed>.pth; existing results are skipped.
Usage: python ex1_surrogate_study/run.py --stage STAGE [--seeds 0 1 2] [--n-surf 1000000] [--only NAME...]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import ex1lib as L
from configs import STAGES

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True); ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--n-surf", type=int, default=1_000_000); ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.set_num_threads(2)
if L.DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.4)
tb = L.TestBed()
print("test bed ready", flush=True)
for name, cfg in STAGES[a.stage].items():
    if a.only and name not in a.only:
        continue
    for seed in a.seeds:
        tag = f"{name}{a.tag}_s{seed}"
        f = L.HERE / "results" / f"{tag}.json"
        if f.exists():
            print("skip", tag, flush=True); continue
        print(f"=== {tag}: {cfg}", flush=True)
        for attempt in range(12):                    # the GPU is shared: wait out a foreign memory peak instead of dying
            try:
                net, info, hist = L.train(cfg, seed)
                L.save_ckpt(net, cfg, L.HERE / "ckpt" / f"{tag}.pth")
                np.save(L.HERE / "results" / f"{tag}_hist.npy", np.array(hist))
                res, P = L.evaluate(tb, net, a.n_surf)
                np.save(L.HERE / "results" / f"{tag}_surface.npy", P)
                res.update(name=name, seed=seed, cfg=cfg, train=info, timing=L.timing(net), oom_retries=attempt)
                f.write_text(json.dumps(L.jsonable(res), indent=1))
                print(f"--- {tag}: {info['train_time_s']:.0f}s train, val rmse {info['val_rmse_mm']:.2e} mm, Pf TT {res['TT']['pf']:.3e} RR {res['RR']['pf']:.3e} EE {res['EE']['pf']:.3e}; "
                      f"bias TT {res['TT']['bias_band']:+.2e} RR {res['RR']['bias_band']:+.2e}; surf med|e| {res['surface']['0.0']['med_abs']:.4f}; eval {res['surface_time_s']:.0f}s", flush=True)
                break
            except torch.OutOfMemoryError as ex:
                print(f"GPU OOM on {tag}, attempt {attempt}, waiting 120 s", flush=True)
                net = None; torch.cuda.empty_cache(); time.sleep(120)
            except Exception as ex:                  # a failed candidate must be reported, not silently lost
                (L.HERE / "results" / f"{tag}.FAILED").write_text(repr(ex)); print("FAILED", tag, ex, flush=True); break
        else:
            (L.HERE / "results" / f"{tag}.FAILED").write_text("GPU OOM, 12 attempts"); print("FAILED OOM", tag, flush=True)
        net = None; torch.cuda.empty_cache()
